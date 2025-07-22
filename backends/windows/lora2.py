# Copyright © 2023-2024 Apple Inc.

from unsloth import FastLanguageModel, is_bfloat16_supported
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import utils as lora_utils
from collections import OrderedDict
from models import LoRALinear
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.cuda.amp import autocast
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# Configure Unsloth defaults
UNSLOTH_DEFAULT_ARGS = {
    "max_seq_length": 2048,
    "dtype": torch.float16,
    "load_in_4bit": False,
    "use_gradient_checkpointing": "unsloth",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],
}

def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    # Generation args
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--add-eos-token",
        type=int,
        default=1,
        help="Enable add_eos_token for tokenizer",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        with open(path, "r") as fid:
            self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def load(args):
    print(args)
    def load_and_check(name):
        print(args)
        dataset_path = Path(args.get('data')) / f"{name}.jsonl"
        try:
            return Dataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset {dataset_path} ({e})")
            raise

    names = ("train", "valid")
    train, valid = (load_and_check(n) for n in names)

    should_train = args.get('train', True)
    
    if should_train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if should_train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    
    return train, valid


def report_gpu_memory(stage_name):
    """Report current GPU memory usage"""
    if torch.cuda.is_available():
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Log detailed info
        logger.info(f"GPU Memory [{stage_name}]: Allocated {allocated:.2f} MB, Reserved {reserved:.2f} MB, Peak {max_allocated:.2f} MB")
        print(f"GPU Memory [{stage_name}]: Allocated {allocated:.2f} MB, Reserved {reserved:.2f} MB, Peak {max_allocated:.2f} MB")
        
        # Return stats for optional use
        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated
        }
    return None


def loss(model, inputs, targets, lengths):
    """Calculate loss with mixed precision and Unsloth optimizations"""
    # Ensure all inputs are on the same device as the model
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)
    lengths = lengths.to(device)
    
    # Run model on inputs with mixed precision
    with autocast(enabled=torch.cuda.is_available()):
        outputs = model(inputs)
        # Handle both tensor and CausalLMOutputWithPast outputs
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Create mask for padding tokens
        length_mask = torch.arange(inputs.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Calculate cross entropy loss with mask
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        ).reshape_as(targets)
        
        ce_loss = ce_loss * length_mask
        ntoks = length_mask.sum()
        ce_loss = ce_loss.sum() / ntoks
    
    return ce_loss, ntoks


def iterate_batches(dset, tokenizer, batch_size, train=False):
    """Batch iterator with optional VRAM tracking for first batch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_batch = True
    
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tokenizer.encode(dset[int(indices[int(i + j)])]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # Check if any sequence is longer than 2048 tokens
            if max(lengths) > 2048:
                print(
                    "[WARNING] Some sequences are longer than 2048 tokens. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
                
            # Convert to PyTorch tensors and move to device
            batch = torch.tensor(batch_arr, dtype=torch.long, device=device)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            
            # Track VRAM for the first batch only
            if first_batch and torch.cuda.is_available():
                report_gpu_memory("First batch creation")
                first_batch = False
                
            yield batch[:, :-1], batch[:, 1:], lengths

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    model.eval()  # Set model to evaluation mode
    all_losses = []
    ntokens = 0

    # num_batches can be -1 to indicate the entire set
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    with torch.no_grad():
        for it, batch in zip(
            index_iterator,
            iterate_batches(dataset, tokenizer, batch_size),
        ):
            losses, toks = loss(model, *batch)
            all_losses.append((losses * toks).item())
            ntokens += toks.item()

    model.train()  # Set model back to training mode
    return np.sum(all_losses) / ntokens


def train(model, train_set, val_set, optimizer, loss, tokenizer, args):
    model.train()
    
    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(args.iters),
        iterate_batches(train_set, tokenizer, args.batch_size, train=True),
    ):
        # Forward pass
        optimizer.zero_grad()
        lvalue, toks = loss(model, *batch)
        
        # Backward pass
        lvalue.backward()
        optimizer.step()

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )

            start = time.perf_counter()

        # Save adapter weights if needed
        if (it + 1) % args.save_every == 0:
            # Save only trainable parameters (LoRA layers)
            state_dict = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    state_dict[name] = param.data
                    
            torch.save(state_dict, args.adapter_file)
            print(f"Iter {it + 1}: Saved adapter weights to {args.adapter_file}.")


def generate(model, prompt, tokenizer, args):
    """Generate text with VRAM tracking"""
    report_gpu_memory("Before generation")
    print(prompt, end="", flush=True)
    
    # Convert prompt to tensor
    device = next(model.parameters()).device
    prompt_tensor = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

    # Set model to evaluation mode
    model.eval()
    
    tokens = []
    skip = 0
    
    # Generate tokens
    with torch.no_grad():
        report_gpu_memory("Generation - first token")
        for i, (token_tensor, n) in enumerate(zip(
            lora_utils.generate(prompt_tensor, model, args.temp),
            range(args.max_tokens),
        )):
            if i % 10 == 0:  # Track every 10 tokens to avoid too much output
                report_gpu_memory(f"Generation - token {i}")
                
            token = token_tensor.item()
            if token == tokenizer.eos_token_id:
                break

            tokens.append(token)
            s = tokenizer.decode(tokens)
            if len(s) - skip > 1:
                print(s[skip:-1], end="", flush=True)
                skip = len(s) - 1
    
    report_gpu_memory("After generation")
    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return
    
def l2_norm(tensor):
    return torch.sqrt(torch.sum(tensor ** 2))

def tensor_difference(tensor_list_1, tensor_list_2):
    return [t1 - t2 for t1, t2 in zip(tensor_list_1, tensor_list_2)]

def weighted_sum(tensors, weights):
    return torch.sum(torch.stack([torch.sum(t) * w for t, w in zip(tensors, weights)]))

def compute_layer_wise_difference(list1, list2):
    weights = [0.25, 0.25, 0.25, 0.25]
    diffs = [tensor_difference(l1, l2) for l1, l2 in zip(list1, list2)]
    norms_1 = [l2_norm(weighted_sum(l1, weights)) for l1 in list1]
    norms_diff = [torch.sum(torch.stack([l2_norm(d) for d in diff])) for diff in diffs]
    rates = [nd / n1 for n1, nd in zip(norms_1, norms_diff)]
    return rates

def apply_layer_targeting(model, number_of_lora_layers_to_train, top_keys):
    """Apply layer targeting by freezing non-selected layers"""
    # Set requires_grad to False for non-selected layers
    for name, param in model.named_parameters():
        if 'lora' in name and name not in top_keys:
            param.requires_grad_(False)
    
    report_gpu_memory("After applying layer targeting")
    return model

def train_with_layer_selection(model, train_set, val_set, optimizer, loss, tokenizer, args):
    """Main training function with comprehensive VRAM tracking and Unsloth optimizations"""
    # Reset peak memory stats at the beginning
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    # Set model to training mode
    model.train()
    
    # Add GPU memory tracking
    report_gpu_memory("Initial state")
    
    # Ensure model is on the correct device
    device = next(model.parameters()).device
    
    # 1) If there's an existing adapter file to resume from, load it first
    resume_adapter_file = args.get('resume_adapter_file')
    if resume_adapter_file and os.path.exists(resume_adapter_file):
        print(f"Resuming from adapter file: {resume_adapter_file}")
        try:
            report_gpu_memory("Before loading adapter")
            state_dict = torch.load(resume_adapter_file, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded existing fine-tuned weights.")
            report_gpu_memory("After loading adapter")
        except Exception as e:
            print(f"Error in loading existing weights: {str(e)}")
            print("Continuing with base model.")
    else:
        print("No adapter file to resume from. Training from base model.")

    # 2) Count total LoRA layers and prepare for layer selection
    total_lora_layers = sum(1 for name, _ in model.named_parameters() if 'lora' in name)
    number_of_lora_layers_to_train = int(total_lora_layers / 2.5)  # Select ~40% of layers
    print(f"Total LoRA layers: {total_lora_layers}, Selected for training: {number_of_lora_layers_to_train}")
    
    report_gpu_memory("Before pre-training iteration")
    
    # 3) Pre-training pass to figure out which layers change most
    pre_weight = hook_lora_flan(model)
    print("Pre-training weights captured")
    
    # Make a single training iteration to measure layer changes
    for it, batch in zip(range(1), iterate_batches(train_set, tokenizer, args.get('batch_size'), train=True)):
        # Ensure all batch tensors are on the correct device
        batch = tuple(t.to(device) if hasattr(t, 'to') else t for t in batch)
        
        report_gpu_memory("Before optimizer step")
        optimizer.zero_grad()
        with autocast(enabled=torch.cuda.is_available()):
            lvalue, toks = loss(model, *batch)
        lvalue.backward()
        optimizer.step()
        report_gpu_memory("After optimizer step")

    # Clear GPU cache if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        report_gpu_memory("After cache clear")

    after_weight = hook_lora_flan(model)
    print("Post-training weights captured")
    
    # 4) Compare weights to find top layers
    report_gpu_memory("Before weight comparison")
    list1 = list(pre_weight.values())
    list2 = list(after_weight.values())
    keys = list(pre_weight.keys())
    
    # Calculate differences with improved numerical stability
    diffs = []
    for i in range(len(list1)):
        # Add a small epsilon to prevent division by zero
        norm_original = torch.norm(list1[i]).item()
        if norm_original < 1e-10:  # If norm is too small, use absolute change instead
            diffs.append(torch.norm(list2[i] - list1[i]).item())
        else:
            diffs.append(torch.norm(list2[i] - list1[i]).item() / norm_original)
    
    # Sort by difference and get top layers
    sorted_indices = np.argsort(diffs)[::-1]
    top_keys = [keys[i] for i in sorted_indices[:number_of_lora_layers_to_train]]
    
    # Apply layer targeting
    model = apply_layer_targeting(model, number_of_lora_layers_to_train, top_keys)
    
    # Report GPU memory after layer selection
    report_gpu_memory("After layer selection")
    
    # 5) Proceed with main fine-tuning using Unsloth's optimizations
    yield {'message': 'Model prepared for training'}

    losses = []
    n_tokens = 0
    yield {'message': 'Starting model fine-tuning'}

    # Setup gradient accumulation parameters
    batch_size = args.get('batch_size', 4)
    micro_batch_size = args.get('micro_batch_size', batch_size)
    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)
    
    print(f"Training with batch_size={batch_size}, micro_batch_size={micro_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
    report_gpu_memory("Before main training loop")

    # Training loop with Unsloth optimizations
    for it, batch in zip(range(args.get('iters')), iterate_batches(train_set, tokenizer, micro_batch_size, train=True)):
        # Track memory at the start of each iteration if it's a reporting step
        if (it + 1) % args.get('steps_per_report', 10) == 0:
            report_gpu_memory(f"Start of iteration {it+1}")
            
        # Ensure all batch tensors are on the correct device
        batch = tuple(t.to(device) if hasattr(t, 'to') else t for t in batch)
        
        # Forward and backward pass with mixed precision
        optimizer.zero_grad()
        with autocast(enabled=torch.cuda.is_available()):
            lvalue, toks = loss(model, *batch)
        
        # Track memory after forward pass on reporting steps
        if (it + 1) % args.get('steps_per_report', 10) == 0:
            report_gpu_memory(f"After forward pass {it+1}")
            
        lvalue.backward()
        
        # Track memory after backward pass on reporting steps
        if (it + 1) % args.get('steps_per_report', 10) == 0:
            report_gpu_memory(f"After backward pass {it+1}")
            
        optimizer.step()
        
        losses.append(lvalue.item())
        n_tokens += toks.item()
        
        # Report training progress
        if (it + 1) % args.get('steps_per_report') == 0:
            train_loss = np.mean(losses)
            logger.info(f"Step {it+1} training loss: {train_loss:.4f}")
            print(f"Step {it+1} training loss: {train_loss:.4f}")
            
            losses = []
            n_tokens = 0
        
        # Save checkpoints
        if (it + 1) % args.get('save_every') == 0:
            report_gpu_memory(f"Before saving checkpoint {it+1}")
            save_adapter(model, tokenizer, args.get('adapter_file'))
            print(f"Training checkpoint: Saved adapter weights.")
            report_gpu_memory(f"After saving checkpoint {it+1}")
            
        # Clear cache periodically
        if torch.cuda.is_available() and (it + 1) % 5 == 0:  # Clear every 5 steps
            report_gpu_memory(f"Before cache clear {it+1}")
            torch.cuda.empty_cache()
            report_gpu_memory(f"After cache clear {it+1}")

    # Final save
    report_gpu_memory("Before final save")
    save_adapter(model, tokenizer, args.get('adapter_file'))
    report_gpu_memory("After final save")
    
    # Final memory report
    if torch.cuda.is_available():
        print(f"Training completed: Saved adapter weights.")
        print(f"Final peak GPU memory: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # CRITICAL: Comprehensive memory cleanup after training completion
    # ──────────────────────────────────────────────────────────────────────────────────
    print("Performing comprehensive memory cleanup after training...")
    
    try:
        # Clear model from GPU memory
        if hasattr(model, 'cpu'):
            model.cpu()
        del model
        print("Model moved to CPU and deleted")
    except Exception as e:
        print(f"Error during model cleanup: {e}")
    
    try:
        # Clear tokenizer
        del tokenizer
        print("Tokenizer deleted")
    except Exception as e:
        print(f"Error during tokenizer cleanup: {e}")
    
    try:
        # Clear optimizer
        del optimizer
        print("Optimizer deleted")
    except Exception as e:
        print(f"Error during optimizer cleanup: {e}")
    
    # Force garbage collection
    import gc
    gc.collect()
    print("Garbage collection completed")
    
    # Clear CUDA cache and synchronize
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB allocated")
        print("GPU cache cleared and synchronized")
    
    print("Memory cleanup completed - ready for model loading")

def do_something():
    logger.info("This is a log message from lora2.py")


def load_model_and_tokenizer(model_name, tokenizer_config={}):
    """Load model and tokenizer with Unsloth optimizations and fallback mechanisms"""
    report_gpu_memory("Before model loading")
    
    print("Loading model with Unsloth optimizations")
    
    try:
        # ── PRIMARY ATTEMPT: Original Unsloth pipeline ───────────────────────────────────────
        # Use Unsloth's optimized model loading - separate base model loading from PEFT config
        base_model_kwargs = {
            "max_seq_length": UNSLOTH_DEFAULT_ARGS["max_seq_length"],
            "dtype": UNSLOTH_DEFAULT_ARGS["dtype"],
            "load_in_4bit": UNSLOTH_DEFAULT_ARGS["load_in_4bit"],
        }
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            **base_model_kwargs
        )
        
        report_gpu_memory("After tokenizer and model loading")
        
        # Apply LoRA adaptations while keeping Unsloth's optimizations
        peft_config = {
            "r": 8,  # Default LoRA rank, can be configured via args
            "lora_alpha": 32,
            "lora_dropout": 0.0,  # Optimized default
            "target_modules": UNSLOTH_DEFAULT_ARGS["target_modules"],
            "use_gradient_checkpointing": UNSLOTH_DEFAULT_ARGS["use_gradient_checkpointing"],
        }
        
        model = FastLanguageModel.get_peft_model(
            model,
            **peft_config
        )
        
        report_gpu_memory("After LoRA adaptation")
        
        # Ensure model is on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"Model loaded on device: {device}")
        report_gpu_memory("After moving model to device")
        
        return model, tokenizer
        
    # ➟ FALLBACK #1: GPU cannot hold quantized weights - try CPU offload
    except (torch.cuda.OutOfMemoryError, ValueError) as e:
        print(f"[WARN] GPU load failed → retrying with CPU off-load: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_config)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                llm_int8_enable_fp32_cpu_offload=True,
                torch_dtype="auto",
            )
            print("Successfully loaded model with CPU offload")
            return model, tokenizer
        except Exception as fallback_e:
            print(f"[WARN] CPU offload also failed: {fallback_e}")
            raise
    
    # ➟ FALLBACK #2: Weight files completely missing - build from config only
    except FileNotFoundError as e:
        print(f"[WARN] Model weights not found – building empty model from config: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_config)
            cfg = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_config(cfg)
            print("Successfully created model from config only")
            return model, tokenizer
        except Exception as config_e:
            print(f"[ERROR] Config-only model creation failed: {config_e}")
            raise
    
    except Exception as e:
        print(f"[ERROR] Unexpected error in model loading: {e}")
        raise


def apply_lora(model, lora_layers):
    """Apply LoRA with VRAM tracking"""
    report_gpu_memory("Before applying LoRA")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Apply LoRA to selected layers
    for i in range(len(model.model.layers) - lora_layers, len(model.model.layers)):
        layer = model.model.layers[i]
        if hasattr(layer, 'self_attn'):
            layer.self_attn.q_proj = LoRALinear.from_linear(layer.self_attn.q_proj)
            layer.self_attn.v_proj = LoRALinear.from_linear(layer.self_attn.v_proj)
            
        if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
            layer.block_sparse_moe.gate = LoRALinear.from_linear(layer.block_sparse_moe.gate)
    
    report_gpu_memory("After applying LoRA")
            
    return model 

def load_datasets(args):
    train_set, valid_set = load(args)
    return train_set, valid_set

def load_adapter(model, adapter_file):
    """
    Load adapter weights with improved error handling and format support
    """
    # Try loading different possible formats
    device = next(model.parameters()).device
    
    # First try the original .npz extension
    if Path(adapter_file).is_file():
        try:
            state_dict = torch.load(adapter_file, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Adapter loaded from {adapter_file} on device: {device}")
            return
        except Exception as e:
            print(f"Failed to load from {adapter_file}: {str(e)}")
    
    # Try with .pt extension
    adapter_file_pt = os.path.splitext(adapter_file)[0] + '.pt'
    if Path(adapter_file_pt).is_file():
        try:
            state_dict = torch.load(adapter_file_pt, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Adapter loaded from {adapter_file_pt} on device: {device}")
            return
        except Exception as e:
            print(f"Failed to load from {adapter_file_pt}: {str(e)}")
    
    # If we got here, neither file could be loaded
    print(f"Adapter files not found or not loadable: {adapter_file}, {adapter_file_pt}")
    print("Continuing with base model weights")

def train_model(model, train_set, valid_set, optimizer, loss, tokenizer, args):
    """Training wrapper with OOM handling"""
    try:
        # Use the original training function
        yield from train_with_layer_selection(model, train_set, valid_set, optimizer, loss, tokenizer, args)
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"[WARN] Training skipped due to GPU OOM: {e}")
        yield {'message': f'Training skipped due to GPU OOM: {e}', 'status': 'warning'}
        
        # Clear GPU cache and continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return  # Stop training loop, but don't crash the whole process
        
    except Exception as e:
        print(f"[ERROR] Training failed with error: {e}")
        yield {'message': f'Training failed: {e}', 'status': 'error'}
        raise  # Re-raise other exceptions

def save_adapter(model, tokenizer, adapter_file):
    """
    Save adapter weights and tokenizer configuration using PEFT's save_pretrained
    Args:
        model: The PEFT/LoRA model
        tokenizer: The tokenizer returned by FastLanguageModel
        adapter_file: Base path for saving adapter
    """
    # Save directly to the main directory instead of creating an adapter subdirectory
    adapter_dir = Path(os.path.dirname(adapter_file))
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights and adapter config using PEFT's save_pretrained
    model.save_pretrained(adapter_dir, safe_serialization=True)
    print(f"Adapter weights and config saved to {adapter_dir}")

    # Save tokenizer configuration
    tokenizer.save_pretrained(adapter_dir)
    print(f"Tokenizer configuration saved to {adapter_dir}")
    
    return adapter_dir

def hook_lora_flan(model):
    report_gpu_memory("Before capturing weights")
    
    LORA_weights_dict = {}
    device = next(model.parameters()).device  # Get model's device
    for name, param in model.named_parameters():
        if 'lora' in name:
            # Clone and ensure all weights are on the same device
            LORA_weights_dict[name] = param.clone().detach().to(device)
    
    report_gpu_memory("After capturing weights")
    return LORA_weights_dict

def load_model_for_inference(model_name, tokenizer_config={}):
    """Load model and tokenizer using standard HuggingFace transformers for inference"""
    report_gpu_memory("Before model loading")
    
    print("Loading model for inference using HuggingFace transformers")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_config)
    
    report_gpu_memory("After tokenizer loading")
    
    # Load model with standard HuggingFace
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    report_gpu_memory("After model loading")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    print(f"Model loaded on device: {model.device}")
    report_gpu_memory("After setup")
    
    return model, tokenizer
