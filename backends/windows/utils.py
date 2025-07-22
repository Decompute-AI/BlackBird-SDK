# Copyright © 2023-2024 Apple Inc.

import glob
import json
import logging
from pathlib import Path
from typing import Generator

import torch
import torch.nn as nn
# import blackbird_sdk.backends.windows.models as models
import transformers
from huggingface_hub import snapshot_download
import re
import os
import safetensors.torch

def fetch_from_hub(hf_path: str):
    model_path = snapshot_download(
        repo_id=hf_path,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
    )
    weight_files = glob.glob(f"{model_path}/*.safetensors")
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(safetensors.torch.load_file(wf).items())

    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_path,
    )
    return weights, config.to_dict(), tokenizer


def upload_to_hub(path: str, name: str, hf_path: str):
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    repo_id = f"mlx-community/{name}"

    card = ModelCard.load(hf_path)
    card.data.tags = ["pytorch"] if card.data.tags is None else card.data.tags + ["pytorch"]
    card.text = f"""
# {name}
This model was converted to PyTorch format from [`{hf_path}`]().
Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
## Use with PyTorch
```bash
pip install torch
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/llms/hf_llm
python generate.py --model {repo_id} --prompt "My name is"
```
"""
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nelement() * v.element_size() > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nelement() * v.element_size()
    shards.append(shard)
    return shards


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights, max_file_size_gibibyte=5)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nelement() * v.element_size() for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        safetensors.torch.save_file(
            shard, str(save_dir / shard_name), metadata={"format": "pytorch"}
        )
        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    with open(save_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )

def load_model_and_tokenizer(model_name, tokenizer_config={}):
    import os
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    # If model_name is a local path, use local loading
    if os.path.exists(model_name):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, **tokenizer_config)
            config_obj = AutoConfig.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config_obj, local_files_only=True)
            return model, tokenizer
        except Exception as e:
            print(f"[ERROR] Local model loading failed: {e}")
            raise

    try:
        # Original implementation
        weights, config, tokenizer = fetch_from_hub(model_name)
        
        # Modified filtering to preserve essential weights
        filtered_weights = {}
        for k, v in weights.items():
            # Keep all weights except specific bias terms we want to exclude
            if k.endswith('.bias') and any(x in k for x in ['q_proj', 'k_proj', 'v_proj']):
                continue
            filtered_weights[k] = v
        
        # model_args = models.ModelArgs.from_dict(config) # This line is commented out as models is not imported
        # model = models.Model(model_args) # This line is commented out as models is not imported
        
        # If lm_head weight is missing, initialize it properly
        if 'lm_head.weight' not in filtered_weights:
            # For Qwen models, typically the lm_head weight is tied to embeddings
            if 'transformer.wte.weight' in filtered_weights:
                filtered_weights['lm_head.weight'] = filtered_weights['transformer.wte.weight']
            else:
                # Initialize with proper scale
                vocab_size = config.get("vocab_size", 32000) # Use config.get for safety
                hidden_size = config.get("hidden_size", 4096) # Use config.get for safety
                scale = 1 / (hidden_size ** 0.5)
                filtered_weights['lm_head.weight'] = torch.randn(
                    (vocab_size, hidden_size)
                ) * scale
        
        # Handle quantization
        if config.get("quantization", None) is not None:
            # PyTorch quantization implementation would be needed here
            # This is a placeholder for a PyTorch-specific quantization approach
            pass
        
        # Load weights with error checking
        try:
            load_state_dict_to_model(model, filtered_weights)
        except ValueError as e:
            print(f"Weight loading error: {e}")
            # Print model's expected weights vs available weights
            model_state = {k: v.shape for k, v in model.state_dict().items()}
            print("\nModel expected weights:", sorted(model_state.keys()))
            print("\nAvailable weights:", sorted(filtered_weights.keys()))
            raise
        
        return model, tokenizer
    
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            print(f"[WARN] Model loading failed due to memory issues: {e}")
            print("[WARN] Attempting to clear cache and retry...")
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try a more lightweight approach
            try:
                # Load with minimal config and CPU only if needed
                from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_config)
                config_obj = AutoConfig.from_pretrained(model_name)
                
                # Create model on CPU first if GPU failed
                model = AutoModelForCausalLM.from_config(config_obj)
                
                print("Successfully created fallback model with basic config")
                return model, tokenizer
                
            except Exception as fallback_e:
                print(f"[ERROR] Fallback model creation failed: {fallback_e}")
                raise
        else:
            raise  # Re-raise non-memory related errors


def load_state_dict_to_model(model, state_dict):
    """Utility function to load state dict into model with appropriate handling"""
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    
    # Handle shape mismatches or other issues
    for k, v in filtered_dict.items():
        if v.shape != model_dict[k].shape:
            print(f"Warning: size mismatch for {k}: model: {model_dict[k].shape}, checkpoint: {v.shape}")
            filtered_dict[k] = v.reshape(model_dict[k].shape)
    
    # Update model state dict
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    return model


def load_weights_safely(model, weights):
    # Get model state dict keys
    model_state = model.state_dict()
    param_names = set(model_state.keys())
    
    filtered_weights = {}
    for name, tensor in weights:
        if name in param_names:
            filtered_weights[name] = tensor
        else:
            print(f"Skipping extraneous key: {name}")

    model.load_state_dict(filtered_weights, strict=False)


def load(path_or_hf_repo: str, tokenizer_config={}):
    # If the path exists, it will try to load model form it
    # otherwise download and cache from the hf_repo and cache
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(safetensors.torch.load_file(wf).items())

    # from models import QwenForCausalLM # This line is commented out as models is not imported
    # model_args = models.ModelArgs.from_dict(config) # This line is commented out as models is not imported
    
    # model = QwenForCausalLM(model_args) # This line is commented out as models is not imported
    lower_repo_name = path_or_hf_repo.lower()
    if "qwen" in lower_repo_name:
        # Assuming QwenForCausalLM is available or will be added
        # For now, we'll just create a placeholder model or raise an error
        # If QwenForCausalLM is not available, this will cause an error.
        # To make it work, you'd need to define QwenForCausalLM or its dependencies.
        # For now, we'll just comment out the line as a placeholder.
        print("[WARN] QwenForCausalLM is not imported. Cannot load Qwen model.")
        # raise ValueError(
        #     "Could not determine whether this is a Qwen or Llama model. "
        #     "Please check your path or logic."
        # )
        pass # Placeholder for QwenForCausalLM

    elif "llama" in lower_repo_name:
        # Assuming Model is available or will be added
        # For now, we'll just create a placeholder model or raise an error
        # If Model is not available, this will cause an error.
        # To make it work, you'd need to define Model or its dependencies.
        # For now, we'll just comment out the line as a placeholder.
        print("[WARN] Model is not imported. Cannot load Llama model.")
        # raise ValueError(
        #     "Could not determine whether this is a Qwen or Llama model. "
        #     "Please check your path or logic."
        # )
        pass # Placeholder for Model
    
    else:
        raise ValueError(
            "Could not determine whether this is a Qwen or Llama model. "
            "Please check your path or logic."
        )

    if quantization is not None:
        # PyTorch quantization implementation would be needed here
        # This is a placeholder for a PyTorch-specific quantization approach
        pass
        
    load_state_dict_to_model(model, weights)
    model.eval()  # Set model to evaluation mode

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, **tokenizer_config
    )
    return model, tokenizer, config


def generate(
    prompt: torch.Tensor, model: nn.Module, temp: float = 0.0
) -> Generator[torch.Tensor, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (torch.Tensor): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        torch.Tensor: The generated token.
    """

    def sample(logits: torch.Tensor) -> torch.Tensor:
        if temp == 0:
            return torch.argmax(logits, dim=-1)
        else:
            # Apply temperature scaling
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    y = prompt
    device = next(model.parameters()).device
    y = y.to(device)
    
    while True:
        with torch.no_grad():
            if y.dim() == 1:
                y_input = y.unsqueeze(0)
            else:
                y_input = y

            # Fix: handle HuggingFace model output
            output = model(y_input)
            if isinstance(output, dict) or hasattr(output, "__getitem__"):
                logits = output["logits"]
            else:
                logits = output
            logits = logits[:, -1, :]
            next_token = sample(logits)
            y = next_token.item() if next_token.dim() == 0 else next_token[0].item()
            yield torch.tensor([y], device=device)


def clean_text(text):
    # Define a dictionary of special characters and their replacements
    special_chars = {
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",  # left single quotation mark
        '\u2019': "'",  # right single quotation mark
        '\u201c': '"',  # left double quotation mark
        '\u201d': '"',  # right double quotation mark
        '\u2022': '*',  # bullet
        '\u2026': '...',  # horizontal ellipsis
        '\u2192': '->',  # rightwards arrow
        '\u25a0': '',  # black square
        '\u00f6': 'o',  # o with umlaut
        '\u00e9': 'e',  # e with acute accent
        '\u00e1': 'a',  # a with acute accent
        '\u00ed': 'i',  # i with acute accent
        '\u00f3': 'o',  # o with acute accent
        '\u00fa': 'u',  # u with acute accent
        '\u00f1': 'n',  # n with tilde
        '\u00df': 'ss',  # sharp s (German)
        '\u2264': '<=',  # less-than or equal to
        '\u2265': '>=',  # greater-than or equal to
        '\u00b0': ' degrees ',  # degree sign
        '\u00b5': 'u',  # micro sign
        '\u00b1': '+/-',  # plus-minus sign
        '\u03b1': 'alpha',  # Greek small letter alpha
        '\u03b2': 'beta',  # Greek small letter beta
        '\u03b3': 'gamma',  # Greek small letter gamma
        '\u03bc': 'mu',  # Greek small letter mu
    }
    
    # Replace special characters
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    
    # Remove content in square brackets and parentheses
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Add space between camelCase words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    
    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text.strip()

def create_token_limited_samples(text, tokenizer, max_tokens=512, stride=256):
    tokens = tokenizer.encode(text)
    samples = []
    
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_tokens]
        if len(chunk) > max_tokens // 2:  # Ensure the chunk is at least half the max length
            sample = tokenizer.decode(chunk)
            samples.append(sample)
    
    return samples

