# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import math
import time
from pathlib import Path
import copy
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils as lora_utils
from mlx.utils import tree_flatten
from models import LoRALinear


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_model_and_tokenizer(model_path):
    model, tokenizer, _ = lora_utils.load(model_path)
    return model, tokenizer

def setup_lora(model, lora_layers):
    model.freeze()
    for l in model.model.layers[len(model.model.layers) - lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)
    return model


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def load_datasets(args):
    def load_and_check(name):
        print("checking dataset patha and all")
        dataset_path = Path(args.data) / f"{name}.jsonl"
        print(dataset_path)
        try:
            return Dataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset {dataset_path} ({e})")
            raise

    names = ("train", "valid", "test")
    print("here in something")
    train, valid, test = (load_and_check(n) for n in names)
    print("done with above totally")

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
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
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0

    # num_batches can be -1 to indicate the entire set
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for it, batch in zip(
        index_iterator,
        iterate_batches(dataset, tokenizer, batch_size),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(model, train_set, val_set, optimizer, loss, tokenizer, args):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(args.iters),
        iterate_batches(train_set, tokenizer, args.batch_size, train=True),
    ):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

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
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )
            print(f"Iter {it + 1}: Saved adapter weights to {args.adapter_file}.")


def generate(model, prompt, tokenizer, args):
    print(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
        lora_utils.generate(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1
    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return
    
def l2_norm(tensor):
    return mx.sqrt(mx.sum(tensor ** 2))

def tensor_difference(tensor_list_1, tensor_list_2):
    return [t1 - t2 for t1, t2 in zip(tensor_list_1, tensor_list_2)]

def hook_lora_flan(model):
    LORA_weights_dict = {}
    for name, param in tree_flatten(model.parameters()):
        if 'lora' in name:
            LORA_weights_dict[name] = mx.array(param)
    return LORA_weights_dict

def weighted_sum(tensors, weights):
    return mx.sum(mx.stack([mx.sum(t) * w for t, w in zip(tensors, weights)]))

def compute_layer_wise_difference(list1, list2):
    weights = [0.25, 0.25, 0.25, 0.25]
    diffs = [tensor_difference(l1, l2) for l1, l2 in zip(list1, list2)]
    norms_1 = [l2_norm(weighted_sum(l1, weights)) for l1 in list1]
    norms_diff = [mx.sum(mx.stack([l2_norm(d) for d in diff])) for diff in diffs]
    rates = [nd / n1 for n1, nd in zip(norms_1, norms_diff)]
    return rates

def train_with_layer_selection(model, train_set, val_set, optimizer, loss, tokenizer, args):
    
    loss_value_and_grad = nn.value_and_grad(model, loss)
    
    # Count total LoRA layers
    total_lora_layers = sum(1 for name, _ in tree_flatten(model.parameters()) if 'lora' in name)
    print(f"Total LoRA layers: {total_lora_layers}")
    
    # Calculate number of layers to train (33%)
    number_of_lora_layers_to_train = int(total_lora_layers / 3)
    print(f"Number of LoRA layers to train: {number_of_lora_layers_to_train}")
    
    # Initial short training to compute gradient changes
    pre_weight = hook_lora_flan(model)
    print("initial training starts")
    for it, batch in zip(range(20), iterate_batches(train_set, tokenizer, args.batch_size, train=True)):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)
    
    after_weight = hook_lora_flan(model)
    
    # Compute layer-wise differences
    layer_wise = compute_layer_wise_difference(list(pre_weight.values()), list(after_weight.values()))
    new_weights_dict = dict(zip(after_weight.keys(), layer_wise))
    sorted_weights_dict = dict(sorted(new_weights_dict.items(), key=lambda item: item[1], reverse=True))
    top_keys = list(sorted_weights_dict.keys())[:number_of_lora_layers_to_train]
    
    # logic to only use the layers in top_keys to be used for main finetuninig 
    
    losses = []
    n_tokens = 0
    start = time.perf_counter()
    
    print("Main trainig starts")
    for it, batch in zip(range(args.iters), iterate_batches(train_set, tokenizer, args.batch_size, train=True)):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        optimizer.update(model, grad)

        for name, param in tree_flatten(model.trainable_parameters()):
            if name in top_keys:
                mx.eval(param, optimizer.state, lvalue)
        
        losses.append(lvalue.item())
        n_tokens += toks.item()
        
        # Report training loss
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)
            stop = time.perf_counter()
            print(f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                  f"It/sec {args.steps_per_report / (stop - start):.3f}, "
                  f"Tokens/sec {float(n_tokens) / (stop - start):.3f}")
            losses = []
            n_tokens = 0
            start = time.perf_counter()
        
        # Report validation loss
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(model, val_set, loss, tokenizer, args.batch_size, args.val_batches)
            print(f"Iter {it + 1}: Val loss {val_loss:.3f}, "
                  f"Val took {(time.perf_counter() - stop):.3f}s")
            start = time.perf_counter()
        
def save_adapter_weights(model, adapter_file):
    mx.savez(adapter_file, **dict(tree_flatten(model.trainable_parameters())))
