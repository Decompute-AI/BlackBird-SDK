# Copyright © 2023 Apple Inc.

import inspect
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    model_type: str = None
    tie_word_embeddings: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        print(self.rope_scaling)
            
        if self.rope_scaling:
            if "rope_type" in self.rope_scaling and self.rope_scaling["rope_type"] == "llama3":
                pass
            elif "type" in self.rope_scaling and self.rope_scaling["type"] == "linear":
                required_keys = {"factor"}
                if not all(key in self.rope_scaling for key in required_keys):
                    raise ValueError(f"rope_scaling must contain keys {required_keys}")

                if self.rope_scaling["type"] != "linear":
                    raise ValueError("rope_scaling 'type' currently only supports 'linear'")
                
            else:
                raise ValueError("Unknown rope_scaling style...")

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        output_dims, input_dims = linear.weight.shape
        if hasattr(linear, 'bits'):  # For quantized linear layers
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self):
        linear = self.linear
        bias = hasattr(linear, 'bias') and linear.bias is not None
        weight = linear.weight
        is_quantized = hasattr(linear, 'bits')

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = torch.float16
            # Dequantization would need a custom implementation for PyTorch
            # This is a placeholder for actual dequantization code
            weight = weight  # Replace with actual dequantization

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.gamma * self.lora_b.T).to(dtype)
        lora_a = self.lora_a.T.to(dtype)
        with torch.no_grad():
            fused_linear.weight = nn.Parameter(weight + lora_b @ lora_a)
            if bias and linear.bias is not None:
                fused_linear.bias = nn.Parameter(linear.bias.clone())

        # Quantization would need to be implemented for PyTorch if needed
        
        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        lora_rank: int = 8,
        bias: bool = False,
        alpha: int = 64,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Scale for low-rank update
        self.gamma = alpha / math.sqrt(lora_rank) 

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = nn.Parameter(torch.empty(input_dims, lora_rank).uniform_(-scale, scale))
        self.lora_b = nn.Parameter(torch.zeros(lora_rank, output_dims))

    def forward(self, x):
        dtype = self.linear.weight.dtype
        if hasattr(self.linear, 'scales'):  # For quantized layers
            dtype = self.linear.scales.dtype
        device = x.device
        
        # Use parameters on the correct device without reassigning them
        y = self.linear(x.to(dtype))
        lora_a = self.lora_a.to(device)
        lora_b = self.lora_b.to(device)
        z = torch.matmul(torch.matmul(x, lora_a), lora_b)
        
        return y + self.gamma * z


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm


class RoPE(nn.Module):
    def __init__(self, dim, traditional=False, base=10000, scale=1.0):
        super().__init__()
        self.dim = dim
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.register_buffer('inv_freq', None, persistent=False)
        self._build_inv_freq()
        
    def _build_inv_freq(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, offset=0):
        seq_len = x.shape[-2]
        pos = torch.arange(seq_len, device=x.device).float() + offset
        pos = pos * self.scale
        
        # Create position embeddings
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if not self.traditional:
            emb_cos = torch.cos(emb).view(1, seq_len, 1, self.dim)
            emb_sin = torch.sin(emb).view(1, seq_len, 1, self.dim)
            
            # Reshape x to apply rotation
            x_reshaped = x.view(*x.shape[:-1], -1, 2)
            x1, x2 = x_reshaped[..., 0::2], x_reshaped[..., 1::2]
            
            # Apply rotation using complex number properties
            rotated_x1 = x1 * emb_cos - x2 * emb_sin
            rotated_x2 = x1 * emb_sin + x2 * emb_cos
            
            # Combine rotated parts
            rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).reshape(*x.shape)
            return rotated_x
        else:
            # Traditional RoPE implementation
            emb_cos = torch.cos(emb).view(1, seq_len, 1, self.dim)
            emb_sin = torch.sin(emb).view(1, seq_len, 1, self.dim)
            
            # Apply rotation directly to x
            return x * emb_cos + torch.roll(x, shifts=1, dims=-1) * emb_sin


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.repeats = n_heads // n_kv_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None
            else 1
        )
        self.rope = RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = torch.cat([key_cache, keys], dim=2)
            values = torch.cat([value_cache, values], dim=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Implement scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
            
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, values)
        
        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        inputs: torch.Tensor,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            # Create causal mask
            seq_len = h.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device) * float('-inf'), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            mask = mask.to(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache

# Minimal illustration
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = LlamaModel(args)

    def forward(self, inputs, cache=None):
        out, cache = self.model(inputs, cache)
        # final projection reuses embedding weight
        logits = torch.matmul(out, self.model.embed_tokens.weight.transpose(0, 1))
        return logits, cache


class QwenAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.repeats = n_heads // n_kv_heads

        head_dim = dim // n_heads
        self.scale = head_dim**-0.5

        # Qwen includes biases in these projections:
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)

        # Typically Qwen has bias=False in output projection, but verify:
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # Qwen can also use RoPE. Check how Qwen does it:
        # e.g., Qwen uses rope_theta=10000, or sometimes offset embeddings.
        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def forward(self, x, mask=None, cache=None):
        B, L, D = x.shape

        q = self.q_proj(x)  # shape [B, L, n_heads*head_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for multi-head attention
        q = q.reshape(B, L, self.n_heads, -1).transpose(1, 2)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(1, 2)

        # Apply RoPE + cache
        if cache:
            key_cache, value_cache = cache
            q = self.rope(q, offset=key_cache.shape[2])
            k = self.rope(k, offset=key_cache.shape[2])
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        # scaled_dot_product_attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores + mask
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        
        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(output), (k, v)

class QwenBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = QwenAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(self, x, mask=None, cache=None):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache

class QwenModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList([QwenBlock(args) for _ in range(args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        mask = None
        if h.shape[1] > 1:
            # Create causal mask
            seq_len = h.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device) * float('-inf'), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            mask = mask.to(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h, cache[i] = layer(h, mask, cache[i])

        return self.norm(h), cache

class QwenForCausalLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = QwenModel(args)

    def forward(self, inputs, cache=None):
        out, cache = self.model(inputs, cache)
        logits = torch.matmul(out, self.model.embed_tokens.weight.transpose(0, 1))
        return logits, cache
    
class CustomQwenForCausalLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = QwenModel(args)
        # Initialize lm_head with bias=False to ensure the bias parameter is created
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


    def forward(self, inputs, cache=None):
        out, cache = self.model(inputs, cache)
        # Use the linear layer directly, it will handle the bias internally
        logits = self.lm_head(out)
        return logits, cache

