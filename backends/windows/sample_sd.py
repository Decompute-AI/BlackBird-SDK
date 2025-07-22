import sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import load_prompt_cache
import time
import platform
import os

def setup_mps_device():
    """Configure MLX for optimal MPS performance"""
    # Enable MPS as the default backend
    mx.set_default_device(mx.gpu)
    
    # Set environment variables for Metal
    os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Get device info
    device_name = platform.processor()
    return f"Using MPS on {device_name}"

def optimize_for_apple_silicon():
    """Apply Apple Silicon specific optimizations"""
    # Enable MLX automatic memory management
    # mx.enable_auto_gc()
    
    # Enable compile-time optimizations
    mx.eval_compile = True
    
    # Enable fast math mode
    # mx.set_fast_math_mode(True)
    
    return "Applied Apple Silicon optimizations"

def generate_with_mps_optimization(
    model_name: str,
    prompt: str,
    max_tokens: int = 100,
    max_kv_size: int = 8192,
    cache_file: str = "prompt_cache.safetensors",
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = True
):
    mx.metal.set_cache_limit(2 * 1024 * 1024 * 1024)
    
    model, tokenizer = load(model_name)
    
    # Ensure prompt is not empty
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    # Format prompt properly
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    # Load or create cache
    prompt_cache = None
    if Path(cache_file).exists():
        prompt_cache = load_prompt_cache(cache_file)
        if verbose:
            print("Using cached prompt")
    
    start = time.time()
    try:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            verbose=verbose,
            max_kv_size=max_kv_size,
            prompt_cache=prompt_cache
        )
        if verbose:
            print(f"Generation time: {time.time() - start:.2f}s")
        return response
    except Exception as e:
        print(f"Generation error: {e}")
        return None

def main():
    # Test same prompt multiple times
    prompt = "Write a short story about a robot learning to paint."
    
    for i in range(3):
        print(f"\nRun {i+1}:")
        response = generate_with_mps_optimization(
            model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
            prompt=prompt,
            max_tokens=200,
            verbose=True
        )

if __name__ == "__main__":
    main()