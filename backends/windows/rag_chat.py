# 1️⃣ Set up ffmpeg paths and audio utilities first
from pathlib import Path
import os
import sys
import subprocess

# Add UTF-8 encoding fix to prevent 'charmap' codec errors with emoji characters
# Configure UTF-8 encoding for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')
# Set environment variable for Python I/O encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
# On Windows, set console code page to UTF-8
if sys.platform == 'win32':
    try:
        # Use subprocess.run directly with shell=True for Windows
        subprocess.run('chcp 65001', check=False, shell=True)
        print("Console code page set to UTF-8 (65001)")
    except Exception as e:
        print(f"Warning: Could not set console code page: {e}")

def check_ffmpeg_version(ffmpeg_path):
    """Check ffmpeg version and available codecs."""
    try:
        # Check version
        version_result = subprocess.run([
            ffmpeg_path, "-version"
        ], capture_output=True, text=True)
        
        if version_result.returncode != 0:
            print(f"Warning: Could not get ffmpeg version: {version_result.stderr}")
            return False
            
        print("FFmpeg version info:")
        print(version_result.stdout.split('\n')[0])  # Print first line of version info
        
        # Check codecs
        codecs_result = subprocess.run([
            ffmpeg_path, "-codecs"
        ], capture_output=True, text=True)
        
        if codecs_result.returncode != 0:
            print(f"Warning: Could not get codec information: {codecs_result.stderr}")
            return False
            
        # Check for required codecs
        required_codecs = ['aac', 'm4a', 'wav', 'mp3']
        codecs_output = codecs_result.stdout.lower()
        
        missing_codecs = [codec for codec in required_codecs 
                         if f" {codec} " not in codecs_output]
        
        if missing_codecs:
            print(f"Warning: Missing required codecs: {', '.join(missing_codecs)}")
            return False
            
        print("All required codecs are available")
        return True
        
    except Exception as e:
        print(f"Error checking ffmpeg: {str(e)}")
        return False

# Set up ffmpeg paths
if getattr(sys, 'frozen', False):
    # Running in PyInstaller bundle
    print("Detected 'frozen' environment. Using sys._MEIPASS")
    base_path = sys._MEIPASS
else:
    # Running in normal Python environment
    print("Non-frozen environment. Using script directory")
    base_path = Path(__file__).resolve().parent

# Get the bin directory containing ffmpeg executables and DLLs
# BIN = Path("/usr/bin")  # .../ffmpeg/bin
# ffmpeg_path = BIN / "ffmpeg"
# ffprobe_path = BIN / "ffmpeg"


BIN = Path(base_path) / "ffmpeg" / "bin"  # .../ffmpeg/bin
ffmpeg_path = BIN / "ffmpeg.exe"
ffprobe_path = BIN / "ffprobe.exe"
# # Verify binaries exist
# if not ffmpeg_path.exists() or not ffprobe_path.exists():
#     raise RuntimeError(f"FFmpeg/FFprobe binaries not found in {BIN}")

# Set environment variables
os.environ["FFMPEG_BINARY"] = str(ffmpeg_path)
os.environ["FFPROBE_BINARY"] = str(ffprobe_path)
os.environ["PATH"] = f"{BIN}{os.pathsep}" + os.environ.get("PATH", "")  # make Windows search work

# Check FFmpeg version and codecs
if not check_ffmpeg_version(str(ffmpeg_path)):
    print("Warning: FFmpeg may not have all required codecs")

print("=== FFmpeg/FFprobe setup complete ===")

# 2️⃣ Now import and configure audio libraries
from pydub import AudioSegment
AudioSegment.converter = str(ffmpeg_path)
AudioSegment.ffprobe = str(ffprobe_path)

# Set up whisper with exact exe path
import whisper
from whisper import audio as w_audio
w_audio.FFMPEG_PATH = str(ffmpeg_path)  # tell Whisper the exact exe

# Import unsloth before transformers
from unsloth import FastLanguageModel, is_bfloat16_supported

# Rest of imports
import faiss
import numpy as np
import time
from pprint import pprint
import io
import shutil
import threading  
import csv
import re
from collections import Counter
from docx import Document
from nltk.stem import PorterStemmer
import wave

# Environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

# 1️⃣ Set paths first, then import pydub & whisper
from pathlib import Path
import os
import sys


# Set up pydub
from pydub import AudioSegment
AudioSegment.converter = str(ffmpeg_path)
AudioSegment.ffprobe = str(ffprobe_path)

# Set up whisper with exact exe path
import whisper
from whisper import audio as w_audio
w_audio.FFMPEG_PATH = str(ffmpeg_path)  # 2️⃣ tell Whisper the exact exe

# Import unsloth before transformers
from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import faiss
import numpy as np
import time
import sys
from pprint import pprint
import io
import shutil
import csv
import re
from pathlib import Path
from collections import Counter
from docx import Document
from nltk.stem import PorterStemmer
import subprocess

# Now import pydub after ffmpeg paths are set
from pydub import AudioSegment
AudioSegment.converter = str(ffmpeg_path)
AudioSegment.ffprobe = str(ffprobe_path)

import wave
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import glob
from langchain_community.document_loaders.parsers.language.javascript import JavaScriptSegmenter
import torch
import whisper

os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Generator, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

import torch.nn as nn
import json
import fitz  # PyMuPDF package
import pandas as pd
from ngrams_extract import extract_section_titles, extract_ngrams, extract_keywords_tfidf
from peft import PeftModel, PeftConfig

def setup_ffmpeg():
    """Set up FFmpeg path and environment variables for both system and pydub."""
    print("=== [setup_ffmpeg] Start ===")

    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        print("[setup_ffmpeg] Detected 'frozen' environment. Using sys._MEIPASS")
        base_path = sys._MEIPASS
    else:
        # Running in normal Python environment
        print("[setup_ffmpeg] Non-frozen environment. Using __file__ directory.")
        base_path = os.path.dirname(os.path.abspath(__file__))

    print(f"[setup_ffmpeg] base_path initially: {base_path}")
    print(f"[setup_ffmpeg] base_path forced to sys._MEIPASS: {base_path}")

    # Construct ffmpeg_path using os.path.join for proper Windows path handling
    ffmpeg_path = os.path.join(base_path, 'ffmpeg', 'bin', 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg')
    print(f"[setup_ffmpeg] Constructed ffmpeg_path = {ffmpeg_path}")

    # Check existence of ffmpeg_path
    print(f"[setup_ffmpeg] Checking if ffmpeg_path exists: {ffmpeg_path}")
    if not os.path.exists(ffmpeg_path):
        raise RuntimeError(f"[setup_ffmpeg] FFmpeg not found at {ffmpeg_path}")
    else:
        print("[setup_ffmpeg] FFmpeg file exists. Proceeding.")

    # Create a directory in a known location for FFmpeg
    local_ffmpeg_dir = os.path.join(os.path.expanduser('~'), 'ffmpeg_temp_2')
    print(f"[setup_ffmpeg] Creating local_ffmpeg_dir: {local_ffmpeg_dir}")
    os.makedirs(local_ffmpeg_dir, exist_ok=True)

    # Copy FFmpeg to the new location with proper Windows extension
    local_ffmpeg_path = os.path.join(local_ffmpeg_dir, 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg')
    print(f"[setup_ffmpeg] Copying from {ffmpeg_path} to {local_ffmpeg_path}")
    
    # If file exists, try to remove it first on Windows
    if sys.platform == 'win32' and os.path.exists(local_ffmpeg_path):
        try:
            os.remove(local_ffmpeg_path)
        except Exception as e:
            print(f"[setup_ffmpeg] Warning: Could not remove existing file: {e}")
    
    shutil.copy2(ffmpeg_path, local_ffmpeg_path)
    print("[setup_ffmpeg] Copy completed.")

    # Set permissions - skip chmod on Windows
    if sys.platform != 'win32':
        print(f"[setup_ffmpeg] Setting chmod 0o777 on {local_ffmpeg_path}")
        os.chmod(local_ffmpeg_path, 0o777)

    # Set environment variables and pydub configurations
    print(f"[setup_ffmpeg] Setting os.environ['FFMPEG_BINARY'] = {local_ffmpeg_path}")
    os.environ["FFMPEG_BINARY"] = local_ffmpeg_path

    print(f"[setup_ffmpeg] AudioSegment.converter = {local_ffmpeg_path}")
    AudioSegment.converter = local_ffmpeg_path

    # Add the directory to PATH
    old_path = os.environ.get("PATH", "")
    new_path = f"{local_ffmpeg_dir}{os.pathsep}{old_path}"
    os.environ["PATH"] = new_path
    print(f"[setup_ffmpeg] Updated PATH: {os.environ['PATH']}")

    print("=== [setup_ffmpeg] End ===")
    return local_ffmpeg_path


def setup_ffprobe():
    """Set up FFprobe path and environment variables for both system and pydub."""
    print("=== [setup_ffprobe] Start ===")

    if getattr(sys, 'frozen', False):
        print("[setup_ffprobe] Detected 'frozen' environment. Using sys._MEIPASS")
        base_path = sys._MEIPASS  # type: ignore
    else:
        print("[setup_ffprobe] Non-frozen environment. Using __file__ directory.")
        base_path = os.path.dirname(os.path.abspath(__file__))

    print(f"[setup_ffprobe] base_path initially: {base_path}")
    print(f"[setup_ffprobe] base_path forced to sys._MEIPASS: {base_path}")

    # Construct ffprobe_path using os.path.join for proper Windows path handling
    ffprobe_path = os.path.join(base_path, 'ffmpeg', 'bin', 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe')
    print(f"[setup_ffprobe] Constructed ffprobe_path = {ffprobe_path}")

    # Check existence of ffprobe_path
    print(f"[setup_ffprobe] Checking if ffprobe_path exists: {ffprobe_path}")
    if not os.path.exists(ffprobe_path):
        raise RuntimeError(f"[setup_ffprobe] FFprobe not found at {ffprobe_path}")
    else:
        print("[setup_ffprobe] FFprobe file exists. Proceeding.")

    # Create a directory in a known location for FFprobe
    local_ffprobe_dir = os.path.join(os.path.expanduser('~'), 'ffmpeg_temp_2')
    print(f"[setup_ffprobe] Creating local_ffprobe_dir: {local_ffprobe_dir}")
    os.makedirs(local_ffprobe_dir, exist_ok=True)

    # Copy FFprobe to the new location with proper Windows extension
    local_ffprobe_path = os.path.join(local_ffprobe_dir, 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe')
    print(f"[setup_ffprobe] Copying from {ffprobe_path} to {local_ffprobe_path}")
    
    # If file exists, try to remove it first on Windows
    if sys.platform == 'win32' and os.path.exists(local_ffprobe_path):
        try:
            os.remove(local_ffprobe_path)
        except Exception as e:
            print(f"[setup_ffprobe] Warning: Could not remove existing file: {e}")
    
    shutil.copy2(ffprobe_path, local_ffprobe_path)
    print("[setup_ffprobe] Copy completed.")

    # Set permissions - skip chmod on Windows
    if sys.platform != 'win32':
        print(f"[setup_ffprobe] Setting chmod 0o777 on {local_ffprobe_path}")
        os.chmod(local_ffprobe_path, 0o777)

    # Set environment variables and pydub configurations
    print(f"[setup_ffprobe] Setting os.environ['FFPROBE_BINARY'] = {local_ffprobe_path}")
    os.environ["FFPROBE_BINARY"] = local_ffprobe_path

    print(f"[setup_ffprobe] AudioSegment.ffprobe = {local_ffprobe_path}")
    AudioSegment.ffprobe = local_ffprobe_path

    # Add the directory to PATH
    old_path = os.environ.get("PATH", "")
    new_path = f"{local_ffprobe_dir}{os.pathsep}{old_path}"
    os.environ["PATH"] = new_path
    print(f"[setup_ffprobe] Updated PATH: {os.environ['PATH']}")

    print("=== [setup_ffprobe] End ===")
    return local_ffprobe_path


UPLOAD_FOLDER = os.path.expanduser('~/Documents/Decompute-Files/uploads')

def quantize(model: nn.Module, group_size: int, bits: int) -> nn.Module:
    """
    Applies quantization to the model weights using PyTorch quantization.

    Args:
        model (nn.Module): model to be quantized.
        group_size (int): group size for quantization.
        bits (int): bits per weight for quantization.

    Returns:
        nn.Module: The quantized model
    """
    # In PyTorch, we would use torch.quantization APIs
    # This is a simplified implementation - production code would use 
    # PyTorch's quantization workflow with calibration and proper observers

    # For static quantization we would typically do:
    # 1. Prepare model for quantization
    # 2. Calibrate with a representative dataset
    # 3. Convert to quantized model
    
    # For dynamic quantization:
    if bits == 8:
        # Simple dynamic quantization for linear layers
        try:
            import torch.quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Returning original model instead")
            return model
    else:
        # For other bit configurations, we'd need more complex approaches
        # like custom quantizers or int4 quantization libraries
        print(f"PyTorch built-in quantization only supports 8-bit, requested {bits}-bit")
        print("Returning original model without quantization")
        return model

# python_loader.py
import tokenize
from pathlib import Path
from typing import Union
from langchain_community.document_loaders.text import TextLoader

class PythonLoader(TextLoader):
    """Load Python files, respecting any non-default encoding if specified."""

    def __init__(self, file_path: Union[str, Path]):
        # Detect and use the Python file's declared encoding
        with open(file_path, "rb") as f:
            encoding, _ = tokenize.detect_encoding(f.readline)
        super().__init__(file_path=file_path, encoding=encoding)


@dataclass
class ChatSetup:
    system: str
    history: List[Dict[str, str]] = ()

    def session(self):
        for el in self.history:
            if "question" not in el.keys() or "answer" not in el.keys():
                raise ValueError("Each element in history must contain a question and an answer.")
        return Session(
            questions=[el["question"] for el in self.history],
            answers=[el["answer"] for el in self.history]
        )

class Session:
    def __init__(self, questions=None, answers=None):
        self.questions = questions or []
        self.answers = answers or []

    def add_question(self, question):
        self.questions.append(question)

    def add_answer(self, answer):
        self.answers.append(answer)

    def reset(self):
        self.questions = []
        self.answers = []

class ChatHistory:
    def __init__(self, max_history=3, summarizer=None):
        """
        Args:
            max_history (int): Number of recent user-assistant pairs to keep verbatim.
            summarizer (callable or None): A function or model method that takes a list of strings
                                           and returns a short summary. If None, summarization is skipped.
        """
        self.history = []      # List of {"user": "...", "assistant": "..."}
        self.summary = ""      # A rolling summary of older conversation
        self.max_history = max_history
        self.summarizer = summarizer

    def add_interaction(self, user_input: str, assistant_response: str):
        """
        Add a new user-assistant turn. If the history grows beyond `max_history`,
        summarize older turns and store them in `self.summary`.
        """
        self.history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        if len(self.history) > self.max_history:
            # Summarize the oldest chunk(s) of conversation, except for the last few
            older_part = self.history[:-self.max_history]
            self.history = self.history[-self.max_history:]  # Keep only last few
            if self.summarizer:
                # Convert older_part into a text block
                older_text_blocks = []
                for turn in older_part:
                    older_text_blocks.append(f"User asked: {turn['user']}")
                    older_text_blocks.append(f"Assistant answered: {turn['assistant']}")
                older_text = "\n".join(older_text_blocks)
                # Summarize
                short_summary = self.summarizer(older_text)
                # Append to our existing summary
                self.summary += f"\n{short_summary}\n"
            else:
                # If no summarizer is provided, just store some minimal info
                minimal_info = "Older conversation was truncated (no summarizer set)."
                self.summary += f"\n{minimal_info}\n"

    def get_context_string(self, user_input: str) -> str:
        """
        Combine the stored summary with the last few messages and the current user input.
        """
        context_lines = []
        
        # Include the summary if any
        if self.summary.strip():
            context_lines.append("Summary of previous discussion:")
            context_lines.append(self.summary)
        # Include the last few turns verbatim
        for interaction in self.history:
            context_lines.append(f"Human: {interaction['user']}")
            context_lines.append(f"Assistant: {interaction['assistant']}")
        # Finally, append the *new* user query at the end
        context_lines.append(f"Human: {user_input}")
        return "\n".join(context_lines)

    def get_recent_interactions(self, max_count=3):
        """
        Get the most recent interactions from history.
        
        Args:
            max_count (int): Maximum number of recent interactions to return
            
        Returns:
            List of tuples (user_input, assistant_response)
        """
        # Return at most max_count recent interactions as (user, assistant) tuples
        recent = []
        for interaction in self.history[-max_count:]:
            recent.append((interaction["user"], interaction["assistant"]))
        return recent

    def clear(self):
        """Hard reset of all stored conversation."""
        self.history = []
        self.summary = ""

def simple_summarizer(text: str) -> str:
    """
    Example: Summaries with your same model or a simpler/hard-coded approach.
    This is just a stub. In real usage, you'd call a separate smaller LLM or
    a summarization pipeline, e.g.:
        summarizer = pipeline("summarization", model="some-summarizer-model")
        return summarizer(text, max_length=60, do_sample=False)
    """
    # Minimal naive approach: just truncate or do a random fake summary
    # You can replace this with an actual LLM call:
    lines = text.split("\n")
    if len(lines) > 5:
        return "A short summary of older conversation: " + " | ".join(lines[:5]) + "..."
    return "A short summary of older conversation: " + text

import psutil
import gc
import psutil

def get_memory_usage():
    """Get current memory usage in MB (process and system)"""
    # Get the memory info from the current process
    process = psutil.Process()
    process_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Get system-wide memory info
    system_memory = psutil.virtual_memory()
    print("system memory value is ")
    print(system_memory)
    total_memory = system_memory.total / 1024 / 1024  # MB
    used_memory = system_memory.used / 1024 / 1024  # MB
    available_memory = system_memory.available / 1024 / 1024  # MB
    percent_used = system_memory.percent  # %
    
    return {
        'process_memory_mb': process_memory,
        'total_system_memory_mb': total_memory,
        'used_system_memory_mb': used_memory,
        'available_system_memory_mb': available_memory,
        'system_memory_percent': percent_used
    }

def select_batch_size():
    mem_info = get_memory_usage()
    # Convert available memory in MB to GB
    available_mem_gb = mem_info['total_system_memory_mb'] / 1024
    print(available_mem_gb)
    if available_mem_gb < 10:
        batch_size = 2
    elif available_mem_gb > 30:
        batch_size = 64
    elif available_mem_gb > 20:
        batch_size = 32
    elif available_mem_gb > 15:
        batch_size = 8
    else:
        batch_size = 2
    
    return batch_size

class RAGChat:
    _instance = None
    _is_initialized = False
    SPECIAL_CHARS = {
        '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'", '\u201c': '"',
        '\u201d': '"', '\u2022': '*', '\u2026': '...', '\u2192': '->', '\u25a0': '',
        '\u00f6': 'o', '\u00e9': 'e', '\u00e1': 'a', '\u00ed': 'i', '\u00f3': 'o',
        '\u00fa': 'u', '\u00f1': 'n', '\u00df': 'ss', '\u2264': '<=', '\u2265': '>=',
        '\u00b0': ' degrees ', '\u00b5': 'u', '\u00b1': '+/-', '\u03b1': 'alpha',
        '\u03b2': 'beta', '\u03b3': 'gamma', '\u03bc': 'mu'
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance and thoroughly clean up memory - optimized for MLX on Mac"""
        if cls._instance is not None:
            # Explicitly delete large objects
            if hasattr(cls._instance, 'model'):
                # For MLX models, we need to ensure all weights are properly released
                if hasattr(cls._instance.model, 'parameters'):
                    for param_name in list(cls._instance.model.parameters().keys()):
                        if hasattr(cls._instance.model, param_name):
                            setattr(cls._instance.model, param_name, None)
                del cls._instance.model
                
            if hasattr(cls._instance, 'tokenizer'):
                del cls._instance.tokenizer
                
            if hasattr(cls._instance, 'embedding_model'):
                # For sentence-transformers models
                if hasattr(cls._instance.embedding_model, 'clear_cache'):
                    cls._instance.embedding_model.clear_cache()
                del cls._instance.embedding_model
                
            # Handle FAISS index
            if hasattr(cls._instance, 'index'):
                if hasattr(cls._instance.index, 'reset'):
                    cls._instance.index.reset()
                del cls._instance.index
                
            # Clear all large data collections
            for attr in ['all_chunks', 'all_cleaned_texts', 'chunk_mapping', 
                         'financial_entity_cache', 'chat_history']:
                if hasattr(cls._instance, attr):
                    setattr(cls._instance, attr, None)
        
        # Force multiple garbage collection cycles for more thorough cleanup
        import gc
        import sys
        
        # Run multiple GC cycles
        for _ in range(3):
            gc.collect()
            
        # On macOS, attempt additional memory cleanup if possible
        if sys.platform == 'darwin':
            try:
                import subprocess
                subprocess.run(['sudo', 'purge'], capture_output=True, check=False)
            except:
                pass
        
        # Run one final garbage collection
        gc.collect()
        
        # Reset singleton instance
        cls._instance = None
        cls._is_initialized = False
        print("RAGChat instance has been reset and memory cleaned up for MLX on Mac.")
    

    def __init__(self, model_name, pdf_path, agent, max_tokens=2048, temperature=0.4, use_finetuning=False,
                 chunk_size=1000, chunk_overlap=200, max_chunks=3, force_reinit=False , load_existing=False , finance_toggle=False):
        # Skip initialization if already initialized unless force_reinit is True
        if self._is_initialized and not force_reinit:
            self.model_name = model_name
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.model_folder = pdf_path
            self.finance_toggle=finance_toggle
            self.agent = agent
            self.embedding_model_name = "hkunlp/instructor-large"
            self.dimension = 768
            self.financial_entity_cache = {}
            self.max_tokens = max_tokens
            self.chat_history = ChatHistory(max_history=1, summarizer=simple_summarizer)
            self.setup_file_paths(pdf_path)
            self._initialize_model(use_finetuning)
            final_memory = get_memory_usage()
            print(f"model loading 2nd time process memory: {final_memory['process_memory_mb']:.2f} MB")
            print(f"model loading 2nd time System memory: {final_memory['used_system_memory_mb']:.2f} MB / {final_memory['total_system_memory_mb']:.2f} MB ({final_memory['system_memory_percent']:.1f}%)")
            if load_existing:
                self._load_cached_index()
            return

        # Store configuration
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.model_folder = pdf_path
        self.agent = agent
        self.chat_history = ChatHistory(max_history=2, summarizer=simple_summarizer)
        self.embedding_model_name = "hkunlp/instructor-large"
        self.dimension = 768
        self.financial_entity_cache = {}
        self.finance_toggle=finance_toggle
        # Set up file paths
        initial_memory = get_memory_usage()
        print(f"Initial process memory: {initial_memory['process_memory_mb']:.2f} MB")
        print(f"System memory: {initial_memory['used_system_memory_mb']:.2f} MB / {initial_memory['total_system_memory_mb']:.2f} MB ({initial_memory['system_memory_percent']:.1f}%)")
        
        if load_existing:
            self._initialize_model(use_finetuning)
            self._initialize_rag_components()
            self._load_cached_index()
        else:
            # Full initialization for new files
            self.setup_file_paths(pdf_path)
            final_memory = get_memory_usage()
            print(f"Initial process memory: {final_memory['process_memory_mb']:.2f} MB")
            print(f"System memory: {final_memory['used_system_memory_mb']:.2f} MB / {final_memory['total_system_memory_mb']:.2f} MB ({final_memory['system_memory_percent']:.1f}%)")
            
            self._initialize_rag_components()
            final_memory = get_memory_usage()
            print(f"loading embedding model Initial process memory: {final_memory['process_memory_mb']:.2f} MB")
            print(f"System memory: {final_memory['used_system_memory_mb']:.2f} MB / {final_memory['total_system_memory_mb']:.2f} MB ({final_memory['system_memory_percent']:.1f}%)")

            ##possibly used only for rag
            self._initialize_model(use_finetuning)
            final_memory = get_memory_usage()
            # print(f"loading main model Initial process memory: {final_memory['process_memory_mb']:.2f} MB")
            # print(f"System memory: {final_memory['used_system_memory_mb']:.2f} MB / {final_memory['total_system_memory_mb']:.2f} MB ({final_memory['system_memory_percent']:.1f}%)")
            # # self._process_pdf(pdf_path)
        # self._initialize_chat_setup()
        # Mark as initialized
        RAGChat._is_initialized = True
        
        # Compile model if using PyTorch >= 2.1
        if use_finetuning and torch.__version__ >= "2.1" and not getattr(self.model, "_is_compiled", False):
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.model._is_compiled = True
            print("Model compiled with torch.compile()")

    def setup_file_paths(self, pdf_path):
        """Setup directory structure for saving files"""
        self.upload_folder = os.path.expanduser('~/Documents/Decompute-Files/uploads')
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # Create directory based on PDF filename
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.output_dir = os.path.join(self.upload_folder, self.pdf_name)
        os.makedirs(self.output_dir, exist_ok=True)


    def clean_text(self, text: str) -> str:
        """Clean text by removing/replacing special characters"""
        # Replace special characters
        for special_char, replacement in self.SPECIAL_CHARS.items():
            text = text.replace(special_char, replacement)
        
        # Additional cleaning steps
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
        text = text.strip()
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text


    def _initialize_model(self, use_finetuning=False):
        """Initialize or update the model with comprehensive error handling and fallback options."""
        try:
            # Import torch directly in the function scope
            import torch
            from unsloth import FastLanguageModel, is_bfloat16_supported
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel, PeftConfig
            import os
            from pathlib import Path
            
            print("Initializing/updating model...")
            
            # Clear any existing GPU memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Determine appropriate dtype
            if torch.cuda.is_available():
                dtype = torch.float16
            elif is_bfloat16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16  # Fall back to float16 even on CPU
                
            if use_finetuning and hasattr(self, 'model_folder'):
                try:
                    # Look for adapter files directly in the model folder
                    adapter_dir = Path(self.model_folder)
                    
                    if not adapter_dir.exists():
                        print(f"Adapter directory not found at {adapter_dir}, falling back to base model")
                        use_finetuning = False
                    else:
                        print(f"Loading base model and adapter from {adapter_dir}")
                        
                        # Check if adapter files exist in the directory
                        adapter_model_file = adapter_dir / "adapter_model.safetensors"
                        adapter_config_file = adapter_dir / "adapter_config.json"
                        
                        if adapter_model_file.exists() and adapter_config_file.exists():
                            print(f"Found traditional adapter files at {adapter_dir}")
                            
                            try:
                                # First load base model
                                base_model, tokenizer = FastLanguageModel.from_pretrained(
                                    self.model_name,  # base model
                                    max_seq_length=2048,
                                    dtype=dtype,
                                    device_map="auto",
                                    use_gradient_checkpointing="unsloth"
                                )
                                
                                # Use PeftModel.from_pretrained to load the adapter
                                try:
                                    model = PeftModel.from_pretrained(base_model, adapter_dir)
                                    print(f"Successfully loaded adapter using PeftModel.from_pretrained")
                                except Exception as e:
                                    print(f"Failed to load adapter with PeftModel: {str(e)}")
                                    # Fall back to trying to load all safetensors files directly
                                    for file in adapter_dir.glob("*.safetensors"):
                                        try:
                                            state_dict = torch.load(file, map_location=base_model.device)
                                            base_model.load_state_dict(state_dict, strict=False)
                                            print(f"Loaded weights from {file}")
                                        except Exception as load_err:
                                            print(f"Failed to load {file}: {str(load_err)}")
                                    model = base_model  # Use the base model with any weights we were able to load
                            
                            except torch.cuda.OutOfMemoryError as oom_e:
                                print(f"CUDA Out of Memory loading fine-tuned model: {str(oom_e)}")
                                print("Falling back to base model without fine-tuning")
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                use_finetuning = False
                                
                            except RuntimeError as rt_e:
                                if ("out of memory" in str(rt_e).lower() or 
                                    "insufficient memory" in str(rt_e).lower() or
                                    "dispatch" in str(rt_e).lower()):
                                    print(f"Memory issue loading fine-tuned model: {str(rt_e)}")
                                    print("Falling back to base model without fine-tuning")
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                                    use_finetuning = False
                                else:
                                    raise
                        
                        else:
                            try:
                                # If we don't find the traditional adapter files, try FastLanguageModel's load_adapter method
                                print(f"No traditional adapter files found, trying FastLanguageModel's load_adapter method")
                                model, tokenizer = FastLanguageModel.from_pretrained(
                                    self.model_name,  # base model
                                    max_seq_length=2048,
                                    dtype=dtype,
                                    device_map="auto",
                                    use_gradient_checkpointing="unsloth"
                                )
                                
                                # Look for any adapter-related files
                                adapter_files = list(adapter_dir.glob("adapter*")) + list(adapter_dir.glob("*.safetensors"))
                                if adapter_files:
                                    print(f"Found potential adapter files: {[f.name for f in adapter_files]}")
                                    try:
                                        # Try using FastLanguageModel's built-in adapter loading
                                        model = FastLanguageModel.get_peft_model(
                                            model,
                                            r=8,  # Default LoRA rank
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                                                          "gate_proj", "up_proj", "down_proj"],
                                        )
                                        # Now load the adapter weights
                                        print(f"Trying to load adapter from {adapter_dir}")
                                        model.load_adapter(adapter_dir)
                                    except Exception as e:
                                        print(f"Failed to use FastLanguageModel adapter loading: {str(e)}")
                                        # Just continue with the base model
                                else:
                                    print(f"No adapter files found in {adapter_dir}")
                                    
                            except torch.cuda.OutOfMemoryError as oom_e:
                                print(f"CUDA Out of Memory during adapter loading: {str(oom_e)}")
                                print("Falling back to base model without fine-tuning")
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                use_finetuning = False
                                
                            except RuntimeError as rt_e:
                                if ("out of memory" in str(rt_e).lower() or 
                                    "insufficient memory" in str(rt_e).lower() or
                                    "dispatch" in str(rt_e).lower()):
                                    print(f"Memory issue during adapter loading: {str(rt_e)}")
                                    print("Falling back to base model without fine-tuning")
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                                    use_finetuning = False
                                else:
                                    raise
                                    
                except Exception as adapter_e:
                    print(f"Adapter loading failed: {str(adapter_e)}")
                    print("Falling back to base model without fine-tuning")
                    use_finetuning = False
            
            # Load regular model (either fallback or original request)
            if not use_finetuning:
                try:
                    # Regular model loading without fine-tuning
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        self.model_name,
                        max_seq_length=2048,
                        dtype=dtype,
                        device_map="auto",
                        use_gradient_checkpointing="unsloth"
                    )
                    
                except torch.cuda.OutOfMemoryError as oom_e:
                    print(f"CUDA Out of Memory loading base model: {str(oom_e)}")
                    print("Trying with reduced memory settings...")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    # Try with more conservative settings
                    try:
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            self.model_name,
                            max_seq_length=1024,  # Reduced sequence length
                            dtype=dtype,
                            device_map="auto",
                            load_in_4bit=True,  # Force 4-bit quantization
                            use_gradient_checkpointing="unsloth"
                        )
                        print("Successfully loaded model with reduced memory settings")
                        
                    except Exception as final_e:
                        print(f"Failed to load model even with reduced settings: {str(final_e)}")
                        # Try CPU fallback
                        try:
                            print("Attempting CPU fallback...")
                            model, tokenizer = FastLanguageModel.from_pretrained(
                                self.model_name,
                                max_seq_length=1024,
                                dtype=torch.float32,
                                device_map="cpu"
                            )
                            print("Successfully loaded model on CPU")
                        except Exception as cpu_e:
                            print(f"CPU fallback also failed: {str(cpu_e)}")
                            raise RuntimeError(f"Failed to load model on GPU or CPU: {str(cpu_e)}")
                
                except RuntimeError as rt_e:
                    if ("out of memory" in str(rt_e).lower() or 
                        "insufficient memory" in str(rt_e).lower() or
                        "dispatch" in str(rt_e).lower()):
                        print(f"Memory issue loading base model: {str(rt_e)}")
                        print("Trying with reduced memory settings...")
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        
                        try:
                            model, tokenizer = FastLanguageModel.from_pretrained(
                                self.model_name,
                                max_seq_length=1024,  # Reduced sequence length
                                dtype=dtype,
                                device_map="auto",
                                load_in_4bit=True,  # Force 4-bit quantization
                                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload
                                use_gradient_checkpointing="unsloth"
                            )
                            print("Successfully loaded model with reduced memory settings and CPU offload")
                            
                        except Exception as final_e:
                            print(f"Failed to load model with all fallback options: {str(final_e)}")
                            raise RuntimeError(f"Failed to load model: {str(final_e)}")
                    else:
                        raise
            
            self.model = model
            self.tokenizer = tokenizer
            print("Model initialization complete")

        except Exception as e:
            print(f"Error in model initialization: {str(e)}")
            raise
            
    def _load_adapter_weights(self, adapter_file):
        """Load adapter weights for PyTorch models."""
        try:
            # If weights are in .npz format (from MLX), convert them
            if adapter_file.endswith('.npz'):
                weights = np.load(adapter_file)
                # Convert numpy arrays to PyTorch tensors with proper device placement
                device = self.model.device
                state_dict = {}
                
                for key, value in weights.items():
                    # Handle possible key name differences between MLX and PyTorch
                    # This may need customization based on your model architecture
                    state_dict[key] = torch.tensor(value, device=device)
                
                # Load the weights into the model
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"Missing keys when loading adapter weights: {missing}")
                if unexpected:
                    print(f"Unexpected keys when loading adapter weights: {unexpected}")
            else:
                # If weights are already in PyTorch format
                state_dict = torch.load(adapter_file, map_location=self.model.device)
                self.model.load_state_dict(state_dict, strict=False)
                
            print(f"Successfully loaded adapter weights from {adapter_file}")
        except Exception as e:
            print(f"Failed to load adapter weights: {str(e)}")
            print("Continuing with base model weights")

    def is_model_available_in_cache(self) -> bool:
        """
        Check if the model exists in the Hugging Face cache directory.
        Args:
            model_name (str): Name of the Hugging Face model.
        Returns:
            bool: True if the model exists in the cache, False otherwise.
        """
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        if '/' in self.embedding_model_name:
            org, model_name = self.embedding_model_name.split('/')
            formatted_model_name = f"{org}--{model_name}"
        else:
            formatted_model_name = self.embedding_model_name

        model_dir = cache_dir / f"models--{formatted_model_name}" / "snapshots"

        if not model_dir.exists() or not model_dir.is_dir():
            return False

        for snapshot_dir in model_dir.iterdir():
            if snapshot_dir.is_dir():

                required_files_set1 = ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json"]
                required_files_set2 = ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json"]

                if (all((snapshot_dir / file).exists() for file in required_files_set1) or
                    all((snapshot_dir / file).exists() for file in required_files_set2)):
                    return True
        return False

    def load_embedding_model(self):
        """
        Load the embedding model either from the Hugging Face cache or online.
        Returns:
            SentenceTransformer: The loaded model.
        """
        from sentence_transformers import SentenceTransformer
        
        # Set up offline mode before any model loading attempts
        if self.is_model_available_in_cache():
            print("Embedding Model found in Hugging Face cache. Loading locally...")
            # Force offline mode
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            
            try:
                # Specify the cache directory explicitly
                cache_dir = str(Path.home() / ".cache" / "huggingface" / "hub")
                model = SentenceTransformer(
                    self.embedding_model_name,
                    cache_folder=cache_dir
                )
                print("Model loaded successfully from cache.")
                return model.half()
            except Exception as e: 
                print(f"Detailed error while loading from cache: {str(e)}")
                raise RuntimeError(f"Failed to load the model from cache: {e}")
        else:
            print("Embedding Model not found in cache. Attempting to Download...")
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            os.environ["HF_DATASETS_OFFLINE"] = "0"
            from sentence_transformers import SentenceTransformer
            try:
                model = SentenceTransformer(self.embedding_model_name)
                print("Embedding Model downloaded successfully.")
                return model
            except Exception as e:
                raise RuntimeError(f"Failed to download the embedding model online: {e}")
    
    def _initialize_rag_components(self):
        """Initialize RAG components."""
        print("Initializing RAG components...")
        try:            
            self.embedding_model = self.load_embedding_model()
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(self.dimension)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")
        
    def cleanup_memory(self):
        """Explicitly clean up memory after processing."""
        import gc

        # Clear large data containers
        if hasattr(self, 'all_chunks'):
            self.all_chunks = []

        if hasattr(self, 'all_cleaned_texts'):
            self.all_cleaned_texts = []

        if hasattr(self, 'current_file_table_rows'):
            self.current_file_table_rows = []

        # Run garbage collection
        gc.collect()

        print("Memory cleanup performed")


    def process_input(self, input_path: str, use_finetuning: bool = False):
        """
        Process either a single file or a folder containing multiple files.
        
        Args:
            input_path: Path to either a file or a folder containing supported files
            use_finetuning: Whether to use fine-tuning
        """
        try:
            self.all_chunks = []
            self.all_cleaned_texts = []

            # Define supported file extensions
            self.SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx', '.wav', '.xlsx', '.m4a', '.mp3','.py','.js'}

            if os.path.isfile(input_path):
                # Check if file extension is supported
                file_ext = os.path.splitext(input_path)[1].lower()
                if file_ext in self.SUPPORTED_EXTENSIONS:
                    try:
                        self._process_single_input(input_path)
                    except Exception as e:
                        print(f"Error processing file {input_path}: {str(e)}")
                        raise ValueError(f"Failed to process file {input_path}: {str(e)}")
                else:
                    raise ValueError(f"Unsupported file type. Supported types are: {', '.join(self.SUPPORTED_EXTENSIONS)}")
            elif os.path.isdir(input_path):
                try:
                    self._process_folder_input(input_path)
                except Exception as e:
                    print(f"Error processing folder {input_path}: {str(e)}")
                    raise ValueError(f"Failed to process folder {input_path}: {str(e)}")
            else:
                raise ValueError(f"Input must be either a supported file ({', '.join(self.SUPPORTED_EXTENSIONS)}) or a folder")

            # Create combined index and training data
            if self.all_chunks:
                if not hasattr(self, 'embedding_model'):
                    print("Embedding model not found, initializing RAG components...")
                    self._initialize_rag_components()

                try:
                    self._create_combined_rag_index()
                    self._create_combined_training_data()
                except Exception as e:
                    print(f"Error creating index or training data: {str(e)}")
                    raise ValueError(f"Failed to create index or training data: {str(e)}")
            else:
                print("No chunks were created during processing.")
                raise ValueError("No valid content was processed. Please check the input file(s) and try again.")
            
            self.cleanup_memory()

        except Exception as e:
            self.cleanup_memory()
            raise e


    def _extract_text_and_tables_pdfplumber(self, file_path: str):
        all_page_texts = []
        table_rows = []
        import pdfplumber
        import pandas as pd
        batch_size = 20
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                print(f"Processing pages {batch_start+1} to {batch_end}")

                for page_index in range(batch_start, batch_end):
                    try:
                        page = pdf.pages[page_index]
                        page_text = page.extract_text() or ""
                        all_page_texts.append(f"Page {page_index+1}:\n{page_text}\n")
                        # Extract tables
                        tables = page.extract_tables()
                        for t_idx, table in enumerate(tables):
                            if not table or len(table) <= 1:  # Skip empty tables or header-only tables
                                continue

                            # Convert to DataFrame with better header handling
                            df = pd.DataFrame(table)
                            if not df.empty:
                                # Assume first row is header, but clean it up
                                headers = [str(h).strip() if h is not None else f"Column_{i}" 
                                          for i, h in enumerate(df.iloc[0])]
                                df.columns = headers
                                df = df[1:]  # Remove header row

                                # Process each row while preserving table structure
                                for row_index, row_data in df.iterrows():
                                    # Build structured row data
                                    row_items = []
                                    for col_idx, (col, val) in enumerate(row_data.items()):
                                        if pd.notna(val) and str(val).strip():
                                            row_items.append(f"{str(col).strip()}: {str(val).strip()}")

                                    if row_items:
                                        row_str = "; ".join(row_items)
                                        tagged_row = f"Page {page_index+1}, Table {t_idx+1}, Row {row_index-0}: {row_str}"
                                        table_rows.append(tagged_row)
                        del page 
                    except Exception as e:
                        print(f"Error on page {page_index+1}: {str(e)}")
            gc.collect()

        # Combine text and add a clear section for tables
        joined_text = "\n".join(all_page_texts).strip()
        tables_section = "\n\n--- TABLES SECTION ---\n\n" + "\n".join(table_rows) if table_rows else ""

        del all_page_texts
        del table_rows
        gc.collect()

        return joined_text , tables_section


    def pipe_tables_to_csv(self, text):
        """Convert pipe-delimited tables to CSV format with improved cleaning."""
        # Split the text into individual tables (assuming tables are separated by blank lines)
        table_sections = re.split(r'\n\s*\n', text)
        csv_tables = []

        for section in table_sections:
            if '|' in section:  # This is likely a table
                lines = section.strip().split('\n')
                csv_output = io.StringIO()
                csv_writer = csv.writer(csv_output)

                # First pass: determine the max number of columns for consistent output
                max_columns = 0
                for line in lines:
                    if not re.match(r'^[\|\-\s]+$', line):  # Skip separator lines
                        cells = [cell.strip() for cell in line.split('|')]
                        # Remove empty cells at beginning and end
                        if cells and cells[0] == '':
                            cells = cells[1:]
                        if cells and cells[-1] == '':
                            cells = cells[:-1]
                        max_columns = max(max_columns, len(cells))

                # Second pass: write rows with consistent column count
                for line in lines:
                    if re.match(r'^[\|\-\s]+$', line):  # Skip separator lines
                        continue
                    
                    # Split by pipe and strip whitespace from each cell
                    cells = [cell.strip() for cell in line.split('|')]

                    # Remove empty cells at the beginning and end
                    if cells and cells[0] == '':
                        cells = cells[1:]
                    if cells and cells[-1] == '':
                        cells = cells[:-1]

                    if cells:  # Make sure the row isn't empty
                        # Clean cell content - remove HTML and formatting artifacts
                        cleaned_cells = []
                        for cell in cells:
                            # Remove HTML tags
                            cell = re.sub(r'<[^>]+>', '', cell)
                            # Fix currency symbols
                            cell = re.sub(r'\\$<br>', '$', cell)
                            cell = re.sub(r'\\$', '$', cell)
                            # Remove escape characters
                            cell = cell.replace('\\r', '').replace('\\n', '')
                            cleaned_cells.append(cell)

                        # Pad with empty strings to ensure consistent column count
                        while len(cleaned_cells) < max_columns:
                            cleaned_cells.append('')

                        csv_writer.writerow(cleaned_cells)

                csv_tables.append(csv_output.getvalue())
                csv_output.close()

        return csv_tables

    def identify_table_type(self, csv_content):
        """Attempt to identify the type of financial table based on content."""
        csv_content = csv_content.lower()

        if any(term in csv_content for term in ['revenue', 'sales', 'income']):
            return 'Income Statement'
        elif any(term in csv_content for term in ['asset', 'liability', 'equity']):
            return 'Balance Sheet'
        elif any(term in csv_content for term in ['cash flow', 'operating activities']):
            return 'Cash Flow Statement'
        elif any(term in csv_content for term in ['stock', 'option', 'grant', 'vest']):
            return 'Stock-Based Compensation'
        elif any(term in csv_content for term in ['segment', 'geographic', 'region']):
            return 'Segment Reporting'
        else:
            return 'Financial Table'

    def post_process_csv(self, csv_content):
        """Additional post-processing to ensure clean, consistent CSV output."""
        # Parse the CSV
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        if not rows:
            return csv_content

        # Try to identify header row
        header_row = rows[0]

        # Check if there are date headers in the middle of the table
        date_pattern = re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}')

        # Find all date headers and their row indices
        date_rows = []
        for i, row in enumerate(rows):
            for cell in row:
                if date_pattern.search(cell):
                    date_rows.append(i)
                    break
                
        # If we have multiple date headers, restructure the table
        if len(date_rows) > 1:
            # Create a better structured table
            new_rows = []
            current_date = ""

            # First row is the header
            new_rows.append(header_row)

            for i, row in enumerate(rows):
                if i in date_rows:
                    # This is a date header row, save the date
                    for cell in row:
                        if date_pattern.search(cell):
                            current_date = cell
                            break
                else:
                    # This is a data row, add date as first column if needed
                    if current_date and i > 0:
                        # Check if the date is already in the row
                        date_in_row = any(date_pattern.search(cell) for cell in row)
                        if not date_in_row:
                            row = [current_date] + row
                    new_rows.append(row)

            # Convert back to CSV
            output = io.StringIO()
            writer = csv.writer(output)
            for row in new_rows:
                writer.writerow(row)
            return output.getvalue()

        return csv_content

        # You can either return just the chunks or both chunks and cleaned texts
        # return {
        #     "chunks": table_chunks,
        #     "cleaned_texts": all_cleaned_texts,
        #     "csv_tables": csv_tables  # Original CSV strings
        # }


    def _extract_text(self, file_path: str) -> str:
        """Extract text from different file types including audio files with Windows compatibility."""
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if (file_ext == '.wav' or file_ext == '.m4a' or file_ext == '.mp3'):
                print("Processing audio file...")

                try:
                    # First try: direct loading with pydub
                    print("Attempting direct audio load...")
                    try:
                        if file_ext == '.m4a':
                            audio = AudioSegment.from_file(file_path, format="m4a")
                        else:
                            audio = AudioSegment.from_file(file_path)
                    except Exception as e:
                        print(f"Direct load failed: {e}")
                        print("Attempting fallback conversion...")
                        
                        # Fallback: convert to wav using ffmpeg directly
                        temp_wav = str(Path(file_path).with_suffix(".tmp.wav"))
                        try:
                            # First, check if ffmpeg can read the input file
                            probe_result = subprocess.run([
                                ffprobe_path,
                                "-v", "error",
                                "-show_entries", "format=duration",
                                "-of", "json",
                                file_path
                            ], capture_output=True, text=True)
                            
                            if probe_result.returncode != 0:
                                print(f"FFprobe error: {probe_result.stderr}")
                                raise RuntimeError(f"FFprobe failed to read input file: {probe_result.stderr}")

                            # Now try the conversion
                            print("Running ffmpeg conversion...")
                            convert_result = subprocess.run([
                                ffmpeg_path,
                                "-hide_banner",  # Reduce noise in output
                                "-y",  # Overwrite output file if it exists
                                "-i", file_path,  # Input file
                                "-vn",  # Disable video
                                "-ac", "1",  # Mono audio
                                "-ar", "16000",  # 16kHz sample rate
                                "-acodec", "pcm_s16le",  # Force PCM 16-bit output
                                "-f", "wav",  # Force WAV format
                                temp_wav  # Output file
                            ], capture_output=True, text=True)
                            
                            print("FFmpeg stdout:", convert_result.stdout)
                            print("FFmpeg stderr:", convert_result.stderr)
                            
                            if convert_result.returncode != 0:
                                print(f"FFmpeg conversion error: {convert_result.stderr}")
                                raise RuntimeError(f"FFmpeg conversion failed: {convert_result.stderr}")
                            
                            # Verify the output file exists and has size
                            if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                                raise RuntimeError("FFmpeg produced empty or missing output file")
                                
                            print(f"Conversion successful, loading converted file: {temp_wav}")
                            # Load the converted wav file
                            audio = AudioSegment.from_wav(temp_wav)
                            print("Converted WAV file loaded successfully")
                            
                        except Exception as conv_error:
                            print(f"Conversion error: {type(conv_error).__name__}: {str(conv_error)}")
                            raise
                        finally:
                            # Clean up temporary wav file
                            if os.path.exists(temp_wav):
                                try:
                                    os.remove(temp_wav)
                                    print("Cleaned up temporary wav file")
                                except Exception as e:
                                    print(f"Warning: Could not remove temp wav: {e}")

                    print("Audio file loaded successfully")
                    print(f"Audio properties: Duration={len(audio)}ms, Channels={audio.channels}, Sample width={audio.sample_width}")

                    # Process audio
                    print("Processing audio...")
                    processed_audio = (audio
                        .normalize()  # Normalize volume
                        .set_channels(1)  # Convert to mono
                        .set_frame_rate(16000))  # Set sample rate to 16kHz

                    # Save as temporary WAV file for Whisper
                    temp_dir = os.path.dirname(file_path)
                    temp_path = os.path.join(temp_dir, 'temp_processed.wav')
                    try:
                        print(f"Exporting processed audio to {temp_path}")
                        processed_audio.export(
                            temp_path,
                            format="wav",
                            parameters=["-ac", "1", "-ar", "16000"]
                        )
                        print("Audio export completed")

                        # Load Whisper model
                        print("Loading Whisper model...")
                        model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
                        print("Whisper model loaded")

                        # Transcribe
                        print("Transcribing audio...")
                        result = model.transcribe(temp_path, language="en")
                        text = result["text"]
                        print("Transcription completed")
                        print(f"Transcribed text length: {len(text)} characters")

                        if not text.strip():
                            print("Warning: Transcription produced empty text")
                            return ""

                        return text

                    except Exception as export_error:
                        print(f"Error during audio export/transcription: {str(export_error)}")
                        raise
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                                print("Cleaned up temporary file")
                            except Exception as e:
                                print(f"Warning: Could not remove temp file: {e}")

                        # Clean up model
                        if 'model' in locals():
                            del model
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print("Cleaned up Whisper model")

                except Exception as e:
                    print(f"Audio processing error: {str(e)}")
                    print(f"Full error details: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return ""

            elif file_ext == '.pdf':
                page_text, table_rows = self._extract_text_and_tables_pdfplumber(file_path)
                self.current_file_table_rows = table_rows
                return page_text

            elif file_ext in {'.txt'}:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()

            elif file_ext in {'.doc', '.docx'}:
                doc = Document(file_path)
                return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            print(f"Full error details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    

    def _process_javascript_file(self, file_path: str):
        """Process JavaScript file with code-aware chunking."""
        try:
            # Read the JavaScript file
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            if not source_code.strip():
                print(f"Warning: Empty JavaScript file {file_path}")
                return

            # Initialize raw_chunks as empty list
            raw_chunks = []

            # Use JavaScript segmenter if available, otherwise use basic chunking
            try:
                from langchain_community.document_loaders.parsers.language.javascript import JavaScriptSegmenter
                segmenter = JavaScriptSegmenter(source_code)
                if segmenter.is_valid():
                    raw_chunks = segmenter.extract_functions_classes()
                    print(f"Extracted {len(raw_chunks)} function/class segments from {file_path}")
            except (ImportError, Exception) as e:
                print(f"JavaScript segmenter failed: {str(e)}, falling back to basic chunking")
                # Fallback to basic chunking
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    separators=["\nfunction ", "\nclass ", "\nconst ", "\nlet ", "\n\n", "\n", " ", ""]
                )
                raw_chunks = splitter.split_text(source_code)
                print(f"Used basic chunking for {file_path}, created {len(raw_chunks)} chunks")

            if not raw_chunks:
                # If no chunks were created, create one chunk with the entire file
                print("No chunks created, using entire file as one chunk")
                raw_chunks = [source_code]

            # Add context to chunks
            enhanced_chunks = []
            for chunk in raw_chunks:
                # Try to identify what the chunk contains
                chunk_type = "JavaScript code"
                if "function " in chunk.lower():
                    chunk_type = "JavaScript function"
                elif "class " in chunk.lower():
                    chunk_type = "JavaScript class"

                enhanced_chunks.append(f"{chunk_type} section:\n\n{chunk}")

            # Add to global chunks and cleaned texts
            if enhanced_chunks:
                self.all_chunks.extend(enhanced_chunks)
                self.all_cleaned_texts.append(source_code)
                print(f"Successfully processed JavaScript file {file_path} into {len(enhanced_chunks)} chunks")
            else:
                print(f"Warning: No chunks were created from {file_path}")

            # Save extracted code for reference
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            extracted_file_path = os.path.join(self.output_dir, f'{file_name}_extracted.txt')
            with open(extracted_file_path, 'w', encoding='utf-8') as f:
                f.write(source_code)

            del raw_chunks
            del source_code
            gc.collect()

        except Exception as e:
            print(f"Error processing JavaScript file {file_path}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise

        
    def _process_python_file(self, file_path: str):
        """Process Python file with code-aware chunking."""
        try:
            # Load the Python file
            loader = PythonLoader(file_path)
            docs = loader.load()
            source_code = "\n".join(doc.page_content for doc in docs)

            del docs
            gc.collect()
            
            # Extract code structure information using AST
            import ast
            try:
                tree = ast.parse(source_code)
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                print(f"Found {len(functions)} functions and {len(classes)} classes in {file_path}")
            except Exception as e:
                print(f"AST parsing error: {str(e)}")
                functions, classes = [], []

            # Use language-aware chunking
            from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=1500,
                chunk_overlap=300
            )
            chunks = splitter.split_text(source_code)

            # Add context metadata to each chunk
            enhanced_chunks = []
            for chunk in chunks:
                # Find which functions/classes this chunk contains
                contained_functions = [f for f in functions if f in chunk]
                contained_classes = [c for c in classes if c in chunk]

                # Create metadata header
                header_parts = []
                if contained_classes:
                    header_parts.append(f"Classes: {', '.join(contained_classes)}")
                if contained_functions:
                    header_parts.append(f"Functions: {', '.join(contained_functions)}")

                if header_parts:
                    header = f"Python code section containing {' and '.join(header_parts)}:\n\n"
                    enhanced_chunks.append(header + chunk)
                else:
                    enhanced_chunks.append(f"Python code section:\n\n{chunk}")

            # Add to global chunks and cleaned texts
            self.all_chunks.extend(enhanced_chunks)
            self.all_cleaned_texts.append(source_code)

            # Save extracted code for reference
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            extracted_file_path = os.path.join(self.output_dir, f'{file_name}_extracted.txt')
            with open(extracted_file_path, 'w', encoding='utf-8') as f:
                f.write(source_code)
            
            del chunks
            del functions
            del classes
            gc.collect()

            print(f"Successfully processed Python file {file_path} into {len(enhanced_chunks)} chunks")
        except Exception as e:
            print(f"Error processing Python file {file_path}: {str(e)}")

            
    def _process_single_input(self, file_path: str):
        """Process a single file."""
        print(f"Processing file: {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext == '.py':
                self._process_python_file(file_path)
                return
            elif file_ext == '.js':
                self._process_javascript_file(file_path)
                return
            
            self.current_file_table_rows = []
            raw_text = self._extract_text(file_path)
            if not raw_text.strip():
                print(f"Warning: No text could be extracted from {file_path}")
                return
            
            cleaned_text = self.clean_text(raw_text)
            cleaned_rows = [self.clean_text(row) for row in self.current_file_table_rows]

            self.all_cleaned_texts.append(cleaned_text)

            ## create ngrams json
            ngram_results = extract_ngrams(cleaned_text, n_values=[1, 2, 3], top_n=3)
            ngram_save_path = os.path.join(self.model_folder, "extracted_ngrams.json")
            ngram_results_dict = {}
            for n, grams in ngram_results.items():
                gram_type = "unigrams" if n == 1 else "bigrams" if n == 2 else "trigrams" if n == 3 else f"{n}-grams"
                ngram_results_dict[gram_type] = grams

            with open(ngram_save_path, "w") as f:
              json.dump(ngram_results_dict, f, indent=4)
            print("Ngrams is saved")
            # Create chunks
            text_chunks = self._create_rag_chunks(cleaned_text)

            table_chunks = cleaned_rows

            combined_chunks = text_chunks + table_chunks

            if text_chunks:
                self.all_chunks.extend(combined_chunks)

            # Save cleaned text
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            extracted_file_path = os.path.join(self.output_dir, f'{file_name}_extracted.txt')

            with open(extracted_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            

    def _process_folder_input(self, folder_path: str):
        """Process all supported files in a folder."""
        processed_files = []
        
        # Process all files with supported extensions
        for ext in self.SUPPORTED_EXTENSIONS:
            files = list(Path(folder_path).rglob(f"*{ext}"))
            processed_files.extend(files)

        if not processed_files:
            raise ValueError(f"No supported files found in {folder_path}. "
                           f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}")

        print(f"Found {len(processed_files)} supported files to process")
        for file_path in processed_files:
            self._process_single_input(str(file_path))

    def _create_rag_chunks(self, text: str) -> List[str]:
        """Create RAG chunks with improved splitting strategy."""
        if not text.strip():
            return []

        try:
            # Use more sophisticated text splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", ", ", " ", ""]
            )
            chunks = []
            for chunk in splitter.split_text(text):
                # Preserve document structure
                if chunk.startswith(('# ', '## ', '### ')):  # Headings
                    chunk = f"\nSection: {chunk}\n"
                chunks.append(chunk)

            return chunks
        except Exception as e:
            print(f"Error creating chunks: {str(e)}")
            return []
        
    def _create_combined_rag_index(self):
        """
        Incrementally build or update a FAISS index from self.all_chunks.
        If vector_index.faiss and chunk_mapping.json already exist,
        this function will load them and append any new chunks that
        are not already in the mapping. Otherwise, it creates a new index.
        """
        import faiss
        import os
        import json
        import numpy as np

        if not self.all_chunks:
            raise ValueError("No chunks available for indexing")

        # Prepare the set/list of new chunks from self.all_chunks
        new_unique_chunks = list(set(self.all_chunks))

        del self.all_chunks
        self.all_chunks = []
        gc.collect()

        # Paths where we store the FAISS index + chunk mapping
        index_save_path = os.path.join(self.model_folder, "vector_index.faiss")
        chunk_mapping_path = os.path.join(self.model_folder, "chunk_mapping.json")

        # Check if index + mapping already exist
        print(index_save_path)
        print(chunk_mapping_path)
        index_exists = os.path.exists(index_save_path)
        mapping_exists = os.path.exists(chunk_mapping_path)

        if index_exists and mapping_exists:
            # -----------------------------
            # 1) Load existing index + map
            # -----------------------------
            self.index = faiss.read_index(index_save_path)
            with open(chunk_mapping_path, "r", encoding="utf-8") as f:
                old_mapping = json.load(f)
            # Convert string keys to int (if needed):
            old_mapping = {int(k): v for k, v in old_mapping.items()}

            original_count = self.index.ntotal
            print(f"Loaded existing FAISS index with {original_count} vectors")

            # -----------------------------
            # 2) Filter out chunks already in mapping
            # -----------------------------
            existing_chunks_set = set(old_mapping.values())
            final_new_chunks = [c for c in new_unique_chunks if c not in existing_chunks_set]

            if not final_new_chunks:
                print("No truly new chunks. The index remains unchanged.")
                self.chunk_mapping = old_mapping
                return  # We are done

            print(f"Found {len(final_new_chunks)} new chunks to embed and add.")

            # -----------------------------
            # 3) Embed new chunks & add to index
            # -----------------------------
            batch_size = 4
            new_embeddings_batches = []
            for i in range(0, len(final_new_chunks), batch_size):
                batch = final_new_chunks[i : i + batch_size]
                emb = self.embedding_model.encode(batch)
                new_embeddings_batches.append(emb)

            final_new_embeddings = np.vstack(new_embeddings_batches)

            # Add these embeddings to the existing index
            self.index.add(final_new_embeddings)

            # -----------------------------
            # 4) Update chunk mapping
            # -----------------------------
            start_id = len(old_mapping)  # The next new ID
            for i, chunk_text in enumerate(final_new_chunks):
                old_mapping[start_id + i] = chunk_text

            self.chunk_mapping = old_mapping

            # -----------------------------
            # 5) Re-save index + mapping
            # -----------------------------
            faiss.write_index(self.index, index_save_path)
            with open(chunk_mapping_path, "w", encoding="utf-8") as f:
                json.dump(self.chunk_mapping, f, ensure_ascii=False, indent=2)

            print(
                f"Appended {len(final_new_chunks)} new chunks. "
                f"Index now has {self.index.ntotal} vectors. "
                f"Saved to: {index_save_path}"
            )
        else:
            print("No existing FAISS index found. Creating a new one.")

            unique_chunks = new_unique_chunks
            # Use smaller batch size for large documents

            batch_size = select_batch_size()
            print("batch size is ")
            print(batch_size)
            # Create empty index first
            self.index = faiss.IndexFlatL2(self.dimension)

            # Process in batches and add to index immediately
            chunk_mapping = {}
            total_batches = (len(unique_chunks) + batch_size - 1) // batch_size

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(unique_chunks))
                batch = unique_chunks[start_idx:end_idx]

                print(f"Embedding batch {batch_num+1}/{total_batches} ({len(batch)} chunks)")

                # Embed this batch
                embeddings = self.embedding_model.encode(batch)

                # Add to index
                self.index.add(embeddings)

                # Update mapping
                for i, chunk in enumerate(batch):
                    chunk_mapping[start_idx + i] = chunk

                # Clear memory
                del embeddings
                gc.collect()

            # Save the complete index and mapping
            self.chunk_mapping = chunk_mapping
            del chunk_mapping
            # Save index
            faiss.write_index(self.index, index_save_path)
            print(f"FAISS index saved to {index_save_path}")

            ##deleting to remove it from memory after saving it in a file (NEED TO REVISIT)
            # del self.index

            # Save chunk mapping
            try:
                with open(chunk_mapping_path, "w", encoding="utf-8") as f:
                    json.dump(self.chunk_mapping, f, ensure_ascii=False, indent=2)

                # del self.chunk_mapping
                print("Successfully wrote chunk_mapping.json.")
            except Exception as e:
                print("Error writing chunk_mapping.json:", e)

            print(f"Chunk mapping saved to {chunk_mapping_path}")


    def _create_combined_training_data(self):
        """Create combined training and validation datasets."""
        if not self.all_cleaned_texts:
            raise ValueError("No texts available for creating training data")

        combined_text = "\n\n".join(self.all_cleaned_texts)

        train_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size // 2,
            chunk_overlap=self.chunk_overlap // 2
        )
        all_chunks = train_splitter.split_text(combined_text)

        random.shuffle(all_chunks)

        split_idx = int(len(all_chunks) * 0.8)
        train_chunks = all_chunks[:split_idx]
        valid_chunks = all_chunks[split_idx:]

        train_data = [{"text": chunk} for chunk in train_chunks]
        valid_data = [{"text": chunk} for chunk in valid_chunks]
        if len(train_data) < 5 :
            raise ValueError("Not enough content to finetune")
        
        self._save_jsonl(train_data, os.path.join(self.output_dir, 'train.jsonl'))
        self._save_jsonl(valid_data, os.path.join(self.output_dir, 'valid.jsonl'))

        print(f"Created combined training files in {self.output_dir}")
        print(f"Train samples: {len(train_data)}, Validation samples: {len(valid_data)}")

        del train_data
        del valid_data
        gc.collect()

    def create_combined_training_data_with_feedback(self,file_path):
        """Create combined training and validation datasets with feedback data."""
        # 1. Check if we have texts available
        if not self.all_cleaned_texts:
            raise ValueError("No texts available for creating training data")

        # 2. Get feedback messages
        feedback_file = os.path.join(file_path, 'feedback.json')
        feedback_messages = []

        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                    # Extract assistant messages from the feedback
                    if isinstance(feedback_data, list):
                        feedback_messages = [item.get('assistant_message', '') for item in feedback_data if item.get('is_positive', False)]
                    elif isinstance(feedback_data, dict):
                        feedback_messages = [feedback_data.get('assistant_message', '')] if feedback_data.get('is_positive', False) else []
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read feedback file: {e}")

        # 3. Combine all text sources
        combined_text = "\n\n".join(self.all_cleaned_texts)

        # 4. Add feedback messages to the text if they exist
        if feedback_messages:
            feedback_text = "\n\n".join(feedback_messages)
            combined_text = combined_text + "\n\n" + feedback_text

        # 5. Split text into chunks
        print("the chunk size is now")
        print(self.chunk_size)
        train_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size // 2,
            chunk_overlap=self.chunk_overlap // 2
        )
        all_chunks = train_splitter.split_text(combined_text)

        # 6. Shuffle and split
        random.shuffle(all_chunks)
        split_idx = int(len(all_chunks) * 0.8)
        train_chunks = all_chunks[:split_idx]
        valid_chunks = all_chunks[split_idx:]

        train_data = [{"text": chunk} for chunk in train_chunks]
        valid_data = [{"text": chunk} for chunk in valid_chunks]

        if len(train_data) < 5:
            raise ValueError("Not enough content to finetune")

        # 7. Save to jsonl files
        self._save_jsonl(train_data, os.path.join(self.output_dir, 'train.jsonl'))
        self._save_jsonl(valid_data, os.path.join(self.output_dir, 'valid.jsonl'))

        print(f"Created combined training files in {self.output_dir}")
        print(f"Train samples: {len(train_data)}, Validation samples: {len(valid_data)}")
        print(f"Included {len(feedback_messages)} feedback messages")

        # 8. Clean up
        del train_data
        del valid_data
        gc.collect()

    def _load_cached_index(self):
        """
        Loads FAISS index and chunk mapping if they exist 
        in self.output_dir.
        """
        index_path = os.path.join(self.model_folder, "vector_index.faiss")
        mapping_path = os.path.join(self.model_folder, "chunk_mapping.json")

        if os.path.exists(index_path) and os.path.exists(mapping_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded FAISS index from {index_path}")

            with open(mapping_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.chunk_mapping = {int(k): v for k, v in loaded.items()}
            print(f"Loaded chunk mapping from {mapping_path}")

            self.dimension = self.index.d
        else:
            raise FileNotFoundError("No cached FAISS index found. Make sure to process files first.")

    def _save_jsonl(self, data: List[Dict], filepath: str):
        """Save data to JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def index_chunks(self, chunks):
        """Index chunks for RAG retrieval."""
        try:
            unique_chunks = list(set(chunks))
            batch_size = 16
            all_embeddings = []
            
            for i in range(0, len(unique_chunks), batch_size):
                batch = unique_chunks[i:i + batch_size]
                embeddings = self.embedding_model.encode(batch)
                all_embeddings.append(embeddings)
            
            final_embeddings = np.vstack(all_embeddings)
            self.index.add(final_embeddings)
            self.chunk_mapping = {i: chunk for i, chunk in enumerate(unique_chunks)}
            
        except Exception as e:
            raise RuntimeError(f"Error indexing chunks: {str(e)}")

    def _initialize_chat_setup(self):
        """Initialize chat setup."""
        self.chat_setup = ChatSetup(
            system="You are a helpful AI assistant. Answer questions based on the provided document context. "
                "If the answer cannot be found in the context, say so clearly."
        )
        self.session = self.chat_setup.session()

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance"""
        cls._instance = None
        cls._is_initialized = False

    def retrieve_chunks(self, query, k=None):
        """Enhanced chunk retrieval with better relevance scoring."""
        if k is None:
            k = self.max_chunks

        try:
            # 1. Initial embedding-based retrieval
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)

            print("Query embedding shape:", query_embedding.shape)
            print("Query embedding dtype:", query_embedding.dtype)

            total_indices = self.index.ntotal
            search_k = min(k * 5, total_indices)
            print(f"Searching for {search_k} neighbors out of {total_indices} total vectors")

            if search_k <= 0:
                print("No vectors in index to search")
                return []

            distances, indices = self.index.search(query_embedding, search_k)

            # 2. Enhanced scoring for retrieved chunks
            scored_chunks = []
            for dist, idx in zip(distances[0], indices[0]):
                chunk = self.chunk_mapping[idx]

                # Multiple scoring components
                semantic_score = self.calculate_semantic_relevance(query, chunk)
                keyword_score = self._calculate_keyword_match(query, chunk)
                length_score = self._calculate_length_score(chunk)

                # Combined scoring with weights
                total_score = (
                    0.5 * semantic_score +
                    0.3 * keyword_score +
                    0.2 * length_score
                )

                if (len(chunk) > 20 and 
                    total_score > 0.4 and  
                    dist < self.get_dynamic_distance_threshold(query)):
                    scored_chunks.append((chunk, total_score))

            # Sort by score and take top k
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            del query_embedding
            return [chunk for chunk, _ in scored_chunks[:k]]

        except Exception as e:
            print(f"Error in chunk retrieval: {e}")
            return []

    def _get_content_hash(self, text):
        """Create a simple hash of content to detect near-duplicates."""
        # Remove whitespace and make lowercase
        text = re.sub(r'\s+', ' ', text.lower()).strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Create a sorted list of words to be order-invariant
        words = sorted(text.split())
        # Join back together and hash
        return hash(' '.join(words[:50]))  # Consider first 50 words enough for similarity
    
    def _is_financial_query(self, query):
        """Detect if query is seeking financial information."""
        financial_terms = [
            "sales", "revenue", "profit", "margin", "income", "earnings", 
            "ebitda", "eps", "dividend", "cash flow", "balance", "asset",
            "liability", "expense", "cost", "roi", "return", "growth",
            "quarter", "annual", "fiscal", "budget", "forecast"
        ]
        
        # Entity patterns to check what financial data is being requested
        entity_patterns = {
            "sales": r"sales|revenue",
            "profit": r"profit|income|earnings|margin",
            "growth": r"growth|increase|decrease|change",
            "geographic": r"north america|europe|asia|region|country",
            "temporal": r"quarter|annual|year|month|period",
        }
        
        # Check for financial terms
        query_lower = query.lower()
        has_financial_terms = any(term in query_lower for term in financial_terms)
        
        # Determine entity type
        entity_type = None
        if has_financial_terms:
            for entity, pattern in entity_patterns.items():
                if re.search(pattern, query_lower):
                    entity_type = entity
                    break
        
        return has_financial_terms, entity_type
        
    def _expand_financial_query(self, query):
        """Expand financial queries with related terms for better retrieval."""
        expansions = {
            "sales": ["revenue", "turnover", "net sales", "gross sales"],
            "profit": ["income", "earnings", "net income", "profit margin", "gross profit"],
            "north america": ["US", "USA", "United States", "Canada", "Mexico", "NA region"],
            "europe": ["EU", "European Union", "EMEA", "European region"],
            "quarter": ["Q1", "Q2", "Q3", "Q4", "quarterly", "three months"],
            "annual": ["yearly", "fiscal year", "12 months", "year-end"],
        }
        
        expanded = query
        for term, synonyms in expansions.items():
            if term.lower() in query.lower():
                # Add a few synonyms to expand the query
                additions = " OR " + " OR ".join(random.sample(synonyms, min(2, len(synonyms))))
                expanded += additions
        
        return expanded
        
    def _calculate_financial_relevance(self, query, chunk):
        """Calculate financial relevance score with awareness of financial data."""
        # Check for presence of financial data in the chunk
        has_numbers = bool(re.search(r'[$€£]?\s*\d+(?:[.,]\d+)*(?:\s*(?:million|billion|thousand|m|b|k))?', chunk))
        
        # Look for matching financial entities between query and chunk
        query_entities = self._extract_financial_entities(query)
        chunk_entities = self._extract_financial_entities(chunk)
        
        # Calculate overlap score
        entity_match = 0.0
        if query_entities and chunk_entities:
            query_terms = set(e[0].lower() for e in query_entities)
            chunk_terms = set(e[0].lower() for e in chunk_entities)
            overlap = query_terms.intersection(chunk_terms)
            entity_match = len(overlap) / len(query_terms) if query_terms else 0
        
        # Higher score for chunks with both financial terms and numbers
        if has_numbers and entity_match > 0:
            return 0.8 + entity_match * 0.2  # Boost chunks with matching entities and numbers
        elif has_numbers:
            return 0.6  # Decent score for chunks with numbers
        elif entity_match > 0:
            return 0.4 + entity_match * 0.3  # Moderate score for matching entities without numbers
        else:
            return 0.2  # Lower score for chunks without financial data
    
    def _extract_financial_entities(self, text):
        """Extract financial entities (metrics, regions, periods) from text."""
        entities = []
        
        # Financial metrics patterns
        metric_patterns = [
            (r'\b(?:net\s+)?sales\b', 'metric', 'sales'),
            (r'\b(?:net\s+)?revenue\b', 'metric', 'revenue'),
            (r'\b(?:net|gross)\s+profit\b', 'metric', 'profit'),
            (r'\b(?:profit\s+)?margin\b', 'metric', 'margin'),
            (r'\b(?:net\s+)?income\b', 'metric', 'income'),
            (r'\bebitda\b', 'metric', 'ebitda'),
            (r'\bearnings\b', 'metric', 'earnings'),
        ]
        
        # Geographical regions
        region_patterns = [
            (r'\bnorth\s+america\b', 'region', 'north_america'),
            (r'\b(?:united\s+states|us|usa)\b', 'region', 'us'),
            (r'\beurope\b', 'region', 'europe'),
            (r'\basia(?:\s+pacific)?\b', 'region', 'asia'),
            (r'\blatam\b', 'region', 'latam'),
        ]
        
        # Time periods
        period_patterns = [
            (r'\bq[1-4]\b', 'period', 'quarter'),
            (r'\b(?:fiscal\s+)?year\b', 'period', 'year'),
            (r'\b(?:first|second|third|fourth)\s+quarter\b', 'period', 'quarter'),
        ]
        
        # Numerical values with units
        value_pattern = r'[$€£]?\s*(\d+(?:[.,]\d+)*(?:\s*(?:million|billion|thousand|m|b|k))?)'
        
        # Extract all types of entities
        all_patterns = metric_patterns + region_patterns + period_patterns
        
        for pattern, entity_type, entity_subtype in all_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                entities.append((match.group(0), entity_type, entity_subtype))
        
        # Find numerical values
        value_matches = re.finditer(value_pattern, text)
        for match in value_matches:
            entities.append((match.group(0), 'value', 'amount'))
        
        return entities
    
    def _extract_and_cache_financial_entities(self, query, chunks):
        """Extract and cache financial entities from chunks for consistency."""
        # Extract query information
        query_entities = self._extract_financial_entities(query)
        query_metrics = [e for e in query_entities if e[1] == 'metric']
        query_regions = [e for e in query_entities if e[1] == 'region']
        query_periods = [e for e in query_entities if e[1] == 'period']
        
        # Create a key for this specific financial query
        query_key = '_'.join(sorted([e[2] for e in query_entities if e[1] != 'value']))
        if not query_key:
            return
            
        # Extract all values from chunks
        all_values = []
        for chunk in chunks:
            chunk_entities = self._extract_financial_entities(chunk)
            
            # Find sentences with financial values
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            for sentence in sentences:
                sent_entities = self._extract_financial_entities(sentence)
                sent_values = [e for e in sent_entities if e[1] == 'value']
                
                # Only consider sentences with values and matching query entities
                if sent_values:
                    sent_metrics = [e for e in sent_entities if e[1] == 'metric']
                    sent_regions = [e for e in sent_entities if e[1] == 'region']
                    sent_periods = [e for e in sent_entities if e[1] == 'period']
                    
                    # Check if this sentence has entities matching the query
                    metrics_match = not query_metrics or any(m[2] in [qm[2] for qm in query_metrics] for m in sent_metrics)
                    regions_match = not query_regions or any(r[2] in [qr[2] for qr in query_regions] for r in sent_regions)
                    periods_match = not query_periods or any(p[2] in [qp[2] for qp in query_periods] for p in sent_periods)
                    
                    if metrics_match and regions_match and periods_match:
                        # This sentence contains relevant financial data
                        context = {
                            'value': sent_values[0][0],
                            'metric': sent_metrics[0][2] if sent_metrics else None,
                            'region': sent_regions[0][2] if sent_regions else None,
                            'period': sent_periods[0][2] if sent_periods else None,
                            'sentence': sentence
                        }
                        all_values.append(context)
        
        # Store the extracted data with context
        if all_values:
            self.financial_entity_cache[query_key] = all_values
    
    def _get_consistent_financial_data(self, query):
        """Get consistent financial data from cache to avoid contradictions."""
        query_entities = self._extract_financial_entities(query)
        query_key = '_'.join(sorted([e[2] for e in query_entities if e[1] != 'value']))
        
        if query_key in self.financial_entity_cache:
            # Return the most frequently mentioned value for consistency
            values = self.financial_entity_cache[query_key]
            if values:
                # Count occurrences of each value
                value_counter = Counter(v['value'] for v in values)
                most_common_value = value_counter.most_common(1)[0][0]
                
                # Find the best context for this value
                best_context = next((v for v in values if v['value'] == most_common_value), values[0])
                return best_context
                
        return None

    def _calculate_keyword_match(self, query: str, chunk: str) -> float:
        """Calculate keyword matching score with improved handling of financial terms."""
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())

        # Use stemming for better matching
        ps = PorterStemmer()
        query_stems = {ps.stem(word) for word in query_words}
        chunk_stems = {ps.stem(word) for word in chunk_words}

        # Extract financial terms for special handling
        financial_terms = {"sales", "revenue", "profit", "income", "margin", 
                          "growth", "increase", "decrease", "fiscal", "quarter"}
        
        # Financial term stems
        financial_stems = {ps.stem(term) for term in financial_terms}
        
        # Calculate overlap with extra weight for financial terms
        regular_stems = query_stems - financial_stems
        financial_query_stems = query_stems.intersection(financial_stems)
        
        # Check for regular term matches
        regular_matches = regular_stems.intersection(chunk_stems)
        financial_matches = financial_query_stems.intersection(chunk_stems)
        
        # Weight financial terms more heavily
        total_score = 0
        if regular_stems:
            total_score += 0.6 * (len(regular_matches) / len(regular_stems))
        if financial_query_stems:
            total_score += 0.4 * (len(financial_matches) / len(financial_query_stems))
        else:
            total_score += 0.4  # No financial terms in query
            
        return total_score
    
    def _calculate_length_score(self, chunk: str) -> float:
        """Calculate length-based relevance score with preference for concise financial data."""
        # Financial information is often more concise
        optimal_length = 150  # Slightly shorter optimal length for financial text
        current_length = len(chunk)

        # Penalize chunks that are too short or too long
        if current_length < 50:
            return 0.5 * (current_length / 50)
        elif current_length > optimal_length * 2:
            return 0.5
        else:
            return 1.0 - abs(current_length - optimal_length) / optimal_length
        
    def calculate_semantic_relevance(self, query, chunk):
        """Advanced semantic relevance calculation with multiple strategies"""
        try:
            # Normalize embeddings for accurate cosine similarity
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            chunk_embedding = self.embedding_model.encode([chunk], normalize_embeddings=True)[0]

            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding)

            # Additional keyword-based relevance
            query_keywords = set(query.lower().split())
            chunk_keywords = set(chunk.lower().split())
            keyword_overlap = len(query_keywords.intersection(chunk_keywords)) / len(query_keywords) if query_keywords else 0

            # Combined score with more weight on embedding similarity
            combined_score = (0.7 * similarity) + (0.3 * keyword_overlap)
            return combined_score
        except Exception as e:
            print(f"Relevance calculation error: {e}")
            return 0

    def _is_out_of_context(self, query, agent):
        try:
            # More aggressive out-of-context detection
            relevant_chunks = self.retrieve_chunks(query, k=3)

            # If no relevant chunks or extremely low semantic score
            if not relevant_chunks:
                return True

            # Calculate max semantic relevance
            max_relevance = max(
                self.calculate_semantic_relevance(query, chunk) 
                for chunk in relevant_chunks
            )
            print(max_relevance)
            
            if any(word in query.lower() for word in ["summarize", "summary", "overview"]):
                return max_relevance < 0.3  # Lower threshold for summary requests

            if agent=="tech":
                return max_relevance < 0.4
            elif agent == "finance":
                return max_relevance < 0.6
            elif agent == "research":
                return max_relevance < 0.65
            
            return max_relevance < 0.7
            
        except Exception as e:
            print(f"Context detection error: {e}")
            return True

    def get_dynamic_distance_threshold(self, query):
        # Dynamically adjust distance threshold based on query complexity
        query_length = len(query.split())
        base_threshold = 5.0
        return base_threshold * (1 + 0.1 * query_length)

    def verify_response_against_context(self, response, context):
        """
        Reject hallucinated responses by ensuring alignment with context.
        """
        context_lower = context.lower()
        response_lower = response.lower()

        # Verify all key terms in response appear in the context
        key_terms = set(response_lower.split())  # Simplistic; refine for domain
        if not key_terms.intersection(set(context_lower.split())):
            return False  # Hallucination detected
        return True
    
    
    def _detect_broad_query(self, query: str, custom_patterns: list[str] = None) -> tuple[bool, str]:
        """
        Detect if a query is a broad meta-instruction for summarization or explanation.

        This function checks if the given English-language query is asking for a summary or explanation of some content,
        by matching the query against common patterns (using regular expressions) for requests like "summarize this"
        or "what are the main points". It also allows additional custom patterns to be included at runtime.

        Optimizations:
        - The query is lowercased and stripped for normalization.
        - A quick check for very short queries containing known trigger words.
        - Regular expressions are compiled and reused to improve performance.

        Args:
            query (str): The user query string.
            custom_patterns (list[str], optional): Additional regex patterns to treat as broad queries.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if the query is detected as a broad summarization/explanation request, False otherwise.
                - str: The processed query after enhancement.
        """
        if not query:
            return False, query  # Return original query if empty

        # Normalize whitespace and case
        q = query.strip().lower()

        # Heuristic: if the query is very short, check against a set of known triggers directly
        words = q.split()
        if len(words) <= 2:
            short_triggers = {
                "summary", "summarize", "summarise", "tl;dr", "overview",
                "synopsis", "main points", "key points", "gist", "explain"
            }
            if q in short_triggers:
                enhancer = QueryEnhancer()
                processed_query = enhancer.process_input(query)
                return True, processed_query

        # Initialize class-level regex cache if not already done
        if not hasattr(self, "_base_regex_cache"):
            self._base_regex_cache = None
            self._custom_regex_cache = {}

        # Compile base regex on first use (cache as class attribute)
        if self._base_regex_cache is None:
            # Base verbs for summarization/explanation requests
            base_verbs = [
                r"summarize", r"summarise",                 # e.g. "summarize"
                r"outline", r"recap", r"sum\s*it\s*up",     # e.g. "outline", "recap", "sum it up"
                r"synopsize", r"explain", r"highlight",     # e.g. "synopsize", "explain", "highlight"
                r"list", r"provide", r"give"               # e.g. "list (the main points)", "give (an overview)"
            ]
            # Terms that indicate a request for a summary or main points
            summary_terms = [
                r"summary", r"overview", r"synopsis", r"gist",
                r"main points?", r"key points?",           # "main point(s)", "key point(s)"
                r"main ideas?", r"key ideas?", r"takeaways?"
            ]
            # References to a document or content (to anchor the request to given text)
            doc_terms = [
                r"this", r"that", r"it", r"above",
                r"the (?:text|article|document|content|passage|chapter|paragraph)"
            ]

            # (a) Imperative requests (commands), possibly with polite prefixes
            # Examples: "summarize this", "please outline the article", "can you give an overview"
            imperative_pattern = rf"\b(?:(?:please\s*)?(?:can|could)\s*you\s*)?(?:{'|'.join(base_verbs)})\b"
            # After the verb, expect either a document reference or a summary-related noun
            imperative_pattern += rf".*?(?:\b(?:{'|'.join(doc_terms)})\b|\b(?:{'|'.join(summary_terms)})\b)"

            # (b) Question form requests for main points or summary
            # Examples: "what are the main points", "what is the gist of this"
            question_terms = [
                r"main point", r"main points", r"key point", r"key points",
                r"main idea", r"main ideas", r"key idea", r"key ideas",
                r"gist", r"summary", r"synopsis"
            ]
            question_pattern = rf"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:{'|'.join(question_terms)})\b"
            question_pattern += rf"(?:\s+of\s+(?:{'|'.join(doc_terms)})\b)?"  # optionally "of this/that..."

            # (c) Descriptive requests using "need/want/looking for"
            # Examples: "need a summary", "want an overview", "looking for a recap"
            request_pattern = rf"\b(?:need|want|looking for)\s+(?:(?:a|an|the)\s+)?(?:(?:quick|brief)\s+)?(?:{'|'.join(['summary','synopsis','overview','recap'])})\b"

            # (d) Help/explanation requests
            # Examples: "help me understand this"
            help_pattern = rf"\b(?:(?:please\s*)?(?:can|could)\s*you\s*)?help\s*me\s*understand\b"
            help_pattern += rf"(?:\s+(?:{'|'.join(doc_terms)})\b)?"

            # (e) Simple-language explanation requests
            # Examples: "explain in simple terms", "explain like I'm five"
            explain_simple_pattern = rf"\bexplain\b.*\b(?:in simple terms|in plain english|like i'?m\s+five)\b"

            # (f) TL;DR shorthand (too long; didn't read)
            tldr_pattern = r"\btl;?dr\b"

            # Combine all base patterns into one master pattern
            base_patterns = [
                imperative_pattern,
                question_pattern,
                request_pattern,
                help_pattern,
                explain_simple_pattern,
                tldr_pattern
            ]
            import re
            combined_base = r"(?:%s)" % "|".join(base_patterns)
            self._base_regex_cache = re.compile(combined_base, re.IGNORECASE)

        # Start with the precompiled base regex
        regex = self._base_regex_cache

        # If custom patterns are provided, merge them with base patterns (with caching)
        if custom_patterns:
            import re
            key = tuple(custom_patterns)
            if key not in self._custom_regex_cache:
                # Compile a new regex that includes both base patterns and custom patterns
                combined_pattern = self._base_regex_cache.pattern + "|" + "|".join(custom_patterns)
                self._custom_regex_cache[key] = re.compile(r"(?:%s)" % combined_pattern, re.IGNORECASE)
            regex = self._custom_regex_cache[key]

        # Return True if any of the patterns match the query
        match_result = bool(regex.search(q))

        enhancer = QueryEnhancer()
        # Process query through the enhancer
        processed_query = enhancer.process_input(query)

        return match_result, processed_query


#  def generate_response_stream(
#         self, 
#         query: str, 
#         agent: str,
#         max_chunks: int = 3,
#         include_history: bool = True,
#         temperature: float = 0.4,
#         repetition_penalty: float = 1.2,
#         window_size: int = 128
#     ) -> Generator[Tuple[str, float], None, None]:
#         """
#         Generate a streaming response based on the query, relevant context, and chat history.
        
#         Args:
#             query: User's question
#             agent: Type of agent to use ("finance", "tech", or default)
#             max_chunks: Maximum number of context chunks to retrieve
#             include_history: Whether to include chat history in the context
#             temperature: Temperature for text generation
#             repetition_penalty: Penalty for repetition in generated text
#             window_size: Window size for repetition penalty
            
#         Yields:
#             Tuple[str, float]: Generated text chunk and tokens/second metric
#         """
#         # Get conversation history with enhanced context awareness
#         if include_history:
#             # Enhanced history context that helps model understand references
#             history_context = self._format_enhanced_history(query)
#         else:
#             history_context = ""

#         is_financial_query, entity_type = self._is_financial_query(query)

#         print("checking the broad query ")
#         is_broad_query, modified_query = self._detect_broad_query(query)
#         print(modified_query)
#         print(is_broad_query)

#         if is_broad_query:
#             try:
#                 ngram_save_path = os.path.join(self.model_folder, "extracted_ngrams.json")
#                 with open(ngram_save_path, "r") as f:
#                     ngrams = json.load(f)
#                 query = modified_query
#                 bigrams = ", ".join(random.sample(ngrams["bigrams"], min(2, len(ngrams["bigrams"]))))
#                 key_words = '''These are some key words of the document {} '''.format(bigrams)
#                 query += key_words
#             except:
#                 query = modified_query
        

#         if self._is_out_of_context(query, agent):
#             # Load the n-grams from the JSON file
#             if (agent != "finance" and agent != "tech"):
#                 ngram_save_path = os.path.join(self.model_folder, "extracted_ngrams.json")
#                 try:
#                     with open(ngram_save_path, "r") as f:
#                         ngrams = json.load(f)
#                     # Get a few examples from each category
#                     unigrams = ", ".join(random.sample(ngrams["unigrams"], min(2, len(ngrams["unigrams"]))))
#                     bigrams = ", ".join(random.sample(ngrams["bigrams"], min(2, len(ngrams["bigrams"]))))
#                     trigrams = ", ".join(random.sample(ngrams["trigrams"], min(2, len(ngrams["trigrams"]))))

#                     # Create the suggestion message
#                     suggestion = (
#                         f"I apologize, but the query seems to be somewhat incomplete. Could you ask something more specific "
#                         f"based on the most frequent terms in the document?\n\n"
#                         f"Frequent terms:\n\n"
#                         f"• {unigrams}\n\n"
#                         f"• {bigrams}\n\n" 
#                         f"• {trigrams}\n\n"
#                         f"Your query could be something like:\n"
#                         f"\"Based on the following key terms: {bigrams}\n"
#                         f"Please provide a comprehensive summary of the document\""
#                     )

#                     yield suggestion, 0
#                     return
#                 except (FileNotFoundError, json.JSONDecodeError, KeyError):
#                     # Fallback if there's an issue with the JSON file
#                     yield "I apologize, but the query seems to be somewhat incomplete. Could you ask more elaboratively or something more specific about the context?", 0
#                     return
#             else:
#                 yield "I apologize, but the query seems to be somewhat incomplete. Could you ask more elaboratively or something more specific about the context?", 0
#                 return
        
#         # Retrieve relevant chunks based on query
#         relevant_chunks = self.retrieve_chunks(query, k=max_chunks)
#         if not relevant_chunks:
#             yield "I don't have enough context to provide a complete answer. Could you please rephrase or provide more details?", 0
#             return
        
#         # Add financial data context if relevant
#         if is_financial_query and agent == "finance":
#             financial_data = self._get_consistent_financial_data(query)
#             if financial_data:
#                 # Add a specific instruction for consistent financial data
#                 financial_context = (
#                     f"\nIMPORTANT: When answering about financial figures, use the following verified data:\n"
#                     f"- Value: {financial_data['value']}\n"
#                     f"- Metric: {financial_data['metric']}\n"
#                     f"- Region: {financial_data['region'] if financial_data['region'] else 'Overall'}\n"
#                     f"- Time Period: {financial_data['period'] if financial_data['period'] else 'Current'}\n"
#                     f"- Context: \"{financial_data['sentence']}\"\n"
#                 )

#                 # Append the financial context to the first chunk for emphasis
#                 relevant_chunks[0] = relevant_chunks[0] + financial_context

#         # Format context from chunks
#         context = self._format_context(relevant_chunks)
#         del relevant_chunks
#         gc.collect()

#         # Prepare messages based on agent type
#         if agent == "finance":
#             if self.finance_toggle == "true":
#                 # Advanced financial analysis system prompt
#                 messages = [
#                     {
#                         "role": "system",
#                         "content": self._get_finance_advanced_system_prompt()
#                     },
#                     {
#                         "role": "user",
#                         "content": (
#                             f"Context:\n{context}\n\n"
#                             f"{history_context}\n"  
#                             f"Question: {query}\n\n"
#                             "Please analyze the table data and provide a detailed answer to the question."
#                         )
#                     }
#                 ]
#             else:
#                 # Basic financial analysis system prompt
#                 messages = [
#                     {
#                         "role": "system",
#                         "content": self._get_finance_basic_system_prompt()
#                     },
#                     {
#                         "role": "user",
#                         "content": (
#                             f"Context:\n{context}\n\n"
#                             f"{history_context}\n"  
#                             f"Question: {query}\n\n"
#                             "Provide a clear, well-structured response using appropriate markdown formatting.\n\n"
#                             "IMPORTANT: The context may contain financial tables where rows and columns are separated by white spaces. "
#                             "When you encounter tabular data:\n"
#                             "1. Reconstruct the table structure using markdown table formatting\n"
#                             "2. Identify column headers and align numerical values properly\n"
#                             "3. For financial tables, pay attention to units (millions, billions) mentioned\n"
#                             "4. Treat '$' symbols and parentheses (indicating negative values) appropriately\n"
#                             "5. When answering questions about specific financial metrics, reference both the exact value and its location in the reconstructed table"
#                         )
#                     }
#                 ]
            
#         elif agent == "tech":
#             # Technical code analysis system prompt
#             messages = [
#                 {
#                     "role": "system",
#                     "content": self._get_tech_system_prompt()
#                 },
#                 {
#                     "role": "user",
#                     "content": (
#                         f"Technical Context:\n{context}\n\n"
#                         f"Related Code History:\n{history_context}\n\n"
#                         f"Technical Query: {query}\n\n"
#                         "Provide a detailed technical analysis that includes:\n"
#                         "1. Code overview and architecture implications.\n"
#                         "2. Specific implementation details.\n"
#                         "3. Best practices and potential optimizations.\n"
#                         "4. Example usage and edge cases.\n"
#                         "5. Default to python when the language is not specified by the user\n\n"
#                         "Use appropriate code blocks and technical formatting."
#                     )
#                 }
#             ]

#         elif agent == "meetings":
#             # Meeting analysis system prompt
#             messages = self.get_dynamic_meeting_prompt(query, context, history_context)

#         elif agent == "research":
#             messages = [
#                 {
#                     "role": "system",
#                     "content": self._get_document_overview_system_prompt()
#                 },
#                 {
#                     "role": "user",
#                     "content": (
#                         f"Document Overview Context:\n{context}\n\n"
#                         f"Related conversation History:\n{history_context}\n\n"
#                         f"Request: {query}\n\n"
#                         "Please provide a comprehensive overview of the document based on the context provided. "
#                         "Structure your response with clear headings and focus on the main points, themes, and insights "
#                         "from the document. If the document is a research paper, include any findings and conclusions."
#                     )
#                 }
#             ]

#         elif agent=="legal":
#             messages = [
#                 {
#                     "role": "system",
#                     "content": self._get_general_system_prompt()
#                 },
#                 {
#                     "role": "user",
#                     "content": (
#                         f"Context:\n{context}\n\n"
#                         f"Related conversation History:\n{history_context}\n\n"
#                         f"Question: {query}\n\n"
#                         "Important guidelines:\n"
#                         "- Please provide your response without unnecessary repetition. Avoid restating the same ideas, examples, or phrases.\n"
#                         "- You are not an attorney and cannot provide legal advice or legal opinions.\n"
#                         "- When rewriting text, preserve the original meaning but improve clarity and wording.\n"
#                         "- When summarizing, extract and clearly present the key points.\n"
#                         "- When answering questions, base your answers on the provided document text and known general concepts (without speculation).\n"
#                         "- Maintain a neutral, professional tone and explain legal terms if needed for understanding.\n"
#                         "- ALWAYS end every response with the disclaimer: Please consult a qualified attorney. This is an AI-generated suggestion only."
#                     )
#                 }
#             ]
#         else:
#             # Default general system prompt
#             messages = [
#                 {
#                     "role": "system",
#                     "content": self._get_general_system_prompt()
#                 },
#                 {
#                     "role": "user",
#                     "content": (
#                         f"Context:\n{context}\n\n"
#                         f"Related conversation History:\n{history_context}\n\n"
#                         f"Question: {query}\n\n"
#                         " Please provide your response without unnecessary repetition. Avoid restating the same ideas, examples, or phrases.\n"
#                         "Answer the question directly using only information from the context provided. "
#                         "Structure your response clearly with appropriate markdown formatting. "
#                         "Begin with a direct answer to the question before providing supporting details."
#                     )
#                 }
#             ]
        

#         # Apply chat template
#         prompt = self.tokenizer.apply_chat_template(
#             messages, 
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         # Convert to input IDs
#         input_ids = self.tokenizer(
#             prompt, 
#             return_tensors="pt"
#         ).input_ids.to(self.model.device)

#         # Set up streamer for efficient generation
#         from transformers import TextIteratorStreamer
#         streamer = TextIteratorStreamer(
#             self.tokenizer,
#             skip_prompt=True,
#             skip_special_tokens=True
#         )

#         # Configure generation parameters
#         gen_kwargs = dict(
#             input_ids=input_ids,
#             streamer=streamer,
#             max_new_tokens=self.max_tokens,
#             temperature=temperature,
#             repetition_penalty=repetition_penalty,
#             do_sample=temperature > 0,
#             use_cache=True  # Enable KV-cache
#         )

#         # Start generation in a separate thread
#         thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
#         thread.start()

#         # Stream tokens and track metrics
#         start = time.time()
#         generated_text = ""
#         for idx, token_text in enumerate(streamer, 1):
#             generated_text += token_text
#             tps = idx / (time.time() - start) if idx else 0
#             yield token_text, tps

#         # Update chat history
#         self.chat_history.add_interaction(query, generated_text)
    def generate_response_stream(
        self, 
        query: str, 
        agent: str,
        max_chunks: int = 3,
        include_history: bool = True,
        temperature: float = 0.4,
        repetition_penalty: float = 1.2,
        window_size: int = 128
    ) -> Generator[Tuple[str, float], None, None]:
        """
        Generate a streaming response based on the query, relevant context, and chat history.
        
        Args:
            query: User's question
            agent: Type of agent to use ("finance", "tech", or default)
            max_chunks: Maximum number of context chunks to retrieve
            include_history: Whether to include chat history in the context
            temperature: Temperature for text generation
            repetition_penalty: Penalty for repetition in generated text
            window_size: Window size for repetition penalty
            
        Yields:
            Tuple[str, float]: Generated text chunk and tokens/second metric
        """
        # Get conversation history with enhanced context awareness
        if include_history:
            history_context = self._format_enhanced_history(query)
        else:
            history_context = ""

        is_financial_query, entity_type = self._is_financial_query(query)

        print("checking the broad query ")
        is_broad_query, modified_query = self._detect_broad_query(query)
        print(modified_query)
        print(is_broad_query)

        if is_broad_query:
            try:
                ngram_save_path = os.path.join(self.model_folder, "extracted_ngrams.json")
                with open(ngram_save_path, "r") as f:
                    ngrams = json.load(f)
                query = modified_query
                bigrams = ", ".join(random.sample(ngrams["bigrams"], min(2, len(ngrams["bigrams"]))))
                key_words = '''These are some key words of the document {} '''.format(bigrams)
                query += key_words
            except:
                query = modified_query
        
        # Enhanced summary request detection
        cleaned_query = query.lower().strip()
        is_summary_request = any(
            word in cleaned_query for word in [
                "summarize", "summary", "overview", "summarise", 
                "summarize the", "summary of", "overview of",
                "what is this", "what's this", "tell me about", "what is the document about",
                "explain this", "describe this", "brief me",
                "tl;dr", "tldr", "main points"
            ]
        ) or cleaned_query in ["summarize", "summary", "overview"]

        # Helper function to generate suggestion message
        def generate_suggestion(agent: str) -> str:
            if agent in ["finance", "tech"]:
                return "I apologize, but the query seems to be somewhat incomplete. Could you ask more elaboratively or something more specific about the context?"
            
            try:
                ngram_save_path = os.path.join(self.model_folder, "extracted_ngrams.json")
                with open(ngram_save_path, "r") as f:
                    ngrams = json.load(f)
                
                # Get top important terms (first 2 from each category)
                unigrams = ", ".join(ngrams["unigrams"][:min(2, len(ngrams["unigrams"]))])
                bigrams = ", ".join(ngrams["bigrams"][:min(2, len(ngrams["bigrams"]))])
                trigrams = ", ".join(ngrams["trigrams"][:min(2, len(ngrams["trigrams"]))])

                return (
                    f"I apologize, but the query seems to be somewhat incomplete. Could you ask something more specific "
                    f"based on important terms from the document?\n\n"
                    f"Important terms:\n\n"
                    f"• {unigrams}\n\n"
                    f"• {bigrams}\n\n" 
                    f"• {trigrams}\n\n"
                    f"Your query could be something like:\n"
                    f"\"Based on the following key terms: {bigrams}\n"
                    f"Please provide a comprehensive summary of the document\""
                )
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                return "I apologize, but the query seems to be somewhat incomplete. Could you ask more elaboratively or something more specific about the context?"

        # Out-of-context check for non-summary requests
        if not is_summary_request and self._is_out_of_context(query, agent):
            suggestion = generate_suggestion(agent)
            yield suggestion, 0
            return

        # Retrieve relevant chunks based on query
        relevant_chunks = self.retrieve_chunks(query, k=max_chunks)
        
        # Special handling for summary requests across all agent types
        if is_summary_request:
            if not relevant_chunks:
                try:
                    ngram_save_path = os.path.join(self.model_folder, "extracted_ngrams.json")
                    with open(ngram_save_path, "r") as f:
                        ngrams = json.load(f)
                    
                    # Create fallback context from n-grams if no chunks found
                    fallback_context = (
                        "Key terms from the document include:\n"
                        f"- {', '.join(ngrams['unigrams'][:min(5, len(ngrams['unigrams']))])}\n"
                        f"- {', '.join(ngrams['bigrams'][:min(3, len(ngrams['bigrams']))])}\n"
                    )
                    relevant_chunks = [fallback_context]
                except Exception as e:
                    # If we can't even get ngrams, use a minimal fallback
                    relevant_chunks = ["The document contains various topics and information."]

            # Format context from whatever chunks we have
            context = self._format_context(relevant_chunks)
            
            # Prepare agent-specific summary prompts
            if agent == "finance":
                system_prompt = (
                    "You are a financial analyst providing document summaries. "
                    "Include:\n1. Key financial metrics\n2. Important trends\n"
                    "3. Major findings\nStructure with clear sections.\n"
                    "End with: 'NOTE: This summary provides a high-level overview based on available financial data.'"
                )
            elif agent == "tech":
                system_prompt = (
                    "You are a technical writer providing documentation summaries. "
                    "Include:\n1. Main components\n2. Key functions\n3. Architecture overview\n"
                    "Use technical headings. End with: 'NOTE: This technical summary is based on available code documentation.'"
                )
            elif agent == "research":
                system_prompt = (
                    "You are a research assistant providing academic summaries. "
                    "Include:\n1. Research question\n2. Methodology\n3. Key findings\n"
                    "4. Conclusions. End with: 'NOTE: This academic summary captures main points from available content.'"
                )
            elif agent == "legal":
                system_prompt = (
                    "You are a legal assistant providing document summaries. "
                    "Include:\n1. Key provisions\n2. Important clauses\n3. Legal implications\n"
                    "End with: 'NOTE: This legal summary is informational only. Consult an attorney for advice.'"
                )
            else:  # general case
                system_prompt = (
                    "You are an AI assistant providing comprehensive document summaries. "
                    "Include:\n1. Main topics\n2. Key points\n3. Important details\n"
                    "Structure clearly. End with: 'NOTE: The summary request was somewhat general. "
                    "This was the best summary based on available content.'"
                )
            
            messages = [{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Document content to summarize:\n{context}\n\nAdditional context:\n{history_context}"
            }]
            
            # Generate and stream the summary response
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            thread = threading.Thread(target=self.model.generate, kwargs={
                "input_ids": input_ids,
                "streamer": streamer,
                "max_new_tokens": self.max_tokens,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "use_cache": True
            })
            thread.start()
            
            start = time.time()
            generated_text = ""
            for idx, token_text in enumerate(streamer, 1):
                generated_text += token_text
                tps = idx / (time.time() - start) if idx else 0
                yield token_text, tps
            
            self.chat_history.add_interaction(query, generated_text)
            return

        # Handle non-summary requests with no relevant chunks
        elif not relevant_chunks:
            suggestion = generate_suggestion(agent)
            yield suggestion, 0
            return
        
        # Add financial data context if relevant
        if is_financial_query and agent == "finance":
            financial_data = self._get_consistent_financial_data(query)
            if financial_data:
                financial_context = (
                    f"\nIMPORTANT: When answering about financial figures, use the following verified data:\n"
                    f"- Value: {financial_data['value']}\n"
                    f"- Metric: {financial_data['metric']}\n"
                    f"- Region: {financial_data['region'] if financial_data['region'] else 'Overall'}\n"
                    f"- Time Period: {financial_data['period'] if financial_data['period'] else 'Current'}\n"
                    f"- Context: \"{financial_data['sentence']}\"\n"
                )
                relevant_chunks[0] = relevant_chunks[0] + financial_context

        # Format context from chunks
        context = self._format_context(relevant_chunks)
        del relevant_chunks
        gc.collect()

        # Prepare messages based on agent type
        if agent == "finance":
            if self.finance_toggle == "true":
                messages = [
                    {
                        "role": "system",
                        "content": self._get_finance_advanced_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\n"
                            f"{history_context}\n"  
                            f"Question: {query}\n\n"
                            "Please analyze the table data and provide a detailed answer to the question."
                        )
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": self._get_finance_basic_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\n"
                            f"{history_context}\n"  
                            f"Question: {query}\n\n"
                            "Provide a clear, well-structured response using appropriate markdown formatting.\n\n"
                            "IMPORTANT: The context may contain financial tables where rows and columns are separated by white spaces. "
                            "When you encounter tabular data:\n"
                            "1. Reconstruct the table structure using markdown table formatting\n"
                            "2. Identify column headers and align numerical values properly\n"
                            "3. For financial tables, pay attention to units (millions, billions) mentioned\n"
                            "4. Treat '$' symbols and parentheses (indicating negative values) appropriately\n"
                            "5. When answering questions about specific financial metrics, reference both the exact value and its location in the reconstructed table"
                        )
                    }
                ]
            
        elif agent == "tech":
            messages = [
                {
                    "role": "system",
                    "content": self._get_tech_system_prompt()
                },
                {
                    "role": "user",
                    "content": (
                        f"Technical Context:\n{context}\n\n"
                        f"Related Code History:\n{history_context}\n\n"
                        f"Technical Query: {query}\n\n"
                        "Provide a detailed technical analysis that includes:\n"
                        "1. Code overview and architecture implications.\n"
                        "2. Specific implementation details.\n"
                        "3. Best practices and potential optimizations.\n"
                        "4. Example usage and edge cases.\n"
                        "5. Default to python when the language is not specified by the user\n\n"
                        "Use appropriate code blocks and technical formatting."
                    )
                }
            ]

        elif agent == "meetings":
            messages = self.get_dynamic_meeting_prompt(query, context, history_context)

        elif agent == "research":
            messages = [
                {
                    "role": "system",
                    "content": self._get_document_overview_system_prompt()
                },
                {
                    "role": "user",
                    "content": (
                        f"Document Overview Context:\n{context}\n\n"
                        f"Related conversation History:\n{history_context}\n\n"
                        f"Request: {query}\n\n"
                        "Please provide a comprehensive overview of the document based on the context provided. "
                        "Structure your response with clear headings and focus on the main points, themes, and insights "
                        "from the document. If the document is a research paper, include any findings and conclusions."
                    )
                }
            ]

        elif agent=="legal":
            messages = [
                {
                    "role": "system",
                    "content": self._get_general_system_prompt()
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Related conversation History:\n{history_context}\n\n"
                        f"Question: {query}\n\n"
                        "Important guidelines:\n"
                        "- Please provide your response without unnecessary repetition. Avoid restating the same ideas, examples, or phrases.\n"
                        "- You are not an attorney and cannot provide legal advice or legal opinions.\n"
                        "- When rewriting text, preserve the original meaning but improve clarity and wording.\n"
                        "- When summarizing, extract and clearly present the key points.\n"
                        "- When answering questions, base your answers on the provided document text and known general concepts (without speculation).\n"
                        "- Maintain a neutral, professional tone and explain legal terms if needed for understanding.\n"
                        "- ALWAYS end every response with the disclaimer: Please consult a qualified attorney. This is an AI-generated suggestion only."
                    )
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": self._get_general_system_prompt()
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Related conversation History:\n{history_context}\n\n"
                        f"Question: {query}\n\n"
                        " Please provide your response without unnecessary repetition. Avoid restating the same ideas, examples, or phrases.\n"
                        "Answer the question directly using only information from the context provided. "
                        "Structure your response clearly with appropriate markdown formatting. "
                        "Begin with a direct answer to the question before providing supporting details."
                    )
                }
            ]
        

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )

        # Convert to input IDs
        input_ids = self.tokenizer(
            prompt, 
            return_tensors="pt"
        ).input_ids.to(self.model.device)

        # Set up streamer for efficient generation
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Configure generation parameters
        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=self.max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            use_cache=True  # Enable KV-cache
        )

        # Start generation in a separate thread
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Stream tokens and track metrics
        start = time.time()
        generated_text = ""
        for idx, token_text in enumerate(streamer, 1):
            generated_text += token_text
            tps = idx / (time.time() - start) if idx else 0
            yield token_text, tps

        # Update chat history
        self.chat_history.add_interaction(query, generated_text)
    def _format_enhanced_history(self, current_query: str) -> str:
        """
        Format conversation history with enhanced context awareness for follow-up questions.
        
        Args:
            current_query: The current user query
            
        Returns:
            str: Formatted history context with reference tracking
        """
        # Get raw history from chat history manager (assuming implementation exists)
        raw_history = self.chat_history.get_recent_interactions(max_count=1)  # Limit to recent interactions
        
        if not raw_history:
            return ""
            
        # Extract potential reference entities from current query
        potential_references = self._extract_reference_entities(current_query)
        
        # Format with reference highlighting
        formatted_history = "\nPrevious conversation with reference tracking:\n"
        
        for i, (past_query, past_answer) in enumerate(raw_history, 1):
            # Truncate long answers (preserving critical information)
            truncated_answer = self._smart_truncate(past_answer, max_length=1024)
            
            # Highlight key entities in previous Q&A that might be referenced
            highlighted_entities = self._highlight_key_entities(past_query, past_answer, potential_references)
            
            formatted_history += f"Previous Question {i}: {past_query}\n"
            
            if highlighted_entities:
                formatted_history += f"Key entities from this exchange: {', '.join(highlighted_entities)}\n"
                
            formatted_history += f"Previous Answer {i}: {truncated_answer}\n\n"
        
        # Add explicit instructions for reference resolution
        if potential_references:
            formatted_history += f"IMPORTANT - Current query contains potential references: {', '.join(potential_references)}\n"
            formatted_history += "Use the previous conversation to resolve these references correctly.\n\n"
            
        return formatted_history.strip()
    
    def _extract_reference_entities(self, query: str) -> List[str]:
        """
        Extract potential reference entities from current query.
        
        Args:
            query: Current user query
            
        Returns:
            List[str]: List of potential reference terms
        """
        # Simple pattern matching for common reference terms
        reference_patterns = [
            r'\b(it|this|that|these|those|they|them|their)\b',
            r'\b(the figure|the value|the number|the percentage|the ratio|the metric)\b',
            r'\b(percent change|change|difference|increase|decrease|growth|decline)\b',
            r'\b(how much|how many|compared to|versus|vs\.)\b',
            r'\b(previous|earlier|last|next|following)\b'
        ]
        
        found_references = []
        for pattern in reference_patterns:
            matches = re.findall(pattern, query.lower())
            found_references.extend(matches)
            
        return list(set(found_references))  # Remove duplicates
    
    def _highlight_key_entities(self, query: str, answer: str, potential_references: List[str]) -> List[str]:
        """
        Identify key entities in previous exchanges that might be referenced.
        
        Args:
            query: Previous query
            answer: Previous answer
            potential_references: List of reference terms in current query
            
        Returns:
            List[str]: Key entities identified
        """
        # If no potential references, don't process
        if not potential_references:
            return []
            
        # Patterns for financial metrics, dates, and quantities
        entity_patterns = [
            (r'\b(revenue|sales|income|profit|expense|cost|margin|ebitda|eps|dividend)\b', 'financial_metric'),
            (r'\b(20\d\d|q[1-4]|quarter|fiscal year|fy\d\d)\b', 'time_period'),
            (r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|m|b))?', 'monetary_value'),
            (r'\b\d+(?:\.\d+)?%\b', 'percentage')
        ]
        
        found_entities = []
        combined_text = query + " " + answer
        
        for pattern, entity_type in entity_patterns:
            matches = re.findall(pattern, combined_text.lower())
            for match in matches:
                if isinstance(match, tuple):  # Some regex patterns return tuples
                    match = match[0]
                found_entities.append(match)
        
        return list(set(found_entities))  # Remove duplicates
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """
        Smartly truncate text while preserving important information.
        
        Args:
            text: Text to truncate
            max_length: Maximum length to keep
            
        Returns:
            str: Truncated text
        """
        if len(text) <= max_length:
            return text
            
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Priority to keep first 2 and last 2 sentences
        if len(sentences) <= 4:
            return text[:max_length] + "..."
            
        # Keep first 2 sentences
        result = sentences[0] + " " + sentences[1]
        
        # Add marker for truncation
        result += " [...] "
        
        # Add last 2 sentences
        result += sentences[-2] + " " + sentences[-1]
        
        # If still too long, hard truncate
        if len(result) > max_length:
            return result[:max_length] + "..."
            
        return result
    

    def _get_document_overview_system_prompt(self) -> str:
        """Returns specialized system prompt for document overview queries."""
        return """You are a precise and helpful AI assistant specializing in document analysis. Follow these guidelines:    

        1. **Document Structure Analysis**
           - Identify key sections, their focus and purpose
           - Note the overall organization and flow of information
           - Highlight important transitions between topics

        2. **Main Points Extraction**
           - Identify and summarize the key arguments, findings, or claims
           - Extract major themes or through-lines in the document
           - Pay special attention to conclusions and their supporting evidence

        3. **Language and Formatting**
           - Use clear headings and subheadings to organize your response
           - Begin with a concise overview before providing section-by-section details
           - Use bullet points for key takeaways where appropriate

        4. **Academic Content Handling**
           - For research papers, clearly state the research question, methodology, and findings
           - For technical documents, emphasize key processes, requirements, or specifications
           - For informational texts, focus on main concepts and their explanations

        5. **Response Structure**
           - Start with "Document Overview:" followed by a 2-3 sentence high-level summary
           - Include "Main Themes:" with 3-5 bullet points on key topics
           - Follow with "Document Breakdown:" with section-by-section analysis
           - End with "Key Takeaways:" summarizing the most important points

        6. **Citation and Attribution**
           - When quoting directly from the document, use quotation marks
           - Attribute findings, statistics, or specific claims to the document
           - Be transparent about areas where the context may be incomplete
        """

    
    def get_dynamic_meeting_prompt(self,query, context, history_context=""):
        import re
        """
        Generate a dynamic meeting analysis prompt based on the user's query.

        Args:
            query (str): The user's query about the meeting
            context (str): The meeting transcript
            history_context (str): Related meeting history (optional)

        Returns:
            list: Messages list for the LLM with the appropriate prompt
        """
        # Convert query to lowercase for easier pattern matching
        q = query.lower().strip()

        # Base user content that will be in every prompt
        base_content = (
            f"Meeting Transcript:\n{context}\n\n"
        )

        # Add history context if available
        if history_context:
            base_content += f"Related Meeting History:\n{history_context}\n\n"

        base_content += f"Query: {query}\n\n"

        # Additional instructions based on query type
        specific_instructions = ""

        # Pattern matching for different query types
        if (any(term in q for term in ["summarize", "summary", "overview", "main points", 
                                      "key points", "highlights", "recap"]) 
            or q.startswith("give me a quick recap") 
            or q.startswith("give me a quick summary") 
            or re.search(r"^what\s+was\s+(?:this|the)\s+meeting\s+about", q)):
            # Summarization request
            specific_instructions = (
                "Provide a concise summary that captures the essence of the meeting. "
                "Focus on the main topics discussed, key decisions made, and important action items."
            )
        elif any(term in q for term in ["everything", "full analysis", "complete breakdown", 
                                        "all details", "comprehensive"]):
            # Comprehensive analysis
            specific_instructions = (
                "Provide a comprehensive analysis of this meeting, including:\n"
                "1. A summary of all key points discussed\n"
                "2. All action items (with assignees)\n"
                "3. Decisions made (and their context)\n"
                "4. Any timelines/deadlines mentioned\n"
                "5. Planned follow-ups or next steps\n\n"
                "Structure the response with clear headings and use bullet points where appropriate."
            )
        elif any(term in q for term in ["action", "action item", "task", "to-do", "to do", "assigned", "responsibility"]):
            # Action items query
            specific_instructions = (
                "Identify all action items mentioned in the meeting. For each action item:\n"
                "- State what needs to be done\n"
                "- Who is responsible for it (if mentioned)\n"
                "- Any deadline or timeline noted\n"
                "Present the action items in a clear, bulleted list."
            )
        elif any(term in q for term in ["decision", "decided", "agreement", "conclusion", "agreed", "resolved"]):
            # Decisions made query
            specific_instructions = (
                "Extract all decisions made during the meeting. For each decision:\n"
                "- State the decision clearly\n"
                "- Provide context or reasons if mentioned\n"
                "- Note any dissent or alternatives discussed\n"
                "Present these decisions in a clear, structured format."
            )
        elif any(term in q for term in ["timeline", "deadline", "due date", "by when", "schedule", "due", "when"]):
            # Deadlines/timelines query
            specific_instructions = (
                "List all deadlines or timelines mentioned in the meeting. Include:\n"
                "- The task or deliverable and its deadline\n"
                "- Who is responsible (if specified)\n"
                "Present this information chronologically or by urgency."
            )
        elif any(term in q for term in ["follow-up", "follow up", "next steps", "next meeting", "future"]):
            # Follow-ups and next steps query
            specific_instructions = (
                "Identify any follow-up actions and upcoming meetings discussed. Include:\n"
                "- Planned follow-up meetings (with dates if given)\n"
                "- Next steps or future actions for the team\n"
                "Provide these details in a clear, organized manner."
            )
        elif any(term in q for term in ["who", "attendee", "participant", "said", "mentioned"]):
            # Participant-specific query
            specific_instructions = (
                "Focus on what specific individuals said or were assigned during the meeting. "
                "Organize the information by person, mentioning their contributions or responsibilities."
            )
        elif any(term in q for term in ["topic", "discuss", "subject", "about"]):
            # Topic-specific query
            import re
            topic = None
            match = re.search(r'about\s+(.*)$', q)
            if match:
                topic = match.group(1).strip().rstrip('?.')

            if topic:
                specific_instructions = (
                    f"Focus on the discussion about \"{topic}\" during the meeting. "
                    "Summarize what was said, any decisions made, and relevant action items on that topic."
                )
            else:
                specific_instructions = (
                    "Identify the main topics discussed in the meeting and provide a brief overview of each. "
                    "Structure the response by topic."
                )
        else:
            # Default fallback
            if not re.search(r'\b(who|what|when|where|why|how)\b', q):
                # No explicit wh-question words – treat as a generic "overview" query
                specific_instructions = (
                    "Provide a brief summary of the meeting, highlighting the key points, decisions, and any action items."
                )
            else:
                # There is a question word, but it didn't match specific categories
                specific_instructions = (
                    "Analyze the meeting and answer the user's query directly, using only relevant details from the transcript."
                )

        # Final user content
        user_content = base_content + specific_instructions + "\n\nUse appropriate formatting with headers, bullet points, and clear structure. Please provide your response without unnecessary repetition. Avoid restating the same ideas, examples, or phrases.\n"

        # Construct the messages list
        messages = [
            {
                "role": "system",
                "content": self._get_meetings_system_prompt()
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        return messages

    def _get_meetings_system_prompt(self) -> str:
        """Returns the meetings system prompt with contextual understanding."""
        return """You are a specialized meeting assistant with expertise in analyzing meeting transcripts and audio recordings. 
        Follow these guidelines:

        1. **Meeting Summary**
           - Distill key topics, discussions, and outcomes into a concise overview.
           - Maintain chronological flow of major discussion points.
           - Highlight pivotal moments and critical information exchanges.
           - Focus on substantive content rather than casual exchanges.

        2. **Action Item Extraction**
           - Identify and clearly list all tasks/action items mentioned.
           - Include who is responsible for each action item when specified.
           - Note deadlines and timeframes for completion when mentioned.
           - Distinguish between confirmed action items and suggested tasks.

        3. **Decision Documentation**
           - Precisely document decisions made during the meeting.
           - Note approval processes and voting outcomes if applicable.
           - Identify consensus points and areas of agreement.
           - Highlight any postponed decisions or items requiring further discussion.

        4. **Participant Analysis**
           - Identify key contributors and their main points.
           - Note questions raised and responses provided.
           - Recognize when participants express concerns or enthusiasm.
           - Track discussion ownership and handoffs between speakers.

        5. **Communication Patterns**
           - Identify information sharing, problem-solving, and decision-making segments.
           - Note discussion intensity and focus areas based on transcript length/detail.
           - Recognize topic transitions and discussion evolution.
           - Identify recurring themes or frequently mentioned topics.

        6. **Response Formatting**
           - Use clear headings for different sections (Summary, Action Items, Decisions, etc.).
           - Present action items in bulleted or numbered lists for clarity.
           - Organize information in order of importance or chronology as appropriate.
           - Use bold formatting for action item owners and deadlines.
           
        7. **Handling Conversation Context**
           - When a user asks follow-up questions about the meeting, refer to your previous analysis.
           - If asked about specific participants, connect to their contributions identified earlier.
           - Maintain awareness of previously discussed meeting sections when focusing on specifics.
           - For queries about "key takeaways" or "main points," prioritize decisions and action items.

        """
        
    def _get_finance_advanced_system_prompt(self) -> str:
        """Returns the advanced finance system prompt with contextual understanding."""
        return """You are an expert financial analyst specializing in SEC filings, annual reports, and corporate financial disclosures. Your task is to analyze tables extracted from these documents and provide precise, professional insights.
Below data is extracted from a formatted csv , convert it into a table format and answer the following question
                """
        
    def _get_finance_basic_system_prompt(self) -> str:
        """Returns the basic finance system prompt with contextual understanding."""
        return """You are a precise and knowledgeable financial analysis AI assistant specializing in SEC filings and annual reports. Follow these guidelines:

        1. **Extract and reconstruct tables** from unstructured text by identifying column patterns, alignments, and data relationships.
        2. **Focus on extracting precise numerical data** from financial statements, especially revenue, income, and growth metrics.
        3. **Structure your response clearly** with headings, tables, and bullet points using markdown.
        4. **Always cite specific sections** of the document when providing financial figures.
        5. **Reconstruct tables** when answering questions about tabular data.
        6. **Be precise with financial terminology** and maintain accounting accuracy.
        7. **For year-over-year comparisons**, clearly state both values and the percentage change.
        8. **When uncertain about exact figures**, indicate this rather than approximating.
        9. **Do not hallucinate data** - if information is not in the provided context, state this clearly.
        10. **For trend analysis**, use data from multiple years when available in the context.
        
        IMPORTANT - HANDLING CONVERSATION CONTEXT:
        - When a user asks follow-up questions, pay attention to what was discussed in previous messages
        - If the current question contains references like "it", "this figure", or "percent change" without specifying the metric, refer to the previous questions and answers
        - Always specify which financial metric you're discussing based on conversation history
        - When calculating changes or comparisons, clearly identify which values from previous discussions you're using
        - Begin responses to follow-up questions by stating what you understand the user is referring to
        """
        
    def _get_tech_system_prompt(self) -> str:
        """Returns the tech system prompt with contextual understanding."""
        return """You are a specialized coding assistant with expertise in software development and technical documentation. 
        Follow these guidelines:

        1. **Code Analysis**
           - Identify programming patterns, dependencies, and potential issues.
           - Explain complex code segments with clear examples.
           - Highlight best practices and suggest improvements.
           - Consider performance implications.

        2. **Documentation Focus**
           - Extract key technical requirements and specifications.
           - Maintain technical accuracy in explanations.
           - Preserve important code comments and documentation structure.
           - Reference relevant API docs or technical standards.

        3. **Response Structure**
           - Use proper code formatting with language-specific syntax highlighting.
           - Include input/output examples where relevant.
           - Break down complex implementations step-by-step.
           - Distinguish between different code components (e.g., functions, classes, modules).

        4. **Context Awareness**
           - Consider the codebase's overall architecture.
           - Note dependencies between different files.
           - Maintain version compatibility awareness.
           - Respect existing coding standards and patterns.
           
        5. **Handling Conversation Context**
           - When a user asks about specific code elements ("this function", "that class", etc.), refer to the conversation history
           - Connect new questions to previously discussed code elements
           - Maintain continuity in technical discussions by referencing previous code explanations
           - For follow-up questions about implementation details, recall the code context from earlier in the conversation
        """
        
    def _get_general_system_prompt(self) -> str:
        """Returns the general system prompt with contextual understanding."""
        return """You are a precise and efficient AI assistant. Follow these guidelines:

        1. **Analyze the question first** to identify exactly what information is needed.
        2. **Provide direct, concise answers** that address the core question immediately.
        3. **Structure your response with clear headings and organization** using markdown.
        4. **Use bullet points and numbered lists** for clarity where appropriate.
        5. **Focus exclusively on relevant information** from the provided context.
        6. **Include specific quotes or data points** from the context to support key statements.
        7. **Avoid unnecessary explanations** of your reasoning process.
        8. **Only use information present in the context** - do not add external knowledge.
        9. **If the context doesn't contain the answer**, clearly state this rather than guessing.
        10. **Keep explanations brief but complete** - prioritize accuracy over verbosity.
        
        IMPORTANT - HANDLING CONVERSATION CONTEXT:
        - When the user asks follow-up questions, refer to previous messages to understand what they're referring to
        - If the user asks about "it", "this", "that" or uses other pronouns, identify what these refer to from the conversation history
        - Always begin by clarifying what you understand the question to be asking based on the conversation context
        - Maintain continuity between responses by connecting new information to previously discussed topics
        """

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history.clear()

    def _format_context(self, chunks: List[str]) -> str:
        """Format context chunks for better prompt structure."""
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            # Strip out 'Human:' or 'Assistant:' from the chunk
            cleaned_chunk = re.sub(r'^(Human|Assistant):\s*', '', chunk, flags=re.MULTILINE).strip()

            if cleaned_chunk.startswith(('# ', '## ', '### ')):
                formatted_chunks.append(f"\n{cleaned_chunk}")
            else:
                formatted_chunks.append(f"Excerpt {i}:\n{cleaned_chunk}")

        return "\n\n".join(formatted_chunks)
       

    def _format_history(self, query: str) -> str:
        """Format chat history for better context integration, removing role labels."""
        history = self.chat_history.get_context_string(query)
        if not history:
            return ""

        # Remove lines that start with "Human:" or "Assistant:" (case-insensitive if needed)
        cleaned_history = re.sub(r'^(Human|Assistant):\s*', '', history, flags=re.MULTILINE)

        return f"Previous relevant discussion:\n{cleaned_history}" 

    # 3. Improved Response Post-processing

import torch

class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float = 1.4, window_size: int = 128):
        """
        Initialize repetition penalty processor with penalty factor and window size.
        """
        self.penalty = penalty
        self.window_size = window_size
        
    def __call__(self, input_ids, scores):
        """
        Apply repetition penalty to the scores (logits) using element-wise operations.
        
        Args:
            input_ids: Tensor of token IDs in the sequence so far
            scores: Logits tensor to modify
            
        Returns:
            Modified logits tensor with repetition penalty applied
        """
        # If the sequence is smaller than window size, just use the whole thing
        seq_length = input_ids.shape[1]
        window_size = min(self.window_size, seq_length)
        
        # Get recent tokens within the specified window
        recent_tokens = input_ids[:, -window_size:].reshape(-1)
        unique_tokens = torch.unique(recent_tokens).tolist()
        
        # Prepare to update the scores
        scores_copy = scores.clone()
        
        # Apply penalty to each unique token in recent history
        for token in unique_tokens:
            # Select scores corresponding to the token from the scores tensor
            token_scores = scores[:, token]
            
            # Apply the penalty (different calculation for positive and negative scores)
            penalized_scores = torch.where(
                token_scores < 0,
                token_scores * self.penalty,  # Make negative scores more negative
                token_scores / self.penalty   # Make positive scores smaller
            )
            
            # Replace the score for that token with the penalized score
            scores_copy[:, token] = penalized_scores
            
        return scores_copy
    

import os
import gc
import time


def create_training_files_with_feedback(model_folder, output_dir, chunk_size=1000, chunk_overlap=200):
    """
    Independent function to create training and validation files by combining existing text files
    with positive feedback responses.
    
    Args:
        model_folder (str): Path to the folder containing the feedback.json file
        output_dir (str): Path to the folder containing text files and where train.jsonl/valid.jsonl will be saved
        chunk_size (int): Size of text chunks for training data
        chunk_overlap (int): Overlap between chunks
    """
    import os
    import glob
    import json
    import random
    import gc
    import re
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    print(f"Creating training files with feedback from {model_folder}")
    
    # Function to clean text
    def clean_text(text):
        """
        Gently clean text by removing formatting and structural elements while preserving
        the actual content and information.

        Args:
            text (str): The text to clean

        Returns:
            str: Cleaned text that preserves important content
        """
        import re

        # Remove markdown formatting but preserve content

        # Replace header formatting (# Header) with plain text
        text = re.sub(r'^#+\s+(.*?)$', r'\1', text, flags=re.MULTILINE)

        # Remove horizontal rules
        text = re.sub(r'={3,}|-{3,}|\*{3,}', '', text)

        # Convert bullet points to simple text
        text = re.sub(r'^\s*[\*\-\+]\s+(.*?)$', r'\1', text, flags=re.MULTILINE)

        # Remove section markers but keep the content
        text = re.sub(r'(?i)^(?:direct answer|supporting details|evidence|reasoning|document summary|key terms and definitions|summary explanation|example answer)(?:\s+\(.*?\))?:?$', '', text, flags=re.MULTILINE)

        # Remove but preserve content from Roman numeral headers
        text = re.sub(r'^[IVXLCDM]+\.\s+(.*?)$', r'\1', text, flags=re.MULTILINE)

        # Remove but preserve content from numeric headers
        text = re.sub(r'^\d+\.\s+(.*?)$', r'\1', text, flags=re.MULTILINE)

        # Remove markdown bold and italic formatting while keeping the content
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)       # Italic

        # Remove AI-specific phrases
        text = re.sub(r'(?i)(?:as an ai language model,?|i apologize,? but|as a language model,?|here\'s what|based on the (?:available )?context)', '', text)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with 2

        # Remove empty lines
        text = re.sub(r'^\s*$\n', '', text, flags=re.MULTILINE)

        # Strip whitespace
        text = text.strip()

        return text
    
    # Load all .txt files in the output directory
    all_cleaned_texts = []
    txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
    
    if txt_files:
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    if file_content:
                        all_cleaned_texts.append(file_content)
                        print(f"Loaded text from {txt_file}")
            except Exception as e:
                print(f"Error loading text from {txt_file}: {str(e)}")
        
        print(f"Loaded content from {len(txt_files)} text files")
    else:
        print("No .txt files found in the output directory")
    
    # Load feedback data
    feedback_file = os.path.join(model_folder, 'feedback.json')
    feedback_messages = []
    
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
                # Extract assistant messages from the feedback
                if isinstance(feedback_data, list):
                    feedback_messages = [clean_text(item.get('assistant_message', '')) for item in feedback_data if item.get('is_positive', False)]
                elif isinstance(feedback_data, dict):
                    if feedback_data.get('is_positive', False):
                        feedback_messages = [clean_text(feedback_data.get('assistant_message', ''))]
                    else:
                        feedback_messages = []
                
                # Filter out empty messages after cleaning
                feedback_messages = [msg for msg in feedback_messages if msg.strip()]
                print(feedback_messages)
                print(f"Loaded {len(feedback_messages)} positive feedback messages")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read feedback file: {e}")
    
    # Process the feedback messages
    if feedback_messages:
        feedback_text = "\n\n".join(feedback_messages)
        all_cleaned_texts.append(feedback_text)
        print(f"Added {len(feedback_messages)} feedback messages to training data")
    
    # Verify we have content to train on
    if not all_cleaned_texts:
        raise ValueError("No text content or feedback messages available for training")
    
    # Now create the training data
    combined_text = "\n\n".join(all_cleaned_texts)
    
    train_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size // 2,
        chunk_overlap=chunk_overlap // 2
    )
    all_chunks = train_splitter.split_text(combined_text)
    
    # Shuffle for better training
    random.shuffle(all_chunks)
    
    # Split into train and validation sets
    split_idx = int(len(all_chunks) * 0.8)
    train_chunks = all_chunks[:split_idx]
    valid_chunks = all_chunks[split_idx:]
    
    train_data = [{"text": chunk} for chunk in train_chunks]
    valid_data = [{"text": chunk} for chunk in valid_chunks]
    
    if len(train_data) < 5:
        raise ValueError("Not enough content to finetune (need at least 5 training samples)")
    
    # Save the new training files
    save_jsonl(train_data, os.path.join(output_dir, 'train.jsonl'))
    save_jsonl(valid_data, os.path.join(output_dir, 'valid.jsonl'))
    
    print(f"Created combined training files in {output_dir}")
    print(f"Train samples: {len(train_data)}, Validation samples: {len(valid_data)}")
    print(f"Included {len(feedback_messages)} feedback messages")
    
    # Clean up memory
    del train_data
    del valid_data
    del all_chunks
    gc.collect()

def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

import os

def derive_output_dir(model_folder):
    """
    Automatically derive the output_dir from the model_folder path structure.
    
    Args:
        model_folder (str): Path to the folder containing the feedback.json file
        
    Returns:
        str: The derived output_dir path
    """
    # Get the base directory (up to Decompute-Files)
    base_parts = model_folder.split('Decompute-Files')
    if len(base_parts) < 2:
        raise ValueError("Invalid model_folder path structure. Expected 'Decompute-Files' in path.")
    
    base_dir = base_parts[0] + 'Decompute-Files'
    
    # Extract the model identifier (e.g., Decompute_Safe_79)
    folder_name = os.path.basename(model_folder)
    
    # Construct the output directory
    output_dir = os.path.join(base_dir, 'uploads', folder_name)
    
    return output_dir

def run_training_with_derived_paths(model_folder, chunk_size=1000, chunk_overlap=200):
    """
    Run create_training_files_with_feedback with automatically derived output_dir
    
    Args:
        model_folder (str): Path to the folder containing the feedback.json file
        chunk_size (int): Size of text chunks for training data
        chunk_overlap (int): Overlap between chunks
    """
    output_dir = derive_output_dir(model_folder)
    print(f"Derived output_dir: {output_dir}")
    
    # Now call the original function with the derived path
    create_training_files_with_feedback(model_folder, output_dir, chunk_size, chunk_overlap)
    
    return output_dir



class QueryEnhancer:
    def __init__(self, stopwords=None):
        try:
            if stopwords is None:
                stopwords = set([
                    "the", "and", "or", "if", "in", "on", "of", "for", "to", "is", "are", "am",
                    "was", "were", "this", "that", "it", "its", "those", "these",
                    "a", "an", "above", "below", "i", "me", "you", "your", "yours",
                    "we", "our", "us", "can", "could", "should", "would", "may", "might", "will",
                    "please", "what", "when", "where", "who", "why", "how",
                    "do", "does", "did", "done", "doing", "have", "has", "had", "having",
                    "be", "being", "been", "at", "as", "by", "with", "not", "too", "also",
                    "any", "some", "no", "nor", "all", "just", "but", "there", "their",
                    "then", "than", "so", "such", "because", "about", "into", "it's", "they", "them",
                    "help", "need", "want", "looking", "give", "provide", "list", "show", "tell", "say", "which"
                ])
                broad_triggers = {
                    "summary", "summarize", "summarise", "overview", "synopsis", "gist",
                    "main", "points", "point", "key", "keys", "ideas", "idea", "takeaways",
                    "explain", "explanation", "outline", "recap", "highlight", "highlights",
                    "tl", "dr", "tl;dr", "tldr"
                }
                stopwords |= broad_triggers
            self.stopwords = stopwords
        except Exception as e:
            print(f"[Init Error] Failed to initialize stopwords: {e}")
            self.stopwords = set()

    def detect_broad_query(self, query: str) -> bool:
        try:
            if not query:
                return False
            q = query.strip().lower()
            if len(q.split()) <= 2:
                short_triggers = {
                    "summary", "summarize", "summarise", "tl;dr", "overview",
                    "synopsis", "main points", "key points", "gist", "explain"
                }
                if q in short_triggers:
                    return True
            broad_verbs = ["summarize", "summarise", "outline", "recap", "synopsize",
                          "explain", "highlight", "list", "provide", "give", "sum it up"]
            vague_terms = ["this", "that", "it", "above", "below",
                          "the text", "the article", "the document", "the content",
                          "the passage", "the chapter", "the paragraph"]
            summary_terms = ["summary", "overview", "synopsis", "gist",
                           "main points", "key points", "main ideas", "key ideas", "takeaways"]
            verb_present = any(v in q for v in broad_verbs)
            vague_present = any(vt in q for vt in vague_terms)
            summary_present = any(st in q for st in summary_terms)
            if verb_present and (vague_present or summary_present):
                return True
            if re.search(r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:" + "|".join(map(re.escape, summary_terms)) + r")\b", q):
                return True
            if re.search(r"help\s*me\s*understand", q):
                return True
            if "tl;dr" in q or q.replace(" ", "") == "tldr":
                return True
            return False
        except Exception as e:
            print(f"[Detect Error] Failed to detect broad query: {e}")
            return False

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        try:
            if not text:
                return []
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
            words = clean_text.split()
            words = [w for w in words if w not in self.stopwords and len(w) > 2 and not w.isdigit()]
            if not words:
                return []
            freq = Counter(words)
            bigram_freq = Counter()
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                bigram_freq[f"{w1} {w2}"] += 1
            common_words = [w for w, _ in freq.most_common(max_keywords * 2)]
            common_bigrams = [b for b, count in bigram_freq.most_common(max_keywords * 2) if count >= 2]
            keywords = []
            for w in common_words:
                base = w.rstrip('s')
                if base in keywords or w in keywords:
                    continue
                keywords.append(w)
                if len(keywords) >= max_keywords:
                    break
            final_keywords = []
            kw_set = set(keywords)
            for phrase in common_bigrams:
                w1, w2 = phrase.split()
                if w1 in kw_set and w2 in kw_set:
                    final_keywords.append(phrase)
                    kw_set.discard(w1)
                    kw_set.discard(w2)
                elif (w1 in kw_set) ^ (w2 in kw_set):
                    if w1 in kw_set: kw_set.discard(w1)
                    if w2 in kw_set: kw_set.discard(w2)
                    final_keywords.append(phrase)
            for w in kw_set:
                final_keywords.append(w)
            if len(final_keywords) < max_keywords:
                used_words = {w for kw in final_keywords for w in kw.split()}
                extras = []
                for b in common_bigrams:
                    if b in final_keywords: continue
                    w1, w2 = b.split()
                    if w1 in used_words or w2 in used_words:
                        continue
                    extras.append((b, bigram_freq[b]))
                for w in common_words:
                    if w in used_words:
                        continue
                    extras.append((w, freq[w]))
                extras.sort(key=lambda x: x[1], reverse=True)
                for term, _ in extras:
                    final_keywords.append(term)
                    used_words.update(term.split())
                    if len(final_keywords) >= max_keywords:
                        break
            return final_keywords[:max_keywords]
        except Exception as e:
            print(f"[Keyword Extraction Error] Failed to extract keywords: {e}")
            return []

    def enhance(self, query: str, context_text: str = None, prev_query: str = None) -> str:
        try:
            if not self.detect_broad_query(query):
                return query
            q_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower())
            tokens = [t for t in q_clean.split() if t and t not in self.stopwords]
            remaining = " ".join(tokens).strip()
            keywords = []
            if context_text:
                keywords = self._extract_keywords(context_text, max_keywords=5)
            elif prev_query:
                keywords = self._extract_keywords(prev_query, max_keywords=5)
            if not keywords:
                return f"summary of {remaining or 'the content'}"
            extra_words = []
            for w in remaining.split():
                if not any(w in kw for kw in keywords):
                    extra_words.append(w)
            if extra_words:
                keywords += extra_words
            unique_keywords = []
            seen = set()
            for kw in keywords:
                if kw in seen:
                    continue
                if any((kw != phrase and kw in phrase.split()) for phrase in keywords):
                    seen.add(kw)
                    continue
                unique_keywords.append(kw)
                seen.add(kw)
            keywords = unique_keywords[:5]
            if len(keywords) == 1:
                return f"summary of {keywords[0]}"
            elif len(keywords) == 2:
                return f"summary of {keywords[0]} and {keywords[1]}"
            else:
                return "summary of " + ", ".join(keywords[:-1]) + ", and " + keywords[-1]
        except Exception as e:
            print(f"[Enhance Error] Failed to enhance query: {e}")
            return query

    # Compatibility methods to maintain existing interface
    def enhance_query(self, query: str) -> str:
        """Compatibility method to match old interface"""
        return self.enhance(query)

    def process_input(self, user_input: str) -> str:
        """Process user input and return enhanced query"""
        return self.enhance(user_input)
