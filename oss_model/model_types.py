"""Model management types and configurations for the Decompute SDK."""

import os
import time
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
"""Model types and configurations for the Blackbird SDK."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

class ModelType(Enum):
    """Supported model types."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    FINE_TUNED = "fine_tuned"

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    model_name: str
    model_type: ModelType = ModelType.CHAT
    temperature: float = 0.7
    max_tokens: int = 2000
    context_length: int = 4096
    stop_sequences: List[str] = None
    custom_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
        if self.custom_parameters is None:
            self.custom_parameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'context_length': self.context_length,
            'stop_sequences': self.stop_sequences,
            'custom_parameters': self.custom_parameters
        }

@dataclass
class ModelCapabilities:
    """Model capabilities and limitations."""
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    max_context_length: int = 4096
    pricing_per_token: float = 0.0

class ModelType(Enum):
    """Types of AI models supported."""
    LLM = "llm"  # Large Language Models
    EMBEDDING = "embedding"
    IMAGE_GEN = "image_generation"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

class ModelFormat(Enum):
    """Model file formats."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    GGML = "ggml"
    TENSORFLOW = "tensorflow"
    HUGGINGFACE = "huggingface"

class ModelPrecision(Enum):
    """Model precision levels."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    DYNAMIC = "dynamic"

class ModelSource(Enum):
    """Sources for model downloads."""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    OPENAI = "openai"
    LOCAL = "local"
    URL = "url"
    DECOMPUTE = "decompute"

class ModelStatus(Enum):
    """Model lifecycle status."""
    DOWNLOADING = "downloading"
    AVAILABLE = "available"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    QUANTIZING = "quantizing"
    OPTIMIZING = "optimizing"

@dataclass
class ModelInfo:
    """Comprehensive model information."""
    model_id: str
    name: str
    model_type: ModelType
    format: ModelFormat
    source: ModelSource
    status: ModelStatus
    
    # Model specifications
    size_bytes: int = 0
    precision: ModelPrecision = ModelPrecision.FP32
    context_length: int = 2048
    parameter_count: Optional[int] = None
    
    # File information
    local_path: Optional[str] = None
    download_url: Optional[str] = None
    checksum: Optional[str] = None
    
    # Metadata
    description: str = ""
    version: str = "1.0.0"
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    
    # Performance data
    load_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    inference_speed_tokens_per_sec: float = 0.0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size_mb(self) -> float:
        """Get model size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        """Get model size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)
    
    @property
    def age_days(self) -> float:
        """Get model age in days."""
        return (time.time() - self.created_at) / 86400
    
    @property
    def is_local(self) -> bool:
        """Check if model is stored locally."""
        return self.local_path is not None and os.path.exists(self.local_path)
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = time.time()
        self.use_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'type': self.model_type.value,
            'format': self.format.value,
            'source': self.source.value,
            'status': self.status.value,
            'size': {
                'bytes': self.size_bytes,
                'mb': self.size_mb,
                'gb': self.size_gb
            },
            'precision': self.precision.value,
            'specs': {
                'context_length': self.context_length,
                'parameter_count': self.parameter_count,
                'description': self.description,
                'version': self.version
            },
            'files': {
                'local_path': self.local_path,
                'download_url': self.download_url,
                'checksum': self.checksum,
                'is_local': self.is_local
            },
            'usage': {
                'created_at': self.created_at,
                'last_used': self.last_used,
                'use_count': self.use_count,
                'age_days': self.age_days
            },
            'performance': {
                'load_time_ms': self.load_time_ms,
                'memory_usage_mb': self.memory_usage_mb,
                'inference_speed': self.inference_speed_tokens_per_sec
            },
            'config': self.config,
            'metadata': self.metadata
        }

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    target_precision: ModelPrecision
    calibration_dataset: Optional[str] = None
    num_calibration_samples: int = 100
    preserve_accuracy: bool = True
    target_size_reduction: float = 0.5  # 50% size reduction target
    optimization_level: str = "balanced"  # fast, balanced, quality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'target_precision': self.target_precision.value,
            'calibration_dataset': self.calibration_dataset,
            'num_calibration_samples': self.num_calibration_samples,
            'preserve_accuracy': self.preserve_accuracy,
            'target_size_reduction': self.target_size_reduction,
            'optimization_level': self.optimization_level
        }

@dataclass
class ModelStats:
    """Model service statistics."""
    total_models: int = 0
    loaded_models: int = 0
    total_size_gb: float = 0.0
    cache_hit_rate: float = 0.0
    average_load_time_ms: float = 0.0
    downloads_completed: int = 0
    downloads_failed: int = 0
    quantizations_completed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_models': self.total_models,
            'loaded_models': self.loaded_models,
            'total_size_gb': self.total_size_gb,
            'cache_hit_rate': self.cache_hit_rate,
            'average_load_time_ms': self.average_load_time_ms,
            'downloads': {
                'completed': self.downloads_completed,
                'failed': self.downloads_failed,
                'success_rate': self.downloads_completed / max(1, self.downloads_completed + self.downloads_failed)
            },
            'quantizations_completed': self.quantizations_completed
        }

def generate_model_id(name: str, version: str = "1.0.0") -> str:
    """Generate a unique model ID."""
    identifier = f"{name}:{version}:{time.time()}"
    return hashlib.md5(identifier.encode()).hexdigest()

def calculate_checksum(file_path: str) -> str:
    """Calculate file checksum for integrity verification."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""

def estimate_model_memory(parameter_count: int, precision: ModelPrecision) -> float:
    """Estimate model memory usage in MB."""
    if parameter_count is None:
        return 0.0
    
    bytes_per_param = {
        ModelPrecision.FP32: 4,
        ModelPrecision.FP16: 2,
        ModelPrecision.INT8: 1,
        ModelPrecision.INT4: 0.5,
        ModelPrecision.DYNAMIC: 2  # Average estimate
    }
    
    param_bytes = parameter_count * bytes_per_param.get(precision, 4)
    # Add overhead for activations, gradients, etc.
    overhead_multiplier = 1.5
    
    return (param_bytes * overhead_multiplier) / (1024 * 1024)
