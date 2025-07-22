"""Platform and hardware detection types for the Decompute SDK."""

import platform
import subprocess
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

class PlatformType(Enum):
    """Supported platform types."""
    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"
    UNKNOWN = "unknown"

class ProcessorType(Enum):
    """Processor architectures."""
    APPLE_SILICON = "apple_silicon"  # M1, M2, M3
    INTEL_X64 = "intel_x64"
    AMD_X64 = "amd_x64"
    ARM64 = "arm64"
    UNKNOWN = "unknown"

class GPUType(Enum):
    """GPU types for acceleration."""
    NVIDIA_CUDA = "nvidia_cuda"
    APPLE_METAL = "apple_metal"
    AMD_ROCM = "amd_rocm"
    INTEL_GPU = "intel_gpu"
    NONE = "none"

class InferenceFramework(Enum):
    """Inference optimization frameworks."""
    PYTORCH = "pytorch"
    MLX = "mlx"  # Apple MLX
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    COREML = "coreml"
    OPENVINO = "openvino"
    CPU_ONLY = "cpu_only"

class OptimizationLevel(Enum):
    """Optimization levels for inference."""
    FASTEST = "fastest"  # Maximum speed, may sacrifice accuracy
    BALANCED = "balanced"  # Balance speed and accuracy
    QUALITY = "quality"  # Maximum accuracy, slower
    MEMORY_EFFICIENT = "memory_efficient"  # Optimize for low memory

@dataclass
class HardwareInfo:
    """Detailed hardware information."""
    platform: PlatformType
    processor: ProcessorType
    gpu_type: GPUType
    cpu_cores: int
    memory_gb: float
    gpu_memory_gb: float = 0.0
    gpu_name: str = ""
    cpu_name: str = ""
    architecture: str = ""
    
    # Capabilities
    supports_cuda: bool = False
    supports_mps: bool = False
    supports_mlx: bool = False
    supports_opencl: bool = False
    
    # Performance hints
    recommended_batch_size: int = 1
    max_sequence_length: int = 2048
    parallel_workers: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'platform': self.platform.value,
            'processor': self.processor.value,
            'gpu_type': self.gpu_type.value,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_memory_gb': self.gpu_memory_gb,
            'gpu_name': self.gpu_name,
            'cpu_name': self.cpu_name,
            'architecture': self.architecture,
            'capabilities': {
                'cuda': self.supports_cuda,
                'mps': self.supports_mps,
                'mlx': self.supports_mlx,
                'opencl': self.supports_opencl
            },
            'performance': {
                'recommended_batch_size': self.recommended_batch_size,
                'max_sequence_length': self.max_sequence_length,
                'parallel_workers': self.parallel_workers
            }
        }

@dataclass
class InferenceConfig:
    """Configuration for inference optimization."""
    framework: InferenceFramework = InferenceFramework.PYTORCH  # ADD DEFAULT VALUE
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED  # ADD DEFAULT VALUE
    device: str = "auto"  # "cpu", "cuda", "mps", "mlx", "auto"
    batch_size: int = 1
    max_length: int = 2048
    use_half_precision: bool = False
    use_quantization: bool = False
    cache_size_mb: int = 512
    parallel_workers: int = 1
    
    # Advanced options
    enable_compilation: bool = True
    enable_graph_optimization: bool = True
    memory_pool_size_mb: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'framework': self.framework.value,
            'optimization_level': self.optimization_level.value,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'use_half_precision': self.use_half_precision,
            'use_quantization': self.use_quantization,
            'cache_size_mb': self.cache_size_mb,
            'parallel_workers': self.parallel_workers,
            'advanced': {
                'enable_compilation': self.enable_compilation,
                'enable_graph_optimization': self.enable_graph_optimization,
                'memory_pool_size_mb': self.memory_pool_size_mb
            }
        }

@dataclass
class InferenceStats:
    """Statistics for inference operations."""
    total_inferences: int = 0
    total_time_ms: float = 0.0
    average_time_ms: float = 0.0
    tokens_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_peak_mb: float = 0.0
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.total_time_ms == 0:
            return 0.0
        return (self.tokens_processed * 1000) / self.total_time_ms
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0
    
    def update_inference(self, time_ms: float, tokens: int = 0, memory_mb: float = 0.0):
        """Update inference statistics."""
        self.total_inferences += 1
        self.total_time_ms += time_ms
        self.tokens_processed += tokens
        self.average_time_ms = self.total_time_ms / self.total_inferences
        self.memory_peak_mb = max(self.memory_peak_mb, memory_mb)

def detect_platform() -> PlatformType:
    """Detect the current platform."""
    system = platform.system().lower()
    if system == "darwin":
        return PlatformType.MACOS
    elif system == "windows":
        return PlatformType.WINDOWS
    elif system == "linux":
        return PlatformType.LINUX
    else:
        return PlatformType.UNKNOWN

def detect_processor() -> ProcessorType:
    """Detect processor type."""
    try:
        # Check for Apple Silicon
        if platform.system() == "Darwin":
            machine = platform.machine().lower()
            if machine == "arm64" or "apple" in platform.processor().lower():
                return ProcessorType.APPLE_SILICON
            else:
                return ProcessorType.INTEL_X64
        
        # Check for x64 architecture
        machine = platform.machine().lower()
        if machine in ["x86_64", "amd64"]:
            # Try to distinguish Intel vs AMD
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                if "intel" in cpu_info.get("brand_raw", "").lower():
                    return ProcessorType.INTEL_X64
                elif "amd" in cpu_info.get("brand_raw", "").lower():
                    return ProcessorType.AMD_X64
                else:
                    return ProcessorType.INTEL_X64  # Default assumption
            except:
                return ProcessorType.INTEL_X64
        elif machine in ["arm64", "aarch64"]:
            return ProcessorType.ARM64
        else:
            return ProcessorType.UNKNOWN
            
    except Exception:
        return ProcessorType.UNKNOWN

def detect_gpu() -> Tuple[GPUType, str, float]:
    """Detect GPU type and capabilities."""
    gpu_type = GPUType.NONE
    gpu_name = ""
    gpu_memory = 0.0
    
    try:
        # Check for NVIDIA CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_type = GPUType.NVIDIA_CUDA
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return gpu_type, gpu_name, gpu_memory
        except:
            pass
        
        # Check for Apple Metal
        if platform.system() == "Darwin":
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_type = GPUType.APPLE_METAL
                    gpu_name = "Apple Metal Performance Shaders"
                    # Estimate memory based on system memory (unified memory)
                    import psutil
                    gpu_memory = psutil.virtual_memory().total / (1024**3) * 0.6  # Rough estimate
                    return gpu_type, gpu_name, gpu_memory
            except:
                pass
        
        # Could add AMD ROCm, Intel GPU detection here
        
    except Exception:
        pass
    
    return gpu_type, gpu_name, gpu_memory

def get_memory_info() -> Tuple[float, int]:
    """Get system memory information."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # Get CPU core count
        cpu_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        
        return memory_gb, cpu_cores
    except:
        # Fallback
        return 8.0, os.cpu_count() or 1

def get_cpu_name() -> str:
    """Get CPU name."""
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "Unknown CPU")
    except:
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        except:
            pass
        
        return f"{platform.processor()} CPU"
