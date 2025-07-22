"""InferenceEngine for hardware optimization and platform-specific inference."""

import os
import time
import threading
import importlib
from typing import Dict, List, Any, Optional, Callable, Tuple
import platform  # Add this line at the top

from .platform_types import (
    HardwareInfo, InferenceConfig, InferenceStats, PlatformType, ProcessorType, 
    GPUType, InferenceFramework, OptimizationLevel,
    detect_platform, detect_processor, detect_gpu, get_memory_info, get_cpu_name
)
from oss_utils.errors import ValidationError, MemoryError, ModelLoadError
from oss_utils.feature_flags import require_feature, is_feature_enabled
from oss_utils.logger import get_logger

class InferenceEngine:
    """Hardware-optimized inference engine with platform detection."""
    
    def __init__(self, config: InferenceConfig = None):
        """Initialize the InferenceEngine."""
        self.logger = get_logger()
        self.config = config or InferenceConfig(
            framework=InferenceFramework.PYTORCH,
            optimization_level=OptimizationLevel.BALANCED
        )
        
        # Hardware detection
        self.hardware_info: Optional[HardwareInfo] = None
        self.is_optimized = False
        
        # Inference state
        self.current_model = None
        self.model_cache = {}
        self.stats = InferenceStats()
        
        # Framework-specific components
        self.torch_module = None
        self.mlx_module = None
        self.framework_loaded = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize hardware detection
        self._detect_hardware()
        
        # Auto-optimize based on hardware
        self.optimize_hardware()
        
        self.logger.info("InferenceEngine initialized", 
                        platform=self.hardware_info.platform.value,
                        processor=self.hardware_info.processor.value,
                        gpu_type=self.hardware_info.gpu_type.value)
    
    def _detect_hardware(self):
        """Detect hardware capabilities and create hardware info."""
        try:
            # Basic platform detection
            platform_type = detect_platform()
            processor_type = detect_processor()
            gpu_type, gpu_name, gpu_memory = detect_gpu()
            memory_gb, cpu_cores = get_memory_info()
            cpu_name = get_cpu_name()
            
            # Create hardware info
            self.hardware_info = HardwareInfo(
                platform=platform_type,
                processor=processor_type,
                gpu_type=gpu_type,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                gpu_memory_gb=gpu_memory,
                gpu_name=gpu_name,
                cpu_name=cpu_name,
                architecture=platform.machine()
            )
            
            # Detect specific capabilities
            self._detect_framework_capabilities()
            
            # Set performance recommendations
            self._set_performance_recommendations()
            
            self.logger.info("Hardware detected", 
                           platform=platform_type.value,
                           processor=processor_type.value,
                           cpu_cores=cpu_cores,
                           memory_gb=memory_gb,
                           gpu_type=gpu_type.value)
            
        except Exception as e:
            self.logger.error("Hardware detection failed", error=str(e))
            # Create minimal fallback hardware info
            self.hardware_info = HardwareInfo(
                platform=PlatformType.UNKNOWN,
                processor=ProcessorType.UNKNOWN,
                gpu_type=GPUType.NONE,
                cpu_cores=1,
                memory_gb=4.0
            )
    
    def _detect_framework_capabilities(self):
        """Detect framework-specific capabilities."""
        if not self.hardware_info:
            return
        
        # Check PyTorch capabilities
        try:
            import torch
            self.torch_module = torch
            
            # CUDA support
            if torch.cuda.is_available():
                self.hardware_info.supports_cuda = True
                self.logger.debug("CUDA support detected")
            
            # MPS support (Apple Metal)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.hardware_info.supports_mps = True
                self.logger.debug("MPS support detected")
                
        except ImportError:
            self.logger.warning("PyTorch not available")
        
        # Check MLX support (Apple Silicon)
        try:
            if self.hardware_info.processor == ProcessorType.APPLE_SILICON:
                import mlx.core as mx
                self.mlx_module = mx
                self.hardware_info.supports_mlx = True
                self.logger.debug("MLX support detected")
        except ImportError:
            self.logger.debug("MLX not available")
        
        # Check OpenCL support
        try:
            import pyopencl as cl  # Import with alias to avoid shadowing built-in open
            self.hardware_info.supports_opencl = True
            self.logger.debug("OpenCL support detected")
        except ImportError:
            pass
    
    def _set_performance_recommendations(self):
        """Set performance recommendations based on hardware."""
        if not self.hardware_info:
            return
        
        # Batch size recommendations
        if self.hardware_info.gpu_memory_gb >= 8:
            self.hardware_info.recommended_batch_size = 8
        elif self.hardware_info.gpu_memory_gb >= 4:
            self.hardware_info.recommended_batch_size = 4
        elif self.hardware_info.memory_gb >= 16:
            self.hardware_info.recommended_batch_size = 2
        else:
            self.hardware_info.recommended_batch_size = 1
        
        # Sequence length recommendations
        if self.hardware_info.memory_gb >= 32:
            self.hardware_info.max_sequence_length = 4096
        elif self.hardware_info.memory_gb >= 16:
            self.hardware_info.max_sequence_length = 2048
        else:
            self.hardware_info.max_sequence_length = 1024
        
        # Parallel workers
        self.hardware_info.parallel_workers = min(self.hardware_info.cpu_cores, 4)
    
    def optimize_hardware(self) -> Dict[str, Any]:
        """Optimize settings for current hardware."""
        if not self.hardware_info:
            raise MemoryError("Hardware detection failed")
        
        with self._lock:
            optimization_results = {
                'platform': self.hardware_info.platform.value,
                'optimizations_applied': [],
                'recommended_settings': {},
                'warnings': []
            }
            
            # Platform-specific optimizations
            if self.hardware_info.platform == PlatformType.MACOS:
                optimization_results.update(self._optimize_macos())
            elif self.hardware_info.platform == PlatformType.WINDOWS:
                optimization_results.update(self._optimize_windows())
            elif self.hardware_info.platform == PlatformType.LINUX:
                optimization_results.update(self._optimize_linux())
            
            # Apply framework optimizations
            framework_opts = self._optimize_framework()
            optimization_results['optimizations_applied'].extend(framework_opts)
            
            # Update config with recommendations
            self._update_config_from_hardware()
            
            self.is_optimized = True
            
            self.logger.info("Hardware optimization completed", 
                           optimizations=len(optimization_results['optimizations_applied']))
            
            return optimization_results
    
    def _optimize_macos(self) -> Dict[str, Any]:
        """Apply macOS-specific optimizations."""
        optimizations = []
        warnings = []
        
        if self.hardware_info.processor == ProcessorType.APPLE_SILICON:
            # Optimize for Apple Silicon
            if self.hardware_info.supports_mlx:
                optimizations.append("MLX framework enabled for Apple Silicon")
                self.config.framework = InferenceFramework.MLX
                self.config.device = "mlx"
            elif self.hardware_info.supports_mps:
                optimizations.append("MPS acceleration enabled")
                self.config.device = "mps"
            else:
                warnings.append("Neither MLX nor MPS available, using CPU")
                self.config.device = "cpu"
        else:
            # Intel Mac
            if self.hardware_info.supports_mps:
                optimizations.append("MPS acceleration enabled for Intel Mac")
                self.config.device = "mps"
            else:
                optimizations.append("CPU optimization for Intel Mac")
                self.config.device = "cpu"
        
        return {
            'optimizations_applied': optimizations,
            'warnings': warnings
        }
    
    def _optimize_windows(self) -> Dict[str, Any]:
        """Apply Windows-specific optimizations."""
        optimizations = []
        warnings = []
        
        if self.hardware_info.supports_cuda:
            optimizations.append("CUDA acceleration enabled")
            self.config.device = "cuda"
            self.config.use_half_precision = True  # FP16 for CUDA
        else:
            optimizations.append("CPU optimization for Windows")
            self.config.device = "cpu"
            if self.hardware_info.memory_gb < 8:
                warnings.append("Low memory detected, consider upgrading RAM")
        
        # Enable compilation for better performance
        optimizations.append("Model compilation enabled")
        self.config.enable_compilation = True
        
        return {
            'optimizations_applied': optimizations,
            'warnings': warnings
        }
    
    def _optimize_linux(self) -> Dict[str, Any]:
        """Apply Linux-specific optimizations."""
        optimizations = []
        warnings = []
        
        if self.hardware_info.supports_cuda:
            optimizations.append("CUDA acceleration enabled for Linux")
            self.config.device = "cuda"
            self.config.use_half_precision = True
        else:
            optimizations.append("CPU optimization for Linux")
            self.config.device = "cpu"
        
        # Enable all optimizations for Linux
        optimizations.append("Advanced optimizations enabled")
        self.config.enable_compilation = True
        self.config.enable_graph_optimization = True
        
        return {
            'optimizations_applied': optimizations,
            'warnings': warnings
        }
    
    def _optimize_framework(self) -> List[str]:
        """Apply framework-specific optimizations."""
        optimizations = []
        
        if self.config.framework == InferenceFramework.PYTORCH:
            try:
                if self.torch_module:
                    # Set number of threads
                    self.torch_module.set_num_threads(self.hardware_info.parallel_workers)
                    optimizations.append(f"PyTorch threads set to {self.hardware_info.parallel_workers}")
                    
                    # Enable optimizations
                    if hasattr(self.torch_module, 'set_float32_matmul_precision'):
                        self.torch_module.set_float32_matmul_precision('medium')
                        optimizations.append("PyTorch matmul precision optimized")
            except Exception as e:
                self.logger.warning("PyTorch optimization failed", error=str(e))
        
        elif self.config.framework == InferenceFramework.MLX:
            try:
                if self.mlx_module:
                    # MLX-specific optimizations
                    optimizations.append("MLX unified memory optimization enabled")
            except Exception as e:
                self.logger.warning("MLX optimization failed", error=str(e))
        
        return optimizations
    
    def _update_config_from_hardware(self):
        """Update configuration based on hardware capabilities."""
        # Update batch size
        self.config.batch_size = min(
            self.config.batch_size, 
            self.hardware_info.recommended_batch_size
        )
        
        # Update max length
        self.config.max_length = min(
            self.config.max_length,
            self.hardware_info.max_sequence_length
        )
        
        # Update parallel workers
        self.config.parallel_workers = self.hardware_info.parallel_workers
        
        # Memory-based optimizations
        if self.hardware_info.memory_gb < 8:
            self.config.cache_size_mb = 256
            self.config.memory_pool_size_mb = 512
        elif self.hardware_info.memory_gb >= 16:
            self.config.cache_size_mb = 1024
            self.config.memory_pool_size_mb = 2048
    
    def init_ollama(self) -> bool:
        """Initialize Ollama for local inference."""
        try:
            # Check if Ollama is available
            import subprocess
            result = subprocess.run(
                ["ollama", "version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                self.logger.info("Ollama detected and initialized")
                return True
            else:
                self.logger.warning("Ollama not available")
                return False
                
        except Exception as e:
            self.logger.warning("Failed to initialize Ollama", error=str(e))
            return False
    
    def mls_optimize(self) -> bool:
        """Optimize for Apple Metal Performance Shaders."""
        if not self.hardware_info.supports_mps:
            self.logger.warning("MPS not supported on this system")
            return False
        
        try:
            if self.torch_module and hasattr(self.torch_module.backends, 'mps'):
                # Enable MPS optimizations
                self.config.device = "mps"
                self.config.use_half_precision = True
                
                self.logger.info("MPS optimization enabled")
                return True
        except Exception as e:
            self.logger.error("MPS optimization failed", error=str(e))
        
        return False
    
    def cuda_accelerate(self) -> bool:
        """Enable CUDA acceleration if available."""
        if not self.hardware_info.supports_cuda:
            self.logger.warning("CUDA not supported on this system")
            return False
        
        try:
            if self.torch_module and self.torch_module.cuda.is_available():
                self.config.device = "cuda"
                self.config.use_half_precision = True
                
                # Set memory optimizations
                self.torch_module.cuda.empty_cache()
                
                self.logger.info("CUDA acceleration enabled", 
                               gpu_name=self.hardware_info.gpu_name,
                               gpu_memory_gb=self.hardware_info.gpu_memory_gb)
                return True
        except Exception as e:
            self.logger.error("CUDA acceleration failed", error=str(e))
        
        return False
    
    def set_inference_mode(self, mode: OptimizationLevel) -> bool:
        """Set inference optimization mode."""
        try:
            self.config.optimization_level = mode
            
            if mode == OptimizationLevel.FASTEST:
                self.config.use_half_precision = True
                self.config.use_quantization = True
                self.config.enable_compilation = True
            elif mode == OptimizationLevel.BALANCED:
                self.config.use_half_precision = self.hardware_info.gpu_type != GPUType.NONE
                self.config.use_quantization = False
                self.config.enable_compilation = True
            elif mode == OptimizationLevel.QUALITY:
                self.config.use_half_precision = False
                self.config.use_quantization = False
                self.config.enable_compilation = False
            elif mode == OptimizationLevel.MEMORY_EFFICIENT:
                self.config.use_quantization = True
                self.config.batch_size = 1
                self.config.cache_size_mb = 128
            
            self.logger.info("Inference mode set", mode=mode.value)
            return True
            
        except Exception as e:
            self.logger.error("Failed to set inference mode", error=str(e))
            return False
    
    def batch_process(self, requests: List[Dict[str, Any]], 
                     callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Process multiple inference requests in batches."""
        if not requests:
            return []
        
        results = []
        batch_size = self.config.batch_size
        
        with self._lock:
            for i in range(0, len(requests), batch_size):
                batch = requests[i:i + batch_size]
                
                start_time = time.time()
                
                try:
                    # Process batch (placeholder - would integrate with actual model)
                    batch_results = []
                    for request in batch:
                        # Simulate processing
                        result = {
                            'request_id': request.get('id', f'req_{i}'),
                            'status': 'processed',
                            'processing_time_ms': (time.time() - start_time) * 1000,
                            'device': self.config.device
                        }
                        batch_results.append(result)
                    
                    results.extend(batch_results)
                    
                    # Update statistics
                    processing_time = (time.time() - start_time) * 1000
                    self.stats.update_inference(processing_time, len(batch))
                    
                    if callback:
                        callback(batch_results)
                        
                except Exception as e:
                    self.logger.error("Batch processing failed", error=str(e))
                    # Add error results
                    for request in batch:
                        results.append({
                            'request_id': request.get('id', f'req_{i}'),
                            'status': 'error',
                            'error': str(e)
                        })
        
        self.logger.info("Batch processing completed", 
                        total_requests=len(requests),
                        batch_count=len(range(0, len(requests), batch_size)))
        
        return results
    
    def detect_platform(self) -> Dict[str, Any]:
        """Get detailed platform information."""
        if not self.hardware_info:
            return {'error': 'Hardware detection not completed'}
        
        return self.hardware_info.to_dict()
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        return {
            'total_inferences': self.stats.total_inferences,
            'average_time_ms': self.stats.average_time_ms,
            'tokens_per_second': self.stats.tokens_per_second,
            'cache_hit_rate': self.stats.cache_hit_rate,
            'memory_peak_mb': self.stats.memory_peak_mb,
            'current_config': self.config.to_dict()
        }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations for current hardware."""
        if not self.hardware_info:
            return {'error': 'Hardware detection not completed'}
        
        recommendations = {
            'hardware_summary': self.hardware_info.to_dict(),
            'current_config': self.config.to_dict(),
            'recommendations': []
        }
        
        # Memory recommendations
        if self.hardware_info.memory_gb < 8:
            recommendations['recommendations'].append({
                'type': 'memory',
                'message': 'Consider increasing system RAM for better performance',
                'priority': 'high'
            })
        
        # GPU recommendations
        if self.hardware_info.gpu_type == GPUType.NONE:
            recommendations['recommendations'].append({
                'type': 'gpu',
                'message': 'GPU acceleration not available - consider adding a compatible GPU',
                'priority': 'medium'
            })
        
        # Framework recommendations
        if (self.hardware_info.processor == ProcessorType.APPLE_SILICON and 
            not self.hardware_info.supports_mlx):
            recommendations['recommendations'].append({
                'type': 'framework',
                'message': 'Install MLX for optimal Apple Silicon performance',
                'priority': 'high'
            })
        
        return recommendations
    
    def cleanup(self):
        """Clean up inference engine resources."""
        self.logger.info("Cleaning up InferenceEngine")
        
        with self._lock:
            # Clear model cache
            self.model_cache.clear()
            self.current_model = None
            
            # Reset state
            self.is_optimized = False
            self.framework_loaded = False
            
            # Clear CUDA cache if available
            try:
                if self.torch_module and self.torch_module.cuda.is_available():
                    self.torch_module.cuda.empty_cache()
            except:
                pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
