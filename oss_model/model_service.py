"""ModelService for comprehensive model lifecycle management."""

import os
import json
import time
import shutil
import threading
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Generator
from urllib.parse import urlparse
from .model_downloader import ModelDownloader
from .model_types import (
    ModelInfo, ModelType, ModelFormat, ModelSource, ModelStatus, ModelPrecision,
    QuantizationConfig, ModelStats, generate_model_id, calculate_checksum, estimate_model_memory
)
from oss_utils.errors import ValidationError, ModelLoadError, FileProcessingError
from oss_utils.feature_flags import require_feature, is_feature_enabled
from oss_utils.logger import get_logger

class ModelService:
    """Comprehensive model lifecycle management service."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ModelService."""
        self.logger = get_logger()
        self.config = config or {}
        
        # Storage configuration
        self.models_dir = Path(self.config.get('models_dir', './models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(self.config.get('cache_dir', './cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}  # Actual model objects
        
        # Configuration
        self.max_cache_size_gb = self.config.get('max_cache_size_gb', 50.0)
        self.max_loaded_models = self.config.get('max_loaded_models', 3)
        self.download_timeout = self.config.get('download_timeout', 3600)  # 1 hour
        
        # Add model downloader
        self.downloader = ModelDownloader(config)
        
        # Check and download default models on initialization
        self._ensure_default_models()

        # Statistics
        self.stats = ModelStats()
        
        # Background tasks
        self._download_callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        # Load existing models registry
        self._load_models_registry()
        
        self.logger.info("ModelService initialized", 
                        models_dir=str(self.models_dir),
                        max_cache_size_gb=self.max_cache_size_gb)
    
    def _ensure_default_models(self):
        """Ensure default models are available."""
        try:
            cached_models = self.downloader.get_cached_models()
            default_models = set(self.downloader.default_models)
            missing_models = default_models - set(cached_models)
            
            if missing_models:
                self.logger.info(f"Downloading {len(missing_models)} default models...")
                self.downloader.download_default_models(
                    progress_callback=self._download_progress_callback
                )
        except Exception as e:
            self.logger.warning(f"Failed to ensure default models: {e}")
    
    def _download_progress_callback(self, model_id: str, progress: int, message: str):
        """Handle download progress updates."""
        self.logger.info(f"Download progress for {model_id}: {progress}% - {message}")

    def _load_models_registry(self):
        """Load models registry from storage."""
        registry_file = self.models_dir / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for model_data in registry_data.get('models', []):
                    model_info = ModelInfo(
                        model_id=model_data['model_id'],
                        name=model_data['name'],
                        model_type=ModelType(model_data['type']),
                        format=ModelFormat(model_data['format']),
                        source=ModelSource(model_data['source']),
                        status=ModelStatus(model_data['status']),
                        size_bytes=model_data.get('size_bytes', 0),
                        precision=ModelPrecision(model_data.get('precision', 'fp32')),
                        context_length=model_data.get('context_length', 2048),
                        parameter_count=model_data.get('parameter_count'),
                        local_path=model_data.get('local_path'),
                        download_url=model_data.get('download_url'),
                        checksum=model_data.get('checksum'),
                        description=model_data.get('description', ''),
                        version=model_data.get('version', '1.0.0'),
                        created_at=model_data.get('created_at', time.time()),
                        last_used=model_data.get('last_used', time.time()),
                        use_count=model_data.get('use_count', 0),
                        config=model_data.get('config', {}),
                        metadata=model_data.get('metadata', {})
                    )
                    
                    self.models[model_info.model_id] = model_info
                
                self.logger.info("Models registry loaded", 
                               models_count=len(self.models))
                
            except Exception as e:
                self.logger.error("Failed to load models registry", error=str(e))
    
    def _save_models_registry(self):
        """Save models registry to storage."""
             
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            registry_file = self.models_dir / "registry.json"
            registry_data = {
                'version': '1.0.0',
                'updated_at': time.time(),
                'models': [model.to_dict() for model in self.models.values()]
            }
            # Write to temporary file first, then move to avoid corruption
            temp_file = registry_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                json.dump(registry_data, f, indent=2)

            # Atomic move
            temp_file.replace(registry_file)   
        except Exception as e:
            self.logger.error("Failed to save models registry", error=str(e))
    
    def register_model(self, name: str, model_type: ModelType, source: ModelSource,
                  download_url: str = None, local_path: str = None,
                  **kwargs) -> str:
        """Register a new model in the service."""
        
        # Add this validation at the beginning of the method
        if not name or name.strip() == "":
            raise ValidationError("Model name cannot be empty", field_name="name")
        # Generate unique model ID
        model_id = generate_model_id(name, kwargs.get('version', '1.0.0'))
        
        # FIXED: Extract conflicting parameters from kwargs before passing to ModelInfo
        model_format = kwargs.pop('format', ModelFormat.PYTORCH)  # Remove from kwargs
        model_precision = kwargs.pop('precision', ModelPrecision.FP32)  # Remove from kwargs
        
        # Determine status
        model_status = ModelStatus.AVAILABLE if local_path else ModelStatus.DOWNLOADING
        
        # Create model info with explicit parameters (NO **kwargs unpacking)
        model_info = ModelInfo(
            model_id=model_id,
            name=name,
            model_type=model_type,
            format=model_format,  # Explicit parameter
            source=source,
            status=model_status,
            precision=model_precision,  # Explicit parameter
            download_url=download_url,
            local_path=local_path,
            # Only pass safe parameters individually
            size_bytes=kwargs.get('size_bytes', 0),
            context_length=kwargs.get('context_length', 2048),
            parameter_count=kwargs.get('parameter_count'),
            description=kwargs.get('description', ''),
            version=kwargs.get('version', '1.0.0'),
            config=kwargs.get('config', {}),
            metadata=kwargs.get('metadata', {})
        )
        
        with self._lock:
            self.models[model_id] = model_info
            self.stats.total_models += 1
            self._save_models_registry()
        
        self.logger.info("Model registered", 
                        model_id=model_id,
                        name=name,
                        type=model_type.value)
        
        return model_id

    
    def download_model(self, model_id: str, progress_callback: Callable = None) -> bool:
        """Download a model from its source."""
        if model_id not in self.models:
            raise ValidationError(f"Model {model_id} not found", field_name="model_id")
        
        model_info = self.models[model_id]
        
        if model_info.source == ModelSource.LOCAL:
            raise ValidationError("Cannot download local model", field_name="source")
        
        if not model_info.download_url:
            raise ValidationError("No download URL specified", field_name="download_url")
        
        # Set status to downloading
        model_info.status = ModelStatus.DOWNLOADING
        
        try:
            # Create model directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Determine filename
            parsed_url = urlparse(model_info.download_url)
            filename = Path(parsed_url.path).name or f"{model_info.name}.bin"
            local_path = model_dir / filename
            
            # Download with progress tracking
            response = requests.get(model_info.download_url, stream=True, timeout=self.download_timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(model_id, progress, downloaded, total_size)
            
            # Verify download
            if total_size > 0 and downloaded != total_size:
                raise FileProcessingError("Download incomplete")
            
            # Calculate checksum
            checksum = calculate_checksum(str(local_path))
            
            # Update model info
            model_info.local_path = str(local_path)
            model_info.size_bytes = downloaded
            model_info.checksum = checksum
            model_info.status = ModelStatus.AVAILABLE
            
            self.stats.downloads_completed += 1
            self._save_models_registry()
            
            self.logger.info("Model downloaded successfully", 
                           model_id=model_id,
                           size_mb=model_info.size_mb,
                           checksum=checksum[:8])
            
            return True
            
        except Exception as e:
            model_info.status = ModelStatus.ERROR
            self.stats.downloads_failed += 1
            self._save_models_registry()
            
            self.logger.error("Model download failed", 
                            model_id=model_id,
                            error=str(e))
            raise FileProcessingError(f"Download failed: {str(e)}")
    
    def load_model(self, model_id: str, **kwargs) -> Any:
        """Load a model into memory for inference."""
        if model_id not in self.models:
            raise ValidationError(f"Model {model_id} not found", field_name="model_id")
        
        model_info = self.models[model_id]
        
        # Check if already loaded
        if model_id in self.loaded_models:
            model_info.update_usage()
            self.logger.debug("Model already loaded", model_id=model_id)
            return self.loaded_models[model_id]
        
        # Check cache limits
        if len(self.loaded_models) >= self.max_loaded_models:
            self._evict_oldest_model()
        
        # Verify model file exists
        if not model_info.is_local:
            raise ModelLoadError(f"Model file not found: {model_info.local_path}")
        
        model_info.status = ModelStatus.LOADING
        load_start = time.time()
        
        try:
            # Load model based on format
            if model_info.format == ModelFormat.PYTORCH:
                model = self._load_pytorch_model(model_info, **kwargs)
            elif model_info.format == ModelFormat.ONNX:
                model = self._load_onnx_model(model_info, **kwargs)
            elif model_info.format == ModelFormat.HUGGINGFACE:
                model = self._load_huggingface_model(model_info, **kwargs)
            else:
                raise ModelLoadError(f"Unsupported format: {model_info.format.value}")
            
            # Store loaded model
            self.loaded_models[model_id] = model
            
            # Update statistics
            load_time = (time.time() - load_start) * 1000
            model_info.load_time_ms = load_time
            model_info.status = ModelStatus.LOADED
            model_info.update_usage()
            
            self.stats.loaded_models += 1
            self.stats.average_load_time_ms = (
                (self.stats.average_load_time_ms * (self.stats.loaded_models - 1) + load_time) 
                / self.stats.loaded_models
            )
            
            self.logger.info("Model loaded successfully", 
                           model_id=model_id,
                           load_time_ms=load_time)
            
            return model
            
        except Exception as e:
            model_info.status = ModelStatus.ERROR
            self.logger.error("Model loading failed", 
                            model_id=model_id,
                            error=str(e))
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def _load_pytorch_model(self, model_info: ModelInfo, **kwargs) -> Any:
        """Load a PyTorch model."""
        try:
            import torch
            model = torch.load(model_info.local_path, map_location='cpu')
            
            # Apply optimizations
            if kwargs.get('optimize', True):
                model.eval()
                if hasattr(torch, 'jit') and kwargs.get('compile', False):
                    model = torch.jit.script(model)
            
            return model
            
        except ImportError:
            raise ModelLoadError("PyTorch not available")
        except Exception as e:
            raise ModelLoadError(f"PyTorch loading error: {str(e)}")
    
    def _load_onnx_model(self, model_info: ModelInfo, **kwargs) -> Any:
        """Load an ONNX model."""
        try:
            import onnxruntime as ort
            
            # Configure session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = ['CPUExecutionProvider']
            if kwargs.get('use_gpu', False):
                providers.insert(0, 'CUDAExecutionProvider')
            
            model = ort.InferenceSession(model_info.local_path, 
                                       sess_options=session_options,
                                       providers=providers)
            return model
            
        except ImportError:
            raise ModelLoadError("ONNX Runtime not available")
        except Exception as e:
            raise ModelLoadError(f"ONNX loading error: {str(e)}")
    
    def _load_huggingface_model(self, model_info: ModelInfo, **kwargs) -> Any:
        """Load a Hugging Face model."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_path = Path(model_info.local_path).parent
            
            model = AutoModel.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            return {'model': model, 'tokenizer': tokenizer}
            
        except ImportError:
            raise ModelLoadError("Transformers library not available")
        except Exception as e:
            raise ModelLoadError(f"Hugging Face loading error: {str(e)}")
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        if model_id not in self.loaded_models:
            return False
        
        try:
            # Clean up model resources
            model = self.loaded_models[model_id]
            if hasattr(model, 'cpu'):
                model.cpu()
            
            del self.loaded_models[model_id]
            
            # Update statistics
            self.stats.loaded_models -= 1
            
            # Update model status
            if model_id in self.models:
                self.models[model_id].status = ModelStatus.AVAILABLE
            
            self.logger.info("Model unloaded", model_id=model_id)
            return True
            
        except Exception as e:
            self.logger.error("Model unload failed", 
                            model_id=model_id,
                            error=str(e))
            return False
    
    def _evict_oldest_model(self):
        """Evict the least recently used model."""
        if not self.loaded_models:
            return
        
        # Find least recently used model
        oldest_model_id = min(
            self.loaded_models.keys(),
            key=lambda mid: self.models[mid].last_used if mid in self.models else 0
        )
        
        self.unload_model(oldest_model_id)
        self.logger.debug("Evicted oldest model", model_id=oldest_model_id)
    
    def delete_model(self, model_id: str, delete_files: bool = False) -> bool:
        """Delete a model from the service."""
        if model_id not in self.models:
            raise ValidationError(f"Model {model_id} not found", field_name="model_id")
        
        try:
            # Unload if loaded
            if model_id in self.loaded_models:
                self.unload_model(model_id)
            
            model_info = self.models[model_id]
            
            # Delete files if requested
            if delete_files and model_info.local_path:
                model_dir = Path(model_info.local_path).parent
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    self.logger.info("Model files deleted", path=str(model_dir))
            
            # Remove from registry
            del self.models[model_id]
            self.stats.total_models -= 1
            self._save_models_registry()
            
            self.logger.info("Model deleted", model_id=model_id)
            return True
            
        except Exception as e:
            self.logger.error("Model deletion failed", 
                            model_id=model_id,
                            error=str(e))
            return False
    
    def quantize_model(self, model_id: str, config: QuantizationConfig,
                      output_model_id: str = None) -> str:
        """Quantize a model to reduce size and improve performance."""
        if model_id not in self.models:
            raise ValidationError(f"Model {model_id} not found", field_name="model_id")
        
        source_model = self.models[model_id]
        
        if not source_model.is_local:
            raise ValidationError("Source model must be local for quantization")
        
        # Generate output model ID if not provided
        if not output_model_id:
            output_model_id = f"{model_id}_quantized_{config.target_precision.value}"
        
        source_model.status = ModelStatus.QUANTIZING
        
        try:
            # Create output directory
            output_dir = self.models_dir / output_model_id
            output_dir.mkdir(exist_ok=True)
            
            # Perform quantization based on model format
            if source_model.format == ModelFormat.PYTORCH:
                output_path = self._quantize_pytorch_model(source_model, config, output_dir)
            elif source_model.format == ModelFormat.ONNX:
                output_path = self._quantize_onnx_model(source_model, config, output_dir)
            else:
                raise ValidationError(f"Quantization not supported for format: {source_model.format.value}")
            
            # Create quantized model info
            quantized_model = ModelInfo(
                model_id=output_model_id,
                name=f"{source_model.name}_quantized",
                model_type=source_model.model_type,
                format=source_model.format,
                source=ModelSource.LOCAL,
                status=ModelStatus.AVAILABLE,
                local_path=str(output_path),
                precision=config.target_precision,
                context_length=source_model.context_length,
                parameter_count=source_model.parameter_count,
                description=f"Quantized version of {source_model.name}",
                version=source_model.version,
                size_bytes=output_path.stat().st_size,
                checksum=calculate_checksum(str(output_path)),
                metadata={
                    'quantization_config': config.to_dict(),
                    'source_model_id': model_id,
                    'quantization_timestamp': time.time()
                }
            )
            
            # Register quantized model
            self.models[output_model_id] = quantized_model
            self.stats.quantizations_completed += 1
            self._save_models_registry()
            
            # Reset source model status
            source_model.status = ModelStatus.AVAILABLE
            
            self.logger.info("Model quantization completed", 
                           source_model_id=model_id,
                           output_model_id=output_model_id,
                           precision=config.target_precision.value,
                           size_reduction=f"{(1 - quantized_model.size_mb / source_model.size_mb) * 100:.1f}%")
            
            return output_model_id
            
        except Exception as e:
            source_model.status = ModelStatus.ERROR
            self.logger.error("Model quantization failed", 
                            model_id=model_id,
                            error=str(e))
            raise ModelLoadError(f"Quantization failed: {str(e)}")
    
    def _quantize_pytorch_model(self, model_info: ModelInfo, config: QuantizationConfig, output_dir: Path) -> Path:
        """Quantize a PyTorch model."""
        try:
            import torch
            from torch.quantization import quantize_dynamic, QConfig
            
            # Load original model
            model = torch.load(model_info.local_path, map_location='cpu')
            model.eval()
            
            # Apply quantization
            if config.target_precision == ModelPrecision.INT8:
                quantized_model = quantize_dynamic(
                    model, 
                    {torch.nn.Linear, torch.nn.Conv2d}, 
                    dtype=torch.qint8
                )
            else:
                raise ValidationError(f"PyTorch quantization to {config.target_precision.value} not implemented")
            
            # Save quantized model
            output_path = output_dir / "model_quantized.pt"
            torch.save(quantized_model, output_path)
            
            return output_path
            
        except ImportError:
            raise ModelLoadError("PyTorch not available for quantization")
        except Exception as e:
            raise ModelLoadError(f"PyTorch quantization error: {str(e)}")
    
    def _quantize_onnx_model(self, model_info: ModelInfo, config: QuantizationConfig, output_dir: Path) -> Path:
        """Quantize an ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Map precision to ONNX types
            quant_type_map = {
                ModelPrecision.INT8: QuantType.QInt8,
                ModelPrecision.INT4: QuantType.QUInt8  # Approximate
            }
            
            if config.target_precision not in quant_type_map:
                raise ValidationError(f"ONNX quantization to {config.target_precision.value} not supported")
            
            output_path = output_dir / "model_quantized.onnx"
            
            quantize_dynamic(
                model_input=model_info.local_path,
                model_output=str(output_path),
                weight_type=quant_type_map[config.target_precision]
            )
            
            return output_path
            
        except ImportError:
            raise ModelLoadError("ONNX Runtime not available for quantization")
        except Exception as e:
            raise ModelLoadError(f"ONNX quantization error: {str(e)}")
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        if model_id not in self.models:
            return None
        
        model_info = self.models[model_id]
        info_dict = model_info.to_dict()
        
        # Add runtime information
        info_dict['runtime'] = {
            'is_loaded': model_id in self.loaded_models,
            'estimated_memory_mb': estimate_model_memory(
                model_info.parameter_count, 
                model_info.precision
            ) if model_info.parameter_count else 0
        }
        
        return info_dict
    
    def list_models(self, model_type: ModelType = None, status: ModelStatus = None) -> List[Dict[str, Any]]:
        """List all models with optional filtering."""
        models = []
        
        for model_info in self.models.values():
            # Apply filters
            if model_type and model_info.model_type != model_type:
                continue
            if status and model_info.status != status:
                continue
            
            models.append(self.get_model_info(model_info.model_id))
        
        # Sort by last used (most recent first)
        models.sort(key=lambda m: m['usage']['last_used'], reverse=True)
        
        return models
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        # Update current statistics
        self.stats.total_size_gb = sum(
            model.size_gb for model in self.models.values()
        )
        
        return self.stats.to_dict()
    
    def cleanup_cache(self, max_age_days: int = 30) -> int:
        """Clean up old unused models."""
        current_time = time.time()
        cleaned_count = 0
        
        models_to_delete = []
        
        for model_id, model_info in self.models.items():
            # Skip loaded models
            if model_id in self.loaded_models:
                continue
            
            # Check age and usage
            age_days = (current_time - model_info.last_used) / 86400
            
            if age_days > max_age_days and model_info.use_count == 0:
                models_to_delete.append(model_id)
        
        # Delete old models
        for model_id in models_to_delete:
            if self.delete_model(model_id, delete_files=True):
                cleaned_count += 1
        
        self.logger.info("Cache cleanup completed", 
                        cleaned_models=cleaned_count)
        
        return cleaned_count
    
    def check_filename_exists(self, filename: str) -> bool:
        """Check if a filename exists in the models directory."""
        for model_info in self.models.values():
            if model_info.local_path and Path(model_info.local_path).name == filename:
                return True
        return False
    
    def update_query(self, model_id: str, query_updates: Dict[str, Any]) -> bool:
        """Update model configuration or metadata."""
        if model_id not in self.models:
            return False
        
        try:
            model_info = self.models[model_id]
            
            # Update allowed fields
            if 'config' in query_updates:
                model_info.config.update(query_updates['config'])
            
            if 'metadata' in query_updates:
                model_info.metadata.update(query_updates['metadata'])
            
            if 'description' in query_updates:
                model_info.description = query_updates['description']
            
            self._save_models_registry()
            
            self.logger.info("Model query updated", model_id=model_id)
            return True
            
        except Exception as e:
            self.logger.error("Model query update failed", 
                            model_id=model_id,
                            error=str(e))
            return False
    
    def cleanup(self):
        """Clean up service resources."""
        self.logger.info("Cleaning up ModelService")
        
        # Unload all models
        for model_id in list(self.loaded_models.keys()):
            self.unload_model(model_id)
        
        # Save final registry state
        self._save_models_registry()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
