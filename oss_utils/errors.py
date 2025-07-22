from __future__ import annotations
"""Enhanced error classes for the Blackbird SDK with detailed error handling."""

import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

class BlackbirdError(Exception):
    """Base exception for all Blackbird SDK errors."""
    def __init__(self, message: str, status_code: int = 500, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "timestamp": self.timestamp
        }

class AuthenticationError(BlackbirdError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, api_key_status: str = None):
        details = {"api_key_status": api_key_status} if api_key_status else {}
        super().__init__(message, 401, "AUTH_001")

class APIError(BlackbirdError):
    """Raised when the API returns an error."""
    
    def __init__(self, message: str, status_code: int = None, response: Any = None, endpoint: str = None):
        self.status_code = status_code
        self.response = response
        details = {
            "status_code": status_code,
            "endpoint": endpoint,
            "response_preview": str(response)[:200] if response else None
        }
        error_code = f"API_{status_code}" if status_code else "API_GENERAL_001"
        super().__init__(message, status_code or 500, error_code)

class NetworkError(BlackbirdError):
    """Raised when a network error occurs."""
    
    def __init__(self, message: str, retry_count: int = 0, endpoint: str = None):
        details = {
            "retry_count": retry_count,
            "endpoint": endpoint,
            "suggested_action": "Check network connection and server availability"
        }
        super().__init__(message, 503, "NETWORK_001")

class ValidationError(BlackbirdError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_name: str = None, field_value: Any = None):
        details = {
            "field_name": field_name,
            "field_value": str(field_value) if field_value else None,
            "suggested_action": "Check input parameters and their formats"
        }
        super().__init__(message, 400, "VALIDATION_001")

class AgentInitializationError(BlackbirdError):
    """Raised when agent initialization fails."""
    
    def __init__(self, message: str, agent_type: str = None, model_name: str = None, 
                 memory_usage: str = None, cleanup_attempted: bool = False):
        details = {
            "agent_type": agent_type,
            "model_name": model_name,
            "memory_usage": memory_usage,
            "cleanup_attempted": cleanup_attempted,
            "suggested_action": "Try cleanup() before initialization or use a different model"
        }
        super().__init__(message, 500, "AGENT_INIT_001")

class ModelLoadError(BlackbirdError):
    """Raised when model loading fails."""
    
    def __init__(self, message: str, model_name: str = None, memory_available: str = None):
        details = {
            "model_name": model_name,
            "memory_available": memory_available,
            "suggested_action": "Free memory or try a smaller model"
        }
        super().__init__(message, 500, "MODEL_LOAD_001")

class StreamingResponseError(BlackbirdError):
    """Raised when streaming response parsing fails."""
    
    def __init__(self, message: str, response_format: str = None, fallback_available: bool = True):
        details = {
            "response_format": response_format,
            "fallback_available": fallback_available,
            "suggested_action": "Retry request or use non-streaming mode"
        }
        super().__init__(message, 500, "STREAMING_001")

class MemoryError(BlackbirdError):
    """Raised when memory-related errors occur."""
    
    def __init__(self, message: str, memory_usage: str = None, cleanup_recommended: bool = True):
        details = {
            "memory_usage": memory_usage,
            "cleanup_recommended": cleanup_recommended,
            "suggested_action": "Run cleanup() to free memory"
        }
        super().__init__(message, 500, "MEMORY_001")

class FileProcessingError(BlackbirdError):
    """Raised when file processing fails."""
    
    def __init__(self, message: str, file_path: str = None, file_type: str = None, file_size: str = None):
        details = {
            "file_path": file_path,
            "file_type": file_type,
            "file_size": file_size,
            "suggested_action": "Check file format and size limits"
        }
        super().__init__(message, 400, "FILE_PROC_001")

class TimeoutError(BlackbirdError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, timeout_duration: int = None, operation: str = None):
        details = {
            "timeout_duration": timeout_duration,
            "operation": operation,
            "suggested_action": "Increase timeout or check server status"
        }
        super().__init__(message, 504, "TIMEOUT_001")

# NEW ERROR CLASSES FOR SESSION MANAGER
class QuotaExceededError(BlackbirdError):
    """Raised when quota limits are exceeded."""
    
    def __init__(self, message: str, quota_type: str = None, used: int = None, limit: int = None):
        details = {
            "quota_type": quota_type,
            "used": used,
            "limit": limit,
            "suggested_action": "Wait for quota reset or upgrade plan"
        }
        super().__init__(message, 400, "QUOTA_001")

class RateLimitError(BlackbirdError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, limit_type: str = None, retry_after: float = None):
        details = {
            "limit_type": limit_type,
            "retry_after": retry_after,
            "suggested_action": f"Wait {retry_after} seconds before retrying" if retry_after else "Wait before retrying"
        }
        super().__init__(message, 429, "RATE_LIMIT_001")

# NEW ERROR CLASS FOR ATLASTUNE INTEGRATION
class TrainingError(BlackbirdError):
    """Raised when model training fails."""
    
    def __init__(self, message: str, model_name: str = None, training_step: int = None, 
                 epoch: int = None, loss_value: float = None):
        details = {
            "model_name": model_name,
            "training_step": training_step,
            "epoch": epoch,
            "loss_value": loss_value,
            "suggested_action": "Check training data and model configuration"
        }
        super().__init__(message, 500, "TRAINING_001")

# ADDITIONAL ERROR CLASSES FOR COMPREHENSIVE ATLASTUNE SUPPORT
class ExecutionError(BlackbirdError):
    """Raised when execution of operations fails."""
    
    def __init__(self, message: str, operation: str = None, stage: str = None):
        details = {
            "operation": operation,
            "stage": stage,
            "suggested_action": "Check operation parameters and system resources"
        }
        super().__init__(message, 500, "EXECUTION_001")

class FineTuningError(Exception):
    """Exception raised for fine-tuning related errors."""
    
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message

class WebScrapingError(Exception):
    """Exception raised for web scraping errors."""
    
    def __init__(self, message: str, url: str = None):
        self.message = message
        self.url = url
        super().__init__(self.message)

class LicensingError(Exception):
    """Exception raised for licensing errors."""
    
    def __init__(self, message: str, license_key: str = None):
        self.message = message
        self.license_key = license_key
        super().__init__(self.message)
