"""User-centric logger for displaying only relevant information to end users."""

import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import threading


class UserLogLevel(Enum):
    """User-facing log levels."""
    SUCCESS = "SUCCESS"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"


class UserLogFormatter:
    """Clean formatter for user-facing messages."""
    
    def __init__(self, show_timestamps: bool = True, colored: bool = True):
        self.show_timestamps = show_timestamps
        self.colored = colored
        self.colors = {
            UserLogLevel.SUCCESS: '\033[92m',  # Green
            UserLogLevel.INFO: '\033[94m',     # Blue
            UserLogLevel.WARNING: '\033[93m',  # Yellow
            UserLogLevel.ERROR: '\033[91m',    # Red
        }
        self.reset_color = '\033[0m'
    
    def format_message(self, level: UserLogLevel, message: str, **kwargs) -> str:
        """Format user message with clean styling."""
        timestamp = ""
        if self.show_timestamps:
            timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] "
        
        # Create clean message
        formatted_msg = f"{timestamp}{message}"
        
        # Add context if provided
        if kwargs:
            context_parts = []
            for key, value in kwargs.items():
                if key in ['agent_type', 'model', 'operation', 'status']:
                    context_parts.append(f"{key}={value}")
            
            if context_parts:
                formatted_msg += f" ({', '.join(context_parts)})"
        
        # Apply colors for console only
        if self.colored and level in self.colors:
            console_msg = f"{self.colors[level]}{formatted_msg}{self.reset_color}"
            return console_msg, formatted_msg  # Return both colored and plain
        
        return formatted_msg, formatted_msg


class UserLogger:
    """User-centric logger for clean, relevant messaging."""
    
    def __init__(self, enabled: bool = True, show_timestamps: bool = True, 
                 colored: bool = True, file_path: Optional[str] = None):
        self.enabled = enabled
        self.formatter = UserLogFormatter(show_timestamps, colored)
        self.file_path = file_path
        self._file_lock = threading.Lock()
        
        # Track conversation context
        self.current_agent = None
        self.current_model = None
        self.conversation_active = False
    
    def _write_to_file(self, message: str):
        """Safely write to file."""
        if not self.file_path:
            return
        
        try:
            with self._file_lock:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
                    f.flush()  # Ensure immediate write
        except Exception as e:
            # Silently fail for user logger, but print to stderr for debugging
            print(f"UserLogger file write error: {e}", file=sys.stderr)
    
    def _log(self, level: UserLogLevel, message: str, **kwargs):
        """Internal logging method."""
        if not self.enabled:
            return
        
        # Get formatted messages
        result = self.formatter.format_message(level, message, **kwargs)
        if isinstance(result, tuple):
            console_msg, file_msg = result
        else:
            console_msg = file_msg = result
        
        # Print to console
        print(console_msg)
        
        # Write to file
        self._write_to_file(file_msg)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        self._log(UserLogLevel.SUCCESS, f"✅ {message}", **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(UserLogLevel.INFO, f"ℹ️  {message}", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(UserLogLevel.WARNING, f"⚠️  {message}", **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log user-facing error message."""
        self._log(UserLogLevel.ERROR, f"❌ {message}", **kwargs)
    
    def chat_response(self, response: str, agent: str = None, model: str = None):
        """Display chat response cleanly."""
        if not self.enabled:
            return
        
        agent_info = f" ({agent})" if agent else ""
        console_msg = f"\n🤖 Assistant{agent_info}:\n{response}\n"
        file_msg = f"🤖 Assistant{agent_info}:\n{response}"
        
        print(console_msg)
        self._write_to_file(file_msg)
    
    def user_message(self, message: str):
        """Display user message."""
        if not self.enabled:
            return
        
        console_msg = f"👤 You: {message}"
        file_msg = f"👤 You: {message}"
        
        print(console_msg)
        self._write_to_file(file_msg)
    
    def agent_ready(self, agent_type: str, model: str = None):
        """Notify user that agent is ready."""
        self.current_agent = agent_type
        self.current_model = model
        model_info = f" with {model}" if model else ""
        self.success(f"{agent_type.title()} agent ready{model_info}")
    
    def operation_start(self, operation: str, **kwargs):
        """Notify operation start."""
        self.info(f"Starting {operation}...", **kwargs)
    
    def operation_complete(self, operation: str, **kwargs):
        """Notify operation completion."""
        self.success(f"{operation.title()} completed", **kwargs)
    
    def file_processed(self, filename: str, status: str = "processed"):
        """Notify file processing."""
        self.success(f"File {status}: {filename}")
    
    def progress_update(self, message: str, progress: int = None):
        """Show progress updates."""
        if progress is not None:
            self.info(f"{message} ({progress}%)")
        else:
            self.info(message)
    
    def set_context(self, agent: str = None, model: str = None):
        """Update current context."""
        if agent:
            self.current_agent = agent
        if model:
            self.current_model = model
    
    def clear_context(self):
        """Clear current context."""
        self.current_agent = None
        self.current_model = None
        self.conversation_active = False
    
    def enable(self):
        """Enable user logging."""
        self.enabled = True
    
    def disable(self):
        """Disable user logging."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if user logging is enabled."""
        return self.enabled
    
    def set_file_path(self, file_path: str):
        """Set or change the file path for logging."""
        with self._file_lock:
            self.file_path = file_path


# Global user logger instance
_user_logger = None


def get_user_logger(enabled: bool = True, show_timestamps: bool = True, 
                   colored: bool = True, file_path: Optional[str] = None) -> UserLogger:
    """Get or create the global user logger."""
    global _user_logger
    if _user_logger is None:
        _user_logger = UserLogger(enabled, show_timestamps, colored, file_path)
    return _user_logger


def configure_user_logging(enabled: bool = True, show_timestamps: bool = True, 
                          colored: bool = True, file_path: Optional[str] = None):
    """Configure global user logging settings."""
    global _user_logger
    _user_logger = UserLogger(enabled, show_timestamps, colored, file_path)


def disable_user_logging():
    """Disable user logging globally."""
    global _user_logger
    if _user_logger:
        _user_logger.disable()


def enable_user_logging():
    """Enable user logging globally."""
    global _user_logger
    if _user_logger:
        _user_logger.enable()
