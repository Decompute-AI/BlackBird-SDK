"""User-centric logger for displaying only relevant information to end users."""

import sys
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from .logger import get_logger
from .user_logger import get_user_logger, UserLogger
from .errors import BlackbirdError
from .logger import get_logger

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
        
        # Apply colors
        if self.colored and level in self.colors:
            formatted_msg = f"{self.colors[level]}{formatted_msg}{self.reset_color}"
        
        return formatted_msg


class UserLogger:
    """User-centric logger for clean, relevant messaging."""
    
    def __init__(self, enabled: bool = True, show_timestamps: bool = True, 
                 colored: bool = True, file_path: Optional[str] = None):
        self.enabled = enabled
        self.formatter = UserLogFormatter(show_timestamps, colored)
        self.file_path = file_path
        # self.internal_logger.logger.propagate = False
        # for h in self.internal_logger.logger.handlers:
        #     h.setLevel(logging.WARNING if development_mode else logging.ERROR)

        # Track conversation context
        self.current_agent = None
        self.current_model = None
        self.conversation_active = False
    
    def _log(self, level: UserLogLevel, message: str, **kwargs):
        """Internal logging method."""
        if not self.enabled:
            return
        
        formatted_message = self.formatter.format_message(level, message, **kwargs)
        
        # Print to console
        print(formatted_message)
        
        # Write to file if specified
        if self.file_path:
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(formatted_message + '\n')
            except Exception:
                pass  # Silently fail for user logger
    
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
        print(f"\n🤖 Assistant{agent_info}:")
        print(f"{response}\n")
        
        if self.file_path:
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n🤖 Assistant{agent_info}:\n{response}\n\n")
            except Exception:
                pass
    
    def user_message(self, message: str):
        """Display user message."""
        if not self.enabled:
            return
        
        print(f"👤 You: {message}")
        
        if self.file_path:
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(f"👤 You: {message}\n")
            except Exception:
                pass
    
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


# --- DisplayManager Implementation and Singleton Getter ---

class DisplayManager:
    """Manages what gets displayed to users vs logged internally."""
    
    def __init__(self, development_mode: bool = False, user_logging_enabled: bool = True):
        self.development_mode = development_mode
        self.internal_logger = get_logger(force_console= development_mode)
        # Initialize user logger
        self.user_logger = get_user_logger(enabled=user_logging_enabled and not development_mode)
        self._user_logging_enabled = user_logging_enabled

    def log_operation(self, operation: str, internal_message: str, 
                     user_message: str = None, success: bool = True, **kwargs):
        """Log operation to both internal and user loggers appropriately."""
        # Always log internally
        if success:
            self.internal_logger.info(internal_message, operation=operation, **kwargs)
        else:
            self.internal_logger.error(internal_message, operation=operation, **kwargs)
        # Log to user only if enabled and message provided
        if user_message and self.user_logger.is_enabled():
            if success:
                self.user_logger.success(user_message, **kwargs)
            else:
                self.user_logger.error(user_message, **kwargs)

    def log_chat_interaction(self, user_message: str, assistant_response: str, 
                           agent: str = None, model: str = None):
        """Log chat interaction."""
        # Internal logging
        self.internal_logger.info("Chat interaction", 
                                agent=agent, 
                                model=model,
                                user_message_length=len(user_message),
                                response_length=len(assistant_response))
        # User display - only if not in development mode
        if self.user_logger.is_enabled() and not self.development_mode:
            self.user_logger.user_message(user_message)
            self.user_logger.chat_response(assistant_response, agent, model)

    def log_initialization(self, component: str, success: bool = True, 
                          error: Exception = None, show_to_user: bool = False):
        """Log component initialization."""
        # Internal logging
        if success:
            self.internal_logger.info(f"{component} initialized successfully")
        else:
            self.internal_logger.error(f"{component} initialization failed", error=error)
        # User notification only if explicitly requested and not in dev mode
        if show_to_user and self.user_logger.is_enabled() and not self.development_mode:
            if success:
                self.user_logger.success(f"{component} ready")
            else:
                self.user_logger.error(f"Failed to initialize {component}")

    def log_agent_ready(self, agent_type: str, model: str = None):
        """Log agent readiness."""
        # Internal logging
        self.internal_logger.info("Agent ready", 
                                agent_type=agent_type, 
                                model=model)
        # User notification - only if not in development mode
        if self.user_logger.is_enabled() and not self.development_mode:
            self.user_logger.agent_ready(agent_type, model)

    def log_error(self, error: Exception, user_friendly_message: str = None, 
                 show_to_user: bool = True):
        """Log error with appropriate visibility."""
        # Always log internally with full details
        self.internal_logger.error("Error occurred", error=error)
        # Show user-friendly message or generic message to user
        if show_to_user and self.user_logger.is_enabled() and not self.development_mode:
            if user_friendly_message:
                self.user_logger.error(user_friendly_message)
            elif isinstance(error, BlackbirdError):
                # Show clean error message for SDK errors
                self.user_logger.error(f"Operation failed: {error.message}")
            else:
                # Generic message for unexpected errors
                self.user_logger.error("An unexpected error occurred")

    def log_file_operation(self, filename: str, operation: str, success: bool = True):
        """Log file operations."""
        # Internal logging
        self.internal_logger.info(f"File {operation}", 
                                filename=filename, 
                                success=success)
        # User notification
        if self.user_logger.is_enabled() and not self.development_mode:
            if success:
                self.user_logger.file_processed(filename, operation)
            else:
                self.user_logger.error(f"Failed to {operation} file: {filename}")

    # In display_manager.py, modify the development_mode handling:

    def set_development_mode(self, enabled: bool):
        """Toggle development mode."""
        self.development_mode = enabled
        
        if enabled:
            # In development mode, disable user logger but don't print everything
            self.user_logger.disable()
            # Set internal logger to WARNING level to reduce noise
            self.internal_logger.logger.setLevel(logging.WARNING)
            print("🔧 Development mode enabled - reduced logging")
        else:
            # In production mode, enable user logger
            if self._user_logging_enabled:
                self.user_logger.enable()
            # Set internal logger to INFO level
            self.internal_logger.logger.setLevel(logging.INFO)
            print("👤 User mode enabled - clean interface")

    def progress_update(self, message: str, progress: int = None, internal_details: str = None):
        """Update progress with different messages for internal/user."""
        # Internal logging with details
        log_message = internal_details or message
        self.internal_logger.info(log_message, progress=progress)
        # User progress update - only if not in development mode
        if self.user_logger.is_enabled() and not self.development_mode:
            self.user_logger.progress_update(message, progress)

# Singleton instance
_display_manager = None

def get_display_manager(development_mode: bool = False, user_logging_enabled: bool = True):
    """Get or create the global display manager."""
    global _display_manager
    if _display_manager is None:
        _display_manager = DisplayManager(
            development_mode=development_mode,
            user_logging_enabled=user_logging_enabled
        )
    return _display_manager
