"""Event types and configurations for the Decompute SDK streaming functionality."""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import time
import json


class EventType(Enum):
    """Types of events that can be streamed."""
    CHAT_MESSAGE = "chat_message"
    TRAINING_PROGRESS = "training_progress"
    IMAGE_GENERATION = "image_generation"
    MODEL_LOADING = "model_loading"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    COMPLETION = "completion"


class StreamStatus(Enum):
    """Status of a stream connection."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    PAUSED = "paused"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for a stream connection."""
    endpoint: str
    event_type: EventType
    reconnect: bool = True
    max_retries: int = 5
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    timeout: float = 30.0
    error_timeout: float = 5.0  # Add this line - timeout for error scenarios
    headers: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Set default headers for SSE connections."""
        if self.headers is None:
            self.headers = {}
        
        # Ensure SSE headers are set
        default_headers = {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
        
        for key, value in default_headers.items():
            if key not in self.headers:
                self.headers[key] = value


@dataclass
class StreamEvent:
    """Represents a single event from a stream."""
    event_type: str  # Changed from enum to string to match test expectations
    data: Dict[str, Any]
    timestamp: float
    stream_id: str
    raw_data: Optional[str] = None
    
    @classmethod
    def from_sse_line(cls, line: str, stream_id: str) -> Optional['StreamEvent']:
        """Parse an SSE line into a StreamEvent."""
        if not line.startswith('data: '):
            return None
        
        try:
            raw_data = line[6:].strip()  # Remove 'data: ' prefix
            data = json.loads(raw_data)
            
            # Determine event type from data content - match test expectations
            event_type = "message"  # Default for valid JSON
            if 'response' in data:
                event_type = "message"
            elif 'status' in data:
                event_type = "status"
            elif 'error' in data:
                event_type = "error"
            elif 'type' in data:
                event_type = data['type']
            
            return cls(
                event_type=event_type,
                data=data,
                timestamp=time.time(),
                stream_id=stream_id,
                raw_data=raw_data
            )
        except (json.JSONDecodeError, Exception):
            # Return as raw text event - match test expectations
            return cls(
                event_type='raw',
                data={'message': line[6:].strip()},
                timestamp=time.time(),
                stream_id=stream_id,
                raw_data=line[6:].strip()
            )


class StreamConfigPresets:
    """Pre-configured stream configurations for common use cases."""
    
    @staticmethod
    def chat_stream(endpoint: str = '/chat') -> StreamConfig:
        """Configuration for chat streaming."""
        return StreamConfig(
            endpoint=endpoint,
            event_type=EventType.CHAT_MESSAGE,
            reconnect=True,
            max_retries=3,
            retry_delay=1.0,
            timeout=30.0
        )
    
    @staticmethod
    def training_progress(endpoint: str = '/training') -> StreamConfig:
        """Configuration for training progress streaming."""
        return StreamConfig(
            endpoint=endpoint,
            event_type=EventType.TRAINING_PROGRESS,
            reconnect=True,
            max_retries=5,
            retry_delay=2.0,
            timeout=60.0
        )
    
    @staticmethod
    def image_generation(endpoint: str = '/generate') -> StreamConfig:
        """Configuration for image generation progress streaming."""
        return StreamConfig(
            endpoint=endpoint,
            event_type=EventType.IMAGE_GENERATION,
            reconnect=True,
            max_retries=3,
            retry_delay=1.0,
            timeout=45.0
        )
    
    @staticmethod
    def model_loading(endpoint: str = '/model') -> StreamConfig:
        """Configuration for model loading progress streaming."""
        return StreamConfig(
            endpoint=endpoint,
            event_type=EventType.MODEL_LOADING,
            reconnect=False,  # Model loading is typically one-time
            max_retries=2,
            retry_delay=1.0,
            timeout=120.0
        )


# Type definitions for callbacks
StreamCallback = Callable[[StreamEvent], None]
ErrorCallback = Callable[[Exception, str], None]
StatusCallback = Callable[[StreamStatus, str], None]

EventData = StreamEvent
