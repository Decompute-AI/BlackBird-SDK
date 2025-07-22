"""Real-time streaming response handler."""

from typing import Callable, Optional, List
import threading
import time

class StreamingResponse:
    """Handles real-time streaming responses."""
    
    def __init__(self, sdk, message: str, agent: str, model: str):
        self.sdk = sdk
        self.message = message
        self.agent = agent
        self.model = model
        
        self.chunks: List[str] = []
        self.complete_response = ""
        self.is_streaming = False
        self.is_complete = False
        self.error = None
        
        self._lock = threading.Lock()
        self._chunk_callbacks: List[Callable] = []
        self._complete_callbacks: List[Callable] = []
    
    def on_chunk(self, callback: Callable[[str], None]):
        """Add callback for each chunk received."""
        self._chunk_callbacks.append(callback)
        return self
    
    def on_complete(self, callback: Callable[[str], None]):
        """Add callback for completion."""
        self._complete_callbacks.append(callback)
        return self
    
    def start(self) -> 'StreamingResponse':
        """Start the streaming response."""
        def chunk_handler(chunk_text: str):
            with self._lock:
                self.chunks.append(chunk_text)
                for callback in self._chunk_callbacks:
                    try:
                        callback(chunk_text)
                    except Exception as e:
                        print(f"Chunk callback error: {e}")
        
        def complete_handler(full_response: str):
            with self._lock:
                self.complete_response = full_response
                self.is_complete = True
                self.is_streaming = False
                
                for callback in self._complete_callbacks:
                    try:
                        callback(full_response)
                    except Exception as e:
                        print(f"Complete callback error: {e}")
        
        def error_handler(error):
            with self._lock:
                self.error = error
                self.is_streaming = False
        
        self.is_streaming = True
        
        # Start streaming
        self.sdk._handle_streaming_message(
            self.message,
            {'agent': self.agent, 'model': self.model},
            on_chunk=chunk_handler,
            on_complete=complete_handler,
            on_error=error_handler
        )
        
        return self
    
    def wait_for_completion(self, timeout: float = 30.0) -> str:
        """Wait for streaming to complete and return full response."""
        start_time = time.time()
        
        while self.is_streaming and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if self.error:
            raise self.error
        
        return self.complete_response
    
    def get_current_response(self) -> str:
        """Get current accumulated response."""
        with self._lock:
            return ''.join(self.chunks)
