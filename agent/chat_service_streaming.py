"""Chat service for the Decompute SDK with streaming support."""

from typing import Dict, Any, Optional, Callable, List

from oss_utils.constants import CHAT, INITIALIZE_RAG, GET_SUGGESTIONS, CHAT_STREAM
from oss_utils.errors import ValidationError, StreamingResponseError
from oss_session.event_types import EventType, EventData
from oss_utils.feature_flags import is_feature_enabled
import platform
import requests
system = platform.machine().lower()
class ChatService:
    """Handles chat interactions with the Decompute API."""
    
    def __init__(self, http_client, event_source_manager=None):
        """Initialize the chat service.
        
        Args:
            http_client: The HTTP client to use for requests
            event_source_manager: Optional EventSourceManager for streaming
        """
        self.http_client = http_client
        self.event_source_manager = event_source_manager
        self.active_streams = {}
        
    def send_message(self, message, options=None):
        if system=="darwin":
            data = {
            'message': message,
            'agent': options.get('agent', '') if options else '',
            'model': options.get('model', 'mlx-community/Qwen3-4B-4bit') if options else 'mlx-community/Qwen3-4B-4bit'
        }
        else:
            data = {
                'message': message,
                'agent': options.get('agent', '') if options else '',
                'model': options.get('model', 'unsloth/Qwen3-1.7B-bnb-4bit') if options else 'unsloth/Qwen3-1.7B-bnb-4bit'
            }
        if options:
            data.update(options)
        return self.http_client.post(CHAT, data=data)
    
    def send_streaming_message(
        self,
        message: str,
        *,
        on_chunk: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a message to the agent with streaming response."""
        if not is_feature_enabled("streaming_responses"):
            raise StreamingResponseError(
                "Streaming responses are not enabled. "
                "Enable the 'streaming_responses' feature flag to use this functionality.",
                fallback_available=True
            )
        if self.event_source_manager is None:
            raise StreamingResponseError(
                "EventSourceManager not available. "
                "Initialize the SDK with streaming support to use this functionality.",
                fallback_available=True
            )
        # Prepare query parameters
        if system=="darwin":
            query_params = {
                'message': message,
                'agent': options.get('agent', '') if options else '',
                'model': options.get('model', 'mlx-community/Qwen3-4B-4bit') if options else 'mlx-community/Qwen3-4B-4bit'
            }
        else:
            query_params = {
            'message': message,
            'agent': options.get('agent', '') if options else '',
            'model': options.get('model', 'unsloth/Qwen3-1.7B-bnb-4bit') if options else 'unsloth/Qwen3-1.7B-bnb-4bit'
        }
            
        if options:
            for key, value in options.items():
                query_params[key] = str(value)
                
        # Prepare event handler
        def handle_event(event_data: EventData):
            if event_data.event_type == EventType.MESSAGE_CHUNK:
                if on_chunk:
                    on_chunk(event_data.data)
            elif event_data.event_type == EventType.COMPLETE:
                if on_complete:
                    on_complete()
                    
        # Create the stream
        stream_id = self.event_source_manager.create_stream(
            endpoint=CHAT_STREAM,
            on_event=handle_event,
            on_error=on_error,
            query_params=query_params
        )
        
        # Store the stream ID
        self.active_streams[stream_id] = True
        
        return stream_id
    
    def cancel_streaming_message(self, stream_id: str) -> bool:
        """Cancel a streaming message response.
        
        Args:
            stream_id: The ID of the stream to cancel
            
        Returns:
            bool: True if the stream was cancelled, False otherwise
        """
        if self.event_source_manager is None:
            return False
            
        result = self.event_source_manager.close_stream(stream_id)
        
        if result and stream_id in self.active_streams:
            del self.active_streams[stream_id]
            
        return result
    
    def send_message_with_files(self, message, files, options=None):
        """Send a message with files for RAG processing.
        
        Args:
            message: The message to send
            files: File or list of files to upload
            options: Optional parameters
            
        Returns:
            dict: The response from the agent
        """
        # First upload files for RAG processing
        upload_data = {}
        
        if options:
            for key, value in options.items():
                upload_data[key] = value
                
        # Prepare files for upload
        file_objects = {}
        
        if isinstance(files, list):
            for i, file in enumerate(files):
                if isinstance(file, str):
                    # It's a file path
                    with open(file, 'rb') as f:
                        file_objects[f'files[{i}]'] = f.read()
                else:
                    # Assume it's a file-like object
                    file_objects[f'files[{i}]'] = file
        else:
            # Single file case
            if isinstance(files, str):
                with open(files, 'rb') as f:
                    file_objects['files[0]'] = f.read()
            else:
                file_objects['files[0]'] = files
                
        # Initialize RAG with the files
        rag_response = self.http_client.post(
            INITIALIZE_RAG,
            data=upload_data,
            files=file_objects
        )
        
        # Then send the message
        chat_data = {
            'message': message
        }
        
        if options:
            chat_data.update(options)
            
        return self.http_client.post(CHAT, data=chat_data)
    
    def send_streaming_message_with_files(
        self,
        message: str,
        files,
        *,
        on_chunk: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a message with files for RAG processing with streaming response.
        
        Args:
            message: The message to send
            files: File or list of files to upload
            on_chunk: Callback for each message chunk
            on_error: Callback for errors
            on_complete: Callback when response is complete
            options: Optional parameters
            
        Returns:
            str: Stream ID for the streaming response
        """
        # First upload files for RAG processing (same as non-streaming version)
        upload_data = {}
        
        if options:
            for key, value in options.items():
                upload_data[key] = value
                
        # Prepare files for upload
        file_objects = {}
        
        if isinstance(files, list):
            for i, file in enumerate(files):
                if isinstance(file, str):
                    # It's a file path
                    with open(file, 'rb') as f:
                        file_objects[f'files[{i}]'] = f.read()
                else:
                    # Assume it's a file-like object
                    file_objects[f'files[{i}]'] = file
        else:
            # Single file case
            if isinstance(files, str):
                with open(files, 'rb') as f:
                    file_objects['files[0]'] = f.read()
            else:
                file_objects['files[0]'] = files
                
        # Initialize RAG with the files
        rag_response = self.http_client.post(
            INITIALIZE_RAG,
            data=upload_data,
            files=file_objects
        )
        
        # Then send the streaming message (using the streaming method)
        return self.send_streaming_message(
            message=message,
            on_chunk=on_chunk,
            on_error=on_error,
            on_complete=on_complete,
            options=options
        )
        
    def load_chat(self, agent_id, file_path):
        """Load a previous chat history.
        
        Args:
            agent_id: The ID of the agent
            file_path: Path to the chat history file
            
        Returns:
            dict: The loaded chat history
        """
        data = {
            'agent_id': agent_id,
            'file_path': file_path
        }
        
        return self.http_client.post(f'/api/load-conversation/{agent_id}', data=data)
    
    def get_message(self, message_id):
        """Get a specific message by ID.
        
        Args:
            message_id: The ID of the message to retrieve
            
        Returns:
            dict: The message
        """
        return self.http_client.get(f'/api/messages/{message_id}')
    
    def get_suggestions(self, agent_type, input_text, max_suggestions=5):
        """Get input suggestions for auto-completion.
        
        Args:
            agent_type: The type of agent
            input_text: The input text to get suggestions for
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            list: List of suggestions
        """
        data = {
            'agent_type': agent_type,
            'input': input_text,
            'max_suggestions': max_suggestions
        }
        
        return self.http_client.post(GET_SUGGESTIONS, data=data)

    def stream_chat_response(self, message, agent="general", model=None, include_history=True):
        """
        Stream chat response from the backend as a generator.
        Yields each chunk of the response as it arrives.
        """
        url = self.http_client.base_url + "/chat"
        payload = {
            "message": message,
            "agent": agent,
            "model": model or "unsloth/Qwen3-1.7B-bnb-4bit",
            "include_history": include_history
        }
        with requests.post(url, json=payload, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    yield line.decode("utf-8")