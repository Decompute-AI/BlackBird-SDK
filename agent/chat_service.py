"""Enhanced Chat service for the Blackbird SDK - Fixed Version"""

from oss_utils.constants import CHAT, INITIALIZE_RAG, GET_SUGGESTIONS
from oss_utils.errors import ValidationError, StreamingResponseError
from oss_utils.feature_flags import is_feature_enabled, require_feature
from typing import Optional, Callable, Dict, Any, Union
import json
import os
import platform
import requests
system= platform.machine().lower()
class ChatService:
    """Handles chat interactions with the Decompute API, including streaming support."""
    
    def __init__(self, http_client, event_source_manager=None):
        """Initialize the chat service."""
        self.http_client = http_client
        self.event_source_manager = event_source_manager
    
    def send_message(self, message: str, options: Dict[str, Any]=None, streaming=False, on_chunk=None, on_complete=None, on_error=None, **kwargs):
        """Send a message to the agent with optional streaming support."""
        # Prepare data payload - Keep it simple like the original
        data = { 
            'message': message
        }
        
        options = options or {}
        if not options.get("agent"):
            raise ValidationError("agent is required", field_name="agent")
        data.update(options)
        
        # Use regular HTTP request - exactly like the original
        try:
            if streaming and self.event_source_manager and is_feature_enabled("streaming_responses"):
                # return self._send_streaming_message(data, on_chunk, on_complete, on_error)
            # if streaming and self.event_source_manager:
                return self._handle_streaming_response(data, **kwargs)
            response = self.http_client.post(CHAT, data=data)
            return response
        except Exception as e:
            # Log additional context for debugging
            if hasattr(self.http_client, 'logger'):
                self.http_client.logger.error(f"Chat request failed for message: {message[:50]}...")
            raise e
    
    def _extract_response_text(self, response: Union[Dict, str]) -> str:
        """Extract clean response text from API response."""
        
        if isinstance(response, str):
            return response
            
        if isinstance(response, dict):
            # Try different possible response keys
            for key in ['response', 'message', 'text', 'content', 'answer']:
                if key in response:
                    return str(response[key])
            
            # If no recognized key, return the full response as JSON string
            return json.dumps(response, indent=2)
        
        # Fallback to string conversion
        return str(response)

    def _handle_streaming_response(self, data: Dict[str, Any], 
                                 on_chunk: Optional[Callable] = None,
                                 on_complete: Optional[Callable] = None,
                                 on_error: Optional[Callable] = None) -> str:
        """Handle streaming response with user callbacks."""
        
        if not self.event_source_manager:
            raise StreamingResponseError("EventSourceManager not available")

        collected_response = []
        
        def default_chunk_handler(event):
            """Default chunk handler that collects response."""
            try:
                chunk_text = ""
                if hasattr(event, 'data') and isinstance(event.data, dict):
                    if 'response' in event.data:
                        chunk_text = event.data['response']
                    elif 'message' in event.data:
                        chunk_text = event.data['message']
                
                if chunk_text:
                    collected_response.append(chunk_text)
                    
                    # Call user's chunk handler if provided
                    if on_chunk:
                        on_chunk(chunk_text)
                        
            except Exception as e:
                if on_error:
                    on_error(e)

        def completion_handler():
            """Handle stream completion."""
            if on_complete:
                on_complete()

        def error_handler(error):
            """Handle stream errors."""
            if on_error:
                on_error(error)

        # Start streaming
        stream_id = self.event_source_manager.start_chat_stream(
            message=data['message'],
            agent=data.get('agent', ''),
            model=data.get('model', ''),
            event_callback=default_chunk_handler
        )

        # Wait for completion (with timeout)
        import time
        timeout = 30  # 30 second timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.event_source_manager.get_stream_status(stream_id)
            if status in ['CLOSED', 'ERROR', 'COMPLETED']:
                break
            time.sleep(0.1)

        # Return collected response
        return ''.join(collected_response)

    def _send_streaming_message(self, data: Dict[str, Any],
                               on_chunk: Optional[Callable] = None,
                               on_complete: Optional[Callable] = None,
                               on_error: Optional[Callable] = None) -> str:
        """Send a streaming message using EventSourceManager."""
        if not self.event_source_manager:
            raise StreamingResponseError("EventSourceManager not available")
        
        def handle_event(event):
            """Handle incoming stream events."""
            try:
                event_data = event.data
                # Handle different event types
                if event_data.get('status') == 'complete':
                    if on_complete:
                        on_complete()
                elif event_data.get('status') == 'error':
                    error_msg = event_data.get('error', 'Stream error')
                    if on_error:
                        on_error(Exception(error_msg))
                elif 'response' in event_data:
                    # Regular message chunk
                    if on_chunk:
                        chunk_data = {
                            'message': event_data['response'],
                            'tokens_per_second': event_data.get('tokens_per_second', 0)
                        }
                        on_chunk(chunk_data)
            except Exception as e:
                if on_error:
                    on_error(e)
        
        def handle_error(error, stream_id):
            """Handle stream errors."""
            if on_error:
                on_error(error)
        
        # Start the streaming chat
        stream_id = self.event_source_manager.start_chat_stream(
            message=data['message'],
            agent=data.get('agent', ''),
            model=data.get('model', ''),
            event_callback=handle_event
        )
        return stream_id
    
    @require_feature("streaming_responses")
    def send_streaming_message(self, message, agent=None, model=None,
                              on_chunk=None, on_complete=None, on_error=None):
        """Simplified streaming message interface."""
        if not self.event_source_manager:
            raise StreamingResponseError("Streaming not available - EventSourceManager not initialized")
        if system!="darwin":
            data = {
                'message': message,
                'agent': agent or '',
                'model': model or 'unsloth/Qwen3-1.7B-bnb-4bit'
            }
        else:
            data = {
                'message': message,
                'agent': agent or '',
                'model': model or 'mlx-community/Qwen3-1.7B-bnb-4bit'
            }

        return self._send_streaming_message(data, on_chunk, on_complete, on_error)
    def send_message_with_files(self, message, files, options=None):
        """Send a message with files for RAG processing."""
        
        # Prepare files for upload using the corrected format
        if isinstance(files, list):
            file_data = {}
            for i, file_path in enumerate(files):
                if isinstance(file_path, str):
                    with open(file_path, 'rb') as f:
                        filename = os.path.basename(file_path)
                        file_data['files'] = (filename, f.read(), 'application/octet-stream')
                else:
                    file_data['files'] = file_path
        else:
            # Single file case
            if isinstance(files, str):
                with open(files, 'rb') as f:
                    filename = os.path.basename(files)
                    file_data = {'files': (filename, f.read(), 'application/octet-stream')}
            else:
                file_data = {'files': files}
        if system!="darwin":
        # Prepare form data for RAG initialization
            upload_data = {
                'agent': options.get('agent', 'general') if options else 'general',
                'model': options.get('model', 'unsloth/Qwen3-1.7B-bnb-4bit') if options else 'unsloth/Qwen3-1.7B-bnb-4bit',
                'use_finetuning': False
            }
        else:
            upload_data = {
                'agent': options.get('agent', 'general') if options else 'general',
                'model': options.get('model', 'mlx-community/Qwen3-1.7B-bnb-4bit') if options else 'mlx-community/Qwen3-1.7B-bnb-4bit',
                'use_finetuning': False
            }
        if options:
            for key, value in options.items():
                if key not in ['agent', 'model']:
                    upload_data[key] = value

        # Initialize RAG with the files
        rag_response = self.http_client.post(
            INITIALIZE_RAG,
            data=upload_data,
            files=file_data
        )

        # Then send the message
        chat_data = {
            'message': message,
            'agent': upload_data['agent'],
            'model': upload_data['model']
        }

        return self.http_client.post(CHAT, data=chat_data)

    
    def load_chat(self, agent_id, file_path):
        """Load a previous chat history."""
        data = {
            'agent_id': agent_id,
            'file_path': file_path
        }
        
        return self.http_client.post(f'/api/load-conversation/{agent_id}', data=data)
    
    def get_message(self, message_id):
        """Get a specific message by ID."""
        return self.http_client.get(f'/api/messages/{message_id}')
    
    def get_suggestions(self, agent_type, input_text, max_suggestions=5):
        """Get input suggestions for auto-completion."""
        data = {
            'agent_type': agent_type,
            'input': input_text,
            'max_suggestions': max_suggestions
        }
        
        return self.http_client.post(GET_SUGGESTIONS, data=data)
    
    def stop_stream(self, stream_id: str):
        """Stop a specific chat stream."""
        if self.event_source_manager:
            self.event_source_manager.stop_stream(stream_id)
    
    def pause_stream(self, stream_id: str):
        """Pause a specific chat stream."""
        if self.event_source_manager:
            self.event_source_manager.pause_stream(stream_id)
    
    def resume_stream(self, stream_id: str):
        """Resume a specific chat stream."""
        if self.event_source_manager:
            self.event_source_manager.resume_stream(stream_id)
    
    def get_stream_status(self, stream_id: str):
        """Get the status of a specific chat stream."""
        if self.event_source_manager:
            return self.event_source_manager.get_stream_status(stream_id)
        return None
    
    def get_active_streams(self):
        """Get list of active chat streams."""
        if self.event_source_manager:
            return self.event_source_manager.get_active_streams()
        return []

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