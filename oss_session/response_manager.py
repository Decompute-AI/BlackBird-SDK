"""Response management for storing and manipulating chat responses."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path

class ChatResponse:
    """Represents a single chat response."""
    
    def __init__(self, message: str, response: str, agent: str, model: str, 
                 timestamp: Optional[datetime] = None, metadata: Dict[str, Any] = None):
        self.message = message
        self.response = response
        self.agent = agent
        self.model = model
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'message': self.message,
            'response': self.response,
            'agent': self.agent,
            'model': self.model,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatResponse':
        """Create from dictionary."""
        return cls(
            message=data['message'],
            response=data['response'],
            agent=data['agent'],
            model=data['model'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

class ResponseManager:
    """Manages chat responses with storage and retrieval capabilities."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.responses: List[ChatResponse] = []
        self.storage_path = storage_path or str(Path.home() / ".blackbird" / "responses.json")
        self._ensure_storage_dir()
        self.load_responses()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
    
    def add_response(self, message: str, response: str, agent: str, model: str, 
                    metadata: Dict[str, Any] = None) -> ChatResponse:
        """Add a new response."""
        chat_response = ChatResponse(message, response, agent, model, metadata=metadata)
        self.responses.append(chat_response)
        self.save_responses()
        return chat_response
    
    def get_responses(self, limit: Optional[int] = None, 
                     agent: Optional[str] = None) -> List[ChatResponse]:
        """Get responses with optional filtering."""
        filtered_responses = self.responses
        
        if agent:
            filtered_responses = [r for r in filtered_responses if r.agent == agent]
        
        if limit:
            filtered_responses = filtered_responses[-limit:]
        
        return filtered_responses
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history in a chat format."""
        recent_responses = self.responses[-limit:] if limit else self.responses
        
        history = []
        for response in recent_responses:
            history.append({"role": "user", "content": response.message})
            history.append({"role": "assistant", "content": response.response})
        
        return history
    
    def search_responses(self, query: str) -> List[ChatResponse]:
        """Search responses by content."""
        query_lower = query.lower()
        return [
            r for r in self.responses 
            if query_lower in r.message.lower() or query_lower in r.response.lower()
        ]
    
    def clear_responses(self):
        """Clear all responses."""
        self.responses.clear()
        self.save_responses()
    
    def save_responses(self):
        """Save responses to storage."""
        try:
            data = [r.to_dict() for r in self.responses]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save responses: {e}")
    
    def load_responses(self):
        """Load responses from storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.responses = [ChatResponse.from_dict(item) for item in data]
        except Exception as e:
            print(f"Failed to load responses: {e}")
            self.responses = []
    
    def export_responses(self, format: str = 'json', output_path: Optional[str] = None) -> str:
        """Export responses in different formats."""
        if not output_path:
            output_path = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        if format == 'json':
            data = [r.to_dict() for r in self.responses]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'txt':
            with open(output_path, 'w') as f:
                for response in self.responses:
                    f.write(f"Timestamp: {response.timestamp}\n")
                    f.write(f"Agent: {response.agent}\n")
                    f.write(f"User: {response.message}\n")
                    f.write(f"Assistant: {response.response}\n")
                    f.write("-" * 50 + "\n")
        
        return output_path
