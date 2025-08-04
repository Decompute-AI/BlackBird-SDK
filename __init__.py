"""

"""

from typing import Dict, List, Any, Optional, Union, Callable
import platform
import atexit

# Agent management
from .agent.agent_manager import AgentManager
from .agent.chat_service import ChatService

# Server backend management
from .server.backend_manager import BackendManager

class BlackbirdSDK:
    """Enhanced SDK focused on agent creation and chat functionality.
    
    Args:
        runasync (bool): If True, runs the backend in a separate terminal in keepalive mode. The backend will persist and be reused by other SDK sessions. Default is False.
    """
    
    def __init__(self, log_level='INFO', 
                 development_mode=False, 
                 runasync=False):
        """Initialize the SDK for agent creation and chat.
        
        Args:
            runasync (bool): If True, runs the backend in a separate terminal in keepalive mode. The backend will persist and be reused by other SDK sessions. Default is False.
        """
        self.current_agent: str | None = None
        self.current_model: str | None = None
        self._runasync = runasync
        
        if runasync:
            # Start backend in keepalive async mode and skip normal backend init
            print("\n[INFO] Backend is running in keepalive async mode in a separate terminal. You can reuse this backend in other SDK sessions.\n")
            return
        
        self._initialize_sdk_components(log_level, development_mode)
        
        # Register cleanup
        atexit.register(self._cleanup_on_exit)
        
        # Automatically start the backend if not running
        self.backend_manager = BackendManager.get_instance()
        backend_status = self.backend_manager.get_backend_status()
        if not (backend_status['is_running'] and backend_status['health_check']):
            print("[SDK] Backend not running or unhealthy, starting now...")
            self.backend_manager.start()
        else:
            print("[SDK] Backend already running and healthy")
    
    def _initialize_sdk_components(self, log_level, development_mode):
        """Initialize SDK components."""
        print(f"[SDK] Initializing Blackbird SDK (development_mode={development_mode})")
        
        # Initialize backend manager
        self.backend_manager = BackendManager.get_instance()
        backend_status = self.backend_manager.get_backend_status()
        
        if (backend_status['is_running'] and backend_status['health_check']):
            print("Using existing healthy backend")
            backend_url = f"http://localhost:{backend_status['port']}"
        else:
            print("Starting backend server...")
            if not self.backend_manager.start():
                raise RuntimeError("Failed to start backend server")
            backend_url = f"http://localhost:5012"
        
        # Initialize HTTP client and services
        from .backends.windows.utils import HTTPClient
        self.http_client = HTTPClient(backend_url)
        self.agent_manager = AgentManager(self.http_client)
        self.chat_service = ChatService(self.http_client)
        
        print(f"Backend ready at: {backend_url}")
    
    def _cleanup_on_exit(self):
        """Cleanup resources on exit."""
        try:
            if hasattr(self, 'backend_manager') and not self._runasync:
                self.backend_manager.stop()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def initialize_agent(self, agent_type, model_name=None, options=None):
        """Initialize an AI agent.
        
        Args:
            agent_type (str): Type of agent to initialize
            model_name (str, optional): Model to use
            options (dict, optional): Additional options
            
        Returns:
            dict: Agent initialization response
        """
        return self.agent_manager.initialize_agent(agent_type, model_name, options)
    
    def send_message(self, message, streaming=False, return_full_response=False, **kwargs):
        """Send a message to the current agent.
        
        Args:
            message (str): Message to send
            streaming (bool): Whether to use streaming response
            return_full_response (bool): Whether to return full response object
            **kwargs: Additional options
            
        Returns:
            str or dict: Response from the agent
        """
        if not self.current_agent:
            raise ValueError("No agent initialized. Call initialize_agent() first.")
        
        options = {
            'agent': self.current_agent,
            **kwargs
        }
        
        if self.current_model:
            options['model'] = self.current_model
        
        return self.chat_service.send_message(message, options, streaming)
    
    def stream_message(self, message, on_chunk=None, on_complete=None, on_error=None, **kwargs):
        """Send a streaming message to the current agent.
        
        Args:
            message (str): Message to send
            on_chunk (callable): Callback for each chunk
            on_complete (callable): Callback when complete
            on_error (callable): Callback for errors
            **kwargs: Additional options
        """
        if not self.current_agent:
            raise ValueError("No agent initialized. Call initialize_agent() first.")
        
        options = {
            'agent': self.current_agent,
            **kwargs
        }
        
        if self.current_model:
            options['model'] = self.current_model
        
        return self.chat_service.send_message(
            message, options, streaming=True,
            on_chunk=on_chunk, on_complete=on_complete, on_error=on_error
        )
    
    def get_available_agents(self):
        """Get list of available agent types.
        
        Returns:
            list: Available agent types
        """
        return self.agent_manager.get_available_agents()
    
    def get_agent_capabilities(self, agent_type):
        """Get capabilities of a specific agent.
        
        Args:
            agent_type (str): Type of agent
            
        Returns:
            dict: Agent capabilities
        """
        return self.agent_manager.get_agent_capabilities(agent_type)
    
    def switch_agent(self, new_agent_type):
        """Switch to a different agent type.
        
        Args:
            new_agent_type (str): New agent type to switch to
            
        Returns:
            dict: Agent initialization response
        """
        self.current_agent = new_agent_type
        return self.agent_manager.switch_agent(new_agent_type)
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'backend_manager') and not self._runasync:
            self.backend_manager.stop()
    
    def get_status(self):
        """Get SDK status.
        
        Returns:
            dict: Status information
        """
        return {
            'current_agent': self.current_agent,
            'current_model': self.current_model,
            'backend_running': self.backend_manager.get_backend_status()['is_running'] if hasattr(self, 'backend_manager') else False
        }
