"""
Enhanced Blackbird SDK with production licensing and feature flags.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import platform
# Core configuration and utilities
from .oss_utils.config import ConfigManager
from .oss_utils.http_client import HTTPClient
from .oss_utils.logger import get_logger, clear_logs
from .oss_utils.feature_flags import get_feature_flags, configure_features, require_feature, is_feature_enabled
from .oss_utils.errors import *
from .oss_utils.web_search import WebSearchBackend
from .oss_session.response_manager import ResponseManager
from .oss_session.streaming_response import StreamingResponse
import platform
system= platform.system().lower()

# Open Source Blackbird SDK - Inference, RAG, and File Upload Only

# No licensing, authentication, or blackbird_sdk dependencies

# (You can add any open source SDK initialization code here if needed)

# Agent management
from .agent.agent_manager import AgentManager
from .agent.chat_service import ChatService

# File and data services
from .oss_data_pipeline.file_service import FileService

# Session and memory management
from .oss_session.event_source_manager import EventSourceManager
from .oss_session.event_types import EventType, StreamStatus, StreamConfig
from .oss_session.session_manager import SessionManager
from .oss_session.session_types import QuotaType, RateLimitType, SessionStatus
from .oss_session.memory_store import MemoryStore
from .oss_session.memory_types import MemoryStoreConfig, generate_cache_key

# Model and acceleration
from .oss_acceleration.platform_manager import PlatformManager
from .oss_acceleration.inference_engine import InferenceEngine
from .oss_acceleration.platform_types import InferenceConfig, OptimizationLevel
from .oss_model.model_service import ModelService
from .oss_model.model_types import ModelType, ModelFormat, ModelSource
from .oss_utils.display_manager import get_display_manager, DisplayManager
from .oss_utils.user_logger import get_user_logger

from .creation.builder import AgentBuilder, CustomAgent, create_agent
from .creation.templates import AgentTemplates
import atexit
from .server.backend_manager import BackendManager

class BlackbirdSDK:
    """Enhanced SDK with production licensing and feature management.
    
    Args:
        runasync (bool): If True, runs the backend in a separate terminal in keepalive mode. The backend will persist and be reused by other SDK sessions. Default is False.
    """
    
    def __init__(self, license_server_url: Optional[str] = None, log_level='INFO', 
                 structured_logging=True, feature_config=None, 
                 development_mode=False, user_logging=True, 
                 offline_mode=False, web_search_backend=None,
                 skip_licensing=False,
                 runasync=False):
        """Initialize the SDK with production licensing.
        
        Args:
            runasync (bool): If True, runs the backend in a separate terminal in keepalive mode. The backend will persist and be reused by other SDK sessions. Default is False.
        """
        self.license_manager = None
        self.user_config = None
        self.display_manager = get_display_manager(
            development_mode=development_mode,
            user_logging_enabled=user_logging
        )
        self.current_agent: str | None = None
        self.current_model: str | None = None
        self._runasync = runasync
        if runasync:
            # Start backend in keepalive async mode and skip normal backend init
            from .oss_acceleration.platform_manager import PlatformManager
            self.platform_manager = PlatformManager()
            self.platform_manager.run_backend_keepalive_async()
            print("\n[INFO] Backend is running in keepalive async mode in a separate terminal. You can reuse this backend in other SDK sessions.\n")
            return
        
        # Ensure user and license are set up
        # ensure_user_and_license() # This line is removed as per the edit hint

        # Initialize licensing system
        # if not skip_licensing and LICENSING_AVAILABLE: # This block is removed as per the edit hint
        #     try:
        #         self.user_config = get_user_config()
        #         self.license_manager = get_license_manager(license_server_url)
                
        #         # Initialize license manager
        #         if self.license_manager.initialize():
        #             license_info = self.license_manager.get_license_info()
        #             user_info = self.license_manager.get_user_info()
                    
        #             self.display_manager.user_logger.info(
        #                 f"Licensed user: {user_info.get('user_id', 'unknown')} "
        #                 f"(Tier: {user_info.get('tier', 'basic')})"
        #             )
                    
        #             if license_info:
        #                 self.display_manager.user_logger.info(
        #                     f"License valid until: {license_info.get('expires_at', 'unknown')}"
        #                 )
        #         else:
        #             self.display_manager.user_logger.warning(
        #                 "License initialization failed - running in limited mode"
        #             )
                    
        #     except Exception as e:
        #         self.display_manager.user_logger.error(f"License system error: {e}")
        #         if not development_mode:
        #             raise
        # else:
        #     self.display_manager.user_logger.info("Running without licensing system")
        
        self._initialize_sdk_components(log_level, structured_logging, feature_config, development_mode, user_logging)
        self.web_search_backend = web_search_backend or WebSearchBackend()
        
        # Register cleanup
        atexit.register(self._cleanup_on_exit)
        
        # Automatically start the backend if not running
        self.backend_manager = BackendManager.get_instance()
        backend_status = self.backend_manager.get_backend_status()
        if not (backend_status['is_running'] and backend_status['health_check']):
            print("[SDK] Backend not running or unhealthy, starting now...")
            self.backend_manager.start()
        else:
            print("[SDK] Backend already running and healthy.")
        
    def _initialize_sdk_components(self, log_level, structured_logging, feature_config, development_mode, user_logging):
        """Initialize SDK components with backend conflict prevention."""
        
        try:
            # Check for keepalive backend first
            from .oss_acceleration.platform_manager import PlatformManager
            platform_manager = PlatformManager()
            if platform_manager.is_keepalive_backend(5012):
                self.logger = get_logger(force_console=development_mode)
                self.display_manager.log_operation(
                    operation="backend_check",
                    internal_message="Detected keepalive backend running on port 5012. Will reuse it.",
                    user_message="Using existing backend (keepalive mode)" if not development_mode else None
                )
                backend_url = f"http://localhost:5012"
                config = {'base_url': backend_url}
                self.config = ConfigManager(config)
                self.platform_manager = platform_manager
                # Continue with the rest of initialization, skipping backend_manager.start()
                self.http_client = HTTPClient(self.config, self.logger)
                self.agent_manager = AgentManager(self.http_client)
                self._initialize_optional_components()
                if is_feature_enabled("streaming_responses"):
                    self.event_source_manager = EventSourceManager(self.http_client)
                else:
                    self.event_source_manager = None
                self.chat_service = ChatService(
                    self.http_client, 
                    self.event_source_manager
                )
                self.web_search_backend = self.web_search_backend if hasattr(self, 'web_search_backend') else WebSearchBackend()
                self.display_manager.log_operation(
                    operation="sdk_init",
                    internal_message="SDK initialized successfully (keepalive backend)",
                    user_message="✨ Blackbird SDK ready! (keepalive backend)" if not development_mode else None,
                    success=True,
                    base_url=self.config.get('base_url'),
                    enabled_features=len(self.feature_flags.get_enabled_features()) if hasattr(self, 'feature_flags') else 0
                )
                return
            # Initialize core components first
            self.config = ConfigManager()
            self.logger = get_logger(force_console=development_mode)
            self.response_manager = ResponseManager()
            # Initialize display manager
            self.display_manager = get_display_manager(
                development_mode=development_mode,
                user_logging_enabled=user_logging
            )
            
            # Initialize feature flags based on license
            # if self.license_manager: # This block is removed as per the edit hint
            #     available_features = self.license_manager.get_available_features()
            #     # Override feature config with license-based features
            #     if feature_config is None:
            #         feature_config = {}
            #     feature_config['enabled_features'] = available_features
            
            # self.feature_flags = configure_features(feature_config) # This line is removed as per the edit hint
            
            # Backend initialization with conflict prevention
            self.display_manager.log_operation(
                operation="backend_check",
                internal_message="Checking backend availability...",
                user_message="Checking services..." if not development_mode else None
            )
            
            # Import the enhanced backend manager
            from .server import BackendManager
            self.backend_manager = BackendManager.get_instance()
            
            # Check backend status before starting
            backend_status = self.backend_manager.get_backend_status()
            self.logger.info("Backend status check", **backend_status)
            # PATCH: Treat keepalive backend as valid even if is_running is None
            if (backend_status['is_running'] and backend_status['health_check']) or platform_manager.is_keepalive_backend(backend_status['port']):
                self.logger.info("Using existing healthy backend (or keepalive backend)")
                backend_url = f"http://localhost:{backend_status['port']}"
            else:
                # Start backend with improved logic
                self.display_manager.log_operation(
                    operation="backend_start",
                    internal_message="Starting backend server...",
                    user_message="Starting services..." if not development_mode else None
                )
                
                if not self.backend_manager.start():
                    raise RuntimeError("Failed to start backend server")
                
                # Get backend URL
                backend_url = f"http://localhost:5012"
            
            # Update config with backend URL
            config = {'base_url': backend_url}
            self.config = ConfigManager(config)
            
            # Initialize platform manager reference
            self.platform_manager = self.backend_manager.get_platform_manager()
            
            self.display_manager.log_operation(
                operation="backend_ready",
                internal_message=f"Backend ready at: {backend_url}",
                user_message="Services ready" if not development_mode else None,
                success=True
            )
            
            # Continue with rest of initialization...
            self.http_client = HTTPClient(self.config, self.logger)
            # self.auth_service = AuthenticationService(self.http_client, self.config)
            self.agent_manager = AgentManager(self.http_client)
            
            # Initialize remaining components
            self._initialize_optional_components()
            
            # Initialize chat service
            if is_feature_enabled("streaming_responses"):
                self.event_source_manager = EventSourceManager(self.http_client)
            else:
                self.event_source_manager = None
                
            self.chat_service = ChatService(
                self.http_client, 
                self.event_source_manager
            )
            
            # Register cleanup handler
            atexit.register(self._cleanup_on_exit)
            
            # Final success message
            self.display_manager.log_operation(
                operation="sdk_init",
                internal_message="SDK initialized successfully",
                user_message="✨ Blackbird SDK ready!" if not development_mode else None,
                success=True,
                base_url=self.config.get('base_url'),
                enabled_features=len(self.feature_flags.get_enabled_features()) if hasattr(self, 'feature_flags') else 0
            )
            
        except Exception as e:
            error = BlackbirdError(f"Failed to initialize SDK: {str(e)}")
            if hasattr(self, 'display_manager'):
                self.display_manager.log_error(
                    error=error,
                    user_friendly_message="Failed to start SDK. Please check your configuration.",
                    show_to_user=True
                )
            raise error

    def _initialize_optional_components(self):
        """Initialize optional components based on feature flags."""
        
        # Initialize file service if file upload is enabled
        if is_feature_enabled("file_upload"):
            self.file_service = FileService(self.http_client, self.logger)
            self.display_manager.log_initialization("File Service", success=True)
        else:
            self.file_service = None
        
        # Initialize session manager if enabled
        if is_feature_enabled("core_sdk"):
            session_config = self.config.get('session_config', {})
            self.session_manager = SessionManager(session_config)
            self.display_manager.log_initialization("Session Manager", success=True)
        else:
            self.session_manager = None
        
        # Initialize memory store if enabled
        if is_feature_enabled("core_sdk"):
            memory_config_dict = self.config.get('memory_config', {})
            memory_config = MemoryStoreConfig(**memory_config_dict)
            self.memory_store = MemoryStore(memory_config)
            self.display_manager.log_initialization("Memory Store", success=True)
        else:
            self.memory_store = None
        
        # Initialize inference engine if enabled
        if is_feature_enabled("core_sdk"):
            inference_config_dict = self.config.get('inference_config', {})
            
            # Handle string values from config for enum conversion
            if 'framework' in inference_config_dict and isinstance(inference_config_dict['framework'], str):
                try:
                    from .oss_acceleration.platform_types import InferenceFramework
                    inference_config_dict['framework'] = InferenceFramework(inference_config_dict['framework'])
                except (ValueError, ImportError):
                    self.logger.warning(f"Invalid framework: {inference_config_dict['framework']}, using default")
                    inference_config_dict.pop('framework', None)
            
            if 'optimization_level' in inference_config_dict and isinstance(inference_config_dict['optimization_level'], str):
                try:
                    from .oss_acceleration.platform_types import OptimizationLevel
                    inference_config_dict['optimization_level'] = OptimizationLevel(inference_config_dict['optimization_level'])
                except (ValueError, ImportError):
                    self.logger.warning(f"Invalid optimization_level: {inference_config_dict['optimization_level']}, using default")
                    inference_config_dict.pop('optimization_level', None)
            
            inference_config = InferenceConfig(**inference_config_dict)
            self.inference_engine = InferenceEngine(inference_config)
            self.display_manager.log_initialization("Inference Engine", success=True)
        else:
            self.inference_engine = None
        
        # Initialize model service if enabled
        if is_feature_enabled("core_sdk"):
            model_config = self.config.get('model_config', {})
            self.model_service = ModelService(model_config)
            self.display_manager.log_initialization("Model Service", success=True)
        else:
            self.model_service = None
        
        # Future feature placeholders
        if is_feature_enabled("mas_router"):
            self.logger.info("MAS router feature enabled - will be initialized when implemented")
        
        if is_feature_enabled("knowledge_graphs"):
            self.logger.info("Knowledge graphs feature enabled - will be initialized when implemented")
        
        if is_feature_enabled("function_calling"):
            self.logger.info("Function calling feature enabled - will be initialized when implemented")
    
    def create_custom_agent(self, name: str, description: str) -> AgentBuilder:
        """Create a custom agent using the builder pattern."""
        return create_agent(name, description)
    
    def get_agent_templates(self) -> List[str]:
        """Get list of available agent templates."""
        return AgentTemplates.list_templates()
    
    def create_agent_from_template(self, template_name: str) -> AgentBuilder:
        """Create agent from a template."""
        return AgentBuilder().from_template(template_name)
    
    def load_custom_agent(self, file_path: str) -> CustomAgent:
        """Load a custom agent from configuration file."""
        builder = AgentBuilder().from_file(file_path)
        return builder.build(sdk_instance=self)
    
    def deploy_custom_agent(self, agent: CustomAgent) -> bool:
        """Deploy a custom agent for use."""
        try:
            # Integrate with existing agent system
            self.current_agent = agent.config.name
            self.current_model = getattr(agent.config.model_config, 'model_name', None) or self.current_model
            
            # Store reference to custom agent
            if not hasattr(self, 'custom_agents'):
                self.custom_agents = {}
            self.custom_agents[agent.config.name] = agent
            
            self.display_manager.user_logger.success(f"Custom agent '{agent.config.name}' deployed successfully")
            return True
            
        except Exception as e:
            self.display_manager.log_error(
                error=e,
                user_friendly_message=f"Failed to deploy custom agent: {e}",
                show_to_user=True
            )
            return False
    def send_message_to_custom_agent(self, agent_name: str, message: str, **kwargs) -> str:
        if not hasattr(self, 'custom_agents') or agent_name not in self.custom_agents:
            raise ValidationError(f"Custom agent '{agent_name}' not found")
        import platform
        custom_agent = self.custom_agents[agent_name]
        # Determine model to use
        model_name = None
        if hasattr(custom_agent.config, 'model_config') and custom_agent.config.model_config:
            # model_config can be a dict or object with model_name
            if isinstance(custom_agent.config.model_config, dict):
                model_name = custom_agent.config.model_config.get('model_name')
            else:
                model_name = getattr(custom_agent.config.model_config, 'model_name', None)
        if not model_name:
            if platform.system().lower() == 'darwin':
                model_name = 'mlx-community/Qwen3-4B-4bit'
            else:
                model_name = 'unsloth/Qwen3-1.7B-bnb-4bit'
        # Merge model into kwargs/options
        options = dict(kwargs) if kwargs else {}
        options['model'] = model_name
        return custom_agent.process_message(message, options)

    def stream_message(self, message: str) -> StreamingResponse:
        """Create a streaming response object for real-time access."""
        
        
        return StreamingResponse(
            self, message, self.current_agent, self.current_model
        )

    def send_message_async(self, message: str, callback: Callable[[str], None]):
        """Send message asynchronously with callback."""
        import threading
        
        def async_send():
            try:
                response = self.send_message(message)
                callback(response)
            except Exception as e:
                callback(f"Error: {e}")
        
        thread = threading.Thread(target=async_send)
        thread.daemon = True
        thread.start()
        return thread

    def chat_interactive(self):
        """Start an interactive chat session."""
        print(f"🤖 Interactive chat with {self.current_agent} agent")
        print("Type 'quit' to exit, 'clear' to clear history")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    self.clear_chat_history()
                    print("🗑️ Chat history cleared")
                    continue
                elif not user_input:
                    continue
                
                print("🤖 Assistant: ", end="")
                response = self.send_message(user_input, streaming=True)
                
            except KeyboardInterrupt:
                print("\n👋 Chat session ended")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    
    def get_response_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.response_manager.get_conversation_history(limit)

    def search_responses(self, query: str) -> List:
        """Search through previous responses."""
        return [r.to_dict() for r in self.response_manager.search_responses(query)]

    def export_chat_history(self, format: str = 'json', output_path: str = None) -> str:
        """Export chat history to file."""
        return self.response_manager.export_responses(format, output_path)

    def clear_chat_history(self):
        """Clear all chat history."""
        self.response_manager.clear_responses()
    def _log_enabled_features(self):
        """Log which features are enabled for debugging."""
        enabled_features = self.feature_flags.get_enabled_features()
        self.logger.info(f"Feature flags initialized with {len(enabled_features)} enabled features",
                        enabled_features=enabled_features)
        
        # Log feature categories
        from .oss_utils.feature_flags import FeatureCategory
        for category in FeatureCategory:
            category_features = self.feature_flags.get_features_by_category(category)
            enabled_in_category = [name for name, enabled in category_features.items() if enabled]
            if enabled_in_category:
                self.logger.debug(f"Category {category.value} features enabled: {enabled_in_category}")
    def _cleanup_on_exit(self):
        """Enhanced cleanup with backend management."""
        try:
            if hasattr(self, 'backend_manager') and self.backend_manager:
                self.logger.info("Cleaning up backend on exit...")
                self.backend_manager.stop()
                backend_status = self.backend_manager.get_backend_status()
                self.logger.info("Backend status during cleanup", **backend_status)
            if hasattr(self, 'platform_manager') and self.platform_manager:
                self.logger.info("Cleaning up platform manager...")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _log_enabled_features(self):
        """Log which features are enabled."""
        enabled_features = self.feature_flags.get_enabled_features()
        self.logger.info(f"Feature flags initialized with {len(enabled_features)} enabled features",
                        enabled_features=enabled_features)
        
        # Log feature categories
        from .oss_utils.feature_flags import FeatureCategory
        for category in FeatureCategory:
            category_features = self.feature_flags.get_features_by_category(category)
            enabled_in_category = [name for name, enabled in category_features.items() if enabled]
            if enabled_in_category:
                self.logger.debug(f"Category {category.value} features enabled: {enabled_in_category}")
    
    def clear_internal_logs(self, *, force: bool = False, keep_latest: int = 3) -> int:
        """
        Delete SDK log files from disk.
        • force=True  → delete everything
        • keep_latest → how many recent files to preserve
        Returns number of files removed.
        """
        removed = clear_logs(force=force, keep_latest=keep_latest)
        self.display_manager.user_logger.success(f"Cleared {removed} log files")
        return removed
    
    def _retry_with_cleanup(self, operation, *args, **kwargs):
        """Execute operation with automatic cleanup on memory errors."""
        max_retries = kwargs.pop('max_retries', 2)
        
        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except MemoryError as e:
                if attempt < max_retries:
                    self.logger.warning(f"Memory error, attempting cleanup and retry {attempt + 1}")
                    try:
                        self.cleanup()
                    except Exception as cleanup_error:
                        self.logger.error("Cleanup failed during retry", error=cleanup_error)
                    continue
                else:
                    raise e
            except Exception as e:
                # Don't retry for non-memory errors
                raise e
    
    @require_feature("basic_agents")
    def initialize_agent(self, agent_type, model_name=None):
        """Initialize agent with user-friendly logging."""
        
        # # Check if agent type is allowed by license
        # if self.license_manager and not self.license_manager.is_feature_enabled('agent_' + agent_type):
        #     if not self.license_manager.is_feature_enabled('all'):
        #         raise LicensingError(f"Agent type '{agent_type}' not enabled in your license")
        
        # Validate inputs first
        if agent_type not in self.get_available_agents():
            error = ValidationError(f"Invalid agent type: {agent_type}")
            self.display_manager.log_error(
                error=error,
                user_friendly_message=f"Agent type '{agent_type}' is not available",
                show_to_user=True
            )
            raise error
        
        if system== "darwin":
            model_name = model_name or 'mlx-community/Qwen3-4B-4bit'
        else:
            model_name = model_name or 'unsloth/Qwen3-1.7B-bnb-4bit'
        
        # Show progress to user
        self.display_manager.log_operation(
            operation="agent_init",
            internal_message=f"Initializing {agent_type} agent with {model_name}",
            user_message=f"Initializing {agent_type} agent...",
            success=True
        )
        
        try:
            response = self.agent_manager.initialize_agent(agent_type, model_name)
            
            if response and response.get('status') in ['success', 'partial_success']:
                self.current_agent = agent_type
                self.current_model = model_name
                
                # Notify user of success
                self.display_manager.log_agent_ready(agent_type, model_name)
                
                return response
            else:
                error_msg = response.get('message', 'Unknown initialization error') if response else 'No response received'
                raise AgentInitializationError(error_msg, agent_type=agent_type, model_name=model_name)
                
        except Exception as e:
            self.display_manager.log_error(
                error=e,
                user_friendly_message=f"Failed to initialize {agent_type} agent",
                show_to_user=True
            )
            raise e
    
    @require_feature("core_sdk")
    def send_message(self, message, streaming=False,return_full_response: bool = False, **kwargs) -> Union[str, Dict[str, Any]]:
        """Send message with clean user interface."""
        
        if not self.current_agent:
            error = ValidationError("No agent initialized. Call initialize_agent() first.")
            self.display_manager.log_error(
                error=error,
                user_friendly_message="Please initialize an agent first",
                show_to_user=True
            )
            raise error
        
        if not message or not message.strip():
            error = ValidationError("Message cannot be empty")
            self.display_manager.log_error(
                error=error,
                user_friendly_message="Please provide a message",
                show_to_user=True
            )
            raise error
        
        # Prepare message options
        import platform
        system= platform.machine().lower()
        if system != "darwin":
            options = {
                'agent': self.current_agent,
                'model': self.current_model or 'unsloth/Qwen3-1.7B-bnb-4bit'
            }
        else:
            options = {
                'agent': self.current_agent,
                'model': self.current_model or"mlx-community/Qwen3-4B-4bit"}
        options.update(kwargs)
        
        try:
            if streaming:
                # Use streaming chat service
                # response = self.chat_service.send_message(
                #     message, 
                #     options=options, 
                #     streaming=True,
                #     on_chunk=kwargs.get('on_chunk'),
                #     on_complete=kwargs.get('on_complete'),
                #     on_error=kwargs.get('on_error')
                # )
                return self._handle_streaming_message(message, options, **kwargs)
            else:
                # Use regular chat service
                # response = self.chat_service.send_message(message, options=options)
                # Regular message
                response = self.chat_service.send_message(message, options=options)
                
                # Extract clean response text
                response_text = self.chat_service._extract_response_text(response)
                
                # Log the interaction for display manager
                self.display_manager.log_chat_interaction(
                    user_message=message,
                    assistant_response=response_text,
                    agent=self.current_agent,
                    model=self.current_model
                )
            
                # Return based on user preference
                if return_full_response:
                    return {
                        'response': response_text,
                        'agent': self.current_agent,
                        'model': self.current_model,
                        'raw_response': response
                    }
                else:
                    return response_text
                
        except Exception as e:
            self.display_manager.log_error(
                error=e,
                user_friendly_message="Failed to send message. Please try again.",
                show_to_user=True
            )
            raise e

    def _handle_streaming_message(self, message: str, options: Dict[str, Any], **kwargs):
        """Handle streaming message with user-friendly interface."""
        
        collected_chunks = []
        
        def chunk_handler(chunk_text: str):
            """Handle incoming chunks."""
            collected_chunks.append(chunk_text)
            
            # Display to user in real-time
            print(chunk_text, end='', flush=True)
            
            # Call user's custom chunk handler if provided
            if 'on_chunk' in kwargs:
                kwargs['on_chunk'](chunk_text)
        
        def completion_handler():
            """Handle stream completion."""
            print()  # New line after streaming
            
            # Log the complete interaction
            complete_response = ''.join(collected_chunks)
            self.display_manager.log_chat_interaction(
                user_message=message,
                assistant_response=complete_response,
                agent=self.current_agent,
                model=self.current_model
            )
            
            if 'on_complete' in kwargs:
                kwargs['on_complete'](complete_response)
        
        def error_handler(error):
            """Handle streaming errors."""
            self.display_manager.log_error(
                error=error,
                user_friendly_message="Streaming failed",
                show_to_user=True
            )
            
            if 'on_error' in kwargs:
                kwargs['on_error'](error)
        
        # Start streaming
        self.chat_service.send_message(
            message,
            options=options,
            streaming=True,
            on_chunk=chunk_handler,
            on_complete=completion_handler,
            on_error=error_handler
        )
        
        # Return collected response
        return ''.join(collected_chunks)
        
    def cleanup(self):
        """Enhanced cleanup with comprehensive error handling."""
        self.logger.info("Starting cleanup operation")
        
        try:
            # Stop all active streams first
            if self.event_source_manager:
                self.event_source_manager.stop_all_streams()
            
            # Cleanup model service before backend cleanup
            if self.model_service:
                self.model_service.cleanup()
            
            # Cleanup inference engine
            if self.inference_engine:
                self.inference_engine.cleanup()
            
            # Cleanup memory store
            if self.memory_store:
                self.memory_store.cleanup()
            
            # Call backend cleanup
            result = self.http_client.post('/api/cleanup', data={})
            
            # Reset state
            old_agent = self.current_agent
            old_model = self.current_model
            self.current_agent = None
            self.current_model = None
            
            self.logger.info("Cleanup completed successfully", 
                        previous_agent=old_agent,
                        previous_model=old_model)
            
            return result
            
        except Exception as e:
            self.logger.error("Cleanup operation failed", error=e)
            
            # Still reset state even if cleanup fails
            self.current_agent = None
            self.current_model = None
            
            raise BlackbirdError(f"Cleanup error: {str(e)}")


    @require_feature("image_generation")
    def generate_image(self, prompt: str, **kwargs):
        """Generate image using FLUX model with automatic download."""
        flux_model = 'black-forest-labs/FLUX.1-schnell'
        
        # Download FLUX model if not cached
        if not self.model_service.downloader.is_model_cached(flux_model):
            self.logger.info("🎨 Downloading FLUX model for image generation...")
            
            def image_progress(model_id, progress, message):
                if progress >= 0:
                    print(f"🖼️  Image Model: {progress}% - {message}")
                else:
                    print(f"❌ Image Model: {message}")
            
            success = self.model_service.downloader.download_model(
                flux_model, progress_callback=image_progress
            )
            
            if not success:
                raise BlackbirdError("Failed to download image generation model")
        
        # Generate image using the model
        return self._generate_image_with_flux(prompt, **kwargs)

    def _generate_image_with_flux(self, prompt: str, **kwargs):
        """Generate image using FLUX model."""
        try:
            # Implementation for FLUX image generation
            response = self.http_client.post('/api/generate-image', json={
                'prompt': prompt,
                'model': 'black-forest-labs/FLUX.1-schnell',
                **kwargs
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise BlackbirdError(f"Image generation error: {str(e)}")


    @require_feature("basic_agents")
    def initialize_agent_with_cleanup(self, agent_type, model_name=None):
        """Initialize agent with automatic cleanup and comprehensive error handling."""
        try:
            self.logger.info("Starting cleanup and initialization sequence", 
                           target_agent=agent_type,
                           target_model=model_name)
            
            # Step 1: Cleanup
            cleanup_result = self.cleanup()
            
            # Step 2: Initialize
            init_result = self.initialize_agent(agent_type, model_name)
            
            return {
                'status': 'success',
                'cleanup': cleanup_result,
                'initialization': init_result,
                'agent': agent_type
            }
            
        except Exception as e:
            self.logger.error("Cleanup and initialization sequence failed", 
                            error=e,
                            target_agent=agent_type)
            raise BlackbirdError(f"Agent initialization with cleanup failed: {str(e)}")
    
    def get_available_agents(self):
        """Get list of available agent types based on enabled features."""
        return self.agent_manager.get_available_agents()
    
    def get_enabled_features(self):
        """Get list of enabled features."""
        return self.feature_flags.get_enabled_features()
    
    def get_feature_info(self, feature_name: str):
        """Get information about a specific feature."""
        return self.feature_flags.get_feature_info(feature_name)
    
    def get_status(self):
        """Get current SDK status and configuration."""
        return {
            'current_agent': self.current_agent,
            'current_model': self.current_model,
            'base_url': self.config.get('base_url'),
            'timeout': self.config.get('timeout'),
            'initialization_attempts': self._initialization_attempts,
            'enabled_features': self.get_enabled_features(),
            'available_agents': self.get_available_agents(),
            'streaming_enabled': is_feature_enabled("streaming_responses"),
            'active_streams': self.event_source_manager.get_active_streams() if self.event_source_manager else []
        }

    # FILE UPLOAD METHODS
    @require_feature("file_upload")
    def upload_file(self, file_path, agent_type=None, options=None):
        """Upload a single file for document processing."""
        if not self.file_service:
            raise ValidationError(
                "File upload service not available. Enable 'file_upload' feature.",
                field_name="file_service"
            )
        
        # Use current agent if not specified
        if agent_type is None:
            if not self.current_agent:
                raise ValidationError(
                    "No agent specified and no current agent initialized",
                    field_name="agent_type"
                )
            agent_type = self.current_agent
        
        self.logger.info("Uploading file", 
                        file_path=file_path,
                        agent_type=agent_type)
        
        try:
            result = self.file_service.upload_single_file(file_path, agent_type, options)
            
            self.logger.info("File upload completed successfully",
                            file_path=file_path,
                            agent_type=agent_type)
            
            return result
            
        except Exception as e:
            self.logger.error("File upload failed",
                             error=e,
                             file_path=file_path,
                             agent_type=agent_type)
            raise e

    @require_feature("file_upload")
    def upload_files(self, file_paths, agent_type=None, options=None):
        """Upload multiple files for document processing."""
        if not self.file_service:
            raise ValidationError(
                "File upload service not available. Enable 'file_upload' feature.",
                field_name="file_service"
            )
        
        # Use current agent if not specified
        if agent_type is None:
            if not self.current_agent:
                raise ValidationError(
                    "No agent specified and no current agent initialized",
                    field_name="agent_type"
                )
            agent_type = self.current_agent
        
        self.logger.info("Uploading multiple files", 
                        file_count=len(file_paths),
                        agent_type=agent_type)
        
        try:
            result = self.file_service.upload_multiple_files(file_paths, agent_type, options)
            
            self.logger.info("Multiple file upload completed successfully",
                            file_count=len(file_paths),
                            agent_type=agent_type)
            
            return result
            
        except Exception as e:
            self.logger.error("Multiple file upload failed",
                             error=e,
                             file_count=len(file_paths),
                             agent_type=agent_type)
            raise e

    def get_file_info(self, file_path):
        """Get detailed information about a file."""
        if not self.file_service:
            # Provide basic info even without file service
            from pathlib import Path
            path = Path(file_path)
            return {
                'path': str(path.absolute()),
                'name': path.name,
                'extension': path.suffix.lower(),
                'exists': path.exists(),
                'file_upload_enabled': False
            }
        
        return self.file_service.get_file_info(file_path)

    def get_supported_file_formats(self, agent_type=None):
        """Get information about supported file formats."""
        if not self.file_service:
            return {
                'file_upload_enabled': False,
                'message': 'Enable file_upload feature to see supported formats'
            }
        
        formats = self.file_service.get_supported_formats()
        
        if agent_type:
            formats['current_agent_formats'] = self.file_service.get_supported_extensions_for_agent(agent_type)
        
        return formats

    @require_feature("file_upload")
    def initialize_agent_with_files(self, agent_type, file_paths, 
                                  model_name=None, options=None):
        """Initialize an agent and upload files in a single operation."""
        try:
            self.logger.info("Starting agent initialization with file upload",
                            agent_type=agent_type,
                            file_count=len(file_paths))
            
            # Step 1: Clean up previous state
            cleanup_result = self.cleanup()
            
            # Step 2: Upload files first
            upload_result = self.upload_files(file_paths, agent_type, options)
            
            # Step 3: Initialize agent (the file upload already processes them)
            # The agent should be automatically initialized by the file upload process
            
            # Step 4: Verify agent is working
            status = self.get_status()
            
            result = {
                'status': 'success',
                'cleanup': cleanup_result,
                'file_upload': upload_result,
                'agent': agent_type,
                'files_processed': len(file_paths),
                'current_status': status
            }
            
            self.current_agent = agent_type
            system= platform.system().lower()
            if system== "darwin":
                model_name = model_name or 'mlx-community/Qwen3-4B-4bit'
            else:
                model_name = model_name or 'unsloth/Qwen3-1.7B-bnb-4bit'
            self.logger.info("Agent initialization with files completed successfully",
                            agent_type=agent_type,
                            file_count=len(file_paths))
            
            return result
            
        except Exception as e:
            self.logger.error("Agent initialization with files failed",
                             error=e,
                             agent_type=agent_type,
                             file_count=len(file_paths))
            raise BlackbirdError(f"Agent initialization with files failed: {str(e)}")

    # STREAMING METHODS
    @require_feature("streaming_responses")
    def send_streaming_message(self, message, on_chunk=None, on_complete=None, on_error=None, **kwargs):
        """Send a streaming message with callbacks."""
        if not self.current_agent:
            error = ValidationError("No agent initialized. Call initialize_agent() first.")
            self.display_manager.log_error(
                error=error,
                user_friendly_message="Please initialize an agent first",
                show_to_user=True
            )
            raise error
        
        if not message or not message.strip():
            error = ValidationError("Message cannot be empty")
            self.display_manager.log_error(
                error=error,
                user_friendly_message="Please provide a message",
                show_to_user=True
            )
            raise error
        import platform
        system = platform.system().lower()
        # Prepare message options
        if system != "darwin":
            options = {
                'agent': self.current_agent,
                'model': self.current_model or 'unsloth/Qwen3-1.7B-bnb-4bit'
            }
        else:
            options = {
                'agent': self.current_agent,
                'model': self.current_model or"mlx-community/Qwen3-4B-4bit"}
        
        options.update(kwargs)
        
        try:
            # Use streaming chat service directly
            stream_id = self.chat_service.send_streaming_message(
                message=message,
                agent=options.get('agent'),
                model=options.get('model'),
                on_chunk=on_chunk,
                on_complete=on_complete,
                on_error=on_error
            )
            
            # Log the streaming interaction
            self.display_manager.log_chat_interaction(
                user_message=message,
                assistant_response="[Streaming response]",
                agent=self.current_agent,
                model=self.current_model
            )
            
            return stream_id
                
        except Exception as e:
            self.display_manager.log_error(
                error=e,
                user_friendly_message="Failed to send streaming message. Please try again.",
                show_to_user=True
            )
            raise e
    
    @require_feature("streaming_responses")
    def get_stream_health(self):
        """Get health information for all active streams."""
        if self.event_source_manager:
            return self.event_source_manager.monitor_health()
        return {'streaming_enabled': False}
    
    @require_feature("streaming_responses")
    def stop_stream(self, stream_id: str):
        """Stop a specific stream."""
        if self.event_source_manager:
            self.event_source_manager.stop_stream(stream_id)
    
    @require_feature("streaming_responses")
    def stop_all_streams(self):
        """Stop all active streams."""
        if self.event_source_manager:
            self.event_source_manager.stop_all_streams()

    ## Session Manager Functions 
        
    def create_session(self, user_id: str, tier: str = "free", metadata: Dict[str, Any] = None):
        """Create a new user session."""
        if not self.session_manager:
            raise ValidationError("Session manager not available", field_name="session_manager")
        
        session = self.session_manager.create_session(user_id, tier, metadata)
        
        self.logger.info("User session created", 
                        session_id=session.session_id,
                        user_id=user_id,
                        tier=tier)
        
        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'status': session.status.value,
            'quotas': self.session_manager.get_quota(session.session_id)
        }

    def get_session_info(self, session_id: str):
        """Get session information."""
        if not self.session_manager:
            raise ValidationError("Session manager not available", field_name="session_manager")
        
        return self.session_manager.get_session_data(session_id)

    def terminate_session(self, session_id: str):
        """Terminate a user session."""
        if not self.session_manager:
            raise ValidationError("Session manager not available", field_name="session_manager")
        
        success = self.session_manager.terminate_session(session_id)
        
        if success:
            self.logger.info("Session terminated", session_id=session_id)
        
        return {'success': success}

    def get_quota_status(self, session_id: str, quota_type: str = None):
        """Get quota status for a session."""
        if not self.session_manager:
            raise ValidationError("Session manager not available", field_name="session_manager")
        
        if quota_type:
            try:
                qt = QuotaType(quota_type)
                return self.session_manager.get_quota(session_id, qt)
            except ValueError:
                raise ValidationError(f"Invalid quota type: {quota_type}", field_name="quota_type")
        
        return self.session_manager.get_quota(session_id)

    def get_usage_history(self, user_id: str, limit: int = 100):
        """Get usage history for a user."""
        if not self.session_manager:
            raise ValidationError("Session manager not available", field_name="session_manager")
        
        return self.session_manager.get_usage_history(user_id, limit)

    def track_operation_usage(self, session_id: str, operation: str, tokens: int = 0, 
                            requests: int = 1, metadata: Dict[str, Any] = None):
        """Track usage for an operation."""
        if not self.session_manager:
            return  # Silently skip if session manager not available
        
        try:
            # Track token usage
            if tokens > 0:
                self.session_manager.track_usage(
                    session_id, operation, QuotaType.TOKENS, tokens, metadata
                )
            
            # Track request usage
            if requests > 0:
                self.session_manager.track_usage(
                    session_id, operation, QuotaType.REQUESTS, requests, metadata
                )
            
            # Enforce rate limiting
            self.session_manager.enforce_rate_limit(session_id, operation)
            
        except Exception as e:
            self.logger.warning("Usage tracking failed", 
                            session_id=session_id,
                            operation=operation,
                            error=str(e))

    def get_session_statistics(self):
        """Get session manager statistics."""
        if not self.session_manager:
            return {'error': 'Session manager not available'}
        
        return self.session_manager.get_statistics()
    
    ## Memory Store functions         

    def cache_set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Store data in cache."""
        if not self.memory_store:
            self.logger.warning("Memory store not available for caching")
            return False
        
        return self.memory_store.set_cache(key, value, ttl)

    def cache_get(self, key: str):
        """Retrieve data from cache."""
        if not self.memory_store:
            return None
        
        return self.memory_store.get_cache(key)

    def cache_delete(self, key: str):
        """Delete data from cache."""
        if not self.memory_store:
            return False
        
        return self.memory_store.delete_cache(key)

    def cache_clear(self):
        """Clear all cached data."""
        if not self.memory_store:
            return 0
        
        return self.memory_store.clear_cache()

    def embed_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Embed and index a document for vector search."""
        if not self.memory_store:
            raise ValidationError("Memory store not available", field_name="memory_store")
        
        document = self.memory_store.embed_document(doc_id, content, metadata)
        
        self.logger.info("Document embedded and indexed", 
                        doc_id=doc_id,
                        content_length=len(content),
                        embedding_dimension=document.embedding_dimension)
        
        return {
            'doc_id': document.doc_id,
            'embedding_dimension': document.embedding_dimension,
            'created_at': document.created_at,
            'metadata': document.metadata
        }

    def vector_search(self, query: str, limit: int = 10, threshold: float = 0.0):
        """Perform vector similarity search."""
        if not self.memory_store:
            raise ValidationError("Memory store not available", field_name="memory_store")
        
        results = self.memory_store.vector_search(query, limit, threshold)
        
        self.logger.info("Vector search completed", 
                        query_length=len(query),
                        results_count=len(results))
        
        return [result.to_dict() for result in results]

    def get_cache_stats(self):
        """Get cache statistics."""
        if not self.memory_store:
            return {'error': 'Memory store not available'}
        
        return self.memory_store.get_cache_stats()

    def get_vector_stats(self):
        """Get vector index statistics."""
        if not self.memory_store:
            return {'error': 'Memory store not available'}
        
        return self.memory_store.get_vector_stats()

    def optimize_memory_index(self):
        """Optimize the vector index for better performance."""
        if not self.memory_store:
            return False
        
        return self.memory_store.optimize_index()


    ## INFERENCE ENGINE
    # Add these methods to the BlackbirdSDK class:

    def get_platform_info(self):
        """Get detailed platform and hardware information."""
        if not self.inference_engine:
            return {'error': 'Inference engine not available'}
        
        return self.inference_engine.detect_platform()

    def optimize_for_hardware(self):
        """Optimize SDK for current hardware."""
        if not self.inference_engine:
            raise ValidationError("Inference engine not available", field_name="inference_engine")
        
        optimization_results = self.inference_engine.optimize_hardware()
        
        self.logger.info("Hardware optimization completed", 
                        optimizations_count=len(optimization_results.get('optimizations_applied', [])))
        
        return optimization_results

    def set_inference_mode(self, mode: str):
        """Set inference optimization mode."""
        if not self.inference_engine:
            raise ValidationError("Inference engine not available", field_name="inference_engine")
        
        try:
            opt_level = OptimizationLevel(mode)
            success = self.inference_engine.set_inference_mode(opt_level)
            
            if success:
                self.logger.info("Inference mode updated", mode=mode)
            
            return {'success': success, 'mode': mode}
            
        except ValueError:
            raise ValidationError(f"Invalid optimization mode: {mode}", field_name="mode")

    def enable_cuda_acceleration(self):
        """Enable CUDA acceleration if available."""
        if not self.inference_engine:
            return {'success': False, 'error': 'Inference engine not available'}
        
        success = self.inference_engine.cuda_accelerate()
        
        return {
            'success': success,
            'device': 'cuda' if success else 'cpu',
            'message': 'CUDA acceleration enabled' if success else 'CUDA not available'
        }

    def enable_mps_acceleration(self):
        """Enable Apple Metal Performance Shaders acceleration."""
        if not self.inference_engine:
            return {'success': False, 'error': 'Inference engine not available'}
        
        success = self.inference_engine.mls_optimize()
        
        return {
            'success': success,
            'device': 'mps' if success else 'cpu',
            'message': 'MPS acceleration enabled' if success else 'MPS not available'
        }

    def get_inference_stats(self):
        """Get inference performance statistics."""
        if not self.inference_engine:
            return {'error': 'Inference engine not available'}
        
        return self.inference_engine.get_inference_stats()

    def get_optimization_recommendations(self):
        """Get hardware optimization recommendations."""
        if not self.inference_engine:
            return {'error': 'Inference engine not available'}
        
        return self.inference_engine.get_optimization_recommendations()

    def batch_process_requests(self, requests: List[Dict[str, Any]], callback=None):
        """Process multiple requests in optimized batches."""
        if not self.inference_engine:
            raise ValidationError("Inference engine not available", field_name="inference_engine")
        
        results = self.inference_engine.batch_process(requests, callback)
        
        self.logger.info("Batch processing completed", 
                        requests_count=len(requests),
                        results_count=len(results))
        
        return results

    ## MODEL SERVICE FUNCTIONS
    # Add these methods to your existing BlackbirdSDK class:

    def register_model(self, name: str, model_type: str, source: str, **kwargs):
        """Register a new model in the service."""
        if not self.model_service:
            raise ValidationError("Model service not available", field_name="model_service")
        
        # Convert string enums to proper enum types
        try:
            model_type_enum = ModelType(model_type)
            source_enum = ModelSource(source)
            
            # Handle format conversion if provided
            if 'format' in kwargs:
                kwargs['format'] = ModelFormat(kwargs['format'])
            
            model_id = self.model_service.register_model(
                name=name,
                model_type=model_type_enum,
                source=source_enum,
                **kwargs
            )
            
            self.logger.info("Model registered via SDK", 
                            model_id=model_id,
                            name=name)
            
            return model_id
            
        except ValueError as e:
            raise ValidationError(f"Invalid enum value: {str(e)}", field_name="enum_values")

    def get_model_info(self, model_id: str):
        """Get detailed information about a model."""
        if not self.model_service:
            return {'error': 'Model service not available'}
        
        return self.model_service.get_model_info(model_id)

    def list_models(self, model_type: str = None, status: str = None):
        """List all models with optional filtering."""
        if not self.model_service:
            return {'error': 'Model service not available'}
        
        # Convert string filters to enums if provided
        type_filter = ModelType(model_type) if model_type else None
        status_filter = None  # Add status enum conversion if needed
        
        return self.model_service.list_models(type_filter, status_filter)

    def get_model_stats(self):
        """Get model service statistics."""
        if not self.model_service:
            return {'error': 'Model service not available'}
        
        return self.model_service.get_stats()

    def delete_model(self, model_id: str, delete_files: bool = False):
        """Delete a model from the service."""
        if not self.model_service:
            return {'success': False, 'error': 'Model service not available'}
        
        success = self.model_service.delete_model(model_id, delete_files)
        return {'success': success}

    def update_model_config(self, model_id: str, updates: Dict[str, Any]):
        """Update model configuration."""
        if not self.model_service:
            return {'success': False, 'error': 'Model service not available'}
        
        success = self.model_service.update_query(model_id, updates)
        return {'success': success}

    def cleanup_model_cache(self, max_age_days: int = 30):
        """Clean up old model cache."""
        if not self.model_service:
            return {'cleaned': 0, 'error': 'Model service not available'}
        
        cleaned_count = self.model_service.cleanup_cache(max_age_days)
        return {'cleaned': cleaned_count}


    def _initialize_optional_components(self):
        """Initialize components based on enabled features."""
        
        # Initialize file service if file upload is enabled
        if is_feature_enabled("file_upload"):
            self.file_service = FileService(self.http_client, self.logger)
            self.logger.info("File upload service initialized")
        else:
            self.file_service = None
        
        # Initialize session manager if enabled
        if is_feature_enabled("core_sdk"):
            self.session_manager = SessionManager(self.config.get('session_config', {}))
            self.logger.info("Session manager initialized")
        else:
            self.session_manager = None
        
        # Initialize memory store if enabled
        if is_feature_enabled("core_sdk"):
            memory_config = MemoryStoreConfig(**self.config.get('memory_config', {}))
            self.memory_store = MemoryStore(memory_config)
            self.logger.info("Memory store initialized")
        else:
            self.memory_store = None
        
        # Initialize inference engine if enabled
        if is_feature_enabled("core_sdk"):
            inference_config_dict = self.config.get('inference_config', {})
            
            # Handle string values from config for enum conversion
            if 'framework' in inference_config_dict and isinstance(inference_config_dict['framework'], str):
                try:
                    from .oss_acceleration.platform_types import InferenceFramework
                    inference_config_dict['framework'] = InferenceFramework(inference_config_dict['framework'])
                except (ValueError, ImportError):
                    self.logger.warning(f"Invalid framework: {inference_config_dict['framework']}, using default")
                    inference_config_dict.pop('framework', None)
            
            if 'optimization_level' in inference_config_dict and isinstance(inference_config_dict['optimization_level'], str):
                try:
                    from .oss_acceleration.platform_types import OptimizationLevel
                    inference_config_dict['optimization_level'] = OptimizationLevel(inference_config_dict['optimization_level'])
                except (ValueError, ImportError):
                    self.logger.warning(f"Invalid optimization_level: {inference_config_dict['optimization_level']}, using default")
                    inference_config_dict.pop('optimization_level', None)
            
            inference_config = InferenceConfig(**inference_config_dict)
            self.inference_engine = InferenceEngine(inference_config)
            self.logger.info("Inference engine initialized")
        else:
            self.inference_engine = None
        
        # CRITICAL ADDITION: Initialize model service if enabled (MISSING FROM YOUR VERSION)
        if is_feature_enabled("core_sdk"):
            model_config = self.config.get('model_config', {})
            self.model_service = ModelService(model_config)
            self.logger.info("Model service initialized")
        else:
            self.model_service = None
        
        # Future: Initialize MAS router if enabled
        if is_feature_enabled("mas_router"):
            self.logger.info("MAS router feature enabled - will be initialized when implemented")
        
        # Future: Initialize knowledge graph integration if enabled  
        if is_feature_enabled("knowledge_graphs"):
            self.logger.info("Knowledge graphs feature enabled - will be initialized when implemented")
        
        # Future: Initialize function calling if enabled
        if is_feature_enabled("function_calling"):
            self.logger.info("Function calling feature enabled - will be initialized when implemented")

    def set_development_mode(self, enabled: bool):
        """Toggle development mode to show/hide internal logs."""
        self.display_manager.set_development_mode(enabled)
        
        if enabled:
            print("🔧 Development mode enabled - showing all internal logs")
        else:
            print("👤 User mode enabled - showing clean interface")

    # def fine_tune_model(self, *args, **kwargs):
        # if self.license_manager and not self.license_manager.is_feature_enabled('fine_tuning'):
        #     raise LicensingError("Fine-tuning not enabled in your license")
        # # Continue with fine-tuning
        # pass

    # def get_license_info(self):
    #     if self.license_manager:
    #         return self.license_manager.get_license_info()
    #     return {
    #         'license_type': 'development',
    #         'customer_email': 'dev@blackbird.ai',
    #         'expires_at': '2099-12-31T23:59:59',
    #         'features': ['all'],
    #         'device_limit': 999
    #     }
    def search_web(self, query, max_results=5):
        """Search the web using the configured backend and return results."""
        return self.web_search_backend.search(query, max_results)

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return self.feature_flags.is_enabled(feature_name)

    def get_license_status(self) -> Dict[str, Any]:
        """Get current license status and information"""
        # if not self.license_manager: # This block is removed as per the edit hint
        #     return {
        #         'status': 'no_licensing',
        #         'message': 'Licensing system not available'
        #     }
        
        # try:
        #     license_info = self.license_manager.get_license_info()
        #     user_info = self.license_manager.get_user_info()
        #     is_valid = self.license_manager.validate_license()
            
        #     return {
        #         'status': 'valid' if is_valid else 'invalid',
        #         'user_id': user_info.get('user_id'),
        #         'tier': user_info.get('tier'),
        #         'features': license_info.get('features', []),
        #         'expires_at': license_info.get('expires_at'),
        #         'device_id': license_info.get('device_id'),
        #         'license_server': user_info.get('license_server')
        #     }
        # except Exception as e:
        #     return {
        #         'status': 'error',
        #         'message': str(e)
        #     }
        return {'status': 'no_licensing', 'message': 'Licensing system not available'}
    
    def check_feature_availability(self, feature: str) -> bool:
        """Check if a specific feature is available with current license"""
        # if not self.license_manager: # This block is removed as per the edit hint
        #     return True  # Allow all features if no licensing
        
        # return self.license_manager.check_feature(feature)
        return True # Allow all features if no licensing
    
    def get_available_features(self) -> List[str]:
        """Get list of available features with current license"""
        # if not self.license_manager: # This block is removed as per the edit hint
        #     return ['all']  # All features if no licensing
        
        # return self.license_manager.get_available_features()
        return ['all'] # All features if no licensing
    
    def refresh_license(self) -> bool:
        """Refresh the current license"""
        # if not self.license_manager: # This block is removed as per the edit hint
        #     return False
        
        # try:
        #     return self.license_manager.refresh_license()
        # except Exception as e:
        #     self.display_manager.user_logger.error(f"License refresh failed: {e}")
        #     return False
        return False
    
    def upgrade_license_tier(self, new_tier: str) -> bool:
        """Request license tier upgrade"""
        # if not self.license_manager: # This block is removed as per the edit hint
        #     return False
        
        # try:
        #     success = self.license_manager.upgrade_tier(new_tier)
        #     if success:
        #         self.display_manager.user_logger.info(f"License upgraded to {new_tier} tier")
        #     else:
        #         self.display_manager.user_logger.warning("License upgrade failed")
        #     return success
        # except Exception as e:
        #     self.display_manager.user_logger.error(f"License upgrade error: {e}")
        #     return False
        return False
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get current user information"""
        # if not self.user_config: # This block is removed as per the edit hint
        #     return {'user_id': 'unknown', 'tier': 'development'}
        
        # return self.user_config.get_config_summary()
        return {'user_id': 'unknown', 'tier': 'development'}

# Export enhanced classes and functions
__all__ = [
    'BlackbirdSDK', 
    'configure_logging',
    'configure_features',
    'is_feature_enabled',
    'require_feature',
    'BlackbirdError', 
    'AuthenticationError', 
    'APIError', 
    'NetworkError',
    'ValidationError',
    'AgentInitializationError',
    'ModelLoadError',
    'StreamingResponseError',
    'MemoryError',
    'FileProcessingError',
    'TimeoutError',
    'EventSourceManager',
    'EventType',
    'StreamStatus',
    'StreamConfig',
    'QuotaExceededError',
    'RateLimitError'
]