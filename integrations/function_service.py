"""Function calling service for the Decompute SDK."""

from typing import Dict, List, Any, Optional, Callable
import json
import re
from .function_registry import get_function_registry
from .function_executor import FunctionExecutor
from .function_types import FunctionCall, FunctionResult, FunctionType
from blackbird_sdk.utils.errors import ValidationError, ExecutionError
from blackbird_sdk.utils.feature_flags import require_feature, is_feature_enabled
from blackbird_sdk.utils.logger import get_logger

class FunctionService:
    """Service for managing and executing function calls."""
    
    def __init__(self):
        """Initialize the function service."""
        self.logger = get_logger()
        self.registry = get_function_registry()
        self.executor = FunctionExecutor()
        
        # Initialize plugins
        self._initialize_plugins()
        
        self.logger.info("Function service initialized")
    
    def _initialize_plugins(self):
        """Initialize built-in plugins."""
        try:
            # Force import of plugins package to trigger registration
            import blackbird_sdk.integrations
            
            # Also try direct imports as fallback
            try:
                from .calculator import register_calculator_functions
                register_calculator_functions()
            except ImportError as e:
                self.logger.warning(f"Calculator plugin not loaded: {e}")
            
            try:
                from .calendar import register_calendar_functions
                register_calendar_functions()
            except ImportError as e:
                self.logger.warning(f"Calendar plugin not loaded: {e}")
            
            self.logger.info("Built-in plugins registered successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing plugins: {e}")

    @require_feature("function_calling")
    def call_function(
        self, 
        function_name: str, 
        parameters: Dict[str, Any], 
        agent_type: str = None,
        user_id: str = None
    ) -> FunctionResult:
        """
        Execute a function call.
        
        Args:
            function_name: Name of function to call
            parameters: Function parameters
            agent_type: Type of agent making the call
            user_id: User ID for tracking
            
        Returns:
            Function execution result
        """
        try:
            # Create function call
            call = FunctionCall(
                call_id="",  # Will be auto-generated
                function_name=function_name,
                parameters=parameters,
                agent_type=agent_type or "general",
                user_id=user_id
            )
            
            self.logger.info(f"Executing function call: {function_name}", 
                           agent_type=agent_type, 
                           parameters=list(parameters.keys()))
            
            # Execute the function
            result = self.executor.execute_function(call)
            
            if result.status.value == "completed":
                self.logger.info(f"Function {function_name} completed successfully")
            else:
                self.logger.warning(f"Function {function_name} failed: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calling function {function_name}: {str(e)}")
            raise ExecutionError(f"Function call failed: {str(e)}")
    
    @require_feature("function_calling")
    def parse_and_execute_function_calls(self, text: str, agent_type: str = None) -> List[FunctionResult]:
        """
        Parse function calls from text and execute them.
        
        Args:
            text: Text containing function calls
            agent_type: Type of agent making the calls
            
        Returns:
            List of function execution results
        """
        try:
            # Extract function calls from text
            function_calls = self._extract_function_calls(text)
            
            if not function_calls:
                return []
            
            results = []
            for call_data in function_calls:
                try:
                    result = self.call_function(
                        function_name=call_data['function'],
                        parameters=call_data['parameters'],
                        agent_type=agent_type
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error executing parsed function call: {str(e)}")
                    # Continue with other function calls
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error parsing and executing function calls: {str(e)}")
            raise ExecutionError(f"Function call parsing failed: {str(e)}")
    
    def _extract_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract function calls from text using pattern matching.
        
        Args:
            text: Text to parse
            
        Returns:
            List of extracted function call data
        """
        function_calls = []
        
        # Pattern for function calls: function_name(param1=value1, param2=value2)
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            function_name = match.group(1)
            params_str = match.group(2)
            
            # Check if this is a registered function
            if not self.registry.get_function(function_name):
                continue
            
            # Parse parameters
            parameters = self._parse_parameters(params_str)
            
            function_calls.append({
                'function': function_name,
                'parameters': parameters,
                'raw_match': match.group(0)
            })
        
        return function_calls
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """
        Parse function parameters from string.
        
        Args:
            params_str: Parameter string
            
        Returns:
            Dictionary of parsed parameters
        """
        parameters = {}
        
        if not params_str.strip():
            return parameters
        
        # Simple parameter parsing (can be enhanced)
        param_pairs = params_str.split(',')
        
        for pair in param_pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # Try to convert to appropriate type
                try:
                    # Try integer
                    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        parameters[key] = int(value)
                    # Try float
                    elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
                        parameters[key] = float(value)
                    # Try boolean
                    elif value.lower() in ['true', 'false']:
                        parameters[key] = value.lower() == 'true'
                    # Keep as string
                    else:
                        parameters[key] = value
                except:
                    # If conversion fails, keep as string
                    parameters[key] = value
        
        return parameters
    
    @require_feature("function_calling")
    def get_available_functions(self, agent_type: str = None) -> List[Dict[str, Any]]:
        """
        Get list of available functions for an agent.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            List of function definitions
        """
        try:
            functions = self.registry.list_functions(agent_type=agent_type)
            return [
                {
                    'name': func.name,
                    'description': func.description,
                    'type': func.function_type.value,
                    'parameters': [
                        {
                            'name': param.name,
                            'type': param.type.value,
                            'description': param.description,
                            'required': param.required,
                            'default': param.default
                        }
                        for param in func.parameters
                    ],
                    'examples': func.examples
                }
                for func in functions
            ]
        except Exception as e:
            self.logger.error(f"Error getting available functions: {str(e)}")
            return []
    
    @require_feature("function_calling")
    def get_openai_functions(self, agent_type: str = None) -> List[Dict[str, Any]]:
        """
        Get function definitions in OpenAI format for model integration.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            List of function definitions in OpenAI format
        """
        try:
            return self.registry.get_openai_functions(agent_type=agent_type)
        except Exception as e:
            self.logger.error(f"Error getting OpenAI functions: {str(e)}")
            return []
    
    def register_custom_function(self, definition: Dict[str, Any], handler: Callable) -> bool:
        """
        Register a custom function.
        
        Args:
            definition: Function definition dictionary
            handler: Function handler callable
            
        Returns:
            True if registration successful
        """
        try:
            # Convert dictionary to FunctionDefinition
            from .function_types import FunctionDefinition, FunctionParameter, FunctionType, ParameterType
            
            parameters = []
            for param_data in definition.get('parameters', []):
                param = FunctionParameter(
                    name=param_data['name'],
                    type=ParameterType(param_data['type']),
                    description=param_data['description'],
                    required=param_data.get('required', True),
                    default=param_data.get('default')
                )
                parameters.append(param)
            
            func_def = FunctionDefinition(
                name=definition['name'],
                description=definition['description'],
                function_type=FunctionType(definition.get('type', 'custom')),
                parameters=parameters,
                timeout=definition.get('timeout', 30),
                agent_restrictions=definition.get('agent_restrictions', [])
            )
            
            return self.registry.register_function(func_def, handler)
            
        except Exception as e:
            self.logger.error(f"Error registering custom function: {str(e)}")
            return False
    
    def get_function_stats(self) -> Dict[str, Any]:
        """Get function calling statistics."""
        try:
            registry_stats = self.registry.get_registry_stats()
            executor_stats = self.executor.get_executor_stats()
            
            return {
                'registry': registry_stats,
                'executor': executor_stats,
                'feature_enabled': is_feature_enabled("function_calling")
            }
        except Exception as e:
            self.logger.error(f"Error getting function stats: {str(e)}")
            return {}
    
    def cleanup(self):
        """Clean up function service resources."""
        self.logger.info("Cleaning up function service")
        
        if self.executor:
            self.executor.cleanup()
        
        self.logger.info("Function service cleanup complete")
