"""Function registry for managing callable functions."""

import threading
from typing import Dict, List, Optional, Callable, Any
from .function_types import FunctionDefinition, FunctionType, BUILTIN_FUNCTIONS
from blackbird_sdk.utils.errors import ValidationError
from blackbird_sdk.utils.logger import get_logger

class FunctionRegistry:
    """Registry for managing callable functions."""
    
    def __init__(self):
        """Initialize the function registry."""
        self.logger = get_logger()
        self._functions: Dict[str, FunctionDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        
        # Register built-in functions
        self._register_builtin_functions()
        
        self.logger.info(f"Function registry initialized with {len(self._functions)} built-in functions")
    
    def _register_builtin_functions(self):
        """Register built-in function definitions."""
        for name, definition in BUILTIN_FUNCTIONS.items():
            self._functions[name] = definition
            self.logger.debug(f"Registered built-in function: {name}")
    
    def register_function(self, definition: FunctionDefinition, handler: Callable) -> bool:
        """
        Register a new function with its handler.
        
        Args:
            definition: Function definition
            handler: Callable that implements the function
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            if definition.name in self._functions:
                self.logger.warning(f"Function {definition.name} already registered, overwriting")
            
            self._functions[definition.name] = definition
            self._handlers[definition.name] = handler
            
            self.logger.info(f"Registered function: {definition.name}")
            return True
    
    def unregister_function(self, function_name: str) -> bool:
        """
        Unregister a function.
        
        Args:
            function_name: Name of function to unregister
            
        Returns:
            True if unregistration successful, False if function not found
        """
        with self._lock:
            if function_name not in self._functions:
                return False
            
            # Don't allow unregistering built-in functions
            if function_name in BUILTIN_FUNCTIONS:
                self.logger.warning(f"Cannot unregister built-in function: {function_name}")
                return False
            
            del self._functions[function_name]
            del self._handlers[function_name]
            
            self.logger.info(f"Unregistered function: {function_name}")
            return True
    
    def get_function(self, function_name: str) -> Optional[FunctionDefinition]:
        """Get function definition by name."""
        with self._lock:
            return self._functions.get(function_name)
    
    def get_handler(self, function_name: str) -> Optional[Callable]:
        """Get function handler by name."""
        with self._lock:
            return self._handlers.get(function_name)
    
    def list_functions(self, agent_type: str = None, function_type: FunctionType = None) -> List[FunctionDefinition]:
        """
        List available functions with optional filtering.
        
        Args:
            agent_type: Filter by agent type
            function_type: Filter by function type
            
        Returns:
            List of function definitions
        """
        with self._lock:
            functions = list(self._functions.values())
            
            # Filter by agent type
            if agent_type:
                functions = [
                    f for f in functions 
                    if not f.agent_restrictions or agent_type in f.agent_restrictions
                ]
            
            # Filter by function type
            if function_type:
                functions = [f for f in functions if f.function_type == function_type]
            
            return functions
    
    def get_function_names(self, agent_type: str = None) -> List[str]:
        """Get list of function names available to an agent."""
        functions = self.list_functions(agent_type=agent_type)
        return [f.name for f in functions]
    
    def validate_function_call(self, function_name: str, parameters: Dict[str, Any], agent_type: str = None) -> tuple[bool, List[str]]:
        """
        Validate a function call.
        
        Args:
            function_name: Name of function to call
            parameters: Function parameters
            agent_type: Type of agent making the call
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        with self._lock:
            # Check if function exists
            if function_name not in self._functions:
                return False, [f"Function '{function_name}' not found"]
            
            function_def = self._functions[function_name]
            
            # Check agent restrictions
            if agent_type and function_def.agent_restrictions:
                if agent_type not in function_def.agent_restrictions:
                    return False, [f"Agent '{agent_type}' not authorized to call function '{function_name}'"]
            
            # Validate parameters
            return function_def.validate_parameters(parameters)
    
    def get_openai_functions(self, agent_type: str = None) -> List[Dict[str, Any]]:
        """
        Get function definitions in OpenAI function calling format.
        
        Args:
            agent_type: Filter functions for specific agent type
            
        Returns:
            List of function definitions in OpenAI format
        """
        functions = self.list_functions(agent_type=agent_type)
        return [f.to_openai_format() for f in functions]
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            function_types = {}
            agent_functions = {}
            
            for func in self._functions.values():
                # Count by type
                func_type = func.function_type.value
                function_types[func_type] = function_types.get(func_type, 0) + 1
                
                # Count by agent restrictions
                if func.agent_restrictions:
                    for agent in func.agent_restrictions:
                        agent_functions[agent] = agent_functions.get(agent, 0) + 1
                else:
                    agent_functions['all'] = agent_functions.get('all', 0) + 1
            
            return {
                'total_functions': len(self._functions),
                'builtin_functions': len(BUILTIN_FUNCTIONS),
                'custom_functions': len(self._functions) - len(BUILTIN_FUNCTIONS),
                'function_types': function_types,
                'agent_functions': agent_functions
            }

# Global registry instance
_function_registry = None

def get_function_registry() -> FunctionRegistry:
    """Get the global function registry instance."""
    global _function_registry
    if _function_registry is None:
        _function_registry = FunctionRegistry()
    return _function_registry
