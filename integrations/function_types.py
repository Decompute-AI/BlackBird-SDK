"""Function calling types and configurations for the Decompute SDK."""

import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime

class FunctionType(Enum):
    """Types of callable functions."""
    CALCULATOR = "calculator"
    CALENDAR = "calendar"
    WEB_SEARCH = "web_search"
    FILE_OPERATIONS = "file_operations"
    API_CALL = "api_call"
    SYSTEM_COMMAND = "system_command"
    DATABASE_QUERY = "database_query"
    CUSTOM = "custom"

class ParameterType(Enum):
    """Parameter types for function definitions."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATE = "date"
    FILE = "file"

class ExecutionStatus(Enum):
    """Status of function execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class FunctionParameter:
    """Definition of a function parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    
    def validate(self, value: Any) -> bool:
        """Validate parameter value against constraints."""
        if value is None and self.required:
            return False
        
        if value is None and not self.required:
            return True
        
        # Type validation
        if self.type == ParameterType.STRING and not isinstance(value, str):
            return False
        elif self.type == ParameterType.INTEGER and not isinstance(value, int):
            return False
        elif self.type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.type == ParameterType.ARRAY and not isinstance(value, list):
            return False
        elif self.type == ParameterType.OBJECT and not isinstance(value, dict):
            return False
        
        # Enum validation
        if self.enum_values and value not in self.enum_values:
            return False
        
        # Range validation
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        
        # Pattern validation for strings
        if self.pattern and self.type == ParameterType.STRING:
            import re
            if not re.match(self.pattern, value):
                return False
        
        return True

@dataclass
class FunctionDefinition:
    """Definition of a callable function."""
    name: str
    description: str
    function_type: FunctionType
    parameters: List[FunctionParameter] = field(default_factory=list)
    returns: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    timeout: int = 30
    requires_auth: bool = False
    agent_restrictions: List[str] = field(default_factory=list)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop_def = {
                "type": param.type.value,
                "description": param.description
            }
            
            if param.enum_values:
                prop_def["enum"] = param.enum_values
            if param.min_value is not None:
                prop_def["minimum"] = param.min_value
            if param.max_value is not None:
                prop_def["maximum"] = param.max_value
            if param.pattern:
                prop_def["pattern"] = param.pattern
            
            properties[param.name] = prop_def
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate function call parameters."""
        errors = []
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Missing required parameter: {param.name}")
            elif param.name in params and not param.validate(params[param.name]):
                errors.append(f"Invalid value for parameter {param.name}: {params[param.name]}")
        
        # Check for unexpected parameters
        expected_params = {p.name for p in self.parameters}
        for param_name in params:
            if param_name not in expected_params:
                errors.append(f"Unexpected parameter: {param_name}")
        
        return len(errors) == 0, errors

@dataclass
class FunctionCall:
    """Represents a function call request."""
    call_id: str
    function_name: str
    parameters: Dict[str, Any]
    agent_type: str
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.call_id:
            self.call_id = str(uuid.uuid4())

@dataclass
class FunctionResult:
    """Result of a function execution."""
    call_id: str
    function_name: str
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'call_id': self.call_id,
            'function_name': self.function_name,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'duration': self.duration,
            'metadata': self.metadata
        }

# Predefined function definitions for common tools
BUILTIN_FUNCTIONS = {
    "calculator": FunctionDefinition(
        name="calculator",
        description="Perform mathematical calculations including basic arithmetic, trigonometry, and advanced functions",
        function_type=FunctionType.CALCULATOR,
        parameters=[
            FunctionParameter(
                name="expression",
                type=ParameterType.STRING,
                description="Mathematical expression to evaluate (e.g., '2 + 2', 'sin(pi/2)', 'sqrt(16)')",
                required=True
            ),
            FunctionParameter(
                name="precision",
                type=ParameterType.INTEGER,
                description="Number of decimal places for the result",
                required=False,
                default=10,
                min_value=0,
                max_value=15
            )
        ],
        examples=[
            {"expression": "2 + 2", "result": 4},
            {"expression": "sqrt(16)", "result": 4.0},
            {"expression": "sin(pi/2)", "result": 1.0}
        ],
        agent_restrictions=["finance", "tech", "general"]
    ),
    
    "get_current_time": FunctionDefinition(
        name="get_current_time",
        description="Get current date and time in various formats",
        function_type=FunctionType.CALENDAR,
        parameters=[
            FunctionParameter(
                name="format",
                type=ParameterType.STRING,
                description="Time format string",
                required=False,
                default="%Y-%m-%d %H:%M:%S",
                enum_values=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%H:%M:%S", "iso", "timestamp"]
            ),
            FunctionParameter(
                name="timezone",
                type=ParameterType.STRING,
                description="Timezone (e.g., 'UTC', 'US/Eastern')",
                required=False,
                default="UTC"
            )
        ],
        examples=[
            {"format": "%Y-%m-%d", "result": "2024-01-15"},
            {"format": "iso", "result": "2024-01-15T14:30:00Z"}
        ]
    ),
    
    "web_search": FunctionDefinition(
        name="web_search",
        description="Search the web for information and return relevant results",
        function_type=FunctionType.WEB_SEARCH,
        parameters=[
            FunctionParameter(
                name="query",
                type=ParameterType.STRING,
                description="Search query string",
                required=True
            ),
            FunctionParameter(
                name="num_results",
                type=ParameterType.INTEGER,
                description="Number of search results to return",
                required=False,
                default=5,
                min_value=1,
                max_value=20
            ),
            FunctionParameter(
                name="safe_search",
                type=ParameterType.BOOLEAN,
                description="Enable safe search filtering",
                required=False,
                default=True
            )
        ],
        timeout=15,
        agent_restrictions=["research", "general", "tech"]
    ),
    
    "read_file": FunctionDefinition(
        name="read_file",
        description="Read contents of a text file",
        function_type=FunctionType.FILE_OPERATIONS,
        parameters=[
            FunctionParameter(
                name="file_path",
                type=ParameterType.STRING,
                description="Path to the file to read",
                required=True
            ),
            FunctionParameter(
                name="encoding",
                type=ParameterType.STRING,
                description="File encoding",
                required=False,
                default="utf-8",
                enum_values=["utf-8", "ascii", "latin-1"]
            )
        ],
        agent_restrictions=["tech", "research", "general"]
    )
}
