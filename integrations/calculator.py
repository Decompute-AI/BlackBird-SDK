"""Calculator plugin for mathematical operations."""

import math
import re
from typing import Any, Union, Dict
from .function_registry import get_function_registry
from .function_types import FunctionDefinition, FunctionType, FunctionParameter, ParameterType

def safe_eval(expression: str) -> Union[int, float]:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Numerical result
        
    Raises:
        ValueError: If expression is invalid or unsafe
    """
    # Allowed names for mathematical operations
    allowed_names = {
        # Basic math functions
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'pow': pow,
        
        # Math module functions
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'ceil': math.ceil, 'floor': math.floor,
        'factorial': math.factorial, 'degrees': math.degrees,
        'radians': math.radians,
        
        # Constants
        'pi': math.pi, 'e': math.e, 'tau': math.tau,
        
        # Additional functions
        'gcd': math.gcd, 'lcm': getattr(math, 'lcm', lambda a, b: abs(a * b) // math.gcd(a, b))
    }
    
    # Clean the expression
    expression = expression.strip()
    
    # Basic security check - no double underscores or imports
    if '__' in expression or 'import' in expression or 'exec' in expression or 'eval' in expression:
        raise ValueError("Invalid expression: contains prohibited operations")
    
    # Replace common mathematical notation
    expression = expression.replace('^', '**')  # Power operator
    expression = expression.replace('×', '*')   # Multiplication
    expression = expression.replace('÷', '/')   # Division
    
    try:
        # Compile and evaluate with restricted namespace
        code = compile(expression, '<string>', 'eval')
        
        # Check for prohibited operations
        for name in code.co_names:
            if name not in allowed_names:
                raise ValueError(f"Invalid operation: {name}")
        
        result = eval(code, {"__builtins__": {}}, allowed_names)
        
        # Ensure result is a number
        if not isinstance(result, (int, float, complex)):
            raise ValueError("Expression must evaluate to a number")
        
        # Convert complex to float if imaginary part is zero
        if isinstance(result, complex) and result.imag == 0:
            result = result.real
        
        return result
        
    except Exception as e:
        raise ValueError(f"Invalid mathematical expression: {str(e)}")

def calculator(expression: str, precision: int = 10) -> Dict[str, Any]:
    """
    Perform mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate
        precision: Number of decimal places for the result
        
    Returns:
        Dictionary containing the result and metadata
    """
    try:
        # Evaluate the expression
        raw_result = safe_eval(expression)
        
        # Format result based on precision
        if isinstance(raw_result, float):
            if raw_result.is_integer():
                formatted_result = int(raw_result)
            else:
                formatted_result = round(raw_result, precision)
        else:
            formatted_result = raw_result
        
        return {
            'result': formatted_result,
            'expression': expression,
            'type': type(formatted_result).__name__,
            'precision': precision,
            'raw_result': raw_result
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'expression': expression,
            'result': None
        }

def percentage_calculation(value: float, percentage: float) -> Dict[str, Any]:
    """Calculate percentage of a value."""
    try:
        result = (value * percentage) / 100
        return {
            'result': result,
            'calculation': f"{percentage}% of {value}",
            'formula': f"({value} × {percentage}) ÷ 100"
        }
    except Exception as e:
        return {'error': str(e), 'result': None}

def compound_interest(principal: float, rate: float, time: float, compound_frequency: int = 1) -> Dict[str, Any]:
    """Calculate compound interest."""
    try:
        amount = principal * (1 + rate/100/compound_frequency) ** (compound_frequency * time)
        interest = amount - principal
        
        return {
            'final_amount': round(amount, 2),
            'interest_earned': round(interest, 2),
            'principal': principal,
            'rate': rate,
            'time': time,
            'compound_frequency': compound_frequency
        }
    except Exception as e:
        return {'error': str(e), 'result': None}

# Register calculator functions
def register_calculator_functions():
    """Register calculator functions with the function registry."""
    registry = get_function_registry()
    
    # Main calculator function
    registry.register_function(
        definition=FunctionDefinition(
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
        handler=calculator
    )
    
    # Percentage calculation
    registry.register_function(
        definition=FunctionDefinition(
            name="calculate_percentage",
            description="Calculate percentage of a value",
            function_type=FunctionType.CALCULATOR,
            parameters=[
                FunctionParameter(
                    name="value",
                    type=ParameterType.FLOAT,
                    description="The base value",
                    required=True
                ),
                FunctionParameter(
                    name="percentage",
                    type=ParameterType.FLOAT,
                    description="The percentage to calculate",
                    required=True,
                    min_value=0
                )
            ],
            agent_restrictions=["finance", "general"]
        ),
        handler=percentage_calculation
    )
    
    # Compound interest calculation
    registry.register_function(
        definition=FunctionDefinition(
            name="compound_interest",
            description="Calculate compound interest for investments",
            function_type=FunctionType.CALCULATOR,
            parameters=[
                FunctionParameter(
                    name="principal",
                    type=ParameterType.FLOAT,
                    description="Initial principal amount",
                    required=True,
                    min_value=0
                ),
                FunctionParameter(
                    name="rate",
                    type=ParameterType.FLOAT,
                    description="Annual interest rate (as percentage)",
                    required=True,
                    min_value=0
                ),
                FunctionParameter(
                    name="time",
                    type=ParameterType.FLOAT,
                    description="Time period in years",
                    required=True,
                    min_value=0
                ),
                FunctionParameter(
                    name="compound_frequency",
                    type=ParameterType.INTEGER,
                    description="Number of times interest compounds per year",
                    required=False,
                    default=1,
                    min_value=1,
                    max_value=365
                )
            ],
            agent_restrictions=["finance"]
        ),
        handler=compound_interest
    )
