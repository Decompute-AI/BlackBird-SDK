"""Function execution engine for the Decompute SDK."""

import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Any, Optional, Callable, List
from .function_types import FunctionCall, FunctionResult, ExecutionStatus
from .function_registry import get_function_registry
from blackbird_sdk.utils.errors import ValidationError, TimeoutError, ExecutionError
from blackbird_sdk.utils.logger import get_logger

class FunctionExecutor:
    """Executes function calls with timeout and error handling."""
    
    def __init__(self, max_workers: int = 10):
        """Initialize the function executor."""
        self.logger = get_logger()
        self.registry = get_function_registry()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_calls: Dict[str, FunctionCall] = {}
        self._lock = threading.RLock()
        
        self.logger.info(f"Function executor initialized with {max_workers} workers")
    
    def execute_function(self, call: FunctionCall) -> FunctionResult:
        """
        Execute a function call synchronously.
        
        Args:
            call: Function call to execute
            
        Returns:
            Function execution result
        """
        result = FunctionResult(
            call_id=call.call_id,
            function_name=call.function_name,
            status=ExecutionStatus.PENDING,
            start_time=time.time()
        )
        
        try:
            # Validate function call
            is_valid, errors = self.registry.validate_function_call(
                call.function_name, call.parameters, call.agent_type
            )
            
            if not is_valid:
                result.status = ExecutionStatus.FAILED
                result.error = f"Validation failed: {'; '.join(errors)}"
                result.end_time = time.time()
                return result
            
            # Get function definition and handler
            function_def = self.registry.get_function(call.function_name)
            handler = self.registry.get_handler(call.function_name)
            
            if not function_def or not handler:
                result.status = ExecutionStatus.FAILED
                result.error = f"Function '{call.function_name}' not found or no handler registered"
                result.end_time = time.time()
                return result
            
            # Track active call
            with self._lock:
                self.active_calls[call.call_id] = call
            
            result.status = ExecutionStatus.RUNNING
            
            # Execute function with timeout
            try:
                future = self.executor.submit(handler, **call.parameters)
                function_result = future.result(timeout=function_def.timeout)
                
                result.status = ExecutionStatus.COMPLETED
                result.result = function_result
                result.end_time = time.time()
                
                self.logger.info(f"Function {call.function_name} executed successfully in {result.duration:.2f}s")
                
            except FutureTimeoutError:
                result.status = ExecutionStatus.TIMEOUT
                result.error = f"Function execution timed out after {function_def.timeout} seconds"
                result.end_time = time.time()
                
                self.logger.warning(f"Function {call.function_name} timed out after {function_def.timeout}s")
                
            except Exception as e:
                result.status = ExecutionStatus.FAILED
                result.error = str(e)
                result.end_time = time.time()
                
                self.logger.error(f"Function {call.function_name} failed: {str(e)}")
        
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = f"Execution error: {str(e)}"
            result.end_time = time.time()
            
            self.logger.error(f"Critical error executing function {call.function_name}: {str(e)}")
        
        finally:
            # Remove from active calls
            with self._lock:
                self.active_calls.pop(call.call_id, None)
        
        return result
    
    def execute_function_async(self, call: FunctionCall, callback: Optional[Callable] = None) -> str:
        """
        Execute a function call asynchronously.
        
        Args:
            call: Function call to execute
            callback: Optional callback function for result
            
        Returns:
            Call ID for tracking
        """
        def execute_and_callback():
            result = self.execute_function(call)
            if callback:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Callback error for function {call.function_name}: {str(e)}")
        
        self.executor.submit(execute_and_callback)
        return call.call_id
    
    def get_active_calls(self) -> List[FunctionCall]:
        """Get list of currently active function calls."""
        with self._lock:
            return list(self.active_calls.values())
    
    def cancel_call(self, call_id: str) -> bool:
        """
        Cancel an active function call.
        
        Args:
            call_id: ID of call to cancel
            
        Returns:
            True if call was cancelled, False if not found or already completed
        """
        with self._lock:
            if call_id in self.active_calls:
                # Note: ThreadPoolExecutor doesn't support cancellation of running tasks
                # This is a limitation we acknowledge
                self.logger.warning(f"Cannot cancel running function call {call_id}")
                return False
            return False
    
    def get_executor_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self._lock:
            return {
                'active_calls': len(self.active_calls),
                'max_workers': self.executor._max_workers,
                'active_calls_details': [
                    {
                        'call_id': call.call_id,
                        'function_name': call.function_name,
                        'agent_type': call.agent_type,
                        'created_at': call.created_at
                    }
                    for call in self.active_calls.values()
                ]
            }
    
    def cleanup(self):
        """Clean up executor resources."""
        self.logger.info("Cleaning up function executor")
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        with self._lock:
            self.active_calls.clear()
        
        self.logger.info("Function executor cleanup complete")
