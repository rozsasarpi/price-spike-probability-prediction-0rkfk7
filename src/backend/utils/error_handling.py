"""
Core error handling utilities for the ERCOT RTLMP spike prediction system.

Provides standardized error classes, error handling decorators, retry mechanisms,
and context managers to ensure consistent error management across all system components.
"""

import typing
from typing import Dict, List, Optional, Callable, Any, Tuple, Union, Type, TypeVar, Generic, cast
import functools
import time
import random
import traceback
import sys
import inspect
from contextlib import contextmanager

from ..utils.logging import get_logger
from ..utils import type_definitions

# Set up logger
logger = get_logger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = A = 3
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_JITTER_FACTOR = 0.1

# Retry mechanism
def retry_with_backoff(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    jitter_factor: float = DEFAULT_JITTER_FACTOR,
    log_retries: bool = True
) -> Callable:
    """
    Decorator that retries a function with exponential backoff on specified exceptions.
    
    Args:
        exceptions: Exception or tuple of exceptions to catch
        max_retries: Maximum number of retries
        backoff_factor: Factor to increase delay between retries
        jitter_factor: Random jitter factor to add to delay
        log_retries: Whether to log retry attempts
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Initialize retry counter and delay
            retry_count = 0
            delay = 1.0
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    
                    if retry_count > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise
                    
                    if log_retries:
                        logger.warning(
                            f"Retry {retry_count}/{max_retries} for {func.__name__} after error: {e}"
                        )
                    
                    # Calculate delay with exponential backoff and jitter
                    jitter = random.uniform(-jitter_factor, jitter_factor) * delay
                    sleep_time = delay + jitter
                    
                    time.sleep(sleep_time)
                    delay *= backoff_factor
        
        return wrapper
    
    return decorator

def handle_errors(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
    default_return: Any = None,
    reraise: bool = False,
    error_message: Optional[str] = None
) -> Callable:
    """
    Decorator that handles specified exceptions with custom error handling.
    
    Args:
        exceptions: Exception or tuple of exceptions to catch
        default_return: Value to return if an exception is caught
        reraise: Whether to re-raise the exception after handling
        error_message: Custom error message to log
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                message = error_message or f"Error in {func.__name__}: {e}"
                log_error(e, message, level="ERROR")
                
                if reraise:
                    raise
                
                return default_return
        
        return wrapper
    
    return decorator

def format_exception(exc: Exception, include_traceback: bool = True) -> str:
    """
    Formats an exception with traceback information for logging.
    
    Args:
        exc: Exception to format
        include_traceback: Whether to include traceback information
        
    Returns:
        Formatted exception string
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)
    
    if include_traceback:
        tb_str = traceback.format_exc()
        return f"{exc_type}: {exc_msg}\n{tb_str}"
    
    return f"{exc_type}: {exc_msg}"

def get_error_context(exc: Exception, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Extracts contextual information from an exception for error reporting.
    
    Args:
        exc: Exception to extract context from
        additional_context: Additional context to include
        
    Returns:
        Error context dictionary
    """
    context = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "timestamp": time.time()
    }
    
    # Add traceback if available
    if sys.exc_info()[2] is not None:
        context["traceback"] = traceback.format_exc()
    
    # Add function information if available
    frame = inspect.currentframe()
    if frame and frame.f_back:
        func_frame = frame.f_back
        func_info = inspect.getframeinfo(func_frame)
        context["function"] = func_info.function
        context["filename"] = func_info.filename
        context["lineno"] = func_info.lineno
    
    # Merge with additional context
    if additional_context:
        context.update(additional_context)
    
    return context

def is_retryable_error(exc: Exception) -> bool:
    """
    Determines if an error is retryable based on its type and attributes.
    
    Args:
        exc: Exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    # Check if it's one of our custom retryable errors
    if hasattr(exc, "retryable") and getattr(exc, "retryable"):
        return True
    
    # Check for common retryable error types
    retryable_types = [
        ConnectionError,
        RateLimitError,
        # Standard library/external errors that are typically transient
        TimeoutError,
        ConnectionResetError,
        ConnectionAbortedError,
        ConnectionRefusedError
    ]
    
    for error_type in retryable_types:
        if isinstance(exc, error_type):
            return True
    
    # Check for common HTTP error patterns that might be retryable
    if hasattr(exc, "status_code") and getattr(exc, "status_code") in [429, 500, 502, 503, 504]:
        return True
    
    return False

def circuit_breaker(failure_threshold: int = 5, reset_timeout: float = 60.0) -> Callable:
    """
    Implements the circuit breaker pattern to prevent repeated calls to failing services.
    
    Args:
        failure_threshold: Number of failures before opening the circuit
        reset_timeout: Time in seconds before trying to close the circuit again
        
    Returns:
        Decorator implementing circuit breaker pattern
    """
    def decorator(func: Callable) -> Callable:
        # Static variables for tracking circuit state
        failure_count = 0
        last_failure_time = 0.0
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal failure_count, last_failure_time
            
            # Check if circuit is open
            if failure_count >= failure_threshold:
                # Check if reset timeout has elapsed
                if time.time() - last_failure_time < reset_timeout:
                    reset_time = last_failure_time + reset_timeout
                    raise CircuitOpenError(
                        f"Circuit breaker open for {func.__name__}, will reset at {reset_time}",
                        reset_time
                    )
                else:
                    # Reset timeout has elapsed, try again
                    logger.info(f"Circuit reset for {func.__name__}, attempting to close")
                    failure_count = 0
            
            try:
                result = func(*args, **kwargs)
                # Success, reset failure count
                failure_count = 0
                return result
            except Exception as e:
                # Increment failure count and update timestamp
                failure_count += 1
                last_failure_time = time.time()
                
                if failure_count >= failure_threshold:
                    logger.error(f"Circuit breaker tripped for {func.__name__} after {failure_count} failures")
                
                # Re-raise the original exception
                raise
        
        return wrapper
    
    return decorator

def log_error(
    exc: Exception, 
    message: str, 
    level: str = "ERROR", 
    context: Dict[str, Any] = None
) -> None:
    """
    Logs an error with appropriate severity and context.
    
    Args:
        exc: Exception to log
        message: Message to log
        level: Log level to use
        context: Additional context for the log entry
    """
    # Determine appropriate log level
    log_level = level.upper()
    if not hasattr(logger, log_level.lower()):
        log_level = "ERROR"
    
    # Get the logging function for the specified level
    log_func = getattr(logger, log_level.lower())
    
    # Format the exception
    formatted_exc = format_exception(exc)
    
    # Create the message with exception info
    full_message = f"{message}: {formatted_exc}"
    
    # Add context if provided
    if context:
        error_context = get_error_context(exc, context)
        log_func(full_message, extra={"context": error_context})
    else:
        log_func(full_message)

# Custom exception classes
class BaseError(Exception):
    """Base exception class for all custom errors in the system."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = False):
        """
        Initialize the base error with message and context.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.retryable = retryable
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the error to a dictionary representation.
        
        Returns:
            Dictionary representation of the error
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "retryable": self.retryable
        }
        
        # Add traceback if available
        if sys.exc_info()[2] is not None:
            result["traceback"] = traceback.format_exc()
        
        return result

class DataError(BaseError):
    """Base class for data-related errors."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = False):
        """
        Initialize the data error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried
        """
        super().__init__(message, context, retryable)

class DataFormatError(DataError):
    """Error raised when data has an invalid format."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = False):
        """
        Initialize the data format error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried
        """
        super().__init__(message, context, retryable)

class MissingDataError(DataError):
    """Error raised when required data is missing."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = False):
        """
        Initialize the missing data error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried
        """
        super().__init__(message, context, retryable)

class ConnectionError(BaseError):
    """Error raised when a connection to an external service fails."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = True):
        """
        Initialize the connection error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried (default True for connection errors)
        """
        super().__init__(message, context, retryable)

class RateLimitError(BaseError):
    """Error raised when an API rate limit is exceeded."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retry_after: Optional[float] = None):
        """
        Initialize the rate limit error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retry_after: Time in seconds after which to retry
        """
        super().__init__(message, context, retryable=True)
        self.retry_after = retry_after

class ModelError(BaseError):
    """Base class for model-related errors."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = False):
        """
        Initialize the model error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried
        """
        super().__init__(message, context, retryable)

class ModelLoadError(ModelError):
    """Error raised when a model cannot be loaded."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = False):
        """
        Initialize the model load error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried
        """
        super().__init__(message, context, retryable)

class ModelTrainingError(ModelError):
    """Error raised when model training fails."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = False):
        """
        Initialize the model training error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried
        """
        super().__init__(message, context, retryable)

class InferenceError(BaseError):
    """Error raised when inference fails."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, retryable: bool = False):
        """
        Initialize the inference error.
        
        Args:
            message: Error message
            context: Additional context about the error
            retryable: Whether this error can be retried
        """
        super().__init__(message, context, retryable)

class CircuitOpenError(BaseError):
    """Error raised when a circuit breaker is open."""
    
    def __init__(self, message: str, reset_time: float, context: Dict[str, Any] = None):
        """
        Initialize the circuit open error.
        
        Args:
            message: Error message
            reset_time: Time at which the circuit will reset
            context: Additional context about the error
        """
        super().__init__(message, context, retryable=False)
        self.reset_time = reset_time

# Context managers
class ErrorHandler:
    """Context manager for handling errors with custom error handling."""
    
    def __init__(
        self, 
        exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
        default_return: Any = None,
        reraise: bool = False,
        error_message: Optional[str] = None,
        handler: Optional[Callable[[Exception], Any]] = None
    ):
        """
        Initialize the error handler context manager.
        
        Args:
            exceptions: Exception or tuple of exceptions to catch
            default_return: Value to return if an exception is caught
            reraise: Whether to re-raise the exception after handling
            error_message: Custom error message to log
            handler: Custom handler function to call with the exception
        """
        self._exceptions = exceptions
        self._default_return = default_return
        self._reraise = reraise
        self._error_message = error_message
        self._handler = handler
    
    def __enter__(self) -> 'ErrorHandler':
        """
        Enter the error handler context.
        
        Returns:
            Self reference
        """
        return self
    
    def __exit__(
        self, 
        exc_type: Optional[Type[Exception]], 
        exc_val: Optional[Exception], 
        exc_tb: Optional[traceback.TracebackType]
    ) -> bool:
        """
        Exit the error handler context, handling any exceptions.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
            
        Returns:
            True if exception was handled, False otherwise
        """
        if exc_val is None:
            # No exception occurred
            return False
        
        if isinstance(exc_val, self._exceptions):
            # Log the error
            message = self._error_message or f"Error handled by context manager: {exc_val}"
            log_error(exc_val, message)
            
            # Call custom handler if provided
            if self._handler:
                self._handler(exc_val)
            
            # Determine whether to suppress the exception
            return not self._reraise
        
        # Exception didn't match, let it propagate
        return False

class RetryContext:
    """Context manager for retrying operations with exponential backoff."""
    
    def __init__(
        self, 
        exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]],
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        jitter_factor: float = DEFAULT_JITTER_FACTOR,
        log_retries: bool = True
    ):
        """
        Initialize the retry context manager.
        
        Args:
            exceptions: Exception or tuple of exceptions to catch
            max_retries: Maximum number of retries
            backoff_factor: Factor to increase delay between retries
            jitter_factor: Random jitter factor to add to delay
            log_retries: Whether to log retry attempts
        """
        self._exceptions = exceptions
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._jitter_factor = jitter_factor
        self._log_retries = log_retries
        self._retry_count = 0
        self._delay = 1.0
    
    def __enter__(self) -> 'RetryContext':
        """
        Enter the retry context.
        
        Returns:
            Self reference
        """
        return self
    
    def __exit__(
        self, 
        exc_type: Optional[Type[Exception]], 
        exc_val: Optional[Exception], 
        exc_tb: Optional[traceback.TracebackType]
    ) -> bool:
        """
        Exit the retry context, handling any exceptions with retry logic.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
            
        Returns:
            True if retry should continue, False otherwise
        """
        if exc_val is None:
            # No exception occurred
            return False
        
        if isinstance(exc_val, self._exceptions):
            self._retry_count += 1
            
            if self._retry_count > self._max_retries:
                # Max retries exceeded, let the exception propagate
                logger.error(f"Max retries ({self._max_retries}) exceeded: {exc_val}")
                return False
            
            if self._log_retries:
                logger.warning(
                    f"Retry {self._retry_count}/{self._max_retries} after error: {exc_val}"
                )
            
            # Calculate delay with exponential backoff and jitter
            jitter = random.uniform(-self._jitter_factor, self._jitter_factor) * self._delay
            sleep_time = self._delay + jitter
            
            time.sleep(sleep_time)
            self._delay *= self._backoff_factor
            
            # Suppress the exception to retry the operation
            return True
        
        # Exception didn't match, let it propagate
        return False
    
    def reset(self) -> None:
        """
        Resets the retry count and delay.
        """
        self._retry_count = 0
        self._delay = 1.0
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the retry attempts.
        
        Returns:
            Dictionary with retry statistics
        """
        return {
            "retry_count": self._retry_count,
            "current_delay": self._delay,
            "max_retries": self._max_retries
        }