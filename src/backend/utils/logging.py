"""
Comprehensive logging utilities for the ERCOT RTLMP spike prediction system.

This module provides standardized logging configuration, formatters, context managers,
and decorators to ensure consistent logging across all system components with
appropriate severity levels, contextual information, and performance tracking.
"""

import logging
import typing
from typing import Dict, List, Optional, Callable, Any, Tuple, Union, Type, TypeVar, cast
import functools
import time
import datetime
import inspect
import json
import os
import sys
import traceback
from contextlib import contextmanager

from ..config.default_config import DEFAULT_CONFIG
from . import type_definitions

# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_FILE = "ercot_rtlmp_prediction.log"
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Configures the logging system with appropriate handlers and formatters.
    
    Args:
        config: Optional configuration dictionary with logging settings
        log_file: Optional path to log file (overrides config)
        log_level: Optional log level (overrides config)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        json_format: Whether to format logs as JSON
        
    Returns:
        Root logger configured with specified settings
    """
    # Get logging configuration from provided config or DEFAULT_CONFIG
    log_config = config.get("system", {}) if config else DEFAULT_CONFIG.get("system", {})
    
    # Determine log level
    if log_level:
        level = LOG_LEVELS.get(log_level.upper(), DEFAULT_LOG_LEVEL)
    else:
        level_name = log_config.get("log_level", "INFO")
        level = LOG_LEVELS.get(level_name.upper(), DEFAULT_LOG_LEVEL)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_output:
        # Get log file path
        if log_file:
            log_file_path = log_file
        else:
            log_dir = DEFAULT_CONFIG.get("paths", {}).get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, DEFAULT_LOG_FILE)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Gets a logger for a specific module with consistent configuration.
    
    Args:
        name: Name of the module requesting a logger
        log_level: Optional log level for this specific logger
        
    Returns:
        Configured logger for the specified module
    """
    logger = logging.getLogger(name)
    
    if log_level:
        level = LOG_LEVELS.get(log_level.upper(), None)
        if level is not None:
            logger.setLevel(level)
    
    return logger

def log_execution_time(
    logger: Optional[logging.Logger] = None,
    level: str = "DEBUG",
    message: str = "Executed {func_name} in {duration:.4f} seconds"
) -> Callable:
    """
    Decorator for logging function execution time.
    
    Args:
        logger: Optional logger to use (defaults to function module's logger)
        level: Log level to use
        message: Message template, can include {func_name} and {duration}
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the appropriate logger
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            # Get the log level function
            log_func = getattr(logger, level.lower(), logger.debug)
            
            # Time the function execution
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log the execution time
            log_func(message.format(func_name=func.__name__, duration=duration))
            
            return result
        return wrapper
    return decorator

def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: str = "DEBUG",
    log_args: bool = True,
    log_result: bool = False
) -> Callable:
    """
    Decorator for logging function calls with parameters and return values.
    
    Args:
        logger: Optional logger to use (defaults to function module's logger)
        level: Log level to use
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the appropriate logger
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            # Get the log level function
            log_func = getattr(logger, level.lower(), logger.debug)
            
            # Log function call
            if log_args:
                # Format arguments for logging
                arg_str_parts = []
                
                # Add positional args (excluding 'self' for methods)
                is_method = args and hasattr(args[0], func.__name__) and callable(getattr(args[0], func.__name__))
                pos_args = args[1:] if is_method else args
                
                if pos_args:
                    arg_str_parts.append(f"args: {pos_args}")
                
                # Add keyword args
                if kwargs:
                    arg_str_parts.append(f"kwargs: {kwargs}")
                
                arg_str = ", ".join(arg_str_parts)
                log_func(f"Calling {func.__name__}({arg_str})")
            else:
                log_func(f"Calling {func.__name__}")
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log the result if requested
            if log_result:
                log_func(f"{func.__name__} returned: {result}")
            else:
                log_func(f"{func.__name__} completed")
            
            return result
        return wrapper
    return decorator

def format_log_message(message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Formats log messages with consistent structure and context.
    
    Args:
        message: Base log message
        context: Optional context dictionary to include in the log
        
    Returns:
        Formatted log message with context
    """
    if context is None:
        return message
    
    try:
        context_str = json.dumps(context)
        return f"{message} | Context: {context_str}"
    except (TypeError, ValueError):
        # If context can't be JSON serialized, use repr
        return f"{message} | Context: {repr(context)}"

def sanitize_log_data(data: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]:
    """
    Sanitizes sensitive data from log messages.
    
    Args:
        data: Dictionary of data to sanitize
        sensitive_keys: List of keys to redact
        
    Returns:
        Sanitized data dictionary
    """
    # Create a deep copy to avoid modifying the original
    sanitized = {}
    
    for key, value in data.items():
        if key in sensitive_keys:
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value, sensitive_keys)
        elif isinstance(value, list):
            # Handle lists of dictionaries
            if value and isinstance(value[0], dict):
                sanitized[key] = [sanitize_log_data(item, sensitive_keys) 
                                 if isinstance(item, dict) else item 
                                 for item in value]
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    
    return sanitized

def configure_component_logger(
    component_name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    propagate: bool = True
) -> logging.Logger:
    """
    Configures a logger for a specific component with custom settings.
    
    Args:
        component_name: Name of the component
        log_level: Optional log level for this component
        log_file: Optional separate log file for this component
        propagate: Whether to propagate logs to parent loggers
        
    Returns:
        Configured component logger
    """
    # Get logger for the component
    logger = logging.getLogger(component_name)
    
    # Set log level if provided
    if log_level:
        level = LOG_LEVELS.get(log_level.upper(), None)
        if level is not None:
            logger.setLevel(level)
    
    # Add file handler if log_file is provided
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create handler with formatter
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logger.level)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        file_handler.setFormatter(formatter)
        
        # Add the handler
        logger.addHandler(file_handler)
    
    # Set propagate flag
    logger.propagate = propagate
    
    return logger

class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON objects.
    """
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_name: bool = True,
        include_path: bool = True,
        include_function: bool = True,
        include_process: bool = True
    ):
        """
        Initialize the JSON formatter with configuration options.
        
        Args:
            include_timestamp: Whether to include timestamp in the output
            include_level: Whether to include log level in the output
            include_name: Whether to include logger name in the output
            include_path: Whether to include file path in the output
            include_function: Whether to include function name in the output
            include_process: Whether to include process ID in the output
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_name = include_name
        self.include_path = include_path
        self.include_function = include_function
        self.include_process = include_process
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a log record as a JSON string.
        
        Args:
            record: LogRecord to format
            
        Returns:
            JSON-formatted log message
        """
        # Create base dictionary with message
        log_dict = {
            "message": record.getMessage()
        }
        
        # Add optional fields
        if self.include_timestamp:
            log_dict["timestamp"] = datetime.datetime.fromtimestamp(
                record.created
            ).strftime(DEFAULT_DATE_FORMAT)
        
        if self.include_level:
            log_dict["level"] = record.levelname
        
        if self.include_name:
            log_dict["logger"] = record.name
        
        if self.include_path and hasattr(record, "pathname"):
            log_dict["path"] = record.pathname
            log_dict["line"] = record.lineno
        
        if self.include_function and hasattr(record, "funcName"):
            log_dict["function"] = record.funcName
        
        if self.include_process:
            log_dict["process"] = record.process
        
        # Add any extra attributes
        if hasattr(record, "context") and record.context:
            log_dict["context"] = record.context
        
        # Include exception info if present
        if record.exc_info:
            log_dict["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_dict)

class ContextAdapter:
    """
    Adapter for adding contextual information to logs.
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize the context adapter with a logger and context.
        
        Args:
            logger: Logger to adapt
            context: Context dictionary to add to logs
        """
        self.logger = logger
        self.context = context
    
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a debug message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.debug(format_log_message(msg, self.context), *args, **kwargs)
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log an info message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.info(format_log_message(msg, self.context), *args, **kwargs)
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a warning message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.warning(format_log_message(msg, self.context), *args, **kwargs)
    
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log an error message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.error(format_log_message(msg, self.context), *args, **kwargs)
    
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a critical message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.critical(format_log_message(msg, self.context), *args, **kwargs)
    
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log an exception message with context.
        
        Args:
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.exception(format_log_message(msg, self.context), *args, **kwargs)
    
    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a message with specified level and context.
        
        Args:
            level: Logging level
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.log(level, format_log_message(msg, self.context), *args, **kwargs)

class LoggingContext:
    """
    Context manager for temporarily adding context to logs.
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize the logging context with a logger and context.
        
        Args:
            logger: Logger to use
            context: Context dictionary to add to logs
        """
        self._logger = logger
        self._context = context
        self._adapter = None
    
    def __enter__(self) -> ContextAdapter:
        """
        Enter the logging context.
        
        Returns:
            Context adapter for the logger
        """
        self._adapter = ContextAdapter(self._logger, self._context)
        return self._adapter
    
    def __exit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_val: Optional[Exception],
        exc_tb: Optional[traceback.TracebackType]
    ) -> bool:
        """
        Exit the logging context.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
            
        Returns:
            False to propagate exceptions
        """
        self._adapter = None
        return False  # Propagate exceptions

class PerformanceLogger:
    """
    Utility for logging performance metrics and execution times.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the performance logger.
        
        Args:
            logger: Optional logger to use for logging
        """
        self._logger = logger or logging.getLogger("performance")
        self._timers = {}  # type: Dict[str, Dict[str, float]]
        self._metrics = {}  # type: Dict[str, Dict[str, List[float]]]
    
    def start_timer(self, operation: str, category: Optional[str] = None) -> None:
        """
        Start a timer for a specific operation.
        
        Args:
            operation: Name of the operation being timed
            category: Optional category for grouping operations
        """
        category = category or "default"
        
        if category not in self._timers:
            self._timers[category] = {}
        
        self._timers[category][operation] = time.time()
        self._logger.debug(f"Started timer for {operation} in category {category}")
    
    def stop_timer(
        self,
        operation: str,
        category: Optional[str] = None,
        log_result: bool = True
    ) -> float:
        """
        Stop a timer and record the execution time.
        
        Args:
            operation: Name of the operation being timed
            category: Optional category for grouping operations
            log_result: Whether to log the result
            
        Returns:
            Execution time in seconds
        """
        category = category or "default"
        
        if category not in self._timers or operation not in self._timers[category]:
            self._logger.warning(f"No timer started for {operation} in category {category}")
            return 0.0
        
        start_time = self._timers[category][operation]
        execution_time = time.time() - start_time
        
        # Record the metric
        if category not in self._metrics:
            self._metrics[category] = {}
        
        if operation not in self._metrics[category]:
            self._metrics[category][operation] = []
        
        self._metrics[category][operation].append(execution_time)
        
        # Log the result if requested
        if log_result:
            self._logger.info(f"{operation} completed in {execution_time:.4f} seconds")
        
        return execution_time
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        category: Optional[str] = None
    ) -> None:
        """
        Log a custom performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            category: Optional category for grouping metrics
        """
        category = category or "default"
        
        if category not in self._metrics:
            self._metrics[category] = {}
        
        if metric_name not in self._metrics[category]:
            self._metrics[category][metric_name] = []
        
        self._metrics[category][metric_name].append(value)
        self._logger.info(f"Recorded metric {metric_name} = {value} in category {category}")
    
    def get_metrics(
        self,
        category: Optional[str] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Get all recorded performance metrics.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Performance metrics dictionary
        """
        if category:
            return {category: self._metrics.get(category, {})}
        
        return self._metrics
    
    def get_average_metrics(
        self,
        category: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get average values for all recorded metrics.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Average metrics dictionary
        """
        metrics = self.get_metrics(category)
        avg_metrics = {}
        
        for cat, cat_metrics in metrics.items():
            avg_metrics[cat] = {}
            
            for metric_name, values in cat_metrics.items():
                if values:
                    avg_metrics[cat][metric_name] = sum(values) / len(values)
                else:
                    avg_metrics[cat][metric_name] = 0.0
        
        return avg_metrics
    
    def reset_metrics(self, category: Optional[str] = None) -> None:
        """
        Reset all recorded metrics.
        
        Args:
            category: Optional category to reset
        """
        if category:
            if category in self._metrics:
                del self._metrics[category]
            if category in self._timers:
                del self._timers[category]
            self._logger.debug(f"Reset metrics for category {category}")
        else:
            self._metrics = {}
            self._timers = {}
            self._logger.debug("Reset all metrics")
    
    @contextmanager
    def time_operation(
        self,
        operation: str,
        category: Optional[str] = None,
        log_result: bool = True
    ):
        """
        Context manager for timing an operation.
        
        Args:
            operation: Name of the operation
            category: Optional category for grouping operations
            log_result: Whether to log the result
            
        Yields:
            Control to the with block
        """
        self.start_timer(operation, category)
        try:
            yield
        finally:
            self.stop_timer(operation, category, log_result)