"""
Provides logging utilities for the CLI application of the ERCOT RTLMP spike prediction system.

This module configures logging with appropriate handlers, formatters, and severity levels
to ensure consistent logging across all CLI operations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Union, cast

import click  # version 8.0+

from .exceptions import CLIException
from .cli_types import LogLevel

# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CLI_LOG_FORMAT = "[%(levelname)s] %(message)s"
VERBOSE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Mapping of log level names to logging constants
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Cache for CLI loggers
CLI_LOGGERS: Dict[str, logging.Logger] = {}


class ClickLogFormatter(logging.Formatter):
    """Custom formatter that applies Click styling to log messages for CLI output."""

    def __init__(self, fmt: str, use_colors: bool = True):
        """
        Initialize the Click log formatter with color options.

        Args:
            fmt: Format string for the log message
            use_colors: Whether to apply color formatting
        """
        super().__init__(fmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a log record with Click styling.

        Args:
            record: Log record to format

        Returns:
            str: Formatted log message with styling
        """
        # Get the basic formatted message from the parent class
        formatted_message = super().format(record)

        if self.use_colors:
            # Apply click styling based on log level
            level_name = record.levelname
            colored_level_name = colorize_log_level(level_name)
            
            # Replace the level name with the colored version
            if CLI_LOG_FORMAT in self._fmt:
                # For CLI format, replace the level in the square brackets
                formatted_message = formatted_message.replace(
                    f"[{level_name}]", f"[{colored_level_name}]"
                )
            else:
                # For other formats, replace the level name directly
                formatted_message = formatted_message.replace(
                    level_name, colored_level_name
                )
                
            # Add additional styling for error messages
            if record.levelno >= logging.ERROR:
                formatted_message = click.style(formatted_message, bold=True)
                
        return formatted_message


def configure_cli_logging(
    log_level: Optional[str] = None, 
    log_file: Optional[Path] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Configures the logging system for CLI operations with appropriate handlers and formatters.
    
    Args:
        log_level: String representation of log level (DEBUG, INFO, etc.)
        log_file: Optional file path to write logs to
        verbose: Whether to use verbose logging format
        
    Returns:
        logging.Logger: Root logger configured with specified settings
    """
    # Determine log level from parameter or environment variable
    level_name = log_level or os.environ.get("RTLMP_LOG_LEVEL", "INFO")
    numeric_level = LOG_LEVELS.get(level_name.upper(), DEFAULT_LOG_LEVEL)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler for stderr output
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    
    # Set the appropriate log format based on verbose flag
    if verbose:
        log_format = VERBOSE_LOG_FORMAT
        formatter = logging.Formatter(log_format)
    else:
        log_format = CLI_LOG_FORMAT
        # Use color output if connected to a terminal
        use_colors = sys.stderr.isatty()
        formatter = ClickLogFormatter(log_format, use_colors=use_colors)
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # If log_file is provided, create file handler
    if log_file:
        os.makedirs(log_file.parent, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        # Always use verbose format for file logs
        file_formatter = logging.Formatter(VERBOSE_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_cli_logger(name: str, log_level: Optional[LogLevel] = None) -> logging.Logger:
    """
    Gets a logger for a specific CLI component with consistent configuration.
    
    Args:
        name: Logger name, typically the module or component name
        log_level: Optional specific log level for this logger
        
    Returns:
        logging.Logger: Configured logger for the specified component
    """
    # Check if this logger is already cached
    if name in CLI_LOGGERS:
        logger = CLI_LOGGERS[name]
        # Update log level if specified
        if log_level is not None:
            logger.setLevel(LOG_LEVELS[log_level])
        return logger
    
    # Create a new logger
    logger = logging.getLogger(name)
    
    # Set specific log level if provided, otherwise inherit from root
    if log_level is not None:
        logger.setLevel(LOG_LEVELS[log_level])
    
    # Cache the logger for future use
    CLI_LOGGERS[name] = logger
    
    return logger


def log_command_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Logs an error that occurred during command execution with appropriate context.
    
    Args:
        error: The exception that occurred
        context: Additional context information
        logger: Logger to use (defaults to CLI logger if not provided)
    """
    if logger is None:
        logger = get_cli_logger("cli")
    
    # Extract error details based on error type
    error_details: Dict[str, Any] = {}
    
    if isinstance(error, CLIException):
        error_details = error.details.copy() if error.details else {}
        message = str(error)
    else:
        error_details = {"type": type(error).__name__}
        message = str(error)
    
    # Merge with provided context if any
    if context:
        error_details.update(context)
    
    # Format the error message with context
    formatted_message = format_log_message(message, error_details)
    
    # Log at appropriate level
    if isinstance(error, CLIException):
        logger.error(formatted_message)
    else:
        # For non-CLI exceptions, log as critical with traceback
        logger.critical(formatted_message, exc_info=True)
    
    # If the error has a cause, log that as well
    if isinstance(error, CLIException) and error.cause:
        cause_message = f"Caused by: {error.cause}"
        logger.error(cause_message)


def format_log_message(message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Formats log messages with consistent structure and context for CLI output.
    
    Args:
        message: The primary log message
        context: Optional context dictionary to include
        
    Returns:
        str: Formatted log message with context
    """
    if not context:
        return message
    
    # Format context as a readable string
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    return f"{message} [{context_str}]"


def colorize_log_level(level_name: str) -> str:
    """
    Applies color formatting to log level names for console output.
    
    Args:
        level_name: Name of the log level (DEBUG, INFO, etc.)
        
    Returns:
        str: Colorized log level name
    """
    # Map log levels to appropriate colors
    level_colors = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bright_red"
    }
    
    # Get color for this level, default to white if not found
    color = level_colors.get(level_name, "white")
    
    # Apply click styling
    return click.style(level_name, fg=color)


class CLILoggingContext:
    """Context manager for temporarily changing log level or adding context to logs."""
    
    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[LogLevel] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the CLI logging context.
        
        Args:
            logger: Logger to modify
            level: Optional new log level to set temporarily
            context: Optional context dictionary to use for logging
        """
        self._logger = logger
        self._level = LOG_LEVELS[level] if level else None
        self._context = context
        self._original_level = None
        
    def __enter__(self) -> logging.Logger:
        """
        Enter the logging context, changing log level if specified.
        
        Returns:
            logging.Logger: The logger with modified configuration
        """
        if self._level is not None:
            self._original_level = self._logger.level
            self._logger.setLevel(self._level)
            
        return self._logger
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> bool:
        """
        Exit the logging context, restoring original log level.
        
        Args:
            exc_type: Type of exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Exception traceback, if any
            
        Returns:
            bool: False to propagate exceptions
        """
        # Restore original log level if we changed it
        if self._original_level is not None:
            self._logger.setLevel(self._original_level)
            
        # If an exception occurred and we have context, log it
        if exc_val and self._context:
            log_command_error(exc_val, self._context, self._logger)
            
        # Return False to propagate any exceptions
        return False