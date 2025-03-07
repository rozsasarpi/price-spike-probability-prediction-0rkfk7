"""
Provides error handling utilities for the CLI application of the ERCOT RTLMP spike prediction system.

This module contains functions and classes for consistent error handling, formatting, and reporting
across all CLI commands.
"""

import sys
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, ContextManager, Dict, Optional, TypeVar, Union, cast

import click  # version 8.0+

from ..exceptions import CLIException, get_exit_code
from ..logger import format_log_message, get_cli_logger, log_command_error
from ..ui.colors import bold, red, yellow

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Get a logger for this module
logger = get_cli_logger("error_handlers")


def format_error_message(
    error: Exception, context: Optional[Dict[str, Any]] = None, verbose: bool = False
) -> str:
    """
    Formats an error message with consistent styling and context.

    Args:
        error: The exception to format
        context: Additional context information
        verbose: Whether to include detailed error information

    Returns:
        str: Formatted error message with appropriate styling
    """
    # Get the basic error message
    message = str(error)

    # Apply styling to make errors stand out
    formatted_message = red(bold(message))

    # Add context information if provided
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        formatted_message = f"{formatted_message} [{context_str}]"

    # Add cause information if available and verbose is enabled
    if verbose and isinstance(error, CLIException) and error.cause:
        cause_message = f"\nCaused by: {error.cause}"
        formatted_message = f"{formatted_message}{cause_message}"

    # Add traceback for non-CLI exceptions when verbose is enabled
    if verbose and not isinstance(error, CLIException):
        tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        formatted_message = f"{formatted_message}\n\n{tb_str}"

    return formatted_message


def print_error_message(
    error: Exception, context: Optional[Dict[str, Any]] = None, verbose: bool = False
) -> None:
    """
    Prints a formatted error message to stderr.

    Args:
        error: The exception to print
        context: Additional context information
        verbose: Whether to include detailed error information
    """
    formatted_message = format_error_message(error, context, verbose)
    print(formatted_message, file=sys.stderr, flush=True)


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    exit_on_error: bool = True,
) -> int:
    """
    Handles an error by logging it, printing a message, and optionally exiting.

    Args:
        error: The exception to handle
        context: Additional context information
        verbose: Whether to include detailed error information
        exit_on_error: Whether to exit the program after handling the error

    Returns:
        int: Exit code for the error (only returned if not exiting)
    """
    # Log the error
    log_command_error(error, context, logger)

    # Print the error message
    print_error_message(error, context, verbose)

    # Get the appropriate exit code
    exit_code = get_exit_code(error)

    # Exit if requested
    if exit_on_error:
        sys.exit(exit_code)

    return exit_code


def with_error_handling(verbose: bool = False, exit_on_error: bool = True):
    """
    Decorator that adds error handling to a function.

    Args:
        verbose: Whether to include detailed error information
        exit_on_error: Whether to exit the program after handling an error

    Returns:
        Callable: Decorator function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(e, verbose=verbose, exit_on_error=exit_on_error)
                return None  # This line is only reached if exit_on_error is False

        return cast(F, wrapper)

    return decorator


class ErrorHandler:
    """Class that provides error handling functionality for CLI commands."""

    def __init__(self, verbose: bool = False, exit_on_error: bool = True):
        """
        Initialize the error handler with verbosity and exit behavior settings.

        Args:
            verbose: Whether to include detailed error information
            exit_on_error: Whether to exit the program after handling an error
        """
        self.verbose = verbose
        self.exit_on_error = exit_on_error

    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Handle an error by logging it, printing a message, and optionally exiting.

        Args:
            error: The exception to handle
            context: Additional context information

        Returns:
            int: Exit code for the error (only returned if not exiting)
        """
        return handle_error(error, context, self.verbose, self.exit_on_error)

    def context(self, context: Optional[Dict[str, Any]] = None) -> ContextManager[None]:
        """
        Create a context manager for error handling.

        Args:
            context: Additional context information

        Returns:
            ContextManager: Context manager for error handling
        """

        @contextmanager
        def error_handling_context():
            try:
                yield
            except Exception as e:
                self.handle(e, context)

        return error_handling_context()

    def decorator(self, context: Optional[Dict[str, Any]] = None) -> Callable[[F], F]:
        """
        Create a decorator for adding error handling to a function.

        Args:
            context: Additional context information

        Returns:
            Callable: Decorator function
        """
        return with_error_handling(verbose=self.verbose, exit_on_error=self.exit_on_error)


class ErrorHandlingContext:
    """Context manager for error handling with specific context information."""

    def __init__(
        self, context: Dict[str, Any], verbose: bool = False, exit_on_error: bool = True
    ):
        """
        Initialize the error handling context.

        Args:
            context: Context information to include in error messages
            verbose: Whether to include detailed error information
            exit_on_error: Whether to exit the program after handling an error
        """
        self._context = context
        self._verbose = verbose
        self._exit_on_error = exit_on_error

    def __enter__(self) -> "ErrorHandlingContext":
        """
        Enter the error handling context.

        Returns:
            ErrorHandlingContext: Self reference for use in with statement
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> bool:
        """
        Exit the error handling context, handling any exceptions.

        Args:
            exc_type: Type of exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Exception traceback, if any

        Returns:
            bool: True if exception was handled, False otherwise
        """
        if exc_type is None:
            return False

        handle_error(
            exc_val,  # type: ignore
            context=self._context,
            verbose=self._verbose,
            exit_on_error=self._exit_on_error,
        )
        return True  # Indicate that we've handled the exception