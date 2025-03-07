"""
Exceptions for the ERCOT RTLMP spike prediction system CLI.

This module defines custom exception classes for the CLI application with appropriate
error messages, exit codes, and contextual information for various failure scenarios.
"""

from pathlib import Path
from typing import Any, Dict, Optional

# Exit code constants
EXIT_CODE_SUCCESS = 0
EXIT_CODE_GENERAL_ERROR = 1
EXIT_CODE_CONFIGURATION_ERROR = 2
EXIT_CODE_VALIDATION_ERROR = 3
EXIT_CODE_FILE_ERROR = 4
EXIT_CODE_DATA_FETCH_ERROR = 5
EXIT_CODE_MODEL_ERROR = 6


def get_exit_code(exception: Exception) -> int:
    """
    Return the appropriate exit code for a given exception type.

    Args:
        exception: The exception to get an exit code for

    Returns:
        int: The exit code corresponding to the exception type
    """
    if isinstance(exception, ConfigurationError):
        return EXIT_CODE_CONFIGURATION_ERROR
    elif isinstance(exception, ValidationError):
        return EXIT_CODE_VALIDATION_ERROR
    elif isinstance(exception, FileError):
        return EXIT_CODE_FILE_ERROR
    elif isinstance(exception, DataFetchError):
        return EXIT_CODE_DATA_FETCH_ERROR
    elif isinstance(exception, ModelOperationError):
        return EXIT_CODE_MODEL_ERROR
    elif isinstance(exception, CLIException):
        return EXIT_CODE_GENERAL_ERROR
    else:
        return EXIT_CODE_GENERAL_ERROR


class CLIException(Exception):
    """Base exception class for all CLI-specific exceptions."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the CLI exception with message and optional details.

        Args:
            message: Human-readable error message
            details: Additional structured information about the error
            cause: The underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        """
        Return a string representation of the exception.

        Returns:
            str: String representation of the exception
        """
        return self.message

    def with_context(self, context_message: str) -> "CLIException":
        """
        Create a new exception with additional context information.

        Args:
            context_message: Additional context to prepend to the message

        Returns:
            CLIException: New exception with updated context
        """
        new_message = f"{context_message}: {self.message}"
        new_exception = self.__class__(new_message)
        new_exception.details = self.details.copy()
        new_exception.cause = self
        return new_exception


class ConfigurationError(CLIException):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the configuration error with message and details.

        Args:
            message: Human-readable error message
            config_details: Details about the configuration that caused the error
            cause: The underlying exception that caused this error
        """
        super().__init__(message, config_details, cause)
        self.config_details = config_details or {}


class CommandError(CLIException):
    """Exception raised for command execution failures."""

    def __init__(
        self,
        message: str,
        command: str,
        command_details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the command error with command name and details.

        Args:
            message: Human-readable error message
            command: The name of the command that failed
            command_details: Additional details about the command execution
            cause: The underlying exception that caused this error
        """
        details = {"command": command, "command_details": command_details or {}}
        super().__init__(message, details, cause)
        self.command = command
        self.command_details = command_details or {}


class ValidationError(CLIException):
    """Exception raised for parameter validation failures."""

    def __init__(
        self,
        message: str,
        parameter: str,
        provided_value: Any,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the validation error with parameter name and provided value.

        Args:
            message: Human-readable error message
            parameter: The name of the parameter that failed validation
            provided_value: The invalid value that was provided
            cause: The underlying exception that caused this error
        """
        details = {"parameter": parameter, "provided_value": provided_value}
        super().__init__(message, details, cause)
        self.parameter = parameter
        self.provided_value = provided_value


class FileError(CLIException):
    """Exception raised for file operation failures."""

    def __init__(
        self,
        message: str,
        file_path: Path,
        operation: str,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the file error with file path and operation.

        Args:
            message: Human-readable error message
            file_path: The path to the file that caused the error
            operation: The operation that failed (e.g., "read", "write")
            cause: The underlying exception that caused this error
        """
        details = {"file_path": str(file_path), "operation": operation}
        super().__init__(message, details, cause)
        self.file_path = file_path
        self.operation = operation


class DataFetchError(CLIException):
    """Exception raised for data fetching failures."""

    def __init__(
        self,
        message: str,
        data_source: str,
        fetch_details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the data fetch error with data source and details.

        Args:
            message: Human-readable error message
            data_source: The source of the data (e.g., "ERCOT API")
            fetch_details: Additional details about the data fetch operation
            cause: The underlying exception that caused this error
        """
        details = {"data_source": data_source, "fetch_details": fetch_details or {}}
        super().__init__(message, details, cause)
        self.data_source = data_source
        self.fetch_details = fetch_details or {}


class ModelOperationError(CLIException):
    """Exception raised for model operation failures."""

    def __init__(
        self,
        message: str,
        operation: str,
        model_details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the model operation error with operation and details.

        Args:
            message: Human-readable error message
            operation: The model operation that failed (e.g., "training", "inference")
            model_details: Additional details about the model operation
            cause: The underlying exception that caused this error
        """
        details = {"operation": operation, "model_details": model_details or {}}
        super().__init__(message, details, cause)
        self.operation = operation
        self.model_details = model_details or {}