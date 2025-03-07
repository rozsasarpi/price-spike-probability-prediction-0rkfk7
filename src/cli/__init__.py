"""
Initializes the CLI (Command Line Interface) package for the ERCOT RTLMP spike prediction system.
This file serves as the entry point for the CLI module, exposing key components and functionality to external modules while setting up version information and package metadata.
"""
from .cli_app import CLI, initialize_cli  # ./cli_app
from .exceptions import (  # ./exceptions
    CLIException,
    ConfigurationError,
    CommandError,
    ValidationError,
    FileError,
    DataFetchError,
    ModelOperationError,
    get_exit_code,
)
from .logger import get_cli_logger, log_command_error, CLILoggingContext  # ./logger
from .cli_types import (  # ./cli_types
    CommandType,
    LogLevel,
    DataType,
    VisualizationType,
    OutputFormat,
    CLIConfigDict,
)

__version__ = "0.1.0"
__author__ = "ERCOT RTLMP Prediction Team"
__description__ = "Command Line Interface for ERCOT RTLMP spike prediction system"