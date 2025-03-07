"""
Main entry point for the ERCOT RTLMP spike prediction system's command-line interface.
This file initializes the CLI application, processes command-line arguments, and executes the appropriate commands with error handling.
"""

import sys  # Standard
from pathlib import Path  # Standard
from typing import Optional, List  # typing

from .cli_app import initialize_cli, CLI  # ./cli_app
from .exceptions import CLIException, get_exit_code  # ./exceptions
from .logger import get_cli_logger, log_command_error  # ./logger

logger = get_cli_logger(__name__)


def main() -> int:
    """
    Main entry point for the CLI application that processes command-line arguments and executes commands

    Returns:
        int: Exit code indicating success (0) or failure (non-zero)
    """
    try:
        # Log the start of the CLI application
        logger.info("Starting ERCOT RTLMP spike prediction CLI application")

        # Parse command-line arguments to extract configuration file path if provided
        config_path = parse_config_arg(sys.argv[1:])

        # Initialize the CLI application with the configuration file
        cli_app = initialize_cli(config_file=config_path)

        # Execute the CLI application with the command-line arguments
        exit_code: int = cli_app(sys.argv[1:], standalone_mode=False)

        # Return the exit code from the CLI execution
        return exit_code

    except Exception as e:
        # Handle any exceptions that occur during execution
        # Log errors using log_command_error
        log_command_error(e, logger=logger)

        # Return appropriate exit code based on the exception type
        return get_exit_code(e)


def parse_config_arg(args: List[str]) -> Optional[Path]:
    """
    Parses command-line arguments to extract the configuration file path

    Args:
        args (List[str]): List of command-line arguments

    Returns:
        Optional[Path]: Path to the configuration file if provided, None otherwise
    """
    # Iterate through the command-line arguments
    i = 0
    while i < len(args):
        # Look for '--config' argument
        if args[i] == '--config':
            # If found, get the next argument as the config file path
            if i + 1 < len(args):
                config_file_path_str = args[i + 1]
                # Convert the path string to a Path object
                config_file_path = Path(config_file_path_str)
                return config_file_path
            else:
                logger.warning("No config file path provided after --config argument")
                return None
        i += 1

    # Return None if no config argument was found
    return None


if __name__ == "__main__":
    sys.exit(main())