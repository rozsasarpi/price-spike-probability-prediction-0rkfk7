"""
Core implementation of the CLI application for the ERCOT RTLMP spike prediction system.
This file defines the main CLI application class, command registration, and initialization logic,
providing a unified interface for all command-line operations.
"""

import typing  # Standard
from typing import Dict, List, Optional, Any, cast  # typing
from pathlib import Path  # Standard
import click  # version 8.0+
import sys  # Standard

from .cli_types import CommandType, CLIConfigDict  # ./cli_types
from .exceptions import CLIException, ConfigurationError  # ./exceptions
from .logger import get_cli_logger  # ./logger
from .config.cli_config import ConfigManager  # ./config/cli_config
from .commands import (  # ./commands
    fetch_data_command,
    train_command,
    predict_command,
    backtest_command,
    evaluate_command,
    visualize_command,
)

logger = get_cli_logger(__name__)


def initialize_cli(config_file: Optional[Path] = None) -> click.Group:
    """
    Initializes the CLI application with configuration and commands

    Args:
        config_file (Optional[Path], optional): Path to the configuration file. Defaults to None.

    Returns:
        click.Group: Configured CLI application group
    """
    logger.info("Initializing CLI application")

    # Create a ConfigManager instance with the provided config_file
    config_manager = ConfigManager(config_file=config_file)

    # Get the CLI configuration using config_manager.get_cli_config()
    cli_config = config_manager.get_cli_config()

    # Create a new CLI application using create_cli_app()
    cli_app = create_cli_app(cli_config)

    # Register all commands with the CLI application
    register_commands(cli_app, config_manager)

    # Return the configured CLI application
    return cli_app


def create_cli_app(config: CLIConfigDict) -> click.Group:
    """
    Creates the base CLI application with global options

    Args:
        config (CLIConfigDict): CLI configuration dictionary

    Returns:
        click.Group: Base CLI application group
    """
    # Create a Click group with the name 'rtlmp_predict'
    @click.group(name='rtlmp_predict', help="ERCOT RTLMP spike prediction system")
    @click.option('--config', '-c', type=click.Path(exists=True, dir_okay=False), help='Path to configuration file')
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
    @click.pass_context
    def cli(ctx: click.Context, config: Optional[str], verbose: bool):
        """
        ERCOT RTLMP spike prediction system
        """
        # Ensure that ctx.obj exists and is a dictionary
        ctx.ensure_object(dict)

        # Store the config and verbose flag in the context object
        ctx.obj['config'] = config
        ctx.obj['verbose'] = verbose

    # Configure the group with appropriate settings from the config
    return cli


def register_commands(cli_app: click.Group, config_manager: ConfigManager):
    """
    Registers all command functions with the CLI application

    Args:
        cli_app (click.Group): The Click group to register commands with
        config_manager (ConfigManager): The ConfigManager instance
    """
    # Register fetch_data_command with the CLI application
    cli_app.add_command(fetch_data_command)

    # Register train_command with the CLI application
    cli_app.add_command(train_command)

    # Register predict_command with the CLI application
    cli_app.add_command(predict_command)

    # Register backtest_command with the CLI application
    cli_app.add_command(backtest_command)

    # Register evaluate_command with the CLI application
    cli_app.add_command(evaluate_command)

    # Register visualize_command with the CLI application
    cli_app.add_command(visualize_command)

    # Log the successful registration of all commands
    logger.info("Registered all commands with the CLI application")


class CLI:
    """
    Main CLI application class that manages commands and execution
    """

    def __init__(self, app: click.Group, config_manager: ConfigManager):
        """
        Initializes the CLI application with configuration

        Args:
            app (click.Group): The Click application group
            config_manager (ConfigManager): The ConfigManager instance
        """
        # Store the Click application group
        self._app = app

        # Store the ConfigManager instance
        self._config_manager = config_manager

        # Initialize an empty dictionary for commands
        self._commands: Dict[CommandType, click.Command] = {}

    def add_command(self, command: click.Command, command_type: CommandType):
        """
        Adds a command to the CLI application

        Args:
            command (click.Command): The Click command to add
            command_type (CommandType): The type of the command
        """
        # Add the command to the Click application group
        self._app.add_command(command)

        # Store the command in the _commands dictionary with command_type as key
        self._commands[command_type] = command

        # Log the addition of the command
        logger.info(f"Added command '{command_type}' to the CLI application")

    def get_command(self, command_type: CommandType) -> Optional[click.Command]:
        """
        Gets a command by its type

        Args:
            command_type (CommandType): The type of the command

        Returns:
            Optional[click.Command]: The command if found, None otherwise
        """
        # Return the command from the _commands dictionary if it exists
        if command_type in self._commands:
            return self._commands[command_type]

        # Return None if the command type is not found
        return None

    def execute(self, args: List[str]) -> int:
        """
        Executes the CLI application with the given arguments

        Args:
            args (List[str]): The command-line arguments

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        logger.info(f"Executing CLI with arguments: {args}")
        try:
            # Call the Click application with the provided arguments
            result: int = self._app(args=args, standalone_mode=False)
            return result
        except click.exceptions.ClickException as e:
            # Handle Click exceptions and return appropriate error codes
            logger.error(f"ClickException: {e}")
            return 1
        except Exception as e:
            # Handle any exceptions and return appropriate error codes
            logger.error(f"Exception: {e}")
            return 1

    def get_config_manager(self) -> ConfigManager:
        """
        Gets the configuration manager instance

        Returns:
            ConfigManager: The configuration manager instance
        """
        # Return the stored ConfigManager instance
        return self._config_manager

# Create a global instance of the CLI application
cli = initialize_cli()