"""
Provides helper functions and classes for managing configuration in the CLI application
of the ERCOT RTLMP spike prediction system.

This module handles loading, validating, merging, and accessing configuration from files,
environment variables, and command-line arguments.
"""

import os
import yaml  # version 6.0+
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, cast, TypeVar, Generic

# Internal imports
from ..cli_types import (
    CLIConfigDict, FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict,
    CommandType, LogLevel, DataType, VisualizationType, OutputFormat
)
from ..config.default_config import (
    get_default_cli_config, get_default_command_params
)
from ..exceptions import ConfigurationError, ValidationError, FileError
from ..utils.validators import validate_cli_config, validate_command_params
from ..logger import get_cli_logger

# Initialize logger
logger = get_cli_logger(__name__)

# Default locations to search for configuration files
DEFAULT_CONFIG_LOCATIONS = [
    Path("./rtlmp_predict.yaml"),
    Path("./rtlmp_predict.yml"),
    Path("./rtlmp_predict.json"),
    Path("~/.rtlmp_predict.yaml"),
    Path("~/.rtlmp_predict.yml"),
    Path("~/.rtlmp_predict.json"),
    Path("/etc/rtlmp_predict/config.yaml"),
    Path("/etc/rtlmp_predict/config.yml"),
    Path("/etc/rtlmp_predict/config.json")
]


def find_config_file() -> Optional[Path]:
    """
    Searches for a configuration file in default locations.
    
    Returns:
        Optional[Path]: Path to the found configuration file, or None if not found
    """
    for path in DEFAULT_CONFIG_LOCATIONS:
        expanded_path = Path(os.path.expanduser(str(path)))
        if expanded_path.exists() and expanded_path.is_file():
            logger.debug(f"Found configuration file at {expanded_path}")
            return expanded_path
    
    logger.debug("No configuration file found in default locations")
    return None


def load_config_from_file(file_path: Path) -> Dict[str, Any]:
    """
    Loads configuration from a file in YAML or JSON format.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary loaded from file
        
    Raises:
        FileError: If the file cannot be read or parsed
    """
    if not file_path.exists():
        raise FileError(f"Configuration file does not exist: {file_path}", file_path, "read")
    
    try:
        with open(file_path, 'r') as f:
            file_extension = file_path.suffix.lower()
            if file_extension in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_extension == '.json':
                config = json.load(f)
            else:
                raise FileError(
                    f"Unsupported configuration file format: {file_extension}", 
                    file_path, 
                    "read"
                )
        
        if not isinstance(config, dict):
            raise FileError(
                f"Invalid configuration format: expected dictionary, got {type(config).__name__}", 
                file_path, 
                "read"
            )
        
        logger.debug(f"Loaded configuration from {file_path}")
        return config
    
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise FileError(
            f"Error parsing configuration file: {str(e)}", 
            file_path, 
            "read",
            cause=e
        )
    except Exception as e:
        raise FileError(
            f"Error reading configuration file: {str(e)}", 
            file_path, 
            "read",
            cause=e
        )


def load_config_from_env() -> Dict[str, Any]:
    """
    Loads configuration from environment variables with RTLMP_PREDICT_ prefix.
    
    Returns:
        Dict[str, Any]: Configuration dictionary loaded from environment variables
    """
    config = {}
    prefix = "RTLMP_PREDICT_"
    
    # Look for environment variables with the prefix
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert from environment variable format to config key format
            # e.g., RTLMP_PREDICT_LOG_LEVEL -> log_level
            config_key = key[len(prefix):].lower()
            
            # Convert nested keys (using double underscore as separator)
            # e.g., RTLMP_PREDICT_TRAIN__MODEL_TYPE -> train.model_type
            if "__" in config_key:
                parts = config_key.split("__")
                current_dict = config
                for part in parts[:-1]:
                    if part not in current_dict:
                        current_dict[part] = {}
                    elif not isinstance(current_dict[part], dict):
                        current_dict[part] = {}
                    current_dict = current_dict[part]
                current_dict[parts[-1]] = value
            else:
                config[config_key] = value
    
    logger.debug(f"Loaded configuration from environment variables: {len(config)} variables")
    return config


def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merges multiple configuration dictionaries, with later dictionaries taking precedence.
    
    Args:
        configs: List of configuration dictionaries to merge
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        if not config:
            continue
            
        for key, value in config.items():
            # If the value is a dict and the key already exists in result as a dict,
            # recursively merge them
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = merge_configs([result[key], value])
            else:
                # Otherwise, simply override
                result[key] = value
    
    return result


def save_config_to_file(config: Dict[str, Any], file_path: Path) -> bool:
    """
    Saves a configuration dictionary to a file in YAML or JSON format.
    
    Args:
        config: Configuration dictionary to save
        file_path: Path where to save the configuration
        
    Returns:
        bool: True if the configuration was saved successfully, False otherwise
        
    Raises:
        FileError: If the file cannot be written
    """
    try:
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_extension = file_path.suffix.lower()
        
        with open(file_path, 'w') as f:
            if file_extension in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif file_extension == '.json':
                json.dump(config, f, indent=4, sort_keys=False)
            else:
                raise FileError(
                    f"Unsupported configuration file format: {file_extension}", 
                    file_path, 
                    "write"
                )
        
        logger.debug(f"Saved configuration to {file_path}")
        return True
        
    except Exception as e:
        raise FileError(
            f"Error writing configuration file: {str(e)}", 
            file_path, 
            "write",
            cause=e
        )


def load_cli_config(config_file: Optional[Path] = None) -> CLIConfigDict:
    """
    Loads and validates CLI configuration from file and environment variables.
    
    Args:
        config_file: Optional path to the configuration file
        
    Returns:
        CLIConfigDict: Validated CLI configuration dictionary
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    # Configuration sources in priority order (later overrides earlier)
    configs = []
    
    # Default configuration
    configs.append(get_default_cli_config())
    
    # File configuration
    if config_file:
        try:
            file_config = load_config_from_file(config_file)
            configs.append(file_config)
        except FileError as e:
            raise ConfigurationError(
                f"Error loading configuration file: {str(e)}",
                {"file_path": str(config_file)},
                cause=e
            )
    else:
        # Try to find a configuration file in default locations
        found_config_file = find_config_file()
        if found_config_file:
            try:
                file_config = load_config_from_file(found_config_file)
                configs.append(file_config)
            except FileError as e:
                logger.warning(f"Error loading found configuration file: {str(e)}")
    
    # Environment variables
    env_config = load_config_from_env()
    if env_config:
        configs.append(env_config)
    
    # Merge all configurations
    merged_config = merge_configs(configs)
    
    # Validate the merged configuration
    try:
        validated_config = validate_cli_config(merged_config)
        logger.debug("Configuration validation successful")
        return validated_config
    except ValidationError as e:
        raise ConfigurationError(
            f"Invalid configuration: {str(e)}",
            e.details,
            cause=e
        )


def load_command_config(
    command: CommandType,
    cli_config: Dict[str, Any],
    command_args: Optional[Dict[str, Any]] = None
) -> Union[
    FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict
]:
    """
    Loads and validates command-specific configuration.
    
    Args:
        command: Command type
        cli_config: CLI configuration dictionary
        command_args: Optional command-specific arguments that override configuration
        
    Returns:
        Union[FetchDataParamsDict, TrainParamsDict, PredictParamsDict, 
              BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict]: 
            Validated command-specific configuration dictionary
        
    Raises:
        ConfigurationError: If the command configuration is invalid
    """
    # Configuration sources in priority order (later overrides earlier)
    configs = []
    
    # Default command parameters
    default_params = get_default_command_params(command)
    configs.append(default_params)
    
    # Command-specific configuration from CLI config
    if command in cli_config:
        configs.append(cli_config[command])
    
    # Command arguments (highest priority)
    if command_args:
        configs.append(command_args)
    
    # Merge all configurations
    merged_config = merge_configs(configs)
    
    # Validate the merged configuration
    try:
        validated_config = validate_command_params(command, merged_config)
        logger.debug(f"Configuration validation successful for command: {command}")
        return validated_config
    except ValidationError as e:
        raise ConfigurationError(
            f"Invalid configuration for command {command}: {str(e)}",
            e.details,
            cause=e
        )


def create_default_config_file(config_path: Path) -> bool:
    """
    Creates a default configuration file at the specified location.
    
    Args:
        config_path: Path where to create the default configuration file
        
    Returns:
        bool: True if the file was created successfully, False otherwise
    """
    # Get default CLI configuration
    default_cli_config = get_default_cli_config()
    
    # Create a dictionary with default CLI config and command-specific configs
    default_config = default_cli_config.copy()
    
    # Add default configuration for each command
    for command in ["fetch-data", "train", "predict", "backtest", "evaluate", "visualize"]:
        command_type = cast(CommandType, command)
        default_config[command] = get_default_command_params(command_type)
    
    try:
        # Save the configuration to the specified file
        return save_config_to_file(default_config, config_path)
    except FileError as e:
        logger.error(f"Error creating default configuration file: {str(e)}")
        return False


class ConfigHelper:
    """
    Helper class for manipulating configuration dictionaries with support for nested keys.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ConfigHelper with a configuration dictionary.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config.copy()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the configuration using a dot-notation key.
        
        Args:
            key: Key in dot notation (e.g., "section.subsection.key")
            default: Default value to return if the key is not found
            
        Returns:
            Any: Value from the configuration or default if not found
        """
        parts = key.split('.')
        current = self._config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        
        return current
    
    def set_value(self, key: str, value: Any) -> None:
        """
        Sets a value in the configuration using a dot-notation key.
        
        Args:
            key: Key in dot notation (e.g., "section.subsection.key")
            value: Value to set
            
        Raises:
            ValueError: If the key path is invalid or cannot be created
        """
        parts = key.split('.')
        current = self._config
        
        # Navigate to the parent of the target key, creating intermediate dicts as needed
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Cannot create a nested path if an intermediate node is not a dict
                raise ValueError(
                    f"Cannot set key '{key}' because '{'.'.join(parts[:i+1])}' is not a dictionary"
                )
            current = current[part]
        
        # Set the value at the target key
        current[parts[-1]] = value
    
    def get_config(self) -> Dict[str, Any]:
        """
        Returns the entire configuration dictionary.
        
        Returns:
            Dict[str, Any]: The configuration dictionary
        """
        return self._config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Updates the configuration with a new dictionary.
        
        Args:
            new_config: New configuration dictionary to merge with the current one
        """
        self._config = merge_configs([self._config, new_config])


class ConfigManager:
    """
    Manages configuration loading, validation, and access for the CLI application.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initializes the ConfigManager with optional configuration file.
        
        Args:
            config_file: Optional path to the configuration file
        """
        self._cli_config = load_cli_config(config_file)
        self._command_configs: Dict[CommandType, Dict[str, Any]] = {}
        self._config_helper = ConfigHelper(self._cli_config)
    
    def get_cli_config(self) -> CLIConfigDict:
        """
        Returns the CLI configuration dictionary.
        
        Returns:
            CLIConfigDict: CLI configuration dictionary
        """
        return cast(CLIConfigDict, self._cli_config.copy())
    
    def get_command_config(
        self, 
        command: CommandType, 
        command_args: Optional[Dict[str, Any]] = None
    ) -> Union[
        FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
        BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict
    ]:
        """
        Returns the configuration for a specific command.
        
        Args:
            command: Command type
            command_args: Optional command-specific arguments that override configuration
            
        Returns:
            Union[FetchDataParamsDict, TrainParamsDict, PredictParamsDict, 
                  BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict]: 
                Command-specific configuration dictionary
        """
        # If we have arguments or haven't loaded this command's config yet, load it
        if command_args is not None or command not in self._command_configs:
            config = load_command_config(command, self._cli_config, command_args)
            self._command_configs[command] = config
        
        return self._command_configs[command]
    
    def reload_config(self, config_file: Optional[Path] = None) -> None:
        """
        Reloads the configuration from the specified file.
        
        Args:
            config_file: Optional path to the configuration file
        """
        self._cli_config = load_cli_config(config_file)
        self._command_configs.clear()
        self._config_helper = ConfigHelper(self._cli_config)
    
    def save_config(self, config_path: Path) -> bool:
        """
        Saves the current configuration to a file.
        
        Args:
            config_path: Path where to save the configuration
            
        Returns:
            bool: True if the configuration was saved successfully, False otherwise
        """
        # Create a combined configuration with CLI config and command configs
        combined_config = self._cli_config.copy()
        
        # Add command-specific configurations
        for command, config in self._command_configs.items():
            combined_config[command] = config
        
        try:
            return save_config_to_file(combined_config, config_path)
        except FileError as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the configuration with support for nested keys.
        
        Args:
            key: Key in dot notation (e.g., "section.subsection.key")
            default: Default value to return if the key is not found
            
        Returns:
            Any: Value from the configuration or default if not found
        """
        return self._config_helper.get_value(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """
        Sets a value in the configuration with support for nested keys.
        
        Args:
            key: Key in dot notation (e.g., "section.subsection.key")
            value: Value to set
        """
        self._config_helper.set_value(key, value)
        
        # Update CLI config if the key is part of it
        parts = key.split('.')
        if parts[0] in self._cli_config:
            self._cli_config = self._config_helper.get_config()
        
        # Update command configs if the key is part of any of them
        for command in self._command_configs:
            if parts[0] == command:
                self._command_configs[command] = self._config_helper.get_value(command, {})