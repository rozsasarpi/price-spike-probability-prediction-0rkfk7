"""
Implements configuration loading, validation, and management for the CLI application of the ERCOT RTLMP spike prediction system.

This module provides functions to load configuration from files and environment variables,
validate configuration against schemas, and handle configuration-related operations
for both global CLI settings and command-specific parameters.
"""

from typing import Dict, List, Any, Optional, Union, cast
from pathlib import Path
import os
import logging
import yaml  # version: 6.0
import json

from ..cli_types import (
    CLIConfigDict, FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict,
    CommandType, LogLevel, DataType, VisualizationType, OutputFormat
)
from .default_config import (
    get_default_cli_config, get_default_command_params,
    merge_with_cli_defaults, merge_with_command_defaults
)
from ..exceptions import ConfigurationError, ValidationError, FileError

# Set up logger
logger = logging.getLogger(__name__)

# Default locations to search for configuration files
DEFAULT_CONFIG_LOCATIONS = [
    Path("./rtlmp_predict.yaml"),  # Current directory
    Path("./rtlmp_predict.yml"),
    Path("./rtlmp_predict.json"),
    Path("~/.rtlmp_predict.yaml").expanduser(),  # User home directory
    Path("~/.rtlmp_predict.yml").expanduser(),
    Path("~/.rtlmp_predict.json").expanduser(),
    Path("/etc/rtlmp_predict.yaml"),  # System-wide configuration
    Path("/etc/rtlmp_predict.yml"),
    Path("/etc/rtlmp_predict.json"),
]

# Schema for CLI configuration validation
CLI_CONFIG_SCHEMA = {
    "config_file": {"type": "optional_path", "required": False},
    "log_level": {"type": "log_level", "required": False},
    "log_file": {"type": "optional_path", "required": False},
    "output_dir": {"type": "optional_path", "required": False},
    "verbose": {"type": "bool", "required": True},
}

# Schemas for command-specific configuration validation
COMMAND_CONFIG_SCHEMAS = {
    "fetch-data": {
        "data_type": {"type": "data_type", "required": True},
        "start_date": {"type": "date", "required": True},
        "end_date": {"type": "date", "required": True},
        "nodes": {"type": "node_list", "required": True},
        "output_path": {"type": "optional_path", "required": False},
        "output_format": {"type": "output_format", "required": False},
        "force_refresh": {"type": "bool", "required": True},
    },
    "train": {
        "start_date": {"type": "date", "required": True},
        "end_date": {"type": "date", "required": True},
        "nodes": {"type": "node_list", "required": True},
        "thresholds": {"type": "threshold_list", "required": True},
        "model_type": {"type": "model_type", "required": True},
        "hyperparameters": {"type": "dict", "required": False},
        "optimize_hyperparameters": {"type": "bool", "required": True},
        "cross_validation_folds": {"type": "int", "required": True},
        "model_name": {"type": "optional_string", "required": False},
        "output_path": {"type": "optional_path", "required": False},
    },
    "predict": {
        "threshold": {"type": "threshold", "required": True},
        "nodes": {"type": "node_list", "required": True},
        "model_version": {"type": "optional_string", "required": False},
        "output_path": {"type": "optional_path", "required": False},
        "output_format": {"type": "output_format", "required": False},
        "visualize": {"type": "bool", "required": True},
    },
    "backtest": {
        "start_date": {"type": "date", "required": True},
        "end_date": {"type": "date", "required": True},
        "thresholds": {"type": "threshold_list", "required": True},
        "nodes": {"type": "node_list", "required": True},
        "model_version": {"type": "optional_string", "required": False},
        "output_path": {"type": "optional_path", "required": False},
        "output_format": {"type": "output_format", "required": False},
        "visualize": {"type": "bool", "required": True},
    },
    "evaluate": {
        "model_version": {"type": "optional_string", "required": False},
        "compare_with": {"type": "optional_string_list", "required": False},
        "thresholds": {"type": "threshold_list", "required": True},
        "nodes": {"type": "node_list", "required": True},
        "start_date": {"type": "optional_date", "required": False},
        "end_date": {"type": "optional_date", "required": False},
        "output_path": {"type": "optional_path", "required": False},
        "output_format": {"type": "output_format", "required": False},
        "visualize": {"type": "bool", "required": True},
    },
    "visualize": {
        "visualization_type": {"type": "visualization_type", "required": True},
        "forecast_id": {"type": "optional_string", "required": False},
        "model_version": {"type": "optional_string", "required": False},
        "compare_with": {"type": "optional_string_list", "required": False},
        "threshold": {"type": "optional_threshold", "required": False},
        "nodes": {"type": "optional_node_list", "required": False},
        "start_date": {"type": "optional_date", "required": False},
        "end_date": {"type": "optional_date", "required": False},
        "output_path": {"type": "optional_path", "required": False},
        "output_format": {"type": "output_format", "required": False},
        "interactive": {"type": "bool", "required": True},
    }
}


def find_config_file() -> Optional[Path]:
    """
    Searches for a configuration file in default locations.
    
    Returns:
        Optional[Path]: Path to the found configuration file, or None if not found
    """
    for path in DEFAULT_CONFIG_LOCATIONS:
        if path.exists() and path.is_file():
            logger.debug(f"Found configuration file at {path}")
            return path
    
    logger.debug("No configuration file found in default locations")
    return None


def validate_cli_config(config: Dict[str, Any]) -> bool:
    """
    Validates a CLI configuration dictionary against the schema.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if the configuration is valid, False otherwise
    """
    # Check all required fields
    for field, field_schema in CLI_CONFIG_SCHEMA.items():
        if field_schema["required"] and field not in config:
            logger.error(f"Missing required field '{field}' in CLI configuration")
            return False
        
        # Skip validation for missing optional fields
        if field not in config:
            continue
            
        # Validate field type
        field_type = field_schema["type"]
        value = config[field]
        
        if field_type == "optional_path":
            if value is not None and not isinstance(value, (str, Path)):
                logger.error(f"Field '{field}' must be a Path or string, got {type(value)}")
                return False
        elif field_type == "log_level":
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if value is not None and value not in valid_levels:
                logger.error(f"Field '{field}' must be one of {valid_levels}, got {value}")
                return False
        elif field_type == "bool":
            if not isinstance(value, bool):
                logger.error(f"Field '{field}' must be a boolean, got {type(value)}")
                return False
                
    return True


def validate_command_config(command: CommandType, config: Dict[str, Any]) -> bool:
    """
    Validates a command-specific configuration dictionary against its schema.
    
    Args:
        command: The command type
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if the configuration is valid, False otherwise
        
    Raises:
        ValueError: If an invalid command is provided
    """
    if command not in COMMAND_CONFIG_SCHEMAS:
        raise ValueError(f"Invalid command: {command}")
        
    schema = COMMAND_CONFIG_SCHEMAS[command]
    
    # Check all required fields
    for field, field_schema in schema.items():
        if field_schema["required"] and field not in config:
            logger.error(f"Missing required field '{field}' in '{command}' command configuration")
            return False
            
        # Skip validation for missing optional fields
        if field not in config:
            continue
            
        # Validate field type
        field_type = field_schema["type"]
        value = config[field]
        
        # Validate based on field type
        if field_type == "data_type":
            valid_types = ["rtlmp", "weather", "grid_conditions", "all"]
            if value not in valid_types:
                logger.error(f"Field '{field}' must be one of {valid_types}, got {value}")
                return False
        elif field_type == "date":
            if not hasattr(value, "year") or not hasattr(value, "month") or not hasattr(value, "day"):
                logger.error(f"Field '{field}' must be a date, got {type(value)}")
                return False
        elif field_type == "optional_date":
            if value is not None and (not hasattr(value, "year") or not hasattr(value, "month") or not hasattr(value, "day")):
                logger.error(f"Field '{field}' must be a date or None, got {type(value)}")
                return False
        elif field_type == "node_list":
            if not isinstance(value, list) or not all(isinstance(node, str) for node in value):
                logger.error(f"Field '{field}' must be a list of strings, got {type(value)}")
                return False
        elif field_type == "optional_node_list":
            if value is not None and (not isinstance(value, list) or not all(isinstance(node, str) for node in value)):
                logger.error(f"Field '{field}' must be a list of strings or None, got {type(value)}")
                return False
        elif field_type == "threshold":
            if not isinstance(value, (int, float)) or value <= 0:
                logger.error(f"Field '{field}' must be a positive number, got {value}")
                return False
        elif field_type == "optional_threshold":
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                logger.error(f"Field '{field}' must be a positive number or None, got {value}")
                return False
        elif field_type == "threshold_list":
            if not isinstance(value, list) or not all(isinstance(t, (int, float)) and t > 0 for t in value):
                logger.error(f"Field '{field}' must be a list of positive numbers, got {type(value)}")
                return False
        elif field_type == "model_type":
            valid_types = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]
            if value not in valid_types:
                logger.error(f"Field '{field}' must be one of {valid_types}, got {value}")
                return False
        elif field_type == "dict":
            if value is not None and not isinstance(value, dict):
                logger.error(f"Field '{field}' must be a dictionary or None, got {type(value)}")
                return False
        elif field_type == "optional_string":
            if value is not None and not isinstance(value, str):
                logger.error(f"Field '{field}' must be a string or None, got {type(value)}")
                return False
        elif field_type == "optional_string_list":
            if value is not None and (not isinstance(value, list) or not all(isinstance(s, str) for s in value)):
                logger.error(f"Field '{field}' must be a list of strings or None, got {type(value)}")
                return False
        elif field_type == "optional_path":
            if value is not None and not isinstance(value, (str, Path)):
                logger.error(f"Field '{field}' must be a Path or string or None, got {type(value)}")
                return False
        elif field_type == "output_format":
            valid_formats = ["text", "json", "csv", "html", "png"]
            if value is not None and value not in valid_formats:
                logger.error(f"Field '{field}' must be one of {valid_formats}, got {value}")
                return False
        elif field_type == "visualization_type":
            valid_types = ["forecast", "performance", "calibration", "feature_importance", "roc_curve", "precision_recall"]
            if value not in valid_types:
                logger.error(f"Field '{field}' must be one of {valid_types}, got {value}")
                return False
        elif field_type == "bool":
            if not isinstance(value, bool):
                logger.error(f"Field '{field}' must be a boolean, got {type(value)}")
                return False
        elif field_type == "int":
            if not isinstance(value, int):
                logger.error(f"Field '{field}' must be an integer, got {type(value)}")
                return False
                
    return True


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
        raise FileError(f"Configuration file not found: {file_path}", file_path, "read")
        
    try:
        # Determine file format based on extension
        if file_path.suffix in [".yaml", ".yml"]:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
        elif file_path.suffix == ".json":
            with open(file_path, "r") as f:
                config = json.load(f)
        else:
            raise FileError(f"Unsupported file format: {file_path.suffix}", file_path, "read")
            
        # Validate that the loaded configuration is a dictionary
        if not isinstance(config, dict):
            raise FileError("Invalid configuration format: expected a dictionary", file_path, "read")
            
        logger.debug(f"Loaded configuration from {file_path}")
        return config
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise FileError(f"Error parsing configuration file: {e}", file_path, "read", e)
    except Exception as e:
        raise FileError(f"Error reading configuration file: {e}", file_path, "read", e)


def load_config_from_env() -> Dict[str, Any]:
    """
    Loads configuration from environment variables.
    
    Environment variables are expected to have the prefix RTLMP_PREDICT_
    and use underscores to represent nested configuration keys.
    
    Returns:
        Dict[str, Any]: Configuration dictionary loaded from environment variables
    """
    config = {}
    prefix = "RTLMP_PREDICT_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert environment variable key to configuration key
            config_key = key[len(prefix):].lower()
            
            # Convert value to appropriate type
            if value.lower() == "true":
                config_value = True
            elif value.lower() == "false":
                config_value = False
            elif value.lower() == "none":
                config_value = None
            elif value.isdigit():
                config_value = int(value)
            elif value.replace(".", "", 1).isdigit():
                config_value = float(value)
            else:
                config_value = value
                
            # Support nested keys using '__' separator
            parts = config_key.split("__")
            if len(parts) > 1:
                current_dict = config
                for part in parts[:-1]:
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]
                current_dict[parts[-1]] = config_value
            else:
                config[config_key] = config_value
    
    if config:
        logger.debug(f"Loaded configuration from environment variables: {list(config.keys())}")
    
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
        for key, value in config.items():
            # If both values are dictionaries, merge recursively
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs([result[key], value])
            else:
                # Otherwise, replace the value
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
        
        # Determine file format based on extension
        if file_path.suffix in [".yaml", ".yml"]:
            with open(file_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif file_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(config, f, indent=2)
        else:
            raise FileError(f"Unsupported file format: {file_path.suffix}", file_path, "write")
            
        logger.info(f"Configuration saved to {file_path}")
        return True
    except Exception as e:
        error_msg = f"Error saving configuration to {file_path}: {e}"
        logger.error(error_msg)
        raise FileError(error_msg, file_path, "write", e)


def load_cli_config(config_file: Optional[Path] = None) -> CLIConfigDict:
    """
    Loads and validates CLI configuration from file and environment variables.
    
    Args:
        config_file: Optional path to a specific configuration file
        
    Returns:
        CLIConfigDict: Validated CLI configuration dictionary
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    configs_to_merge = []
    
    # Start with default configuration
    configs_to_merge.append(get_default_cli_config())
    
    # Load from file if provided or found
    file_config = {}
    if config_file is not None:
        try:
            file_config = load_config_from_file(config_file)
            configs_to_merge.append(file_config)
            logger.info(f"Loaded configuration from specified file: {config_file}")
        except FileError as e:
            raise ConfigurationError(f"Error loading configuration file: {e}", {"file_path": str(config_file)}, e)
    else:
        # Try to find a configuration file in default locations
        found_config_file = find_config_file()
        if found_config_file is not None:
            try:
                file_config = load_config_from_file(found_config_file)
                configs_to_merge.append(file_config)
                logger.info(f"Loaded configuration from found file: {found_config_file}")
            except FileError as e:
                logger.warning(f"Error loading found configuration file: {e}")
    
    # Load from environment variables
    env_config = load_config_from_env()
    if env_config:
        configs_to_merge.append(env_config)
    
    # Merge configurations with priority: defaults < file < environment
    merged_config = merge_configs(configs_to_merge)
    
    # Validate the configuration
    if not validate_cli_config(merged_config):
        raise ConfigurationError("Invalid CLI configuration", merged_config)
    
    return cast(CLIConfigDict, merged_config)


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
        command: Command type to load configuration for
        cli_config: CLI configuration dictionary that may contain command-specific configuration
        command_args: Optional command-line arguments to override configuration
        
    Returns:
        Union[FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
            BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict]:
            Validated command-specific configuration dictionary
            
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    configs_to_merge = []
    
    # Start with default command parameters
    configs_to_merge.append(get_default_command_params(command))
    
    # Extract command-specific configuration from CLI config if present
    if command in cli_config:
        configs_to_merge.append(cli_config[command])
    
    # Add command-line arguments if provided
    if command_args is not None:
        configs_to_merge.append(command_args)
    
    # Merge configurations with priority: defaults < cli_config < command_args
    merged_config = merge_configs(configs_to_merge)
    
    # Validate the configuration
    if not validate_command_config(command, merged_config):
        raise ConfigurationError(f"Invalid configuration for command '{command}'", merged_config)
    
    # Return the appropriate type based on the command
    if command == "fetch-data":
        return cast(FetchDataParamsDict, merged_config)
    elif command == "train":
        return cast(TrainParamsDict, merged_config)
    elif command == "predict":
        return cast(PredictParamsDict, merged_config)
    elif command == "backtest":
        return cast(BacktestParamsDict, merged_config)
    elif command == "evaluate":
        return cast(EvaluateParamsDict, merged_config)
    elif command == "visualize":
        return cast(VisualizeParamsDict, merged_config)
    else:
        # This should never happen as we validate the command earlier
        raise ValueError(f"Invalid command: {command}")


def create_default_config_file(config_path: Path) -> bool:
    """
    Creates a default configuration file at the specified location.
    
    Args:
        config_path: Path where to save the default configuration
        
    Returns:
        bool: True if the file was created successfully, False otherwise
    """
    try:
        # Get default CLI configuration
        cli_config = get_default_cli_config()
        
        # Create a comprehensive default configuration with all commands
        complete_config = {
            **cli_config,
            "fetch-data": get_default_command_params("fetch-data"),
            "train": get_default_command_params("train"),
            "predict": get_default_command_params("predict"),
            "backtest": get_default_command_params("backtest"),
            "evaluate": get_default_command_params("evaluate"),
            "visualize": get_default_command_params("visualize"),
        }
        
        # Save the configuration to the specified path
        save_config_to_file(complete_config, config_path)
        logger.info(f"Created default configuration file at {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating default configuration file: {e}")
        return False


class ConfigHelper:
    """
    Helper class for manipulating configuration dictionaries with support for nested keys.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ConfigHelper with a configuration dictionary.
        
        Args:
            config: Configuration dictionary to manipulate
        """
        self._config = config.copy()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the configuration using a dot-notation key.
        
        Args:
            key: Key in dot notation (e.g., "train.hyperparameters.learning_rate")
            default: Default value to return if the key is not found
            
        Returns:
            Any: Value from the configuration or default if not found
        """
        parts = key.split(".")
        current = self._config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
    
    def set_value(self, key: str, value: Any) -> None:
        """
        Sets a value in the configuration using a dot-notation key.
        
        Args:
            key: Key in dot notation (e.g., "train.hyperparameters.learning_rate")
            value: Value to set
            
        Raises:
            ValueError: If the key path cannot be created
        """
        parts = key.split(".")
        current = self._config
        
        # Navigate to the parent of the target key
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            
            if not isinstance(current[part], dict):
                path = ".".join(parts[:i+1])
                raise ValueError(f"Cannot set '{key}': '{path}' is not a dictionary")
                
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
            config_file: Optional path to a specific configuration file
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
            command: Command type to get configuration for
            command_args: Optional command-line arguments to override configuration
            
        Returns:
            Union[FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
                BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict]:
                Command-specific configuration dictionary
        """
        # If command_args is provided or we don't have the config cached, load it
        if command_args is not None or command not in self._command_configs:
            config = load_command_config(command, self._cli_config, command_args)
            self._command_configs[command] = config
            return config
            
        # Return the cached configuration
        return self._command_configs[command]
    
    def reload_config(self, config_file: Optional[Path] = None) -> None:
        """
        Reloads the configuration from the specified file.
        
        Args:
            config_file: Optional path to a specific configuration file
        """
        self._cli_config = load_cli_config(config_file)
        self._command_configs = {}  # Clear cached command configs
        self._config_helper = ConfigHelper(self._cli_config)
    
    def save_config(self, config_path: Path) -> bool:
        """
        Saves the current configuration to a file.
        
        Args:
            config_path: Path where to save the configuration
            
        Returns:
            bool: True if the configuration was saved successfully, False otherwise
        """
        # Create a combined configuration dictionary
        combined_config = {
            **self._cli_config,
            **{command: config for command, config in self._command_configs.items()}
        }
        
        # Save the configuration to the specified path
        return save_config_to_file(combined_config, config_path)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the configuration with support for nested keys.
        
        Args:
            key: Key in dot notation (e.g., "train.hyperparameters.learning_rate")
            default: Default value to return if the key is not found
            
        Returns:
            Any: Value from the configuration or default if not found
        """
        return self._config_helper.get_value(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """
        Sets a value in the configuration with support for nested keys.
        
        Args:
            key: Key in dot notation (e.g., "train.hyperparameters.learning_rate")
            value: Value to set
        """
        self._config_helper.set_value(key, value)
        
        # Update CLI config or command configs as needed
        parts = key.split(".")
        if len(parts) > 0:
            if parts[0] in ["fetch-data", "train", "predict", "backtest", "evaluate", "visualize"]:
                # Update command config if it exists
                if parts[0] in self._command_configs:
                    command_config = self._command_configs[parts[0]]
                    current = command_config
                    
                    for i, part in enumerate(parts[1:-1]):
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    current[parts[-1]] = value
            else:
                # Update CLI config
                self._cli_config = self._config_helper.get_config()