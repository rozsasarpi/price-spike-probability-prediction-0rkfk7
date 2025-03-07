"""
Configuration module for the ERCOT RTLMP spike prediction system CLI.

This module exports configuration management functionality, making key components
available to other parts of the CLI application. It provides functions and classes
for loading, validating, and managing CLI configurations and command parameters.
"""

from .default_config import (
    get_default_cli_config, get_default_command_params,
    merge_with_cli_defaults, merge_with_command_defaults,
    DEFAULT_CLI_CONFIG, DEFAULT_FETCH_DATA_PARAMS, DEFAULT_TRAIN_PARAMS,
    DEFAULT_PREDICT_PARAMS, DEFAULT_BACKTEST_PARAMS, DEFAULT_EVALUATE_PARAMS,
    DEFAULT_VISUALIZE_PARAMS
)

from .cli_config import (
    find_config_file, load_cli_config, load_command_config,
    load_config_from_file, load_config_from_env, merge_configs,
    save_config_to_file, create_default_config_file,
    ConfigHelper, ConfigManager,
    validate_cli_config, validate_command_config
)

__all__ = [
    # Default configuration functions and constants
    'get_default_cli_config',
    'get_default_command_params',
    'merge_with_cli_defaults',
    'merge_with_command_defaults',
    'DEFAULT_CLI_CONFIG',
    'DEFAULT_FETCH_DATA_PARAMS',
    'DEFAULT_TRAIN_PARAMS',
    'DEFAULT_PREDICT_PARAMS',
    'DEFAULT_BACKTEST_PARAMS',
    'DEFAULT_EVALUATE_PARAMS',
    'DEFAULT_VISUALIZE_PARAMS',
    
    # Configuration management functions and classes
    'find_config_file',
    'load_cli_config',
    'load_command_config',
    'load_config_from_file',
    'load_config_from_env',
    'merge_configs',
    'save_config_to_file',
    'create_default_config_file',
    'ConfigHelper',
    'ConfigManager',
    'validate_cli_config',
    'validate_command_config'
]