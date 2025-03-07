"""
Initialization module for the Hydra configuration system in the ERCOT RTLMP spike prediction system.

This module provides utilities for initializing, loading, and managing Hydra-based
configurations, serving as the bridge between the application and the Hydra
configuration framework.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast

import hydra  # version 1.3+
from omegaconf import OmegaConf  # version 2.3+

# Internal imports
from ..validation import (
    validate_config, 
    load_yaml_config, 
    merge_configs, 
    validate_with_defaults
)
from ..default_config import DEFAULT_CONFIG, merge_with_defaults
from ..schema import Config

# Set up logger
logger = logging.getLogger(__name__)

# Default path to the main Hydra configuration file
CONFIG_PATH = Path(__file__).parent / 'config.yaml'

# List of public functions and variables
__all__ = ['initialize_config', 'load_config', 'get_config_value', 'save_config']


def initialize_config(config_path: Optional[str] = None, overrides: Optional[List[str]] = None) -> None:
    """
    Initializes the Hydra configuration system.
    
    Args:
        config_path: Optional path to the configuration file (defaults to CONFIG_PATH)
        overrides: Optional list of configuration overrides
        
    Returns:
        None
    """
    # Set default config_path if not provided
    if config_path is None:
        config_path = str(CONFIG_PATH)
    
    # Initialize empty overrides list if not provided
    if overrides is None:
        overrides = []
    
    # Configure Hydra to use the specified config_path
    hydra.initialize(config_path=config_path)
    
    # Set up Hydra's configuration store with default configuration options
    config_store = hydra.ConfigStore.instance()
    config_store.store(name="config_schema", node=Config)
    
    # Register schema validation for configuration objects
    logger.info(f"Hydra configuration system initialized with config_path: {config_path}")


def load_config(
    config_path: Optional[str] = None, 
    overrides: Optional[List[str]] = None,
    validate: bool = True
) -> Union[Dict[str, Any], Config]:
    """
    Loads configuration from Hydra YAML files with optional overrides.
    
    Args:
        config_path: Optional path to the configuration file (defaults to CONFIG_PATH)
        overrides: Optional list of configuration overrides
        validate: Whether to validate the configuration against schema
        
    Returns:
        Loaded configuration as dictionary or validated Config object
    """
    # Set default config_path if not provided
    if config_path is None:
        config_path = str(CONFIG_PATH)
    
    # Initialize empty overrides list if not provided
    if overrides is None:
        overrides = []
    
    # Initialize Hydra if not already initialized
    try:
        # Use Hydra's compose API to load and compose configuration
        config = hydra.compose(config_name="config", overrides=overrides)
    except Exception as e:
        logger.debug(f"Initializing Hydra: {e}")
        initialize_config(config_path, overrides)
        config = hydra.compose(config_name="config", overrides=overrides)
    
    # Convert OmegaConf container to dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        config_dict = {}
    
    # Merge with default configuration for any missing values
    config_dict = merge_configs([DEFAULT_CONFIG, config_dict])
    
    # If validate is True, validate the configuration using schema
    if validate:
        result = validate_config(config_dict, Config)
        return result
    
    # Return either the raw dictionary or validated Config object
    return config_dict


def get_config_value(config: Dict[str, Any], key_path: str, default_value: Any = None) -> Any:
    """
    Retrieves a specific value from the configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-notated path to the value (e.g., "system.log_level")
        default_value: Value to return if the key doesn't exist
        
    Returns:
        The value at the specified key path or the default value if not found
    """
    # Split the key_path by dots to get the nested keys
    keys = key_path.split('.')
    
    # Start with the root config dictionary
    current = config
    
    # Traverse the nested dictionaries following the key path
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default_value
        current = current[key]
    
    # Return the value found at the specified key path
    return current


def save_config(config: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Saves a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        file_path: Path to save the configuration to
        
    Returns:
        True if saving was successful, False otherwise
    """
    # Convert file_path to Path object if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert dictionary to OmegaConf container
        conf = OmegaConf.create(config)
        
        # Save the configuration to the specified file path using OmegaConf.save
        OmegaConf.save(conf, str(file_path))
        
        # Log successful saving of configuration
        logger.info(f"Configuration saved to {file_path}")
        return True
    except Exception as e:
        # Return True on success, catch and log exceptions and return False on failure
        logger.error(f"Failed to save configuration to {file_path}: {e}")
        return False