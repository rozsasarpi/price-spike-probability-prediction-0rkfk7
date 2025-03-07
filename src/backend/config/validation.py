"""
Provides validation utilities for configuration in the ERCOT RTLMP spike prediction system.

This module implements functions to validate configuration dictionaries against schema models,
load and validate YAML configurations, and merge configurations with defaults.
"""

from typing import Dict, List, Any, Optional, Union, Type, TypeVar, cast
from pathlib import Path
from copy import deepcopy
import yaml  # version 6.0+
from pydantic import ValidationError  # version 2.0+

# Internal imports
from .schema import (
    SystemConfig, PathsConfig, DataConfig, FeatureConfig, 
    ModelConfig, InferenceConfig, VisualizationConfig, Config
)
from .default_config import (
    DEFAULT_CONFIG, DEFAULT_SYSTEM_CONFIG, DEFAULT_PATHS_CONFIG,
    DEFAULT_DATA_CONFIG, DEFAULT_FEATURE_CONFIG, DEFAULT_MODEL_CONFIG,
    DEFAULT_INFERENCE_CONFIG, DEFAULT_VISUALIZATION_CONFIG
)
from ..utils.logging import get_logger
from ..utils.error_handling import handle_errors, DataFormatError

# Set up logger
logger = get_logger(__name__)

# Map config section names to schema classes
CONFIG_SCHEMA_MAP: Dict[str, Type] = {
    "system": SystemConfig,
    "paths": PathsConfig,
    "data": DataConfig,
    "features": FeatureConfig,
    "models": ModelConfig,
    "inference": InferenceConfig,
    "visualization": VisualizationConfig
}


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration validation error.
        
        Args:
            message: Error message describing the validation failure
            context: Optional dictionary with additional context about the error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the error to a dictionary representation.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "error_type": "ConfigValidationError",
            "message": self.message,
            "context": self.context
        }


@handle_errors(ValidationError, None, False, 'Configuration validation failed')
def validate_config(config: Dict[str, Any], schema_class: Optional[Type] = None) -> Union[Dict[str, Any], Any]:
    """
    Validates a configuration dictionary against a schema model.
    
    Args:
        config: Configuration dictionary to validate
        schema_class: Optional Pydantic model class to validate against
        
    Returns:
        Validated configuration object or dictionary
        
    Raises:
        ConfigValidationError: If validation fails and reraise is True
    """
    if not config:
        logger.warning("Empty configuration provided for validation")
        return {}
        
    # If no schema class is provided, try to determine from config structure
    if schema_class is None:
        if len(config) == 1 and next(iter(config)) in CONFIG_SCHEMA_MAP:
            # Config has a single top-level key that corresponds to a section
            section = next(iter(config))
            schema_class = CONFIG_SCHEMA_MAP[section]
            return {section: validate_config(config[section], schema_class)}
        else:
            # Assume this is the root config
            schema_class = Config
            logger.debug(f"No schema class provided, using {schema_class.__name__}")
    
    # Create an instance of the schema class with the config dictionary
    logger.debug(f"Validating config against {schema_class.__name__}")
    validated = schema_class(**config)
    
    # Return the validated model 
    return validated


@handle_errors((FileNotFoundError, yaml.YAMLError), None, False, 'Failed to load configuration file')
def load_yaml_config(file_path: Union[str, Path], validate: bool = True, 
                    schema_class: Optional[Type] = None) -> Union[Dict[str, Any], Any]:
    """
    Loads a YAML configuration file with optional validation.
    
    Args:
        file_path: Path to the YAML configuration file
        validate: Whether to validate the loaded configuration
        schema_class: Optional schema class to validate against
        
    Returns:
        Loaded and optionally validated configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
        ConfigValidationError: If validation fails and reraise is True
    """
    # Convert file_path to Path object if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    # Open and read the YAML file
    logger.debug(f"Loading configuration from {file_path}")
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate the configuration if requested
    if validate:
        config = validate_config(config, schema_class)
    
    return config


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries with the second taking precedence.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary, values override dict1
        
    Returns:
        Merged dictionary
    """
    # Create a deep copy of dict1 to avoid modifying the original
    result = deepcopy(dict1)
    
    for key, value in dict2.items():
        # If the key exists in both dicts and both values are dicts, merge them
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            # Otherwise, simply overwrite the value in result
            result[key] = deepcopy(value)
    
    return result


def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merges multiple configuration dictionaries with later ones taking precedence.
    
    Args:
        configs: List of configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    if not configs:
        return {}
    
    # Start with an empty result
    result: Dict[str, Any] = {}
    
    # Merge each config into the result
    for config in configs:
        if not config:
            continue
            
        result = deep_merge(result, config)
    
    return result


def validate_with_defaults(config: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None,
                          schema_class: Optional[Type] = None) -> Union[Dict[str, Any], Any]:
    """
    Validates a configuration with fallback to default values for missing fields.
    
    Args:
        config: Configuration dictionary to validate
        defaults: Optional default configuration to use as fallback
        schema_class: Optional schema class to validate against
        
    Returns:
        Validated configuration with defaults applied
    """
    # Use the provided defaults or fall back to DEFAULT_CONFIG
    if defaults is None:
        defaults = DEFAULT_CONFIG
    
    # Merge defaults with provided config
    merged_config = merge_configs([defaults, config])
    
    # Validate the merged configuration
    return validate_config(merged_config, schema_class)


def get_config_schema(section_name: str) -> Optional[Type]:
    """
    Gets the appropriate schema class for a configuration section.
    
    Args:
        section_name: Name of the configuration section
        
    Returns:
        Schema class for the specified section or None if not found
    """
    schema_class = CONFIG_SCHEMA_MAP.get(section_name)
    
    if schema_class is None:
        logger.warning(f"No schema found for section: {section_name}")
    
    return schema_class


def get_validated_config(file_path: Union[str, Path], apply_defaults: bool = True) -> Config:
    """
    Gets a validated configuration by loading and validating from a file.
    
    Args:
        file_path: Path to the configuration file
        apply_defaults: Whether to apply default values for missing fields
        
    Returns:
        Validated configuration object
    """
    # Load the configuration from file
    config = load_yaml_config(file_path, validate=False)
    
    # Apply defaults if requested
    if apply_defaults:
        result = validate_with_defaults(config, DEFAULT_CONFIG, Config)
    else:
        result = validate_config(config, Config)
    
    # Ensure we return a Config object
    if not isinstance(result, Config):
        result = Config(**result)
    
    return result