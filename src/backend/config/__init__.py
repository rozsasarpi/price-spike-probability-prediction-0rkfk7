"""
Initialization module for the configuration system in the ERCOT RTLMP spike prediction system.

This file serves as the main entry point for configuration functionality, exposing schema models,
validation utilities, default configurations, and Hydra integration to the rest of the application.
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Internal imports - schema models
from .schema import (
    SystemConfig, PathsConfig, DataConfig, FeatureConfig, 
    ModelConfig, InferenceConfig, VisualizationConfig, Config
)

# Internal imports - default configuration
from .default_config import (
    DEFAULT_CONFIG, DEFAULT_SYSTEM_CONFIG, DEFAULT_PATHS_CONFIG,
    DEFAULT_DATA_CONFIG, DEFAULT_FEATURE_CONFIG, DEFAULT_MODEL_CONFIG,
    DEFAULT_INFERENCE_CONFIG, DEFAULT_VISUALIZATION_CONFIG,
    get_default_config
)

# Internal imports - validation utilities
from .validation import (
    validate_config, load_yaml_config, merge_configs,
    validate_with_defaults, get_validated_config, ConfigValidationError
)

# Internal imports - hydra integration
from .hydra import (
    initialize_config, load_config, get_config_value, save_config
)

# Set up logger
logger = logging.getLogger(__name__)

# Module version
__version__ = "0.1.0"

# List of exported symbols
__all__ = [
    # Schema models
    "SystemConfig", "PathsConfig", "DataConfig", "FeatureConfig",
    "ModelConfig", "InferenceConfig", "VisualizationConfig", "Config",
    
    # Default configurations
    "DEFAULT_CONFIG", "DEFAULT_SYSTEM_CONFIG", "DEFAULT_PATHS_CONFIG",
    "DEFAULT_DATA_CONFIG", "DEFAULT_FEATURE_CONFIG", "DEFAULT_MODEL_CONFIG", 
    "DEFAULT_INFERENCE_CONFIG", "DEFAULT_VISUALIZATION_CONFIG",
    
    # Validation utilities
    "validate_config", "load_yaml_config", "merge_configs",
    "validate_with_defaults", "get_validated_config", "ConfigValidationError",
    
    # Hydra integration
    "initialize_config", "load_config", "get_config_value", "save_config",
    
    # Module functions
    "load_configuration", "get_configuration"
]

# Global variables
_config_cache = None

def load_configuration(
    config_path: str,
    overrides: List[str] = None,
    use_hydra: bool = True,
    validate: bool = True
) -> Union[Dict[str, Any], Config]:
    """
    Loads and validates configuration from a file with optional overrides.
    
    Args:
        config_path: Path to the configuration file
        overrides: Optional list of configuration overrides
        use_hydra: Whether to use Hydra for loading configuration
        validate: Whether to validate the configuration
        
    Returns:
        Loaded and validated configuration
    """
    if overrides is None:
        overrides = []
        
    logger.info(f"Loading configuration from {config_path}")
    
    if use_hydra:
        # Use Hydra to load configuration
        config = load_config(config_path, overrides, validate)
    else:
        # Use standard YAML loading
        config = load_yaml_config(config_path, validate=False)
        
        # Apply default values for missing fields
        config = validate_with_defaults(config, DEFAULT_CONFIG)
        
        # Validate if requested
        if validate:
            config = validate_config(config, Config)
            
    return config

def get_configuration(
    config_path: Optional[str] = None,
    reload: bool = False,
    validate: bool = True
) -> Union[Dict[str, Any], Config]:
    """
    Gets the current configuration or loads a new one if not available.
    
    Args:
        config_path: Path to the configuration file (optional)
        reload: Whether to force reloading the configuration
        validate: Whether to validate the configuration
        
    Returns:
        Current or newly loaded configuration
    """
    global _config_cache
    
    # Return cached configuration if available and reload is not requested
    if _config_cache is not None and not reload:
        return _config_cache
    
    # Load new configuration if needed
    if config_path is not None:
        config = load_configuration(config_path, validate=validate)
        _config_cache = config
        return config
    
    # If no path provided and no cache, return default configuration
    if _config_cache is None:
        config = DEFAULT_CONFIG
        if validate:
            config = validate_config(config, Config)
        _config_cache = config
        
    return _config_cache