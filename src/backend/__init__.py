"""
Main initialization file for the ERCOT RTLMP spike prediction system backend package.
This file serves as the entry point for the entire backend system, exposing key components,
interfaces, and functionality from all submodules to provide a unified API for the prediction system.
"""

import logging  # Standard
from typing import Dict, List, Any, Optional, Union  # Standard

# Internal imports
from .utils import *  # src/backend/utils/__init__.py
from .config import *  # src/backend/config/__init__.py
from .data import *  # src/backend/data/__init__.py
from .features import *  # src/backend/features/__init__.py
from .models import *  # src/backend/models/__init__.py
from .inference import *  # src/backend/inference/__init__.py
from .backtesting import *  # src/backend/backtesting/__init__.py
from .visualization import *  # src/backend/visualization/__init__.py
from .orchestration import *  # src/backend/orchestration/__init__.py
from .api import *  # src/backend/api/__init__.py

# Set up logger
logger = logging.getLogger(__name__)

# Version information
__version__ = "0.1.0"
__author__ = "ERCOT RTLMP Prediction Team"
__description__ = "Backend package for ERCOT RTLMP spike prediction system"

# Expose version and author information
__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "initialize_system",
    "DataAPI",
    "ModelAPI",
    "InferenceAPI",
    "BacktestingAPI",
    "VisualizationAPI",
    "Config",
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    "EnsembleModel",
    "InferenceEngine",
    "BacktestingFramework",
    "FeaturePipeline",
    "Pipeline",
    "Scheduler",
    "DEFAULT_FORECAST_HORIZON",
    "DEFAULT_THRESHOLDS",
]


def initialize_system(config_path: Optional[str] = None, initialize_logging: bool = True) -> bool:
    """
    Initialize the ERCOT RTLMP spike prediction system with default configuration

    Args:
        config_path (Optional[str]): Path to a custom configuration file. If None, default configuration is used.
        initialize_logging (bool): Whether to set up logging.

    Returns:
        bool: True if initialization was successful
    """
    try:
        # Set up logging if initialize_logging is True
        if initialize_logging:
            setup_logging()
            logger.info("Logging initialized")

        # Load configuration from config_path or use default configuration
        if config_path:
            config = load_configuration(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            config = DEFAULT_CONFIG
            logger.info("Using default configuration")

        # Initialize feature registry and feature definitions
        initialize_default_features()
        logger.info("Initialized feature registry and default features")

        # Validate system paths and create directories if needed
        if "paths" in config:
            for path_name, path_value in config["paths"].items():
                if isinstance(path_value, str):
                    path = Path(path_value)
                    if not path.exists():
                        try:
                            path.mkdir(parents=True, exist_ok=True)
                            logger.info(f"Created directory: {path}")
                        except Exception as e:
                            logger.error(f"Failed to create directory {path}: {e}")
                            return False

        logger.info("System initialization complete")
        return True

    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False