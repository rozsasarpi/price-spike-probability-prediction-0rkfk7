"""
Configuration schema definitions for the ERCOT RTLMP spike prediction system.

This module defines Pydantic models that specify the structure, types, and
constraints for all configuration components in the system. These schemas enable
robust validation of configuration data and provide clear type hints throughout
the application.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import logging

# pydantic 2.0+
from pydantic import BaseModel, Field, root_validator, validator

# Internal imports
from ..utils.type_definitions import (
    FeatureGroupType, 
    ModelType,
    ThresholdValue,
    NodeID
)

# Set up logger
logger = logging.getLogger(__name__)

class SystemConfig(BaseModel):
    """Schema model for system-wide configuration settings."""
    
    environment: str = Field(
        default="development",
        description="Execution environment (development, testing, production)"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    parallel_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs (-1 for all available cores)"
    )
    timezone: str = Field(
        default="America/Chicago",
        description="Default timezone for timestamps (ERCOT is in Central Time)"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output"
    )
    
    @validator('environment')
    def validate_environment(cls, value):
        """Validates the environment setting."""
        valid_environments = ["development", "testing", "production"]
        if value not in valid_environments:
            raise ValueError(f"Environment must be one of: {', '.join(valid_environments)}")
        return value
    
    @validator('log_level')
    def validate_log_level(cls, value):
        """Validates the log level setting."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value not in valid_log_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_log_levels)}")
        return value

class PathsConfig(BaseModel):
    """Schema model for file system paths configuration."""
    
    base_dir: Path = Field(
        default=Path("."),
        description="Base directory for the application"
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for all data storage"
    )
    raw_data_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory for raw data storage"
    )
    feature_dir: Path = Field(
        default=Path("data/features"),
        description="Directory for feature storage"
    )
    model_dir: Path = Field(
        default=Path("models"),
        description="Directory for model storage"
    )
    forecast_dir: Path = Field(
        default=Path("forecasts"),
        description="Directory for forecast results"
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for output files"
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )
    
    @validator('data_dir', 'raw_data_dir', 'feature_dir', 'model_dir', 
              'forecast_dir', 'output_dir', 'log_dir')
    def validate_paths(cls, value, values):
        """Validates that paths are properly structured."""
        
        # If it's an absolute path, return it as is
        if value.is_absolute():
            return value
            
        # If base_dir is defined and this is a relative path, 
        # ensure it's properly joined with base_dir
        if 'base_dir' in values and values['base_dir']:
            # Return the path relative to base_dir
            return values['base_dir'] / value
            
        return value
    
    @root_validator
    def validate_path_hierarchy(cls, values):
        """Validates path hierarchy and ensures consistency."""
        # Check if raw_data_dir and feature_dir are under data_dir
        if 'data_dir' in values and 'raw_data_dir' in values and 'feature_dir' in values:
            data_dir = values['data_dir'].resolve()
            raw_data_dir = values['raw_data_dir'].resolve()
            feature_dir = values['feature_dir'].resolve()
            
            # Log a warning if directories are not correctly nested
            if not str(raw_data_dir).startswith(str(data_dir)):
                logger.warning(f"raw_data_dir {raw_data_dir} is not under data_dir {data_dir}")
            
            if not str(feature_dir).startswith(str(data_dir)):
                logger.warning(f"feature_dir {feature_dir} is not under data_dir {data_dir}")
        
        return values

class DataConfig(BaseModel):
    """Schema model for data sources and storage configuration."""
    
    data_sources: Dict[str, Any] = Field(
        default_factory=lambda: {
            "ercot": {
                "api_url": "https://www.ercot.com/api/",
                "auth_required": True,
                "auth_type": "api_key",
                "timeout": 30,
                "retry_attempts": 3,
                "retry_backoff": 2.0
            },
            "weather": {
                "api_url": "https://weather-service-provider.com/api/",
                "auth_required": True,
                "auth_type": "api_key",
                "timeout": 30,
                "retry_attempts": 3,
                "retry_backoff": 2.0
            }
        },
        description="Configuration for external data sources"
    )
    
    storage: Dict[str, Any] = Field(
        default_factory=lambda: {
            "format": "parquet",
            "compression": "snappy",
            "partition_by": ["year", "month"],
            "file_name_template": "{data_type}_{date}_{node_id}.{format}",
            "cache_enabled": True,
            "cache_ttl": 86400  # 24 hours in seconds
        },
        description="Configuration for data storage"
    )
    
    fetching: Dict[str, Any] = Field(
        default_factory=lambda: {
            "batch_size": 100,
            "concurrent_requests": 5,
            "rate_limit": {
                "requests_per_minute": 60,
                "max_retries": 3
            },
            "timeout": 30
        },
        description="Configuration for data fetching operations"
    )
    
    validation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "schema_validation": True,
            "missing_data_threshold": 0.1,  # Allow up to 10% missing data
            "value_range_validation": True,
            "anomaly_detection": {
                "enabled": True,
                "method": "z_score",
                "threshold": 3.0
            }
        },
        description="Configuration for data validation"
    )
    
    @validator('data_sources')
    def validate_data_sources(cls, value):
        """Validates data source configuration."""
        required_sources = ["ercot", "weather"]
        for source in required_sources:
            if source not in value:
                raise ValueError(f"Required data source '{source}' is missing")
                
        # Check that each source has the required fields
        required_fields = ["api_url", "auth_required"]
        for source, config in value.items():
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Data source '{source}' is missing required field '{field}'")
                    
        return value

class FeatureConfig(BaseModel):
    """Schema model for feature engineering configuration."""
    
    feature_groups: Dict[FeatureGroupType, List[str]] = Field(
        default_factory=lambda: {
            "time": [
                "hour_of_day", 
                "day_of_week", 
                "is_weekend", 
                "month",
                "season",
                "is_holiday"
            ],
            "statistical": [
                "rolling_mean_24h",
                "rolling_std_24h",
                "rolling_max_7d",
                "price_volatility"
            ],
            "weather": [
                "temperature",
                "wind_speed",
                "solar_irradiance",
                "humidity"
            ],
            "market": [
                "load_forecast",
                "available_capacity",
                "wind_generation",
                "solar_generation",
                "reserve_margin"
            ]
        },
        description="Groups of features to be created"
    )
    
    feature_selection: Dict[str, Any] = Field(
        default_factory=lambda: {
            "method": "importance",
            "importance_threshold": 0.01,
            "max_features": 50,
            "correlation_threshold": 0.85
        },
        description="Feature selection configuration"
    )
    
    feature_transformation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "scaling": {
                "method": "standard",
                "target_columns": ["*"],
                "exclude": ["is_weekend", "is_holiday"]
            },
            "encoding": {
                "method": "onehot",
                "target_columns": ["season"]
            }
        },
        description="Feature transformation configuration"
    )
    
    feature_registry: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "storage_path": "data/feature_registry.json",
            "track_importance": True,
            "version_features": True
        },
        description="Feature registry configuration"
    )
    
    @validator('feature_groups')
    def validate_feature_groups(cls, value):
        """Validates feature group configuration."""
        valid_groups = ["time", "statistical", "weather", "market"]
        for group in value:
            if group not in valid_groups:
                raise ValueError(f"Feature group '{group}' is invalid. Must be one of: {', '.join(valid_groups)}")

        # Validate feature names
        for group, features in value.items():
            for feature in features:
                if not feature or not isinstance(feature, str):
                    raise ValueError(f"Invalid feature name in group '{group}': {feature}")
                
        return value

class ModelConfig(BaseModel):
    """Schema model for model training and evaluation configuration."""
    
    model_defaults: Dict[str, Any] = Field(
        default_factory=lambda: {
            "random_state": 42,
            "verbose": 0,
            "n_jobs": -1
        },
        description="Default parameters for all models"
    )
    
    model_types: Dict[ModelType, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "xgboost": {
                "package": "xgboost",
                "class": "XGBClassifier",
                "hyperparameters": {
                    "learning_rate": {"default": 0.05, "min": 0.01, "max": 0.3},
                    "max_depth": {"default": 6, "min": 3, "max": 10},
                    "min_child_weight": {"default": 1, "min": 1, "max": 10},
                    "subsample": {"default": 0.8, "min": 0.5, "max": 1.0},
                    "colsample_bytree": {"default": 0.8, "min": 0.5, "max": 1.0},
                    "n_estimators": {"default": 200, "min": 50, "max": 500}
                }
            },
            "lightgbm": {
                "package": "lightgbm",
                "class": "LGBMClassifier",
                "hyperparameters": {
                    "learning_rate": {"default": 0.05, "min": 0.01, "max": 0.3},
                    "num_leaves": {"default": 31, "min": 20, "max": 150},
                    "max_depth": {"default": -1, "min": -1, "max": 15},
                    "min_child_samples": {"default": 20, "min": 10, "max": 100},
                    "subsample": {"default": 0.8, "min": 0.5, "max": 1.0},
                    "colsample_bytree": {"default": 0.8, "min": 0.5, "max": 1.0},
                    "n_estimators": {"default": 200, "min": 50, "max": 500}
                }
            },
            "random_forest": {
                "package": "sklearn.ensemble",
                "class": "RandomForestClassifier",
                "hyperparameters": {
                    "n_estimators": {"default": 200, "min": 50, "max": 500},
                    "max_depth": {"default": 10, "min": 3, "max": 20},
                    "min_samples_split": {"default": 2, "min": 2, "max": 20},
                    "min_samples_leaf": {"default": 1, "min": 1, "max": 10},
                    "max_features": {"default": "sqrt", "options": ["sqrt", "log2", None]}
                }
            },
            "logistic_regression": {
                "package": "sklearn.linear_model",
                "class": "LogisticRegression",
                "hyperparameters": {
                    "C": {"default": 1.0, "min": 0.001, "max": 10.0, "log_scale": True},
                    "penalty": {"default": "l2", "options": ["l1", "l2", "elasticnet", None]},
                    "solver": {"default": "liblinear", "options": ["liblinear", "saga"]},
                    "max_iter": {"default": 100, "min": 50, "max": 1000}
                }
            }
        },
        description="Supported model types and their configurations"
    )
    
    training: Dict[str, Any] = Field(
        default_factory=lambda: {
            "train_test_split": {
                "test_size": 0.2,
                "shuffle": False,
                "stratify": None
            },
            "cross_validation": {
                "method": "time_series_split",
                "n_splits": 5,
                "gap": 24  # 24 hours gap between train and test
            },
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "metric": "auc"
            },
            "class_weight": "balanced"
        },
        description="Model training configuration"
    )
    
    evaluation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "metrics": [
                "auc", 
                "accuracy", 
                "precision", 
                "recall", 
                "f1", 
                "brier_score"
            ],
            "threshold_optimization": {
                "method": "f1",
                "grid_size": 100
            },
            "calibration": {
                "method": "isotonic",
                "cv": 5
            }
        },
        description="Model evaluation configuration"
    )
    
    hyperparameter_tuning: Dict[str, Any] = Field(
        default_factory=lambda: {
            "method": "bayesian",
            "n_trials": 50,
            "timeout": 3600,
            "cv": 3,
            "scoring": "roc_auc",
            "n_jobs": -1
        },
        description="Hyperparameter tuning configuration"
    )
    
    @validator('model_types')
    def validate_model_types(cls, value):
        """Validates model type configuration."""
        if not value:
            raise ValueError("At least one model type must be configured")
            
        for model_type, config in value.items():
            required_fields = ["package", "class", "hyperparameters"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Model type '{model_type}' is missing required field '{field}'")
        
        return value

class InferenceConfig(BaseModel):
    """Schema model for inference and prediction configuration."""
    
    thresholds: List[ThresholdValue] = Field(
        default=[50.0, 100.0, 200.0, 500.0, 1000.0],
        description="Threshold values for price spike definition"
    )
    
    forecast: Dict[str, Any] = Field(
        default_factory=lambda: {
            "horizon": 72,  # 72 hours
            "granularity": "H",  # Hourly
            "uncertainty": {
                "method": "bootstrap",
                "n_iterations": 100,
                "confidence_level": 0.9
            }
        },
        description="Forecast parameters"
    )
    
    calibration: Dict[str, Any] = Field(
        default_factory=lambda: {
            "apply_calibration": True,
            "method": "isotonic",  # or "platt"
            "recalibrate_frequency": "weekly"
        },
        description="Probability calibration configuration"
    )
    
    output: Dict[str, Any] = Field(
        default_factory=lambda: {
            "format": "parquet",
            "save_path_template": "forecasts/{date}/{timestamp}_forecast.parquet",
            "include_metadata": True,
            "include_features": False,
            "export_csv": True
        },
        description="Output configuration for forecasts"
    )
    
    @validator('thresholds')
    def validate_thresholds(cls, value):
        """Validates threshold values for price spike definition."""
        if not value:
            raise ValueError("At least one threshold value must be specified")
            
        for threshold in value:
            if threshold <= 0:
                raise ValueError(f"Threshold value must be positive: {threshold}")
                
        # Check that thresholds are in ascending order
        if sorted(value) != value:
            raise ValueError("Threshold values must be in ascending order")
            
        return value

class VisualizationConfig(BaseModel):
    """Schema model for visualization and reporting configuration."""
    
    plots: Dict[str, Any] = Field(
        default_factory=lambda: {
            "default_figsize": (12, 8),
            "dpi": 100,
            "default_style": "seaborn-whitegrid",
            "save_format": "png",
            "interactive": True
        },
        description="General plot configuration"
    )
    
    dashboards: Dict[str, Any] = Field(
        default_factory=lambda: {
            "forecast_dashboard": {
                "layout": "grid",
                "plots_per_row": 2,
                "show_confidence_intervals": True,
                "show_thresholds": True,
                "include_metrics": True
            },
            "model_dashboard": {
                "layout": "tabs",
                "tabs": ["performance", "calibration", "feature_importance"],
                "show_details": True
            }
        },
        description="Dashboard configuration"
    )
    
    colors: Dict[str, Any] = Field(
        default_factory=lambda: {
            "palette": "viridis",
            "forecast_line": "#1f77b4",
            "actual_line": "#d62728",
            "confidence_interval": "rgba(31, 119, 180, 0.2)",
            "threshold_line": "#ff7f0e",
            "colormap_diverging": "RdBu_r"
        },
        description="Color scheme configuration"
    )
    
    export: Dict[str, Any] = Field(
        default_factory=lambda: {
            "formats": ["png", "pdf", "html"],
            "save_path": "output/visualizations",
            "filename_template": "{plot_type}_{date}_{threshold}.{format}"
        },
        description="Export configuration for visualizations"
    )
    
    @validator('colors')
    def validate_colors(cls, value):
        """Validates color scheme configuration."""
        required_colors = ["forecast_line", "actual_line", "threshold_line"]
        for color in required_colors:
            if color not in value:
                raise ValueError(f"Required color '{color}' is missing")
                
        return value

class Config(BaseModel):
    """Root schema model for the entire configuration."""
    
    system: SystemConfig = Field(
        default_factory=SystemConfig,
        description="System-wide configuration"
    )
    
    paths: PathsConfig = Field(
        default_factory=PathsConfig,
        description="File system paths configuration"
    )
    
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data sources and storage configuration"
    )
    
    features: FeatureConfig = Field(
        default_factory=FeatureConfig,
        description="Feature engineering configuration"
    )
    
    models: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model training and evaluation configuration"
    )
    
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Inference and prediction configuration"
    )
    
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization and reporting configuration"
    )
    
    @root_validator
    def validate_config(cls, values):
        """Validates the complete configuration for consistency across sections."""
        # Cross-validate paths and data storage configuration
        if 'paths' in values and 'data' in values:
            paths = values['paths']
            data = values['data']
            
            # Check storage format is consistent
            if data.storage.get('format') not in ['parquet', 'csv', 'json']:
                logger.warning("Storage format should be one of: parquet, csv, json")
        
        # Validate feature configuration against model requirements
        if 'features' in values and 'models' in values:
            features = values['features']
            models = values['models']
            
            # Check that feature groups have at least some features defined
            for group, feature_list in features.feature_groups.items():
                if not feature_list:
                    logger.warning(f"Feature group '{group}' has no features defined")
        
        # Ensure inference thresholds are compatible with model outputs
        if 'inference' in values and 'models' in values:
            inference = values['inference']
            models = values['models']
            
            # Verify thresholds are reasonable for RTLMP values
            if any(t > 10000 for t in inference.thresholds):
                logger.warning("Some threshold values are unusually high for RTLMP values")
        
        return values