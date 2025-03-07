"""
Default configuration dictionaries for the ERCOT RTLMP spike prediction system.

This file provides fallback values for all configuration parameters used throughout
the system, ensuring consistent behavior when custom configurations are not provided.
"""

from typing import Dict, List, Any, Union, Optional, Literal
from pathlib import Path
import logging

from ..utils.type_definitions import ThresholdValue, NodeID, FeatureGroupType, ModelType

# Set up logger
logger = logging.getLogger(__name__)

def get_default_system_config() -> Dict[str, Any]:
    """
    Returns the default system configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default system configuration dictionary
    """
    return {
        "environment": "development",
        "log_level": "INFO",
        "random_seed": 42,
        "parallel_jobs": 4,
        "timezone": "America/Chicago",  # ERCOT is in Central Time
        "verbose": False
    }

def get_default_paths_config() -> Dict[str, Any]:
    """
    Returns the default paths configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default paths configuration dictionary
    """
    base_dir = Path.cwd()
    data_dir = base_dir / "data"
    
    return {
        "base_dir": str(base_dir),
        "data_dir": str(data_dir),
        "raw_data_dir": str(data_dir / "raw"),
        "feature_dir": str(data_dir / "features"),
        "model_dir": str(base_dir / "models"),
        "forecast_dir": str(base_dir / "forecasts"),
        "output_dir": str(base_dir / "output"),
        "log_dir": str(base_dir / "logs")
    }

def get_default_data_config() -> Dict[str, Any]:
    """
    Returns the default data configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default data configuration dictionary
    """
    return {
        "data_sources": {
            "ercot": {
                "api_url": "https://www.ercot.com/api/",
                "timeout": 60,
                "retry_attempts": 3,
                "retry_delay": 5,
                "api_key_env_var": "ERCOT_API_KEY",
                "default_nodes": ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"]
            },
            "weather": {
                "api_url": "https://api.weather.com/",
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 5,
                "api_key_env_var": "WEATHER_API_KEY",
                "default_locations": [
                    {"name": "NORTH", "lat": 32.7767, "lon": -96.7970},  # Dallas
                    {"name": "SOUTH", "lat": 29.7604, "lon": -95.3698},  # Houston
                    {"name": "WEST", "lat": 31.7619, "lon": -106.4850}   # El Paso
                ]
            }
        },
        "storage": {
            "format": "parquet",
            "compression": "snappy",
            "partition_by": ["year", "month"],
            "cache_ttl": 86400  # 24 hours in seconds
        },
        "fetching": {
            "historical_days": 731,  # ~2 years
            "forecast_horizon_hours": 72,
            "batch_size": 30,  # days per batch
            "parallel_requests": 2
        },
        "validation": {
            "schema_validation": True,
            "completeness_threshold": 0.95,  # 95% of data points must be present
            "value_range_check": True,
            "temporal_consistency_check": True
        }
    }

def get_default_feature_config() -> Dict[str, Any]:
    """
    Returns the default feature engineering configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default feature engineering configuration dictionary
    """
    return {
        "feature_groups": {
            "time": {
                "enabled": True,
                "features": [
                    "hour_of_day",
                    "day_of_week",
                    "is_weekend",
                    "month",
                    "season",
                    "is_holiday"
                ]
            },
            "statistical": {
                "enabled": True,
                "features": [
                    "rolling_mean_24h",
                    "rolling_max_24h",
                    "rolling_std_24h",
                    "rolling_mean_7d",
                    "rolling_max_7d",
                    "rolling_std_7d",
                    "price_volatility_24h"
                ],
                "lookback_windows": {
                    "hours": [1, 3, 6, 12, 24],
                    "days": [2, 7, 14]
                }
            },
            "weather": {
                "enabled": True,
                "features": [
                    "temperature",
                    "temperature_delta_24h",
                    "wind_speed",
                    "solar_irradiance",
                    "humidity"
                ]
            },
            "market": {
                "enabled": True,
                "features": [
                    "load_forecast",
                    "load_forecast_delta_24h",
                    "generation_mix_wind",
                    "generation_mix_solar",
                    "reserve_margin"
                ]
            }
        },
        "feature_selection": {
            "enabled": True,
            "method": "importance",  # or "correlation", "recursive"
            "threshold": 0.01,
            "max_features": 50
        },
        "feature_transformation": {
            "scaling": {
                "enabled": True,
                "method": "standard"  # or "minmax", "robust"
            },
            "encoding": {
                "enabled": True,
                "method": "onehot"  # or "label", "target"
            },
            "imputation": {
                "enabled": True,
                "method": "median"  # or "mean", "knn", "constant"
            }
        },
        "feature_registry": {
            "enabled": True,
            "storage_path": "feature_registry.json",
            "version_features": True
        }
    }

def get_default_model_config() -> Dict[str, Any]:
    """
    Returns the default model configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default model configuration dictionary
    """
    return {
        "model_defaults": {
            "target_variable": "spike_occurred",
            "threshold_values": [50.0, 100.0, 200.0],  # in $/MWh
            "default_threshold": 100.0,  # in $/MWh
            "validation_size": 0.2,
            "test_size": 0.1,
            "random_seed": 42
        },
        "model_types": {
            "xgboost": {
                "enabled": True,
                "package": "xgboost",
                "version": "1.7.0",
                "hyperparameters": {
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "min_child_weight": 1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "n_estimators": 200,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "use_label_encoder": False,
                    "random_state": 42
                }
            },
            "lightgbm": {
                "enabled": True,
                "package": "lightgbm",
                "version": "3.3.0",
                "hyperparameters": {
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "max_depth": -1,
                    "min_child_samples": 20,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "n_estimators": 200,
                    "objective": "binary",
                    "metric": "auc",
                    "random_state": 42
                }
            },
            "random_forest": {
                "enabled": False,
                "package": "sklearn.ensemble",
                "version": "1.2.0",
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "random_state": 42
                }
            },
            "logistic_regression": {
                "enabled": False,
                "package": "sklearn.linear_model",
                "version": "1.2.0",
                "hyperparameters": {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": 42
                }
            }
        },
        "training": {
            "retraining_frequency_days": 2,
            "cv_folds": 5,
            "cv_strategy": "time",  # time-based cross-validation
            "early_stopping_rounds": 20,
            "verbose": 0,
            "save_checkpoints": True,
            "checkpoint_frequency": 10  # save every 10 epochs
        },
        "evaluation": {
            "metrics": ["auc", "brier_score", "precision", "recall", "f1"],
            "primary_metric": "auc",
            "threshold_optimization": {
                "enabled": True,
                "method": "f1",  # optimize threshold based on F1 score
                "step": 0.01
            },
            "calibration": {
                "enabled": True,
                "method": "isotonic"  # or "sigmoid"
            }
        },
        "hyperparameter_tuning": {
            "enabled": True,
            "method": "bayesian",  # or "grid", "random"
            "max_evals": 50,
            "cv_folds": 3,
            "early_stopping_rounds": 10,
            "verbose": 0,
            "parameter_space": {
                "xgboost": {
                    "learning_rate": {"min": 0.01, "max": 0.3},
                    "max_depth": {"min": 3, "max": 10},
                    "min_child_weight": {"min": 1, "max": 10},
                    "subsample": {"min": 0.5, "max": 1.0},
                    "colsample_bytree": {"min": 0.5, "max": 1.0},
                    "n_estimators": {"min": 50, "max": 500}
                },
                "lightgbm": {
                    "learning_rate": {"min": 0.01, "max": 0.3},
                    "num_leaves": {"min": 20, "max": 150},
                    "max_depth": {"min": 3, "max": 12},
                    "min_child_samples": {"min": 5, "max": 50},
                    "subsample": {"min": 0.5, "max": 1.0},
                    "colsample_bytree": {"min": 0.5, "max": 1.0},
                    "n_estimators": {"min": 50, "max": 500}
                }
            }
        }
    }

def get_default_inference_config() -> Dict[str, Any]:
    """
    Returns the default inference configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default inference configuration dictionary
    """
    return {
        "thresholds": {
            "default": 100.0,  # in $/MWh
            "values": [50.0, 100.0, 200.0]  # in $/MWh
        },
        "forecast": {
            "horizon_hours": 72,
            "interval": "1H",
            "generate_before_dam_closure": True,
            "dam_closure_time": "10:00",  # ERCOT Day-Ahead Market closes at 10:00 CT
            "fallback_strategy": "previous_forecast",  # use previous forecast if current fails
            "confidence_interval": 0.9  # 90% confidence interval
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",  # or "sigmoid"
            "recalibrate_on_inference": False
        },
        "output": {
            "format": "parquet",
            "save_intermediates": False,
            "notify_on_completion": True,
            "notification_method": "log"  # or "email", "webhook"
        }
    }

def get_default_visualization_config() -> Dict[str, Any]:
    """
    Returns the default visualization configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default visualization configuration dictionary
    """
    return {
        "plots": {
            "probability_timeline": {
                "enabled": True,
                "show_confidence_interval": True,
                "height": 6,
                "width": 12,
                "dpi": 100
            },
            "calibration_curve": {
                "enabled": True,
                "n_bins": 10,
                "height": 6,
                "width": 6,
                "dpi": 100
            },
            "roc_curve": {
                "enabled": True,
                "height": 6,
                "width": 6,
                "dpi": 100
            },
            "precision_recall_curve": {
                "enabled": True,
                "height": 6,
                "width": 6,
                "dpi": 100
            },
            "feature_importance": {
                "enabled": True,
                "max_features": 20,
                "height": 8,
                "width": 10,
                "dpi": 100
            },
            "confusion_matrix": {
                "enabled": True,
                "normalize": True,
                "height": 6,
                "width": 6,
                "dpi": 100
            }
        },
        "dashboards": {
            "forecast": {
                "enabled": True,
                "default_threshold": 100.0,
                "default_node": "HB_NORTH",
                "days_of_history": 30
            },
            "model_performance": {
                "enabled": True,
                "show_version_comparison": True,
                "metrics_to_display": ["auc", "brier_score", "precision", "recall", "f1"]
            },
            "backtesting": {
                "enabled": True,
                "default_time_range_days": 90,
                "default_thresholds": [100.0],
                "default_nodes": ["HB_NORTH"]
            }
        },
        "colors": {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "threshold_lines": "#d62728",
            "confidence_interval": "rgba(31, 119, 180, 0.2)",
            "colormap": "viridis"
        },
        "export": {
            "formats": ["png", "svg", "pdf", "csv"],
            "default_format": "png",
            "dpi": 300
        }
    }

def get_default_config() -> Dict[str, Any]:
    """
    Returns the complete default configuration dictionary with all sections.
    
    Returns:
        Dict[str, Any]: Complete default configuration dictionary
    """
    return {
        "system": get_default_system_config(),
        "paths": get_default_paths_config(),
        "data": get_default_data_config(),
        "features": get_default_feature_config(),
        "model": get_default_model_config(),
        "inference": get_default_inference_config(),
        "visualization": get_default_visualization_config()
    }

# Initialize the default configuration dictionaries
DEFAULT_SYSTEM_CONFIG = get_default_system_config()
DEFAULT_PATHS_CONFIG = get_default_paths_config()
DEFAULT_DATA_CONFIG = get_default_data_config()
DEFAULT_FEATURE_CONFIG = get_default_feature_config()
DEFAULT_MODEL_CONFIG = get_default_model_config()
DEFAULT_INFERENCE_CONFIG = get_default_inference_config()
DEFAULT_VISUALIZATION_CONFIG = get_default_visualization_config()
DEFAULT_CONFIG = get_default_config()