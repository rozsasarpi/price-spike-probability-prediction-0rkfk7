"""
Defines configuration classes and utilities for backtesting scenarios in the ERCOT RTLMP spike prediction system.

This module provides structured definitions for configuring backtesting scenarios, including
time windows, model parameters, metrics, and evaluation criteria.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from pydantic import BaseModel, Field, validator  # version 2.0+

# Internal imports
from ..inference.threshold_config import ThresholdConfig, get_thresholds
from ..utils.type_definitions import DataFrameType, ModelType, ThresholdValue, NodeID, PathType
from ..utils.date_utils import generate_time_windows, get_forecast_window
from ..utils.logging import get_logger
from ..config.validation import validate_config

# Set up logger for this module
logger = get_logger(__name__)

# Default metrics for evaluation
DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1", "auc", "brier_score"]

# Default window size for backtesting scenarios (30 days)
DEFAULT_WINDOW_SIZE = timedelta(days=30)

# Default forecast horizon in hours
DEFAULT_FORECAST_HORIZON = 72


def validate_scenario_config(config: Dict[str, Any]) -> bool:
    """
    Validates a scenario configuration to ensure it has all required fields and valid values.
    
    Args:
        config: Dictionary containing scenario configuration
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check that config contains required fields
        required_fields = ["name", "start_date", "end_date"]
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field in scenario config: {field}")
                return False
        
        # Validate that start_date is before end_date
        if config["start_date"] >= config["end_date"]:
            logger.error("start_date must be before end_date")
            return False
        
        # If thresholds are specified, validate they are positive numbers
        if "thresholds" in config:
            for threshold in config["thresholds"]:
                if not isinstance(threshold, (int, float)) or threshold <= 0:
                    logger.error(f"Invalid threshold value: {threshold}")
                    return False
        
        # If nodes are specified, validate they are valid node identifiers
        if "nodes" in config:
            if not isinstance(config["nodes"], list) or not all(isinstance(n, str) for n in config["nodes"]):
                logger.error("nodes must be a list of string identifiers")
                return False
        
        # If model_config is specified, validate it has required fields
        if "model_config" in config and isinstance(config["model_config"], dict):
            model_config = config["model_config"]
            if "model_type" not in model_config:
                logger.error("model_config must contain model_type")
                return False
        
        # If metrics_config is specified, validate it has required fields
        if "metrics_config" in config and isinstance(config["metrics_config"], dict):
            metrics_config = config["metrics_config"]
            if "metrics" in metrics_config and not isinstance(metrics_config["metrics"], list):
                logger.error("metrics must be a list")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating scenario config: {e}")
        return False


def generate_time_windows_from_config(config: Dict[str, Any]) -> List[Tuple[datetime, datetime]]:
    """
    Generates time windows for backtesting based on scenario configuration.
    
    Args:
        config: Dictionary containing scenario configuration
        
    Returns:
        List of time window tuples (start, end)
    """
    try:
        # Extract start_date and end_date from config
        start_date = config["start_date"]
        end_date = config["end_date"]
        
        # Extract window_size from config or use DEFAULT_WINDOW_SIZE
        window_size = config.get("window_size", DEFAULT_WINDOW_SIZE)
        
        # Extract window_stride from config or use window_size
        window_stride = config.get("window_stride", window_size)
        
        # Generate time windows using date_utils function
        windows = generate_time_windows(start_date, end_date, window_size, window_stride)
        
        logger.debug(f"Generated {len(windows)} time windows for backtesting")
        return windows
    except Exception as e:
        logger.error(f"Error generating time windows from config: {e}")
        return []


def create_default_model_config() -> Dict[str, Any]:
    """
    Creates a default model configuration for backtesting.
    
    Returns:
        Default model configuration dictionary
    """
    return {
        "model_type": "xgboost",
        "retrain_per_window": False,
        "model_version": None,  # Use latest model
        "hyperparameters": {
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 200,
            "objective": "binary:logistic",
            "eval_metric": "auc"
        }
    }


def create_default_metrics_config() -> Dict[str, Any]:
    """
    Creates a default metrics configuration for backtesting.
    
    Returns:
        Default metrics configuration dictionary
    """
    return {
        "metrics": DEFAULT_METRICS,
        "calibration_curve": True,
        "confusion_matrix": True,
        "threshold_performance": True,
        "additional_metrics": {}
    }


class ModelConfig:
    """
    Configuration class for model parameters in backtesting scenarios.
    """
    
    def __init__(
        self,
        model_type: str,
        model_version: Optional[str] = None,
        retrain_per_window: Optional[bool] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes ModelConfig with model parameters.
        
        Args:
            model_type: Type of model to use for backtesting (e.g., 'xgboost', 'lightgbm')
            model_version: Optional specific model version to use
            retrain_per_window: Whether to retrain the model for each time window
            hyperparameters: Optional dictionary of model hyperparameters
        """
        self.model_type = model_type
        self.model_version = model_version
        self.retrain_per_window = False if retrain_per_window is None else retrain_per_window
        self.hyperparameters = hyperparameters or {}
        
        # Validate the configuration
        if not self.validate():
            logger.warning("Created ModelConfig with invalid parameters")
    
    def validate(self) -> bool:
        """
        Validates the model configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check that model_type is a non-empty string
            if not isinstance(self.model_type, str) or not self.model_type:
                logger.error("model_type must be a non-empty string")
                return False
            
            # If model_version is provided, check that it's a non-empty string
            if self.model_version is not None and (not isinstance(self.model_version, str) or not self.model_version):
                logger.error("model_version must be a non-empty string")
                return False
            
            # Check that retrain_per_window is a boolean
            if not isinstance(self.retrain_per_window, bool):
                logger.error("retrain_per_window must be a boolean")
                return False
            
            # Check that hyperparameters is a dictionary
            if not isinstance(self.hyperparameters, dict):
                logger.error("hyperparameters must be a dictionary")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating ModelConfig: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the model configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "retrain_per_window": self.retrain_per_window,
            "hyperparameters": self.hyperparameters
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Creates a ModelConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing model configuration
            
        Returns:
            ModelConfig instance
        """
        model_type = config_dict["model_type"]
        model_version = config_dict.get("model_version")
        retrain_per_window = config_dict.get("retrain_per_window")
        hyperparameters = config_dict.get("hyperparameters", {})
        
        return cls(
            model_type=model_type,
            model_version=model_version,
            retrain_per_window=retrain_per_window,
            hyperparameters=hyperparameters
        )


class MetricsConfig:
    """
    Configuration class for evaluation metrics in backtesting scenarios.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        calibration_curve: Optional[bool] = None,
        confusion_matrix: Optional[bool] = None,
        threshold_performance: Optional[bool] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes MetricsConfig with evaluation parameters.
        
        Args:
            metrics: List of metric names to calculate
            calibration_curve: Whether to generate calibration curve
            confusion_matrix: Whether to generate confusion matrix
            threshold_performance: Whether to evaluate performance at different thresholds
            additional_metrics: Optional dictionary of additional metric configurations
        """
        self.metrics = metrics or DEFAULT_METRICS.copy()
        self.calibration_curve = True if calibration_curve is None else calibration_curve
        self.confusion_matrix = True if confusion_matrix is None else confusion_matrix
        self.threshold_performance = True if threshold_performance is None else threshold_performance
        self.additional_metrics = additional_metrics or {}
        
        # Validate the configuration
        if not self.validate():
            logger.warning("Created MetricsConfig with invalid parameters")
    
    def validate(self) -> bool:
        """
        Validates the metrics configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check that metrics is a list of strings
            if not isinstance(self.metrics, list) or not all(isinstance(m, str) for m in self.metrics):
                logger.error("metrics must be a list of strings")
                return False
            
            # Check that calibration_curve is a boolean
            if not isinstance(self.calibration_curve, bool):
                logger.error("calibration_curve must be a boolean")
                return False
            
            # Check that confusion_matrix is a boolean
            if not isinstance(self.confusion_matrix, bool):
                logger.error("confusion_matrix must be a boolean")
                return False
            
            # Check that threshold_performance is a boolean
            if not isinstance(self.threshold_performance, bool):
                logger.error("threshold_performance must be a boolean")
                return False
            
            # Check that additional_metrics is a dictionary
            if not isinstance(self.additional_metrics, dict):
                logger.error("additional_metrics must be a dictionary")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating MetricsConfig: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the metrics configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "metrics": self.metrics,
            "calibration_curve": self.calibration_curve,
            "confusion_matrix": self.confusion_matrix,
            "threshold_performance": self.threshold_performance,
            "additional_metrics": self.additional_metrics
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MetricsConfig':
        """
        Creates a MetricsConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing metrics configuration
            
        Returns:
            MetricsConfig instance
        """
        metrics = config_dict.get("metrics")
        calibration_curve = config_dict.get("calibration_curve")
        confusion_matrix = config_dict.get("confusion_matrix")
        threshold_performance = config_dict.get("threshold_performance")
        additional_metrics = config_dict.get("additional_metrics")
        
        return cls(
            metrics=metrics,
            calibration_curve=calibration_curve,
            confusion_matrix=confusion_matrix,
            threshold_performance=threshold_performance,
            additional_metrics=additional_metrics
        )


class ScenarioConfig:
    """
    Configuration class for backtesting scenarios.
    """
    
    def __init__(
        self,
        name: str,
        start_date: datetime,
        end_date: datetime,
        thresholds: List[ThresholdValue],
        nodes: List[NodeID],
        window_size: Optional[timedelta] = None,
        window_stride: Optional[timedelta] = None,
        forecast_horizon: Optional[int] = None,
        model_config: Optional[ModelConfig] = None,
        metrics_config: Optional[MetricsConfig] = None,
        additional_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes ScenarioConfig with scenario parameters.
        
        Args:
            name: Name of the backtesting scenario
            start_date: Start date for the backtesting period
            end_date: End date for the backtesting period
            thresholds: List of price threshold values for spike definition
            nodes: List of node identifiers to backtest
            window_size: Optional size of each time window for backtesting
            window_stride: Optional stride between time windows
            forecast_horizon: Optional forecast horizon in hours
            model_config: Optional model configuration
            metrics_config: Optional metrics configuration
            additional_config: Optional additional configuration parameters
        """
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size or DEFAULT_WINDOW_SIZE
        self.window_stride = window_stride or self.window_size
        self.thresholds = thresholds
        self.nodes = nodes
        self.forecast_horizon = forecast_horizon or DEFAULT_FORECAST_HORIZON
        self.model_config = model_config or ModelConfig(**create_default_model_config())
        self.metrics_config = metrics_config or MetricsConfig(**create_default_metrics_config())
        self.additional_config = additional_config or {}
        
        # Validate the configuration
        if not self.validate():
            logger.warning("Created ScenarioConfig with invalid parameters")
    
    def validate(self) -> bool:
        """
        Validates the scenario configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check that name is a non-empty string
            if not isinstance(self.name, str) or not self.name:
                logger.error("name must be a non-empty string")
                return False
            
            # Check that start_date is before end_date
            if self.start_date >= self.end_date:
                logger.error("start_date must be before end_date")
                return False
            
            # Check that window_size is a positive timedelta
            if not isinstance(self.window_size, timedelta) or self.window_size.total_seconds() <= 0:
                logger.error("window_size must be a positive timedelta")
                return False
            
            # Check that window_stride is a positive timedelta
            if not isinstance(self.window_stride, timedelta) or self.window_stride.total_seconds() <= 0:
                logger.error("window_stride must be a positive timedelta")
                return False
            
            # Check that thresholds is a non-empty list of positive numbers
            if not isinstance(self.thresholds, list) or not self.thresholds:
                logger.error("thresholds must be a non-empty list")
                return False
            
            for threshold in self.thresholds:
                if not isinstance(threshold, (int, float)) or threshold <= 0:
                    logger.error(f"Invalid threshold value: {threshold}")
                    return False
            
            # Check that nodes is a non-empty list of strings
            if not isinstance(self.nodes, list) or not self.nodes:
                logger.error("nodes must be a non-empty list")
                return False
            
            for node in self.nodes:
                if not isinstance(node, str) or not node:
                    logger.error(f"Invalid node identifier: {node}")
                    return False
            
            # Check that forecast_horizon is a positive integer
            if not isinstance(self.forecast_horizon, int) or self.forecast_horizon <= 0:
                logger.error("forecast_horizon must be a positive integer")
                return False
            
            # Validate model_config if provided
            if self.model_config and not self.model_config.validate():
                logger.error("Invalid model_config")
                return False
            
            # Validate metrics_config if provided
            if self.metrics_config and not self.metrics_config.validate():
                logger.error("Invalid metrics_config")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating ScenarioConfig: {e}")
            return False
    
    def get_time_windows(self) -> List[Tuple[datetime, datetime]]:
        """
        Generates time windows for the scenario.
        
        Returns:
            List of time window tuples (start, end)
        """
        try:
            return generate_time_windows(
                self.start_date,
                self.end_date,
                self.window_size,
                self.window_stride
            )
        except Exception as e:
            logger.error(f"Error generating time windows: {e}")
            return []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the scenario configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = {
            "name": self.name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "window_size": self.window_size.total_seconds(),
            "window_stride": self.window_stride.total_seconds(),
            "thresholds": self.thresholds,
            "nodes": self.nodes,
            "forecast_horizon": self.forecast_horizon
        }
        
        # Add model_config if available
        if self.model_config:
            config_dict["model_config"] = self.model_config.to_dict()
        
        # Add metrics_config if available
        if self.metrics_config:
            config_dict["metrics_config"] = self.metrics_config.to_dict()
        
        # Add additional_config if available
        if self.additional_config:
            config_dict["additional_config"] = self.additional_config
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ScenarioConfig':
        """
        Creates a ScenarioConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing scenario configuration
            
        Returns:
            ScenarioConfig instance
        """
        # Extract required fields
        name = config_dict["name"]
        start_date = datetime.fromisoformat(config_dict["start_date"])
        end_date = datetime.fromisoformat(config_dict["end_date"])
        thresholds = config_dict["thresholds"]
        nodes = config_dict["nodes"]
        
        # Extract optional fields with defaults
        window_size_seconds = config_dict.get("window_size")
        window_size = timedelta(seconds=window_size_seconds) if window_size_seconds else None
        
        window_stride_seconds = config_dict.get("window_stride")
        window_stride = timedelta(seconds=window_stride_seconds) if window_stride_seconds else None
        
        forecast_horizon = config_dict.get("forecast_horizon")
        
        # Parse model_config if available
        model_config = None
        if "model_config" in config_dict:
            model_config = ModelConfig.from_dict(config_dict["model_config"])
        
        # Parse metrics_config if available
        metrics_config = None
        if "metrics_config" in config_dict:
            metrics_config = MetricsConfig.from_dict(config_dict["metrics_config"])
        
        # Extract additional_config if available
        additional_config = config_dict.get("additional_config")
        
        return cls(
            name=name,
            start_date=start_date,
            end_date=end_date,
            thresholds=thresholds,
            nodes=nodes,
            window_size=window_size,
            window_stride=window_stride,
            forecast_horizon=forecast_horizon,
            model_config=model_config,
            metrics_config=metrics_config,
            additional_config=additional_config
        )