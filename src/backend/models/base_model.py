"""
Abstract base class for machine learning models in the ERCOT RTLMP spike prediction system.

This module defines a standardized interface for all model implementations, providing
consistent methods for training, prediction, persistence, and evaluation.
"""

import abc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+

from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ModelConfigDict, PathType
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import ModelError
from .persistence import save_model_to_path, load_model_from_path, save_metadata, load_metadata

# Initialize logger
logger = get_logger(__name__)

class BaseModel(abc.ABC):
    """Abstract base class for all prediction models in the system."""
    
    def __init__(
        self, 
        model_id: str, 
        model_type: str, 
        version: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new BaseModel instance.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'xgboost', 'lightgbm')
            version: Optional version identifier
            hyperparameters: Dictionary of model hyperparameters
        """
        self.model_id = model_id
        self.model_type = model_type
        self.version = version
        self.hyperparameters = hyperparameters or {}
        self.performance_metrics = None
        self.training_date = None
        self.feature_names = None
    
    @abc.abstractmethod
    def train(self, features: DataFrameType, targets: SeriesType, params: Optional[Dict[str, Any]] = None) -> 'BaseModel':
        """
        Train the model on provided features and targets.
        
        Args:
            features: DataFrame of input features
            targets: Series of target values
            params: Optional additional parameters for training
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abc.abstractmethod
    def predict(self, features: DataFrameType) -> ArrayType:
        """
        Generate binary predictions using the trained model.
        
        Args:
            features: DataFrame of input features
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        pass
    
    @abc.abstractmethod
    def predict_proba(self, features: DataFrameType) -> ArrayType:
        """
        Generate probability predictions using the trained model.
        
        Args:
            features: DataFrame of input features
            
        Returns:
            Array of probability predictions (values between 0 and 1)
        """
        pass
    
    @abc.abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @log_execution_time(logger, 'INFO')
    def save(self, path: Optional[PathType] = None) -> PathType:
        """
        Save the model to a file with metadata.
        
        Args:
            path: Optional path to save the model, defaults to a path derived from model attributes
            
        Returns:
            Path to the saved model directory
        """
        # Create model configuration dictionary
        model_config = self.get_model_config()
        
        # Determine the save path if not provided
        if path is None:
            # Create a default path based on model_id and version
            base_dir = Path("models")
            model_dir = base_dir / self.model_type / self.model_id
            version_str = self.version if self.version else "latest"
            path = model_dir / version_str
        
        # Ensure the directory exists
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save the model and metadata
        save_model_to_path(self, path)
        save_metadata(model_config, path)
        
        logger.info(f"Saved model {self.model_id} (version: {self.version}) to {path}")
        
        return path
    
    @log_execution_time(logger, 'INFO')
    def load(self, path: PathType) -> 'BaseModel':
        """
        Load the model from a file with metadata.
        
        Args:
            path: Path to the model directory
            
        Returns:
            Self for method chaining
        """
        # Load the model and metadata
        model = load_model_from_path(path)
        metadata = load_metadata(path)
        
        # Update model attributes from metadata
        self.model_id = metadata.get('model_id', self.model_id)
        self.model_type = metadata.get('model_type', self.model_type)
        self.version = metadata.get('version', self.version)
        self.hyperparameters = metadata.get('hyperparameters', self.hyperparameters)
        self.performance_metrics = metadata.get('performance_metrics', self.performance_metrics)
        self.training_date = metadata.get('training_date', self.training_date)
        self.feature_names = metadata.get('feature_names', self.feature_names)
        
        logger.info(f"Loaded model {self.model_id} (version: {self.version}) from {path}")
        
        return self
    
    def is_trained(self) -> bool:
        """
        Check if the model has been trained.
        
        Returns:
            True if the model is trained, False otherwise
        """
        # Default implementation, should be overridden by subclasses
        return False
    
    def validate_features(self, features: DataFrameType) -> bool:
        """
        Validate that features contain required columns.
        
        Args:
            features: DataFrame of input features
            
        Returns:
            True if features are valid, False otherwise
        """
        if self.feature_names is None:
            logger.warning("Feature names not set, skipping validation")
            return True
        
        missing_features = [f for f in self.feature_names if f not in features.columns]
        
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False
        
        return True
    
    def get_model_config(self) -> ModelConfigDict:
        """
        Get the model configuration as a dictionary.
        
        Returns:
            Model configuration dictionary
        """
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'version': self.version,
            'hyperparameters': self.hyperparameters,
            'performance_metrics': self.performance_metrics,
            'training_date': self.training_date,
            'feature_names': self.feature_names
        }
    
    def set_performance_metrics(self, metrics: Dict[str, float]) -> 'BaseModel':
        """
        Set the performance metrics for the model.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Self for method chaining
        """
        self.performance_metrics = metrics
        logger.info(f"Updated performance metrics for model {self.model_id}: {metrics}")
        return self
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get the performance metrics for the model.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics or {}