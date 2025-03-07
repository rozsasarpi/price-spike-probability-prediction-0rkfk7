"""
Implementation of LightGBM model for RTLMP spike prediction.

This module extends the BaseModel abstract class to provide a gradient boosting implementation
optimized for predicting price spike probabilities in the ERCOT market.
"""

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
import lightgbm  # version 3.3+
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, cast

# Internal imports
from .base_model import BaseModel
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ModelConfigDict, PathType
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import ModelError, ModelTrainingError
from .persistence import save_model_to_path, load_model_from_path
from .evaluation import evaluate_model_performance

# Set up logger
logger = get_logger(__name__)

# Default hyperparameters for LightGBM models
DEFAULT_LIGHTGBM_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": 42
}

class LightGBMModel(BaseModel):
    """LightGBM implementation for RTLMP spike prediction."""
    
    model: Optional[lightgbm.LGBMClassifier]
    model_id: str
    model_type: str
    version: Optional[str]
    hyperparameters: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]]
    training_date: Optional[datetime]
    feature_names: Optional[List[str]]
    
    def __init__(self, model_id: str, version: Optional[str] = None, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize a new LightGBMModel instance.
        
        Args:
            model_id: Unique identifier for the model
            version: Optional version identifier
            hyperparameters: Dictionary of model hyperparameters
        """
        # Call the BaseModel constructor
        super().__init__(model_id, 'lightgbm', version, hyperparameters)
        
        # Initialize model to None
        self.model = None
        
        # Merge provided hyperparameters with defaults
        if hyperparameters:
            self.hyperparameters = {**DEFAULT_LIGHTGBM_PARAMS, **hyperparameters}
        else:
            self.hyperparameters = DEFAULT_LIGHTGBM_PARAMS.copy()
        
        logger.info(f"Initialized LightGBM model {model_id} with hyperparameters: {self.hyperparameters}")
    
    @log_execution_time(logger, 'INFO')
    def train(self, features: DataFrameType, targets: SeriesType, params: Optional[Dict[str, Any]] = None) -> 'LightGBMModel':
        """
        Train the LightGBM model on provided features and targets.
        
        Args:
            features: DataFrame of input features
            targets: Series of target values
            params: Optional additional parameters for training
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting training for LightGBM model {self.model_id}")
        
        # Validate that features and targets have the same length
        if len(features) != len(targets):
            raise ModelTrainingError(f"Features and targets must have same length: {len(features)} vs {len(targets)}")
        
        # Merge additional params with hyperparameters if provided
        training_params = self.hyperparameters.copy()
        if params:
            training_params.update(params)
        
        # Store feature names from the DataFrame
        self.feature_names = list(features.columns)
        
        try:
            # Create and train the LightGBM model
            self.model = lightgbm.LGBMClassifier(**training_params)
            self.model.fit(features, targets)
            
            # Update training date
            self.training_date = datetime.now()
            
            logger.info(f"Successfully trained LightGBM model {self.model_id}")
            
            return self
        except Exception as e:
            error_msg = f"Error training LightGBM model {self.model_id}: {str(e)}"
            logger.error(error_msg)
            raise ModelTrainingError(error_msg) from e
    
    def predict(self, features: DataFrameType) -> ArrayType:
        """
        Generate binary predictions using the trained LightGBM model.
        
        Args:
            features: DataFrame of input features
            
        Returns:
            Binary predictions (0 or 1)
        """
        # Validate that the model has been trained
        if not self.is_trained():
            raise ModelError("Model must be trained before prediction")
        
        # Validate that features contain all required columns
        if not self.validate_features(features):
            raise ModelError("Features missing required columns")
        
        # Generate binary predictions
        return self.model.predict(features)
    
    def predict_proba(self, features: DataFrameType) -> ArrayType:
        """
        Generate probability predictions using the trained LightGBM model.
        
        Args:
            features: DataFrame of input features
            
        Returns:
            Probability predictions (values between 0 and 1)
        """
        # Validate that the model has been trained
        if not self.is_trained():
            raise ModelError("Model must be trained before prediction")
        
        # Validate that features contain all required columns
        if not self.validate_features(features):
            raise ModelError("Features missing required columns")
        
        # Generate probability predictions
        # Return the positive class probability (column 1)
        return self.model.predict_proba(features)[:, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained LightGBM model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Validate that the model has been trained
        if not self.is_trained():
            raise ModelError("Model must be trained before getting feature importance")
        
        # Extract feature importance scores
        importance_scores = self.model.feature_importances_
        
        # Create a dictionary mapping feature names to importance scores
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = float(importance_scores[i])
        
        # Sort by importance (descending)
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def is_trained(self) -> bool:
        """
        Check if the LightGBM model has been trained.
        
        Returns:
            True if the model is trained, False otherwise
        """
        return self.model is not None
    
    @log_execution_time(logger, 'INFO')
    def save(self, path: Optional[PathType] = None) -> PathType:
        """
        Save the LightGBM model to a file with metadata.
        
        Args:
            path: Path where the model will be saved
            
        Returns:
            Path to the saved model directory
        """
        # Validate that the model has been trained
        if not self.is_trained():
            raise ModelError("Model must be trained before saving")
        
        # Call the parent class save method to handle common saving logic
        return super().save(path)
    
    @log_execution_time(logger, 'INFO')
    def load(self, path: PathType) -> 'LightGBMModel':
        """
        Load the LightGBM model from a file with metadata.
        
        Args:
            path: Path to the model file
            
        Returns:
            Self for method chaining
        """
        # Call the parent class load method to handle common loading logic
        super().load(path)
        
        # Validate that the loaded model is an LGBMClassifier
        if not isinstance(self.model, lightgbm.LGBMClassifier):
            raise ModelError(f"Loaded model is not an LGBMClassifier: {type(self.model)}")
        
        return self
    
    @log_execution_time(logger, 'INFO')
    def evaluate(self, X_test: DataFrameType, y_test: SeriesType, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate the LightGBM model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            threshold: Optional probability threshold for binary classification
            
        Returns:
            Dictionary of performance metrics
        """
        # Validate that the model has been trained
        if not self.is_trained():
            raise ModelError("Model must be trained before evaluation")
        
        # Use evaluate_model_performance function to calculate metrics
        metrics = evaluate_model_performance(self, X_test, y_test, threshold=threshold)
        
        # Update performance metrics
        self.performance_metrics = metrics
        
        logger.info(f"Model {self.model_id} evaluation results: {metrics}")
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the LightGBM model.
        
        Returns:
            Dictionary of model hyperparameters
        """
        return self.hyperparameters.copy()
    
    def set_params(self, params: Dict[str, Any]) -> 'LightGBMModel':
        """
        Update the hyperparameters of the LightGBM model.
        
        Args:
            params: Dictionary of hyperparameters to update
            
        Returns:
            Self for method chaining
        """
        # Update hyperparameters
        self.hyperparameters.update(params)
        
        # Log warning if the model is already trained
        if self.is_trained():
            logger.warning(f"Model {self.model_id} is already trained. Parameter changes won't affect the current model.")
        
        return self