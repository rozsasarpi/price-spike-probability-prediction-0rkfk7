"""
Implementation of XGBoost model for RTLMP spike prediction.

This module extends the BaseModel abstract class to provide a gradient boosting
implementation optimized for predicting price spike probabilities in the ERCOT market.
"""

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
import xgboost  # version 1.7+
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, cast

from .base_model import BaseModel
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ModelConfigDict, PathType
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import ModelError, ModelTrainingError
from .persistence import save_model_to_path, load_model_from_path
from .evaluation import evaluate_model_performance

# Initialize logger
logger = get_logger(__name__)

# Default XGBoost hyperparameters based on specifications
DEFAULT_XGBOOST_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 200,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42
}

class XGBoostModel(BaseModel):
    """XGBoost implementation for RTLMP spike prediction."""
    
    def __init__(
        self, 
        model_id: str, 
        version: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new XGBoostModel instance.
        
        Args:
            model_id: Unique identifier for the model
            version: Optional version identifier
            hyperparameters: Optional dictionary of model hyperparameters
        """
        # Call the parent class constructor
        super().__init__(model_id, 'xgboost', version, hyperparameters)
        
        # Initialize model attribute
        self.model: Optional[xgboost.XGBClassifier] = None
        
        # Set hyperparameters, using defaults for any not provided
        if hyperparameters:
            self.hyperparameters = {**DEFAULT_XGBOOST_PARAMS, **hyperparameters}
        else:
            self.hyperparameters = DEFAULT_XGBOOST_PARAMS.copy()
            
        logger.debug(f"Initialized XGBoostModel with ID {model_id} and "
                    f"hyperparameters: {self.hyperparameters}")
    
    @log_execution_time(logger, 'INFO')
    def train(self, features: DataFrameType, targets: SeriesType, params: Optional[Dict[str, Any]] = None) -> 'XGBoostModel':
        """
        Train the XGBoost model on provided features and targets.
        
        Args:
            features: DataFrame of input features
            targets: Series of target values
            params: Optional additional parameters for training
            
        Returns:
            Self for method chaining
            
        Raises:
            ModelTrainingError: If training fails
        """
        try:
            logger.info(f"Starting XGBoost model training for {self.model_id}")
            
            # Check that features and targets have the same length
            if len(features) != len(targets):
                msg = f"Features and targets must have the same length: {len(features)} vs {len(targets)}"
                logger.error(msg)
                raise ModelTrainingError(msg)
            
            # Update hyperparameters if additional params are provided
            train_params = self.hyperparameters.copy()
            if params:
                train_params.update(params)
                
            # Create a new XGBoost classifier with the training parameters
            self.model = xgboost.XGBClassifier(**train_params)
            
            # Store feature names from the features DataFrame
            self.feature_names = features.columns.tolist()
            
            # Fit the model to the training data
            self.model.fit(features, targets)
            
            # Update training date
            self.training_date = datetime.now()
            
            logger.info(f"Successfully trained XGBoost model {self.model_id}")
            return self
            
        except Exception as e:
            msg = f"Failed to train XGBoost model {self.model_id}: {str(e)}"
            logger.error(msg)
            raise ModelTrainingError(msg) from e
    
    def predict(self, features: DataFrameType) -> ArrayType:
        """
        Generate binary predictions using the trained XGBoost model.
        
        Args:
            features: DataFrame of input features
            
        Returns:
            Array of binary predictions (0 or 1)
            
        Raises:
            ModelError: If model is not trained or features are invalid
        """
        # Check if model is trained
        if not self.is_trained():
            msg = f"Model {self.model_id} has not been trained yet"
            logger.error(msg)
            raise ModelError(msg)
        
        # Validate features
        if not self.validate_features(features):
            msg = f"Features for model {self.model_id} did not pass validation"
            logger.error(msg)
            raise ModelError(msg)
        
        # Generate predictions
        return self.model.predict(features)
    
    def predict_proba(self, features: DataFrameType) -> ArrayType:
        """
        Generate probability predictions using the trained XGBoost model.
        
        Args:
            features: DataFrame of input features
            
        Returns:
            Array of probability predictions (values between 0 and 1)
            
        Raises:
            ModelError: If model is not trained or features are invalid
        """
        # Check if model is trained
        if not self.is_trained():
            msg = f"Model {self.model_id} has not been trained yet"
            logger.error(msg)
            raise ModelError(msg)
        
        # Validate features
        if not self.validate_features(features):
            msg = f"Features for model {self.model_id} did not pass validation"
            logger.error(msg)
            raise ModelError(msg)
        
        # Generate probabilities and return the positive class probabilities (class 1)
        probabilities = self.model.predict_proba(features)
        return probabilities[:, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained XGBoost model.
        
        Returns:
            Dictionary mapping feature names to importance scores
            
        Raises:
            ModelError: If model is not trained
        """
        # Check if model is trained
        if not self.is_trained():
            msg = f"Model {self.model_id} has not been trained yet"
            logger.error(msg)
            raise ModelError(msg)
        
        # Get feature importances from the model
        importances = self.model.feature_importances_
        
        # Create a dictionary mapping feature names to importance scores
        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            importance_dict[feature] = float(importances[i])
        
        # Sort by importance in descending order
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda item: item[1], 
            reverse=True
        ))
        
        return sorted_importance
    
    def is_trained(self) -> bool:
        """
        Check if the XGBoost model has been trained.
        
        Returns:
            True if the model is trained, False otherwise
        """
        return self.model is not None
    
    @log_execution_time(logger, 'INFO')
    def save(self, path: Optional[PathType] = None) -> PathType:
        """
        Save the XGBoost model to a file with metadata.
        
        Args:
            path: Optional path to save the model, defaults to a path derived from model attributes
            
        Returns:
            Path to the saved model directory
            
        Raises:
            ModelError: If model is not trained or save fails
        """
        # Check if model is trained
        if not self.is_trained():
            msg = f"Cannot save untrained model {self.model_id}"
            logger.error(msg)
            raise ModelError(msg)
        
        # Call the parent class save method to handle common saving logic
        return super().save(path)
    
    @log_execution_time(logger, 'INFO')
    def load(self, path: PathType) -> 'XGBoostModel':
        """
        Load the XGBoost model from a file with metadata.
        
        Args:
            path: Path to the model directory
            
        Returns:
            Self for method chaining
            
        Raises:
            ModelError: If load fails
        """
        # Call the parent class load method to handle common loading logic
        super().load(path)
        
        # Validate that the loaded model is an XGBClassifier
        if not isinstance(self.model, xgboost.XGBClassifier):
            msg = f"Loaded model is not an XGBClassifier: {type(self.model)}"
            logger.error(msg)
            raise ModelError(msg)
        
        return self
    
    @log_execution_time(logger, 'INFO')
    def evaluate(self, X_test: DataFrameType, y_test: SeriesType, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate the XGBoost model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            threshold: Optional probability threshold for binary predictions
            
        Returns:
            Dictionary of performance metrics
            
        Raises:
            ModelError: If model is not trained
        """
        # Check if model is trained
        if not self.is_trained():
            msg = f"Cannot evaluate untrained model {self.model_id}"
            logger.error(msg)
            raise ModelError(msg)
        
        # Use evaluate_model_performance function from evaluation module
        metrics = evaluate_model_performance(self, X_test, y_test, threshold=threshold)
        
        # Update model's performance metrics
        self.performance_metrics = metrics
        
        logger.info(f"Model {self.model_id} evaluation results: {metrics}")
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the XGBoost model.
        
        Returns:
            Dictionary of model hyperparameters
        """
        return self.hyperparameters.copy()
    
    def set_params(self, params: Dict[str, Any]) -> 'XGBoostModel':
        """
        Update the hyperparameters of the XGBoost model.
        
        Args:
            params: Dictionary of hyperparameters to update
            
        Returns:
            Self for method chaining
        """
        # Update hyperparameters
        self.hyperparameters.update(params)
        
        # Log a warning if the model is already trained
        if self.is_trained():
            logger.warning(f"Model {self.model_id} is already trained. "
                          f"Parameter changes will only affect future training.")
        
        return self