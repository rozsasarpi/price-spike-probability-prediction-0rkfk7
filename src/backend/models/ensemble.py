"""
Implementation of ensemble models for RTLMP spike prediction.

This class extends the BaseModel abstract class to provide a meta-model implementation that combines
multiple base models to improve prediction accuracy and robustness.
"""

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
from typing import Dict, List, Optional, Any, Tuple, Union, cast, Type
from datetime import datetime
from enum import Enum
from sklearn.ensemble import VotingClassifier  # version 1.2+

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ModelConfigDict, PathType
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import ModelError, ModelTrainingError
from .persistence import save_model_to_path, load_model_from_path
from .evaluation import evaluate_model_performance, compare_models

# Initialize logger
logger = get_logger(__name__)

# Default ensemble hyperparameters
DEFAULT_ENSEMBLE_PARAMS = {
    "voting": "soft",  # Use probability outputs for voting
    "weights": None,   # Equal weighting by default
    "n_jobs": -1       # Use all available cores
}


class EnsembleMethod(Enum):
    """Enum defining different ensemble methods."""
    VOTING = "voting"     # Simple weighted average of probabilities
    STACKING = "stacking" # Train a meta-model on base model outputs
    BAGGING = "bagging"   # Bootstrap aggregation
    BOOSTING = "boosting" # Sequential training to correct errors


class EnsembleModel(BaseModel):
    """Ensemble model implementation that combines multiple base models."""
    
    def __init__(
        self, 
        model_id: str, 
        version: Optional[str] = None,
        models: Optional[List[BaseModel]] = None,
        weights: Optional[List[float]] = None,
        ensemble_method: str = EnsembleMethod.VOTING.value,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new EnsembleModel instance.
        
        Args:
            model_id: Unique identifier for the model
            version: Optional version identifier
            models: Optional list of pre-trained models to include in the ensemble
            weights: Optional list of weights for the models (must match models length)
            ensemble_method: Method to use for combining models (from EnsembleMethod)
            hyperparameters: Optional dictionary of model hyperparameters
        """
        # Call the parent class constructor
        super().__init__(model_id, 'ensemble', version, hyperparameters)
        
        # Initialize model list
        self.models = models if models is not None else []
        
        # Initialize weights
        self.model_weights = weights
        
        # Set ensemble method
        self.ensemble_method = ensemble_method
        
        # Set hyperparameters
        if hyperparameters:
            self.hyperparameters = {**DEFAULT_ENSEMBLE_PARAMS, **hyperparameters}
        else:
            self.hyperparameters = DEFAULT_ENSEMBLE_PARAMS.copy()
            
        logger.debug(f"Initialized EnsembleModel with ID {model_id}, method {ensemble_method} and "
                    f"hyperparameters: {self.hyperparameters}")
    
    def add_model(self, model: BaseModel, weight: Optional[float] = None) -> 'EnsembleModel':
        """
        Add a model to the ensemble.
        
        Args:
            model: Trained model to add to the ensemble
            weight: Optional weight for this model in the ensemble
            
        Returns:
            Self for method chaining
        """
        # Verify the model is trained
        if not model.is_trained():
            raise ModelError(f"Cannot add untrained model {model.model_id} to ensemble")
        
        # Add model to the list
        self.models.append(model)
        
        # Handle weights
        if weight is not None:
            if self.model_weights is None:
                # Initialize weights list with equal weights for existing models
                self.model_weights = [1.0] * (len(self.models) - 1)
                self.model_weights.append(weight)
            else:
                self.model_weights.append(weight)
                
        logger.debug(f"Added model {model.model_id} to ensemble {self.model_id}")
        return self
    
    def remove_model(self, model_identifier: Union[int, str]) -> Optional[BaseModel]:
        """
        Remove a model from the ensemble by index or model_id.
        
        Args:
            model_identifier: Index or model_id of the model to remove
            
        Returns:
            The removed model or None if not found
        """
        # Check if identifier is an index
        if isinstance(model_identifier, int):
            if 0 <= model_identifier < len(self.models):
                removed_model = self.models.pop(model_identifier)
                
                # Update weights if they exist
                if self.model_weights is not None:
                    self.model_weights.pop(model_identifier)
                    
                logger.debug(f"Removed model at index {model_identifier} from ensemble {self.model_id}")
                return removed_model
            else:
                logger.warning(f"Invalid model index: {model_identifier}")
                return None
                
        # Check if identifier is a model_id
        elif isinstance(model_identifier, str):
            for i, model in enumerate(self.models):
                if model.model_id == model_identifier:
                    removed_model = self.models.pop(i)
                    
                    # Update weights if they exist
                    if self.model_weights is not None:
                        self.model_weights.pop(i)
                        
                    logger.debug(f"Removed model {model_identifier} from ensemble {self.model_id}")
                    return removed_model
                    
            logger.warning(f"Model not found: {model_identifier}")
            return None
            
        else:
            logger.warning(f"Invalid model identifier type: {type(model_identifier)}")
            return None
    
    def set_weights(self, weights: List[float]) -> 'EnsembleModel':
        """
        Set weights for all models in the ensemble.
        
        Args:
            weights: List of weights for each model
            
        Returns:
            Self for method chaining
        """
        # Validate weights length
        if len(weights) != len(self.models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(self.models)})")
        
        # Set weights
        self.model_weights = weights
        
        logger.debug(f"Set model weights for ensemble {self.model_id}: {weights}")
        return self
    
    @log_execution_time(logger, 'INFO')
    def train(self, features: DataFrameType, targets: SeriesType, params: Optional[Dict[str, Any]] = None) -> 'EnsembleModel':
        """
        Train all models in the ensemble on the provided features and targets.
        
        Args:
            features: Input features for training
            targets: Target values for training
            params: Optional parameters to pass to models during training
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting ensemble training for {self.model_id}")
        
        # If no models are provided, create default ensemble with XGBoost and LightGBM
        if not self.models:
            logger.info(f"No models provided, creating default ensemble with XGBoost and LightGBM")
            
            try:
                # Create XGBoost model
                xgb_model = XGBoostModel(f"{self.model_id}_xgb", version=self.version)
                
                # Create LightGBM model
                lgb_model = LightGBMModel(f"{self.model_id}_lgb", version=self.version)
                
                # Add models to the ensemble
                self.models = [xgb_model, lgb_model]
                
                # Initialize weights if needed
                if self.model_weights is None:
                    self.model_weights = [1.0] * len(self.models)
                    
            except Exception as e:
                error_msg = f"Error creating default models for ensemble {self.model_id}: {str(e)}"
                logger.error(error_msg)
                raise ModelTrainingError(error_msg) from e
                
        # Train each model in the ensemble
        for i, model in enumerate(self.models):
            try:
                logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_id}")
                model.train(features, targets, params)
            except Exception as e:
                error_msg = f"Error training model {model.model_id} in ensemble: {str(e)}"
                logger.error(error_msg)
                # Continue with other models even if one fails
                # Alternatively, we could raise an exception here to fail fast
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        # Update training date
        self.training_date = datetime.now()
        
        logger.info(f"Completed training ensemble {self.model_id} with {len(self.models)} models")
        return self
    
    def predict(self, features: DataFrameType) -> ArrayType:
        """
        Generate binary predictions using the ensemble of models.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Binary predictions (0 or 1)
        """
        # Validate that the ensemble is trained
        if not self.is_trained():
            raise ModelError(f"Ensemble {self.model_id} must be trained before prediction")
        
        # Validate features
        if not self.validate_features(features):
            raise ModelError(f"Features for ensemble {self.model_id} failed validation")
        
        # Get probability predictions
        probabilities = self.predict_proba(features)
        
        # Convert to binary predictions (threshold = 0.5)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, features: DataFrameType) -> ArrayType:
        """
        Generate probability predictions using the ensemble of models.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Probability predictions (values between 0 and 1)
        """
        # Validate that the ensemble is trained
        if not self.is_trained():
            raise ModelError(f"Ensemble {self.model_id} must be trained before prediction")
        
        # Validate features
        if not self.validate_features(features):
            raise ModelError(f"Features for ensemble {self.model_id} failed validation")
        
        # No models to predict with
        if not self.models:
            raise ModelError(f"Ensemble {self.model_id} has no models")
        
        # Get predictions from each model
        model_predictions = []
        for model in self.models:
            try:
                # Get probability predictions from the model
                model_preds = model.predict_proba(features)
                model_predictions.append(model_preds)
            except Exception as e:
                logger.error(f"Error getting predictions from model {model.model_id}: {str(e)}")
                # Skip this model's predictions
                continue
        
        # If no valid predictions were made, raise an error
        if not model_predictions:
            raise ModelError(f"No valid predictions from any model in ensemble {self.model_id}")
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == EnsembleMethod.VOTING.value:
            # Simple weighted average
            probabilities = np.zeros(len(features))
            
            if self.model_weights is not None:
                # Validate weights match number of predictions
                valid_weights = self.model_weights[:len(model_predictions)]
                
                # Normalize weights
                weight_sum = sum(valid_weights)
                norm_weights = [w / weight_sum for w in valid_weights]
                
                # Weighted average
                for i, preds in enumerate(model_predictions):
                    probabilities += norm_weights[i] * preds
            else:
                # Simple average
                for preds in model_predictions:
                    probabilities += preds
                probabilities /= len(model_predictions)
        
        # Other ensemble methods could be implemented here...
        else:
            # Default to simple average if method not implemented
            logger.warning(f"Ensemble method {self.ensemble_method} not implemented, using simple average")
            probabilities = np.mean(model_predictions, axis=0)
        
        return probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get aggregated feature importance scores from all models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Validate that the ensemble is trained
        if not self.is_trained():
            raise ModelError(f"Ensemble {self.model_id} must be trained before getting feature importance")
        
        # Initialize empty dictionary for aggregated importance
        aggregated_importance = {}
        
        # Get feature importance from each model
        for i, model in enumerate(self.models):
            try:
                # Get feature importance dictionary
                importance = model.get_feature_importance()
                
                # Weight to apply (default = 1.0)
                weight = 1.0
                if self.model_weights is not None:
                    weight = self.model_weights[i]
                
                # Add to aggregated importance
                for feature, value in importance.items():
                    if feature in aggregated_importance:
                        aggregated_importance[feature] += value * weight
                    else:
                        aggregated_importance[feature] = value * weight
            except Exception as e:
                logger.warning(f"Error getting feature importance from model {model.model_id}: {str(e)}")
                # Continue with other models
        
        # If no feature importance could be aggregated, return empty dict
        if not aggregated_importance:
            logger.warning(f"No feature importance information available for ensemble {self.model_id}")
            return {}
        
        # Normalize importance values to sum to 1.0
        total_importance = sum(aggregated_importance.values())
        if total_importance > 0:
            for feature in aggregated_importance:
                aggregated_importance[feature] /= total_importance
        
        # Sort by importance (descending)
        aggregated_importance = dict(sorted(
            aggregated_importance.items(), 
            key=lambda item: item[1],
            reverse=True
        ))
        
        return aggregated_importance
    
    def is_trained(self) -> bool:
        """
        Check if the ensemble model has been trained.
        
        Returns:
            True if the ensemble has models and all models are trained, False otherwise
        """
        # Check if there are models in the ensemble
        if not self.models:
            return False
        
        # Check if all models are trained
        for model in self.models:
            if not model.is_trained():
                return False
        
        return True
    
    @log_execution_time(logger, 'INFO')
    def save(self, path: Optional[PathType] = None) -> PathType:
        """
        Save the ensemble model to a directory with all component models.
        
        Args:
            path: Optional path to save the ensemble
            
        Returns:
            Path to the saved ensemble directory
        """
        # Validate that the ensemble is trained
        if not self.is_trained():
            raise ModelError(f"Ensemble {self.model_id} must be trained before saving")
        
        # Create a directory for the ensemble if path is provided
        if path is not None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
        
        # Save each individual model
        model_paths = []
        for i, model in enumerate(self.models):
            # Create subdirectory for the model
            model_dir = path / f"model_{i}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            model_path = model.save(model_dir)
            model_paths.append(str(model_path))
        
        # Create metadata for the ensemble
        ensemble_metadata = {
            "model_paths": model_paths,
            "model_weights": self.model_weights,
            "ensemble_method": self.ensemble_method,
        }
        
        # Update hyperparameters with ensemble metadata
        self.hyperparameters.update(ensemble_metadata)
        
        # Use parent class save method to save metadata
        return super().save(path)
    
    @log_execution_time(logger, 'INFO')
    def load(self, path: PathType) -> 'EnsembleModel':
        """
        Load an ensemble model from a directory.
        
        Args:
            path: Path to the saved ensemble directory
            
        Returns:
            Loaded ensemble model
        """
        # Use parent class load method to load metadata
        super().load(path)
        
        # Extract ensemble-specific metadata
        model_paths = self.hyperparameters.get("model_paths", [])
        self.model_weights = self.hyperparameters.get("model_weights")
        self.ensemble_method = self.hyperparameters.get("ensemble_method", EnsembleMethod.VOTING.value)
        
        # Load each individual model
        self.models = []
        for model_path in model_paths:
            try:
                # Determine model type from metadata in the directory
                model_type = None
                model_id = None
                
                # Load model metadata to determine type
                metadata_path = Path(model_path) / "metadata.json"
                if metadata_path.exists():
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        model_type = metadata.get("model_type")
                        model_id = metadata.get("model_id")
                
                # Create appropriate model instance
                if model_type == "xgboost":
                    model = XGBoostModel(model_id or f"{self.model_id}_xgb")
                elif model_type == "lightgbm":
                    model = LightGBMModel(model_id or f"{self.model_id}_lgb")
                else:
                    # Default to BaseModel for unknown types
                    logger.warning(f"Unknown model type {model_type}, loading as generic model")
                    # We can't instantiate BaseModel directly as it's abstract, so use a concrete model
                    model = XGBoostModel(model_id or f"{self.model_id}_unknown")
                
                # Load the model
                model.load(model_path)
                
                # Add to models list
                self.models.append(model)
                
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                # Continue with other models
        
        logger.info(f"Loaded ensemble {self.model_id} with {len(self.models)} models")
        return self
    
    @log_execution_time(logger, 'INFO')
    def evaluate(self, X_test: DataFrameType, y_test: SeriesType, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate the ensemble model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            threshold: Optional probability threshold for binary classification
            
        Returns:
            Dictionary of performance metrics
        """
        # Validate that the ensemble is trained
        if not self.is_trained():
            raise ModelError(f"Ensemble {self.model_id} must be trained before evaluation")
        
        # Use evaluate_model_performance function to calculate metrics
        metrics = evaluate_model_performance(self, X_test, y_test, threshold=threshold)
        
        # Optionally, also evaluate individual models
        # This could be useful for comparing ensemble performance with individual models
        for model in self.models:
            try:
                model_metrics = model.evaluate(X_test, y_test, threshold)
                logger.debug(f"Model {model.model_id} metrics: {model_metrics}")
            except Exception as e:
                logger.warning(f"Error evaluating model {model.model_id}: {str(e)}")
        
        # Update performance metrics
        self.performance_metrics = metrics
        
        logger.info(f"Ensemble {self.model_id} evaluation results: {metrics}")
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the ensemble model.
        
        Returns:
            Dictionary of model hyperparameters
        """
        # Create a copy of hyperparameters
        params = self.hyperparameters.copy()
        
        # Add ensemble-specific parameters
        params["model_count"] = len(self.models)
        params["ensemble_method"] = self.ensemble_method
        
        return params
    
    def set_params(self, params: Dict[str, Any]) -> 'EnsembleModel':
        """
        Update the hyperparameters of the ensemble model.
        
        Args:
            params: Dictionary of hyperparameters to update
            
        Returns:
            Self for method chaining
        """
        # Update hyperparameters
        self.hyperparameters.update(params)
        
        # Update ensemble_method if provided
        if "ensemble_method" in params:
            self.ensemble_method = params["ensemble_method"]
        
        # Log warning if model is already trained
        if self.is_trained():
            logger.warning(f"Ensemble {self.model_id} is already trained. "
                          f"Parameter changes won't affect existing models.")
        
        return self
    
    def get_models(self) -> List[BaseModel]:
        """
        Get the list of models in the ensemble.
        
        Returns:
            List of models
        """
        return self.models.copy()
    
    @log_execution_time(logger, 'INFO')
    def compare_model_performance(
        self, 
        X_test: DataFrameType, 
        y_test: SeriesType,
        threshold: Optional[float] = None
    ) -> DataFrameType:
        """
        Compare performance of individual models and the ensemble.
        
        Args:
            X_test: Test features
            y_test: Test targets
            threshold: Optional probability threshold for binary classification
            
        Returns:
            DataFrame with performance metrics for each model
        """
        # Create a list of all models including the ensemble
        all_models = self.models + [self]
        
        # Use compare_models function to compare all models
        comparison = compare_models(all_models, X_test, y_test)
        
        logger.info(f"Completed model performance comparison for ensemble {self.model_id}")
        
        return comparison