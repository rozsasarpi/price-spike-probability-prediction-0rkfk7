"""
Core module for training machine learning models in the ERCOT RTLMP spike prediction system.
Provides factory functions for model creation, standardized training workflows, and model selection capabilities to support the prediction of price spike probabilities.
"""

import numpy  # version 1.24+
import pandas  # version 2.0+
import typing  # Standard
from typing import Dict, List, Optional, Any, Tuple, Union, Literal, cast
import datetime  # Standard
import pathlib  # Standard
from pathlib import Path
import uuid  # Standard

import sklearn.model_selection  # version 1.2+
from sklearn.model_selection import TimeSeriesSplit  # version 1.2+
from sklearn.model_selection import RandomizedSearchCV  # version 1.2+

from .base_model import BaseModel  # ./base_model
from .xgboost_model import XGBoostModel  # ./xgboost_model
from .lightgbm_model import LightGBMModel  # ./lightgbm_model
from .evaluation import evaluate_model_performance  # ./evaluation
from ..data.storage.model_registry import ModelRegistry  # ../data/storage/model_registry
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ModelType, PathType  # ../utils/type_definitions
from ..utils.logging import get_logger, log_execution_time  # ../utils/logging
from ..utils.error_handling import ModelError, ModelTrainingError  # ../utils/error_handling

# Initialize logger
logger = get_logger(__name__)

# Define supported model types
MODEL_TYPES: Dict[str, Type] = {
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel
}

# Default model type
DEFAULT_MODEL_TYPE: str = "xgboost"

# Default train test split ratio
DEFAULT_TRAIN_TEST_SPLIT: float = 0.2

# Default threshold
DEFAULT_THRESHOLD: float = 0.5

# Default cross validation folds
DEFAULT_CV_FOLDS: int = 5


@log_execution_time(logger, 'INFO')
def create_model(
    model_id: str,
    model_type: ModelType,
    version: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> BaseModel:
    """
    Factory function to create a model instance of the specified type

    Args:
        model_id (str): Unique identifier for the model
        model_type (ModelType): Type of model to create (e.g., 'xgboost', 'lightgbm')
        version (Optional[str], optional): Version of the model. Defaults to None.
        hyperparameters (Optional[Dict[str, Any]], optional): Dictionary of hyperparameters for the model. Defaults to None.

    Returns:
        BaseModel: Initialized model instance
    """
    # Validate that model_type is supported
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {', '.join(MODEL_TYPES.keys())}")

    # Get the model class from MODEL_TYPES dictionary
    model_class = MODEL_TYPES[model_type]

    # Create a new instance of the model class with provided parameters
    model = model_class(model_id=model_id, version=version, hyperparameters=hyperparameters)

    # Log the model creation
    logger.info(f"Created model {model_id} of type {model_type}")

    return model


def generate_model_id(prefix: str) -> str:
    """
    Generate a unique model identifier

    Args:
        prefix (str): Prefix for the model ID

    Returns:
        str: Unique model identifier
    """
    # Generate a UUID
    uuid_str = str(uuid.uuid4())

    # Truncate to first 8 characters
    truncated_uuid = uuid_str[:8]

    # Combine the prefix with the UUID string
    model_id = f"{prefix}_{truncated_uuid}"

    return model_id


def time_based_train_test_split(
    features: DataFrameType,
    targets: SeriesType,
    test_size: Optional[float] = None,
    shuffle: Optional[bool] = False
) -> Tuple[DataFrameType, DataFrameType, SeriesType, SeriesType]:
    """
    Split data into training and testing sets based on time

    Args:
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        test_size (Optional[float], optional): Fraction of data to use for testing. Defaults to DEFAULT_TRAIN_TEST_SPLIT.
        shuffle (Optional[bool], optional): Whether to shuffle the data before splitting. Defaults to False.

    Returns:
        Tuple[DataFrameType, DataFrameType, SeriesType, SeriesType]: Split data as (X_train, X_test, y_train, y_test)
    """
    # Ensure features and targets have the same length
    if len(features) != len(targets):
        raise ValueError(f"Features and targets must have the same length: {len(features)} vs {len(targets)}")

    # Set default test_size if not provided
    if test_size is None:
        test_size = DEFAULT_TRAIN_TEST_SPLIT

    # Sort features and targets by index (assuming time-based index)
    features = features.sort_index()
    targets = targets.sort_index()

    # Calculate split point
    split_index = int(len(features) * (1 - test_size))

    # Split data
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = targets[:split_index], targets[split_index:]

    return X_train, X_test, y_train, y_test


@log_execution_time(logger, 'INFO')
def _optimize_hyperparameters(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    param_grid: Dict[str, Any],
    optimization_method: Optional[str] = None,
    n_iterations: Optional[int] = None
) -> Dict[str, Any]:
    """
    Internal function to optimize hyperparameters for a model

    Args:
        model_type (ModelType): Type of model to optimize
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        param_grid (Dict[str, Any]): Dictionary of hyperparameters to optimize
        optimization_method (Optional[str], optional): Optimization method to use ('random_search', 'grid_search'). Defaults to 'random_search'.
        n_iterations (Optional[int], optional): Number of iterations for random search. Defaults to 10.

    Returns:
        Dict[str, Any]: Optimized hyperparameters
    """
    # Log the start of hyperparameter optimization
    logger.info(f"Starting hyperparameter optimization for model type {model_type}")

    # Set default values if not provided
    if optimization_method is None:
        optimization_method = 'random_search'
    if n_iterations is None:
        n_iterations = 10

    # Prepare the model for hyperparameter optimization
    model = create_model(model_id="temp_model", model_type=model_type)

    if optimization_method == 'random_search':
        # Create a RandomizedSearchCV instance
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iterations, scoring='roc_auc', cv=3, random_state=42)

        # Fit the RandomizedSearchCV on the features and targets
        search.fit(features, targets)

        # Extract the best hyperparameters
        best_params = search.best_params_
    elif optimization_method == 'grid_search':
        # Implement grid search here (not implemented in this example)
        # This is a placeholder for a more complex grid search implementation
        best_params = {}  # Placeholder
        logger.warning("Grid search is not fully implemented. Returning empty hyperparameter set.")
    else:
        raise ValueError(f"Unsupported optimization method: {optimization_method}")

    # Log the best hyperparameters found
    logger.info(f"Best hyperparameters found: {best_params}")

    return best_params


@log_execution_time(logger, 'INFO')
def cross_validate_time_series(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    hyperparameters: Optional[Dict[str, Any]] = None,
    n_splits: Optional[int] = None,
    cv_strategy: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Perform time series cross-validation for model evaluation

    Args:
        model_type (ModelType): Type of model to train
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        hyperparameters (Optional[Dict[str, Any]], optional): Dictionary of hyperparameters for the model. Defaults to None.
        n_splits (Optional[int], optional): Number of cross-validation splits. Defaults to DEFAULT_CV_FOLDS.
        cv_strategy (Optional[str], optional): Cross validation strategy. Defaults to None.

    Returns:
        Dict[str, List[float]]: Dictionary of metric names mapped to lists of scores
    """
    # Set default values if not provided
    if n_splits is None:
        n_splits = DEFAULT_CV_FOLDS

    # Ensure features and targets have the same length
    if len(features) != len(targets):
        raise ValueError(f"Features and targets must have the same length: {len(features)} vs {len(targets)}")

    # Sort features and targets by index (assuming time-based index)
    features = features.sort_index()
    targets = targets.sort_index()

    # Create TimeSeriesSplit with n_splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize empty lists to store metrics for each fold
    auc_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []

    # Perform cross-validation
    for train_index, test_index in tscv.split(features, targets):
        # Split data into training and testing sets
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]

        # Create a model instance
        model = create_model(model_id="cv_model", model_type=model_type, hyperparameters=hyperparameters)

        # Train the model
        model.train(X_train, y_train)

        # Evaluate the model
        metrics = evaluate_model_performance(model, X_test, y_test)

        # Record the performance metrics
        auc_scores.append(metrics.get('auc', 0.0))
        precision_scores.append(metrics.get('precision', 0.0))
        recall_scores.append(metrics.get('recall', 0.0))
        f1_scores.append(metrics.get('f1', 0.0))

    # Aggregate metrics across all folds
    metrics = {
        'auc': auc_scores,
        'precision': precision_scores,
        'recall': recall_scores,
        'f1': f1_scores
    }

    return metrics


@log_execution_time(logger, 'INFO')
def train_model(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    hyperparameters: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
    model_path: Optional[PathType] = None
) -> Tuple[BaseModel, Dict[str, float]]:
    """
    Train a model with the specified parameters

    Args:
        model_type (ModelType): Type of model to train
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        hyperparameters (Optional[Dict[str, Any]], optional): Dictionary of hyperparameters for the model. Defaults to None.
        model_id (Optional[str], optional): Unique identifier for the model. Defaults to None.
        model_path (Optional[PathType], optional): Path to save the trained model. Defaults to None.

    Returns:
        Tuple[BaseModel, Dict[str, float]]: Trained model and performance metrics
    """
    # Log the start of model training
    logger.info(f"Starting training for model type {model_type}")

    # Generate a model ID if not provided
    if model_id is None:
        model_id = generate_model_id(prefix=model_type)

    # Create a model instance
    model = create_model(model_id=model_id, model_type=model_type, hyperparameters=hyperparameters)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = time_based_train_test_split(features, targets)

    # Train the model
    model.train(X_train, y_train)

    # Evaluate the model
    metrics = evaluate_model_performance(model, X_test, y_test)

    # Save the model if a path is provided
    if model_path:
        model.save(path=model_path)

    # Log the completion of model training
    logger.info(f"Completed training for model {model_id} with metrics: {metrics}")

    return model, metrics


@log_execution_time(logger, 'INFO')
def train_and_evaluate(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    hyperparameters: Optional[Dict[str, Any]] = None,
    cv_folds: Optional[int] = None,
    cv_strategy: Optional[str] = None
) -> Tuple[BaseModel, Dict[str, List[float]]]:
    """
    Train a model and evaluate its performance with cross-validation

    Args:
        model_type (ModelType): Type of model to train
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        hyperparameters (Optional[Dict[str, Any]], optional): Dictionary of hyperparameters for the model. Defaults to None.
        cv_folds (Optional[int], optional): Number of cross-validation folds. Defaults to DEFAULT_CV_FOLDS.
        cv_strategy (Optional[str], optional): Cross validation strategy. Defaults to None.

    Returns:
        Tuple[BaseModel, Dict[str, List[float]]]: Trained model and cross-validation metrics
    """
    # Log the start of model training with cross-validation
    logger.info(f"Starting training with cross-validation for model type {model_type}")

    # Generate a model ID
    model_id = generate_model_id(prefix=model_type)

    # Perform cross-validation
    metrics = cross_validate_time_series(model_type, features, targets, hyperparameters, n_splits=cv_folds, cv_strategy=cv_strategy)

    # Train a final model on the full dataset
    model = create_model(model_id=model_id, model_type=model_type, hyperparameters=hyperparameters)
    model.train(features, targets)

    # Log the completion of model training with cross-validation metrics
    logger.info(f"Completed training with cross-validation for model {model_id} with metrics: {metrics}")

    return model, metrics


@log_execution_time(logger, 'INFO')
def optimize_and_train(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    param_grid: Optional[Dict[str, Any]] = None,
    optimization_method: Optional[str] = None,
    n_iterations: Optional[int] = None,
    model_path: Optional[PathType] = None
) -> Tuple[BaseModel, Dict[str, float]]:
    """
    Optimize hyperparameters and train a model

    Args:
        model_type (ModelType): Type of model to train
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        param_grid (Optional[Dict[str, Any]], optional): Dictionary of hyperparameters to optimize. Defaults to None.
        optimization_method (Optional[str], optional): Optimization method to use ('random_search', 'grid_search'). Defaults to None.
        n_iterations (Optional[int], optional): Number of iterations for random search. Defaults to 10.
        model_path (Optional[PathType], optional): Path to save the trained model. Defaults to None.

    Returns:
        Tuple[BaseModel, Dict[str, float]]: Optimized model and performance metrics
    """
    # Log the start of hyperparameter optimization and training
    logger.info(f"Starting hyperparameter optimization and training for model type {model_type}")

    # Generate a model ID
    model_id = generate_model_id(prefix=model_type)

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = time_based_train_test_split(features, targets)

    # Optimize hyperparameters
    best_params = _optimize_hyperparameters(model_type, X_train, y_train, param_grid, optimization_method, n_iterations)

    # Train a model with the optimal hyperparameters
    model = create_model(model_id=model_id, model_type=model_type, hyperparameters=best_params)
    model.train(X_train, y_train)

    # Evaluate the model
    metrics = evaluate_model_performance(model, X_test, y_test)

    # Save the model if a path is provided
    if model_path:
        model.save(path=model_path)

    # Log the completion of optimization and training
    logger.info(f"Completed hyperparameter optimization and training for model {model_id} with metrics: {metrics}")

    return model, metrics


@log_execution_time(logger, 'INFO')
def load_model(
    model_identifier: Union[str, PathType],
    model_type: Optional[str] = None,
    version: Optional[str] = None,
    registry_path: Optional[PathType] = None
) -> BaseModel:
    """
    Load a model from the registry or a file path

    Args:
        model_identifier (Union[str, PathType]): Model ID or path to the model file
        model_type (Optional[str], optional): Type of model to load. Required if loading from a file path. Defaults to None.
        version (Optional[str], optional): Version of the model to load. Defaults to None.
        registry_path (Optional[PathType], optional): Path to the model registry. Defaults to None.

    Returns:
        BaseModel: Loaded model instance
    """
    # Log the start of model loading
    logger.info(f"Loading model with identifier {model_identifier}")

    # If model_identifier is a Path object or a string path to a file
    if isinstance(model_identifier, Path) or isinstance(model_identifier, str) and Path(model_identifier).exists():
        # Determine model type from the file name if not provided
        if model_type is None:
            if "xgboost" in str(model_identifier).lower():
                model_type = "xgboost"
            elif "lightgbm" in str(model_identifier).lower():
                model_type = "lightgbm"
            else:
                raise ValueError("Model type must be specified when loading from a file path.")

        # Create a model instance
        model = create_model(model_id="file_model", model_type=model_type)

        # Load the model from the file path
        model.load(path=Path(model_identifier))
    else:  # model_identifier is a model ID
        # Initialize the ModelRegistry
        model_registry = ModelRegistry(registry_path=registry_path)

        # Get the model from the registry
        model, _ = model_registry.get_model(model_type=model_type, model_id=model_identifier, version=version)

    # Log the successful model loading
    logger.info(f"Successfully loaded model {model.model_id} of type {model.model_type}")

    return model


@log_execution_time(logger, 'INFO')
def get_latest_model(
    model_type: ModelType,
    registry_path: Optional[PathType] = None
) -> Optional[BaseModel]:
    """
    Get the latest model of a specific type from the registry

    Args:
        model_type (ModelType): Type of model to retrieve
        registry_path (Optional[PathType], optional): Path to the model registry. Defaults to None.

    Returns:
        Optional[BaseModel]: Latest model instance or None if not found
    """
    # Log the attempt to get the latest model
    logger.info(f"Attempting to get the latest model of type {model_type}")

    # Initialize the ModelRegistry
    model_registry = ModelRegistry(registry_path=registry_path)

    try:
        # Get the latest model
        model, _ = model_registry.get_latest_model(model_type=model_type)

        # Log the successful retrieval
        logger.info(f"Successfully retrieved the latest model of type {model_type}")
        return model
    except ModelError as e:
        # Log a warning if no model is found
        logger.warning(f"No model found: {e}")
        return None
    except Exception as e:
        # Log the error
        logger.error(f"Error getting the latest model: {e}")
        return None


@log_execution_time(logger, 'INFO')
def compare_models(
    models: List[BaseModel],
    features: DataFrameType,
    targets: SeriesType,
    threshold: Optional[float] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test data

    Args:
        models (List[BaseModel]): List of trained models to compare
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        threshold (Optional[float], optional): Threshold for evaluating the model. Defaults to None.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary of model IDs mapped to their performance metrics
    """
    # Log the start of model comparison
    logger.info(f"Starting comparison of {len(models)} models")

    # Initialize an empty dictionary for results
    results: Dict[str, Dict[str, float]] = {}

    # Evaluate each model
    for model in models:
        # Evaluate the model
        metrics = evaluate_model_performance(model, features, targets, threshold=threshold)

        # Store the performance metrics
        results[model.model_id] = metrics

    # Log the completion of model comparison
    logger.info(f"Completed comparison of {len(models)} models")

    return results


@log_execution_time(logger, 'INFO')
def select_best_model(
    models: List[BaseModel],
    features: DataFrameType,
    targets: SeriesType,
    metric: str,
    higher_is_better: bool
) -> Tuple[BaseModel, float]:
    """
    Select the best model from a list based on a specific metric

    Args:
        models (List[BaseModel]): List of trained models to compare
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        metric (str): Metric to use for comparison
        higher_is_better (bool): Whether a higher metric value is better

    Returns:
        Tuple[BaseModel, float]: Best model and its metric value
    """
    # Log the start of best model selection
    logger.info(f"Starting selection of the best model based on metric {metric}")

    # Compare all models
    model_metrics = compare_models(models, features, targets)

    # Initialize variables to track the best model and metric value
    best_model: Optional[BaseModel] = None
    best_metric_value: float = float('-inf') if higher_is_better else float('inf')

    # Iterate through the models and their metrics
    for model in models:
        # Get the metric value for the current model
        current_metric_value = model_metrics.get(model.model_id, {}).get(metric)

        # Check if the current model is better than the best model so far
        if current_metric_value is not None:
            if higher_is_better and current_metric_value > best_metric_value:
                best_model = model
                best_metric_value = current_metric_value
            elif not higher_is_better and current_metric_value < best_metric_value:
                best_model = model
                best_metric_value = current_metric_value

    # Log the selected best model
    if best_model is not None:
        logger.info(f"Selected best model {best_model.model_id} with {metric} = {best_metric_value}")
    else:
        logger.warning("No model was selected as the best.")

    return best_model, best_metric_value


@log_execution_time(logger, 'INFO')
def retrain_model(
    model: BaseModel,
    features: DataFrameType,
    targets: SeriesType,
    hyperparameters: Optional[Dict[str, Any]] = None,
    model_path: Optional[PathType] = None,
    increment_type: Optional[str] = None
) -> Tuple[BaseModel, Dict[str, float]]:
    """
    Retrain an existing model with new data

    Args:
        model (BaseModel): Existing model to retrain
        features (DataFrameType): DataFrame of input features
        targets (SeriesType): Series of target values
        hyperparameters (Optional[Dict[str, Any]], optional): Dictionary of hyperparameters to update. Defaults to None.
        model_path (Optional[PathType], optional): Path to save the retrained model. Defaults to None.
        increment_type (Optional[str], optional): Type of version increment ('major', 'minor', 'patch'). Defaults to None.

    Returns:
        Tuple[BaseModel, Dict[str, float]]: Retrained model and performance metrics
    """
    # Log the start of model retraining
    logger.info(f"Starting retraining for model {model.model_id}")

    # Get the model's current configuration
    model_type = model.model_type

    # If hyperparameters are provided, update the model's hyperparameters
    if hyperparameters:
        model.hyperparameters.update(hyperparameters)

    # Train the model on the new data
    model.train(features, targets)

    # Evaluate the model's performance
    metrics = evaluate_model_performance(model, features, targets)

    # If model_path is provided, save the model
    if model_path:
        # Initialize the ModelRegistry
        model_registry = ModelRegistry()

        # Register the model with a new version
        model_registry.register_model(model_obj=model, metadata=model.get_model_config(), increment_type=increment_type)

    # Log the completion of model retraining
    logger.info(f"Completed retraining for model {model.model_id} with metrics: {metrics}")

    return model, metrics


@log_execution_time(logger, 'INFO')
def schedule_retraining(
    model_id: str,
    data_provider: Callable[[], Tuple[DataFrameType, SeriesType]],
    days_interval: int,
    registry_path: Optional[PathType] = None
) -> bool:
    """
    Schedule model retraining on a regular cadence

    Args:
        model_id (str): Unique identifier for the model
        data_provider (Callable[[], Tuple[DataFrameType, SeriesType]]): Function that provides the training data
        days_interval (int): Number of days between retraining runs
        registry_path (Optional[PathType], optional): Path to the model registry. Defaults to None.

    Returns:
        bool: True if scheduling was successful, False otherwise
    """
    # Log the scheduling of model retraining
    logger.info(f"Scheduling model retraining for model {model_id} every {days_interval} days")

    # Initialize the ModelRegistry
    model_registry = ModelRegistry(registry_path=registry_path)

    try:
        # Get the latest model
        model, metadata = model_registry.get_latest_model(model_type=metadata['model_type'], model_id=model_id)

        # Get the last training date from the model metadata
        last_training_date = metadata.get('training_date')

        # Calculate the next training date
        next_training_date = last_training_date + datetime.timedelta(days=days_interval)

        # Check if the current date is after the next training date
        if datetime.datetime.now() >= next_training_date:
            # Get new data
            features, targets = data_provider()

            # Retrain the model
            retrain_model(model, features, targets)

            return True
        else:
            logger.info(f"Retraining is not yet due for model {model_id}")
            return False
    except ModelError as e:
        logger.error(f"Model not found: {e}")
        return False