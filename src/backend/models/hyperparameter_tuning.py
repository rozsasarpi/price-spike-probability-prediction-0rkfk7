"""
Module for hyperparameter optimization in the ERCOT RTLMP spike prediction system.
Provides various optimization strategies including grid search, random search, and Bayesian optimization to find optimal hyperparameters for machine learning models.
"""

import typing  # Standard
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Literal

import numpy  # version 1.24+
import pandas  # version 2.0+
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit  # version 1.2+
import joblib  # version 1.2+
import optuna  # version 3.0+

from .base_model import BaseModel  # ./base_model
from .xgboost_model import XGBoostModel  # ./xgboost_model
from .lightgbm_model import LightGBMModel  # ./lightgbm_model
from .training import create_model  # ./training
from .cross_validation import cross_validate_model, time_series_split  # ./cross_validation
from .evaluation import evaluate_model_performance  # ./evaluation
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ModelType  # ../utils/type_definitions
from ..utils.logging import get_logger, log_execution_time  # ../utils/logging
from ..utils.error_handling import ModelError, ModelTrainingError, handle_errors  # ../utils/error_handling

# Initialize logger
logger = get_logger(__name__)

# Default values for hyperparameter optimization
DEFAULT_N_ITERATIONS: int = 50
DEFAULT_CV_FOLDS: int = 5
DEFAULT_SCORING: str = "roc_auc"
DEFAULT_OPTIMIZATION_METHOD: str = "random_search"

# Supported optimization methods
OPTIMIZATION_METHODS: List[str] = ["grid_search", "random_search", "bayesian_optimization"]

# Default parameter grid for XGBoost models
XGBOOST_PARAM_GRID: Dict[str, List[Any]] = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'n_estimators': [100, 200, 300, 400, 500]
}

# Default parameter grid for LightGBM models
LIGHTGBM_PARAM_GRID: Dict[str, List[Any]] = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [20, 31, 50, 70, 100, 150],
    'max_depth': [-1, 5, 10, 15, 20],
    'min_data_in_leaf': [10, 20, 30, 50, 100],
    'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_freq': [0, 5, 10]
}


def get_default_param_grid(model_type: ModelType) -> Dict[str, List[Any]]:
    """
    Get the default parameter grid for a specific model type

    Args:
        model_type: model_type

    Returns:
        Dict[str, List[Any]]: Parameter grid for the specified model type
    """
    if model_type == 'xgboost':
        return XGBOOST_PARAM_GRID
    elif model_type == 'lightgbm':
        return LIGHTGBM_PARAM_GRID
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


@log_execution_time(logger, 'INFO')
@handle_errors(ModelError, reraise=True)
def grid_search_cv(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    param_grid: Dict[str, List[Any]],
    cv_folds: Optional[int] = None,
    scoring: Optional[str] = None,
    verbose: Optional[bool] = None,
    n_jobs: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Perform grid search cross-validation for hyperparameter optimization

    Args:
        model_type: model_type
        features: features
        targets: targets
        param_grid: param_grid
        cv_folds: cv_folds
        scoring: scoring
        verbose: verbose
        n_jobs: n_jobs

    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: Best parameters and corresponding scores
    """
    logger.info("Starting grid search optimization")

    # Set default values if not provided
    cv_folds = cv_folds if cv_folds is not None else DEFAULT_CV_FOLDS
    scoring = scoring if scoring is not None else DEFAULT_SCORING
    verbose = verbose if verbose is not None else 0
    n_jobs = n_jobs if n_jobs is not None else -1

    # Create a model instance
    model = create_model(model_id="grid_search_model", model_type=model_type)

    # Create a TimeSeriesSplit cross-validator
    cv = TimeSeriesSplit(n_splits=cv_folds)

    # Create a GridSearchCV instance
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs
    )

    # Fit the GridSearchCV on the features and targets
    grid_search.fit(features, targets)

    # Extract the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best score found: {best_score}")

    return best_params, {'score': best_score}


@log_execution_time(logger, 'INFO')
@handle_errors(ModelError, reraise=True)
def random_search_cv(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    param_grid: Dict[str, List[Any]],
    n_iterations: Optional[int] = None,
    cv_folds: Optional[int] = None,
    scoring: Optional[str] = None,
    verbose: Optional[bool] = None,
    n_jobs: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Perform random search cross-validation for hyperparameter optimization

    Args:
        model_type: model_type
        features: features
        targets: targets
        param_grid: param_grid
        n_iterations: n_iterations
        cv_folds: cv_folds
        scoring: scoring
        verbose: verbose
        n_jobs: n_jobs

    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: Best parameters and corresponding scores
    """
    logger.info("Starting random search optimization")

    # Set default values if not provided
    n_iterations = n_iterations if n_iterations is not None else DEFAULT_N_ITERATIONS
    cv_folds = cv_folds if cv_folds is not None else DEFAULT_CV_FOLDS
    scoring = scoring if scoring is not None else DEFAULT_SCORING
    verbose = verbose if verbose is not None else 0
    n_jobs = n_jobs if n_jobs is not None else -1

    # Create a model instance
    model = create_model(model_id="random_search_model", model_type=model_type)

    # Create a TimeSeriesSplit cross-validator
    cv = TimeSeriesSplit(n_splits=cv_folds)

    # Create a RandomizedSearchCV instance
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iterations,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=42  # Add random_state for reproducibility
    )

    # Fit the RandomizedSearchCV on the features and targets
    random_search.fit(features, targets)

    # Extract the best parameters and best score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best score found: {best_score}")

    return best_params, {'score': best_score}


@log_execution_time(logger, 'INFO')
@handle_errors(ModelError, reraise=True)
def bayesian_optimization(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    param_grid: Dict[str, List[Any]],
    n_trials: Optional[int] = None,
    cv_folds: Optional[int] = None,
    scoring: Optional[str] = None,
    verbose: Optional[bool] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Perform Bayesian optimization for hyperparameter tuning using Optuna

    Args:
        model_type: model_type
        features: features
        targets: targets
        param_grid: param_grid
        n_trials: n_trials
        cv_folds: cv_folds
        scoring: scoring
        verbose: verbose

    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: Best parameters and corresponding scores
    """
    logger.info("Starting Bayesian optimization")

    # Set default values if not provided
    n_trials = n_trials if n_trials is not None else DEFAULT_N_ITERATIONS
    cv_folds = cv_folds if cv_folds is not None else DEFAULT_CV_FOLDS
    scoring = scoring if scoring is not None else DEFAULT_SCORING
    verbose = verbose if verbose is not None else 0

    # Create a TimeSeriesSplit cross-validator
    cv = TimeSeriesSplit(n_splits=cv_folds)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Suggest hyperparameters based on the parameter grid
        params = {}
        for param_name, param_values in param_grid.items():
            if isinstance(param_values, list):
                # Suggest a categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            else:
                raise ValueError(f"Unsupported parameter type: {type(param_values)}")

        # Create a model with the suggested hyperparameters
        model = create_model(model_id="optuna_model", model_type=model_type, hyperparameters=params)

        # Perform cross-validation
        scores = []
        for train_index, test_index in cv.split(features, targets):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]

            # Train the model
            model.train(X_train, y_train)

            # Evaluate the model
            metrics = evaluate_model_performance(model, X_test, y_test)
            scores.append(metrics.get(scoring, 0.0))

        # Return the negative mean score (Optuna minimizes)
        return -numpy.mean(scores)

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Run the optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=(verbose > 0))

    # Extract the best parameters
    best_params = study.best_params

    # Calculate the best score (negative of the study's best value)
    best_score = -study.best_value

    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best score found: {best_score}")

    return best_params, {'score': best_score}


@log_execution_time(logger, 'INFO')
@handle_errors(ModelError, reraise=True)
def optimize_hyperparameters(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    optimization_method: Optional[str] = None,
    n_iterations: Optional[int] = None,
    cv_folds: Optional[int] = None,
    scoring: Optional[str] = None,
    verbose: Optional[bool] = None,
    n_jobs: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Optimize hyperparameters using the specified method

    Args:
        model_type: model_type
        features: features
        targets: targets
        param_grid: param_grid
        optimization_method: optimization_method
        n_iterations: n_iterations
        cv_folds: cv_folds
        scoring: scoring
        verbose: verbose
        n_jobs: n_jobs

    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: Best parameters and corresponding scores
    """
    logger.info("Starting hyperparameter optimization")

    # Set default values if not provided
    optimization_method = optimization_method if optimization_method is not None else DEFAULT_OPTIMIZATION_METHOD
    n_iterations = n_iterations if n_iterations is not None else DEFAULT_N_ITERATIONS
    cv_folds = cv_folds if cv_folds is not None else DEFAULT_CV_FOLDS
    scoring = scoring if scoring is not None else DEFAULT_SCORING
    verbose = verbose if verbose is not None else 0
    n_jobs = n_jobs if n_jobs is not None else -1

    # Get default param_grid if not provided
    if param_grid is None:
        param_grid = get_default_param_grid(model_type)

    # Validate optimization_method
    if optimization_method not in OPTIMIZATION_METHODS:
        raise ValueError(f"Unsupported optimization method: {optimization_method}. Supported methods are: {', '.join(OPTIMIZATION_METHODS)}")

    if optimization_method == 'grid_search':
        best_params, scores = grid_search_cv(model_type, features, targets, param_grid, cv_folds, scoring, verbose, n_jobs)
    elif optimization_method == 'random_search':
        best_params, scores = random_search_cv(model_type, features, targets, param_grid, n_iterations, cv_folds, scoring, verbose, n_jobs)
    elif optimization_method == 'bayesian_optimization':
        best_params, scores = bayesian_optimization(model_type, features, targets, param_grid, n_trials=n_iterations, cv_folds=cv_folds, scoring=scoring, verbose=verbose)
    else:
        raise ValueError(f"Unsupported optimization method: {optimization_method}")

    logger.info("Completed hyperparameter optimization")

    return best_params, scores


@log_execution_time(logger, 'INFO')
@handle_errors(ModelError, reraise=True)
def optimize_and_train(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    optimization_method: Optional[str] = None,
    n_iterations: Optional[int] = None,
    cv_folds: Optional[int] = None,
    scoring: Optional[str] = None,
    model_id: Optional[str] = None
) -> Tuple[BaseModel, Dict[str, float]]:
    """
    Optimize hyperparameters and train a model with the best parameters

    Args:
        model_type: model_type
        features: features
        targets: targets
        param_grid: param_grid
        optimization_method: optimization_method
        n_iterations: n_iterations
        cv_folds: cv_folds
        scoring: scoring
        model_id: model_id

    Returns:
        Tuple[BaseModel, Dict[str, float]]: Trained model with optimal parameters and performance metrics
    """
    logger.info("Starting optimization and training")

    # Optimize hyperparameters
    best_params, scores = optimize_hyperparameters(
        model_type=model_type,
        features=features,
        targets=targets,
        param_grid=param_grid,
        optimization_method=optimization_method,
        n_iterations=n_iterations,
        cv_folds=cv_folds,
        scoring=scoring
    )

    # Create a model instance with the best parameters
    model = create_model(model_id=model_id or "optimized_model", model_type=model_type, hyperparameters=best_params)

    # Train the model on the full dataset
    model.train(features, targets)

    # Evaluate the model's performance
    metrics = evaluate_model_performance(model, features, targets)

    logger.info("Completed optimization and training")

    return model, metrics


@log_execution_time(logger, 'INFO')
@handle_errors(ModelError, reraise=True)
def cross_validate_with_optimization(
    model_type: ModelType,
    features: DataFrameType,
    targets: SeriesType,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    optimization_method: Optional[str] = None,
    n_iterations: Optional[int] = None,
    cv_folds: Optional[int] = None,
    scoring: Optional[str] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Perform cross-validation with hyperparameter optimization for each fold

    Args:
        model_type: model_type
        features: features
        targets: targets
        param_grid: param_grid
        optimization_method: optimization_method
        n_iterations: n_iterations
        cv_folds: cv_folds
        scoring: scoring

    Returns:
        Dict[int, Dict[str, Any]]: Results for each fold including parameters and metrics
    """
    logger.info("Starting cross-validation with optimization")

    # Set default values if not provided
    optimization_method = optimization_method if optimization_method is not None else DEFAULT_OPTIMIZATION_METHOD
    n_iterations = n_iterations if n_iterations is not None else DEFAULT_N_ITERATIONS
    cv_folds = cv_folds if cv_folds is not None else DEFAULT_CV_FOLDS
    scoring = scoring if scoring is not None else DEFAULT_SCORING

    # Get default param_grid if not provided
    if param_grid is None:
        param_grid = get_default_param_grid(model_type)

    # Create time series splits
    tscv = time_series_split(n_splits=cv_folds)

    # Initialize results dictionary
    results = {}

    # Iterate through each fold
    for fold_index, (train_index, test_index) in enumerate(tscv.split(features, targets)):
        logger.info(f"Starting fold {fold_index + 1}")

        # Split data into training and testing sets
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]

        # Optimize hyperparameters on the training data
        best_params, scores = optimize_hyperparameters(
            model_type=model_type,
            features=X_train,
            targets=y_train,
            param_grid=param_grid,
            optimization_method=optimization_method,
            n_iterations=n_iterations,
            cv_folds=cv_folds,
            scoring=scoring
        )

        # Create a model with the best parameters
        model = create_model(model_id=f"fold_{fold_index}_model", model_type=model_type, hyperparameters=best_params)

        # Train the model on the training data
        model.train(X_train, y_train)

        # Evaluate the model on the test data
        metrics = evaluate_model_performance(model, X_test, y_test)

        # Store the parameters and metrics for this fold
        results[fold_index] = {
            'parameters': best_params,
            'metrics': metrics
        }

        logger.info(f"Completed fold {fold_index + 1} with metrics: {metrics}")

    logger.info("Completed cross-validation with optimization")

    return results


def generate_hyperparameter_report(optimization_results: Dict[str, Any], model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive report on hyperparameter optimization results

    Args:
        optimization_results: optimization_results
        model_info: model_info

    Returns:
        Dict[str, Any]: Hyperparameter optimization report
    """
    report: Dict[str, Any] = {}

    # Initialize report dictionary with timestamp and model info if provided
    report['timestamp'] = str(datetime.datetime.now())
    if model_info:
        report['model_info'] = model_info

    # Extract best parameters and scores
    best_params = optimization_results.get('parameters', {})
    best_score = optimization_results.get('score', None)

    # Add best parameters section
    report['best_parameters'] = best_params

    # Add performance metrics section
    report['performance_metrics'] = {'score': best_score} if best_score is not None else {}

    # If optimization_results contains parameter importance, add that section
    if 'parameter_importance' in optimization_results:
        report['parameter_importance'] = optimization_results['parameter_importance']

    # Add summary section with overall assessment
    report['summary'] = "Hyperparameter optimization completed successfully."

    return report


class HyperparameterOptimizer:
    """
    Class for managing hyperparameter optimization workflows
    """

    _model_type: ModelType
    _param_grid: Optional[Dict[str, List[Any]]]
    _optimization_method: str
    _n_iterations: int
    _cv_folds: int
    _scoring: str
    _optimization_results: Dict[str, Any]
    _best_params: Optional[Dict[str, Any]]

    def __init__(
        self,
        model_type: ModelType,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        optimization_method: Optional[str] = None,
        n_iterations: Optional[int] = None,
        cv_folds: Optional[int] = None,
        scoring: Optional[str] = None
    ):
        """
        Initialize the HyperparameterOptimizer with configuration

        Args:
            model_type: model_type
            param_grid: param_grid
            optimization_method: optimization_method
            n_iterations: n_iterations
            cv_folds: cv_folds
            scoring: scoring
        """
        self._model_type = model_type
        self._param_grid = param_grid if param_grid is not None else get_default_param_grid(model_type)
        self._optimization_method = optimization_method if optimization_method is not None else DEFAULT_OPTIMIZATION_METHOD
        self._n_iterations = n_iterations if n_iterations is not None else DEFAULT_N_ITERATIONS
        self._cv_folds = cv_folds if cv_folds is not None else DEFAULT_CV_FOLDS
        self._scoring = scoring if scoring is not None else DEFAULT_SCORING
        self._optimization_results = {}
        self._best_params = None

    @log_execution_time(logger, 'INFO')
    def optimize(self, features: DataFrameType, targets: SeriesType, verbose: Optional[bool] = None, n_jobs: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization

        Args:
            features: features
            targets: targets
            verbose: verbose
            n_jobs: n_jobs

        Returns:
            Dict[str, Any]: Best hyperparameters
        """
        logger.info("Starting hyperparameter optimization")

        best_params, scores = optimize_hyperparameters(
            model_type=self._model_type,
            features=features,
            targets=targets,
            param_grid=self._param_grid,
            optimization_method=self._optimization_method,
            n_iterations=self._n_iterations,
            cv_folds=self._cv_folds,
            scoring=self._scoring,
            verbose=verbose,
            n_jobs=n_jobs
        )

        self._optimization_results = {'parameters': best_params, 'score': scores['score']}
        self._best_params = best_params

        return best_params

    @log_execution_time(logger, 'INFO')
    def optimize_and_train(self, features: DataFrameType, targets: SeriesType, model_id: Optional[str] = None) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Optimize hyperparameters and train a model

        Args:
            features: features
            targets: targets
            model_id: model_id

        Returns:
            Tuple[BaseModel, Dict[str, float]]: Trained model and performance metrics
        """
        logger.info("Starting optimization and training")

        if self._best_params is None:
            self.optimize(features, targets)

        model, metrics = optimize_and_train(
            model_type=self._model_type,
            features=features,
            targets=targets,
            param_grid=self._param_grid,
            optimization_method=self._optimization_method,
            n_iterations=self._n_iterations,
            model_id=model_id
        )

        return model, metrics

    @log_execution_time(logger, 'INFO')
    def cross_validate(self, features: DataFrameType, targets: SeriesType) -> Dict[int, Dict[str, Any]]:
        """
        Perform cross-validation with hyperparameter optimization

        Args:
            features: features
            targets: targets

        Returns:
            Dict[int, Dict[str, Any]]: Results for each fold
        """
        logger.info("Starting cross-validation")

        results = cross_validate_with_optimization(
            model_type=self._model_type,
            features=features,
            targets=targets,
            param_grid=self._param_grid,
            optimization_method=self._optimization_method,
            n_iterations=self._n_iterations,
            cv_folds=self._cv_folds,
            scoring=self._scoring
        )

        self._optimization_results = results

        return results

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the best hyperparameters found during optimization

        Args:

        Returns:
            Optional[Dict[str, Any]]: Best hyperparameters or None if not optimized
        """
        return self._best_params

    def get_optimization_results(self) -> Dict[str, Any]:
        """
        Get the results of the optimization process

        Args:

        Returns:
            Dict[str, Any]: Optimization results
        """
        return self._optimization_results

    def generate_report(self, model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a report on the optimization results

        Args:
            model_info: model_info

        Returns:
            Dict[str, Any]: Optimization report
        """
        return generate_hyperparameter_report(self._optimization_results, model_info)

    def set_param_grid(self, param_grid: Dict[str, List[Any]]) -> None:
        """
        Set the parameter grid for optimization

        Args:
            param_grid: param_grid

        Returns:
            None
        """
        self._param_grid = param_grid
        self._best_params = None  # Reset best parameters since the parameter space has changed

    def set_optimization_method(self, optimization_method: str) -> None:
        """
        Set the optimization method

        Args:
            optimization_method: optimization_method

        Returns:
            None
        """
        if optimization_method not in OPTIMIZATION_METHODS:
            raise ValueError(f"Unsupported optimization method: {optimization_method}. Supported methods are: {', '.join(OPTIMIZATION_METHODS)}")
        self._optimization_method = optimization_method
        self._best_params = None  # Reset best parameters since the method has changed

    def set_n_iterations(self, n_iterations: int) -> None:
        """
        Set the number of iterations for optimization

        Args:
            n_iterations: n_iterations

        Returns:
            None
        """
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        self._n_iterations = n_iterations

    def set_cv_folds(self, cv_folds: int) -> None:
        """
        Set the number of cross-validation folds

        Args:
            cv_folds: cv_folds

        Returns:
            None
        """
        if cv_folds <= 0:
            raise ValueError("Number of cross-validation folds must be positive")
        self._cv_folds = cv_folds

    def set_scoring(self, scoring: str) -> None:
        """
        Set the scoring metric for optimization

        Args:
            scoring: scoring

        Returns:
            None
        """
        self._scoring = scoring