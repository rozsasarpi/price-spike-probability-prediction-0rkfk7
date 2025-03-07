"""
Implements the train command for the CLI application of the ERCOT RTLMP spike prediction system.
This module provides functionality to train machine learning models with specified parameters,
supporting hyperparameter optimization, cross-validation, and model evaluation.
"""

import typing  # Standard
from typing import Dict, List, Optional, Any, Tuple, cast  # typing
import datetime  # Standard
from pathlib import Path  # Standard
import pandas  # version 2.0+

from ..cli_types import TrainParamsDict  # ../cli_types
from ..exceptions import ModelOperationError, CommandError  # ../exceptions
from ..logger import get_cli_logger  # ../logger
from ..utils.validators import validate_train_params  # ../utils/validators
from ..utils.progress_bars import create_progress_bar, create_indeterminate_spinner  # ../utils/progress_bars
from ...backend.data.fetchers.ercot_api import ERCOTDataFetcher  # ../../backend/data/fetchers/ercot_api
from ...backend.features.feature_pipeline import FeaturePipeline  # ../../backend/features/feature_pipeline
from ...backend.models.training import train_model, train_and_evaluate, optimize_and_train, ModelTrainer  # ../../backend/models/training


logger = get_cli_logger('train_command')

DEFAULT_HYPERPARAMETER_GRID = {
    'xgboost': {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [100, 200, 300]
    },
    'lightgbm': {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [31, 63, 127],
        'max_depth': [3, 5, 7, 9],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [100, 200, 300]
    }
}


def fetch_training_data(start_date: datetime.date, end_date: datetime.date, nodes: List[str]) -> Dict[str, pandas.DataFrame]:
    """
    Fetches historical data for model training

    Args:
        start_date (datetime.date): Start date for training data
        end_date (datetime.date): End date for training data
        nodes (List[str]): List of node IDs to fetch data for

    Returns:
        Dict[str, pandas.DataFrame]: Dictionary of data sources for training
    """
    with create_indeterminate_spinner("Fetching training data...") as spinner:
        data_fetcher = ERCOTDataFetcher()
        rtlmp_df = data_fetcher.fetch_historical_data(start_date, end_date, nodes)
        grid_df = data_fetcher.fetch_historical_data(start_date, end_date, [])
        spinner.update("Training data fetched.")
    return {'rtlmp_df': rtlmp_df, 'grid_df': grid_df}


def prepare_training_features(data_sources: Dict[str, pandas.DataFrame], thresholds: List[float]) -> Dict[float, Tuple[pandas.DataFrame, pandas.Series]]:
    """
    Prepares features and targets for model training

    Args:
        data_sources (Dict[str, pandas.DataFrame]): Dictionary of data sources
        thresholds (List[float]): List of threshold values

    Returns:
        Dict[float, Tuple[pandas.DataFrame, pandas.Series]]: Dictionary mapping thresholds to (features, targets) tuples
    """
    results: Dict[float, Tuple[pandas.DataFrame, pandas.Series]] = {}
    with create_progress_bar(total=len(thresholds), desc="Preparing features") as progress_bar:
        feature_pipeline = FeaturePipeline()
        feature_pipeline.add_data_source('rtlmp_df', data_sources['rtlmp_df'])
        feature_pipeline.add_data_source('grid_df', data_sources['grid_df'])
        for threshold in thresholds:
            target = (data_sources['rtlmp_df']['price'] > threshold).astype(int)
            features = feature_pipeline.create_features()
            results[threshold] = (features, target)
            progress_bar.update()
    return results


def train_models(threshold_data: Dict[float, Tuple[pandas.DataFrame, pandas.Series]], params: TrainParamsDict) -> Dict[float, Tuple[Any, Dict[str, float]]]:
    """
    Trains models for each threshold value

    Args:
        threshold_data (Dict[float, Tuple[pandas.DataFrame, pandas.Series]]): Dictionary mapping thresholds to (features, targets) tuples
        params (TrainParamsDict): Training parameters

    Returns:
        Dict[float, Tuple[Any, Dict[str, float]]]: Dictionary mapping thresholds to (model, metrics) tuples
    """
    results: Dict[float, Tuple[Any, Dict[str, float]]] = {}
    with create_progress_bar(total=len(threshold_data), desc="Training models") as progress_bar:
        for threshold, (features, target) in threshold_data.items():
            progress_bar.update(desc=f"Training model for threshold {threshold}")
            if params['optimize_hyperparameters']:
                hyperparameter_grid = DEFAULT_HYPERPARAMETER_GRID.get(params['model_type'], {})
                model, metrics = optimize_and_train(
                    model_type=params['model_type'],
                    features=features,
                    targets=target,
                    param_grid=hyperparameter_grid
                )
            elif params['cross_validation_folds'] > 1:
                model, metrics = train_and_evaluate(
                    model_type=params['model_type'],
                    features=features,
                    targets=target,
                    cv_folds=params['cross_validation_folds']
                )
            else:
                model, metrics = train_model(
                    model_type=params['model_type'],
                    features=features,
                    targets=target
                )
            results[threshold] = (model, metrics)
            progress_bar.update()
    return results


def save_models(models: Dict[float, Tuple[Any, Dict[str, float]]], output_path: Optional[Path], model_name: Optional[str]) -> Dict[float, str]:
    """
    Saves trained models to the specified output path

    Args:
        models (Dict[float, Tuple[Any, Dict[str, float]]]): Dictionary mapping thresholds to (model, metrics) tuples
        output_path (Optional[Path]): Output path to save models
        model_name (Optional[str]): Model name

    Returns:
        Dict[float, str]: Dictionary mapping thresholds to model file paths
    """
    model_paths: Dict[float, str] = {}
    if output_path is None:
        output_path = Path("models")
    output_path.mkdir(parents=True, exist_ok=True)
    for threshold, (model, metrics) in models.items():
        model_filename = f"{model_name}_{threshold}.joblib" if model_name else f"model_{threshold}.joblib"
        model_path = output_path / model_filename
        joblib.dump(model, model_path)
        model_paths[threshold] = str(model_path)
    return model_paths


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Formats model metrics for display

    Args:
        metrics (Dict[str, float]): Dictionary of model metrics

    Returns:
        str: Formatted metrics string
    """
    formatted_metrics: List[str] = []
    for name, value in metrics.items():
        formatted_metrics.append(f"{name}: {value:.4f}")
    return ", ".join(formatted_metrics)


def train_command(params: Dict[str, Any]) -> int:
    """
    Main function for the train command

    Args:
        params (Dict[str, Any]): Dictionary of parameters for the train command

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    logger.info("Starting train command")
    try:
        train_params = cast(TrainParamsDict, validate_train_params(params))
        data_sources = fetch_training_data(train_params['start_date'], train_params['end_date'], train_params['nodes'])
        threshold_data = prepare_training_features(data_sources, train_params['thresholds'])
        models = train_models(threshold_data, train_params)
        model_paths = save_models(models, train_params['output_path'], train_params['model_name'])
        for threshold, path in model_paths.items():
            model, metrics = models[threshold]
            formatted_metrics = format_metrics(metrics)
            logger.info(f"Model for threshold {threshold} saved to {path} with metrics: {formatted_metrics}")
        return 0
    except Exception as e:
        logger.exception(e)
        return 1