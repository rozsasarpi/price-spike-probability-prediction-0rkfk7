"""
Core module that implements the complete pipeline orchestration for the ERCOT RTLMP spike prediction system.
This module coordinates the execution of data fetching, feature engineering, model training, and inference
operations, providing a unified interface for end-to-end workflow management.
"""

import typing
from typing import Dict, List, Optional, Callable, Any, Tuple, Union, Set, Type, cast
import enum
from datetime import datetime
import pathlib
from pathlib import Path

import pandas  # version 2.0+

from .task_management import TaskManager, Task, TaskStatus, TaskPriority, TaskResult, create_task, execute_task_with_retry
from .scheduler import Scheduler, DailyScheduler, ScheduledTask, ScheduleFrequency
from .error_recovery import ErrorRecoveryManager, RecoveryContext, RecoveryStrategy, PipelineStage
from ..data.fetchers.base import BaseDataFetcher
from ..features.feature_pipeline import FeaturePipeline, create_feature_pipeline
from ..models.training import train_model, get_latest_model, ModelTrainer
from ..inference.prediction_pipeline import PredictionPipeline, MultiThresholdPredictionPipeline
from ..backtesting.framework import BacktestingFramework
from ..config.schema import Config, DataConfig, FeatureConfig, ModelConfig, InferenceConfig, BacktestConfig
from ..utils.type_definitions import DataFrameType, ModelType, PathType
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import retry_with_backoff, handle_errors, PipelineError

# Initialize logger
logger = get_logger(__name__)

# Default values
DEFAULT_PIPELINE_CONFIG = {
    'data_fetch': {'enabled': True, 'timeout': 300, 'max_retries': 3},
    'feature_engineering': {'enabled': True, 'feature_selection': True},
    'model_training': {'enabled': False, 'model_type': 'xgboost', 'cross_validation': True, 'cv_folds': 5},
    'inference': {'enabled': True, 'forecast_horizon': 72, 'thresholds': [50, 100, 200]},
    'backtesting': {'enabled': False}
}


def create_data_fetch_stage(data_fetcher: BaseDataFetcher, fetch_params: Dict[str, Any], task_manager: Optional[TaskManager] = None) -> Task:
    """
    Creates a task for the data fetching stage of the pipeline

    Args:
        data_fetcher (BaseDataFetcher): data_fetcher
        fetch_params (Dict[str, Any]): fetch_params
        task_manager (Optional[TaskManager], optional): task_manager. Defaults to None.

    Returns:
        Task: Data fetching task
    """
    # Create a task for fetching data with the provided data_fetcher and fetch_params
    data_fetch_task = Task(func=data_fetcher.fetch_data, name="data_fetch", kwargs=fetch_params)
    # Set task priority to HIGH
    data_fetch_task.priority = TaskPriority.HIGH
    # If task_manager is provided, add the task to the task manager
    if task_manager:
        task_manager.add_task(data_fetch_task)
    # Return the created task
    return data_fetch_task


def create_feature_engineering_stage(data_sources: Dict[str, DataFrameType], feature_config: Dict[str, Any], task_manager: Optional[TaskManager] = None, dependencies: Optional[Set[str]] = None) -> Task:
    """
    Creates a task for the feature engineering stage of the pipeline

    Args:
        data_sources (Dict[str, DataFrameType]): data_sources
        feature_config (Dict[str, Any]): feature_config
        task_manager (Optional[TaskManager], optional): task_manager. Defaults to None.
        dependencies (Optional[Set[str]], optional): dependencies. Defaults to None.

    Returns:
        Task: Feature engineering task
    """
    # Create a task for feature engineering with the provided data_sources and feature_config
    feature_engineering_task = Task(func=create_feature_pipeline, name="feature_engineering", args=(data_sources, feature_config))
    # Set task priority to HIGH
    feature_engineering_task.priority = TaskPriority.HIGH
    # If dependencies is provided, add them to the task
    if dependencies:
        for dep in dependencies:
            feature_engineering_task.add_dependency(dep)
    # If task_manager is provided, add the task to the task manager
    if task_manager:
        task_manager.add_task(feature_engineering_task)
    # Return the created task
    return feature_engineering_task


def create_model_training_stage(features: DataFrameType, training_config: Dict[str, Any], task_manager: Optional[TaskManager] = None, dependencies: Optional[Set[str]] = None) -> Task:
    """
    Creates a task for the model training stage of the pipeline

    Args:
        features (DataFrameType): features
        training_config (Dict[str, Any]): training_config
        task_manager (Optional[TaskManager], optional): task_manager. Defaults to None.
        dependencies (Optional[Set[str]], optional): dependencies. Defaults to None.

    Returns:
        Task: Model training task
    """
    # Create a task for model training with the provided features and training_config
    model_training_task = Task(func=train_model, name="model_training", kwargs={"features": features, "training_config": training_config})
    # Set task priority to MEDIUM
    model_training_task.priority = TaskPriority.MEDIUM
    # If dependencies is provided, add them to the task
    if dependencies:
        for dep in dependencies:
            model_training_task.add_dependency(dep)
    # If task_manager is provided, add the task to the task manager
    if task_manager:
        task_manager.add_task(model_training_task)
    # Return the created task
    return model_training_task


def create_inference_stage(data_sources: Dict[str, DataFrameType], inference_config: Dict[str, Any], model: Optional[ModelType] = None, task_manager: Optional[TaskManager] = None, dependencies: Optional[Set[str]] = None) -> Task:
    """
    Creates a task for the inference stage of the pipeline

    Args:
        data_sources (Dict[str, DataFrameType]): data_sources
        inference_config (Dict[str, Any]): inference_config
        model (Optional[ModelType], optional): model. Defaults to None.
        task_manager (Optional[TaskManager], optional): task_manager. Defaults to None.
        dependencies (Optional[Set[str]], optional): dependencies. Defaults to None.

    Returns:
        Task: Inference task
    """
    # Create a task for inference with the provided data_sources, inference_config, and model
    inference_task = Task(func=PredictionPipeline.generate_forecast, name="inference", kwargs={"data_sources": data_sources, "inference_config": inference_config, "model": model})
    # Set task priority to CRITICAL
    inference_task.priority = TaskPriority.CRITICAL
    # If dependencies is provided, add them to the task
    if dependencies:
        for dep in dependencies:
            inference_task.add_dependency(dep)
    # If task_manager is provided, add the task to the task manager
    if task_manager:
        task_manager.add_task(inference_task)
    # Return the created task
    return inference_task


def create_backtesting_stage(data_fetcher: BaseDataFetcher, backtest_config: Dict[str, Any], task_manager: Optional[TaskManager] = None, dependencies: Optional[Set[str]] = None) -> Task:
    """
    Creates a task for the backtesting stage of the pipeline

    Args:
        data_fetcher (BaseDataFetcher): data_fetcher
        backtest_config (Dict[str, Any]): backtest_config
        task_manager (Optional[TaskManager], optional): task_manager. Defaults to None.
        dependencies (Optional[Set[str]], optional): dependencies. Defaults to None.

    Returns:
        Task: Backtesting task
    """
    # Create a task for backtesting with the provided data_fetcher and backtest_config
    backtesting_task = Task(func=BacktestingFramework.execute_scenario, name="backtesting", kwargs={"data_fetcher": data_fetcher, "backtest_config": backtest_config})
    # Set task priority to LOW
    backtesting_task.priority = TaskPriority.LOW
    # If dependencies is provided, add them to the task
    if dependencies:
        for dep in dependencies:
            backtesting_task.add_dependency(dep)
    # If task_manager is provided, add the task to the task manager
    if task_manager:
        task_manager.add_task(backtesting_task)
    # Return the created task
    return backtesting_task


@log_execution_time(logger, 'INFO')
def execute_pipeline(pipeline_config: Dict[str, Any], data_fetcher: BaseDataFetcher, model_path: Optional[PathType] = None, output_path: Optional[PathType] = None, with_retry: Optional[bool] = True) -> Dict[str, Any]:
    """
    Executes a complete pipeline from data fetching to inference

    Args:
        pipeline_config (Dict[str, Any]): pipeline_config
        data_fetcher (BaseDataFetcher): data_fetcher
        model_path (Optional[PathType], optional): model_path. Defaults to None.
        output_path (Optional[PathType], optional): output_path. Defaults to None.
        with_retry (Optional[bool], optional): with_retry. Defaults to True.

    Returns:
        Dict[str, Any]: Pipeline execution results
    """
    # Initialize TaskManager for pipeline execution
    task_manager = TaskManager()
    # Initialize ErrorRecoveryManager for handling errors
    recovery_manager = ErrorRecoveryManager()

    # Merge provided pipeline_config with DEFAULT_PIPELINE_CONFIG
    config = deep_merge(DEFAULT_PIPELINE_CONFIG, pipeline_config)

    # Initialize task dependencies set
    dependencies: Set[str] = set()

    # Create data fetch stage task if data_fetch.enabled is True
    if config['data_fetch']['enabled']:
        data_fetch_task = create_data_fetch_stage(data_fetcher, config['data_fetch'], task_manager)
        dependencies.add(data_fetch_task.id)

    # Create feature engineering stage task with dependency on data fetch if feature_engineering.enabled is True
    if config['feature_engineering']['enabled']:
        feature_engineering_task = create_feature_engineering_stage({}, config['feature_engineering'], task_manager, dependencies)
        dependencies = {feature_engineering_task.id}

    # Create model training stage task with dependency on feature engineering if model_training.enabled is True
    if config['model_training']['enabled']:
        model_training_task = create_model_training_stage({}, config['model_training'], task_manager, dependencies)
        dependencies = {model_training_task.id}

    # Create inference stage task with dependencies on feature engineering and optionally model training if inference.enabled is True
    if config['inference']['enabled']:
        inference_stage_task = create_inference_stage({}, config['inference'], None, task_manager, dependencies)
        dependencies = {inference_stage_task.id}

    # Create backtesting stage task with dependencies on model training and inference if backtesting.enabled is True
    if config['backtesting']['enabled']:
        backtesting_stage_task = create_backtesting_stage(data_fetcher, config['backtesting'], task_manager, dependencies)

    # Execute all tasks in dependency order using TaskManager.execute_all()
    results = task_manager.execute_all(parallel=True, with_retry=with_retry)

    # Collect results from each stage
    data_fetch_result = results.get(data_fetch_task.id) if config['data_fetch']['enabled'] else None
    feature_engineering_result = results.get(feature_engineering_task.id) if config['feature_engineering']['enabled'] else None
    model_training_result = results.get(model_training_task.id) if config['model_training']['enabled'] else None
    inference_result = results.get(inference_stage_task.id) if config['inference']['enabled'] else None
    backtesting_result = results.get(backtesting_stage_task.id) if config['backtesting']['enabled'] else None

    # Return dictionary of pipeline execution results
    return {
        "data_fetch": data_fetch_result,
        "feature_engineering": feature_engineering_result,
        "model_training": model_training_result,
        "inference": inference_result,
        "backtesting": backtesting_result
    }


def schedule_pipeline(pipeline_config: Dict[str, Any], data_fetcher: BaseDataFetcher, schedule: Union[str, ScheduleFrequency], model_path: Optional[PathType] = None, output_path: Optional[PathType] = None) -> Scheduler:
    """
    Schedules a pipeline for regular execution

    Args:
        pipeline_config (Dict[str, Any]): pipeline_config
        data_fetcher (BaseDataFetcher): data_fetcher
        schedule (Union[str, ScheduleFrequency]): schedule
        model_path (Optional[PathType], optional): model_path. Defaults to None.
        output_path (Optional[PathType], optional): output_path. Defaults to None.

    Returns:
        Scheduler: Configured scheduler
    """
    # Initialize Scheduler for pipeline scheduling
    scheduler = Scheduler()
    # Create a task that executes the complete pipeline
    pipeline_task = Task(func=execute_pipeline, name="complete_pipeline", kwargs={"pipeline_config": pipeline_config, "data_fetcher": data_fetcher, "model_path": model_path, "output_path": output_path})
    # Add the task to the scheduler with the provided schedule
    scheduler.add_task(pipeline_task, schedule)
    # Return the configured scheduler (not started)
    return scheduler


def setup_daily_pipeline(data_fetcher: BaseDataFetcher, pipeline_config: Optional[Dict[str, Any]] = None, model_path: Optional[PathType] = None, output_path: Optional[PathType] = None) -> DailyScheduler:
    """
    Sets up the standard daily pipeline for RTLMP spike prediction

    Args:
        data_fetcher (BaseDataFetcher): data_fetcher
        pipeline_config (Optional[Dict[str, Any]], optional): pipeline_config. Defaults to None.
        model_path (Optional[PathType], optional): model_path. Defaults to None.
        output_path (Optional[PathType], optional): output_path. Defaults to None.

    Returns:
        DailyScheduler: Configured daily scheduler
    """
    # Initialize DailyScheduler with default schedules for ERCOT operations
    daily_scheduler = DailyScheduler()

    # Merge provided pipeline_config with DEFAULT_PIPELINE_CONFIG
    config = deep_merge(DEFAULT_PIPELINE_CONFIG, pipeline_config or {})

    # Setup daily inference schedule
    if config['inference']['enabled']:
        daily_scheduler.setup_daily_inference(PredictionPipeline.generate_forecast, {"data_fetcher": data_fetcher, "model_path": model_path, "output_path": output_path})

    # Setup bi-daily model retraining schedule
    if config['model_training']['enabled']:
        daily_scheduler.setup_bidaily_retraining(train_model, {"data_fetcher": data_fetcher, "model_path": model_path, "output_path": output_path})

    # Setup data fetching schedule
    if config['data_fetch']['enabled']:
        daily_scheduler.setup_data_fetching(data_fetcher.fetch_data, config['data_fetch'])

    # Return the configured daily scheduler (not started)
    return daily_scheduler