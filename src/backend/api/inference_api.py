"""
Provides a unified API for inference operations in the ERCOT RTLMP spike prediction system.
This module acts as a facade over the inference engine, offering a clean interface for generating forecasts,
retrieving forecast data, and managing inference-related operations.
"""

import os  # Standard
import pandas as pd  # version 2.0+
from datetime import datetime  # Standard
from typing import Dict, List, Optional, Union, Tuple, Any  # Standard

# Internal imports
from ..inference.engine import InferenceEngine  # src/backend/inference/engine.py
from ..inference.prediction_pipeline import PredictionPipeline, MultiThresholdPredictionPipeline  # src/backend/inference/prediction_pipeline.py
from ..inference.calibration import ProbabilityCalibrator  # src/backend/inference/calibration.py
from ..inference.threshold_config import ThresholdConfig  # src/backend/inference/threshold_config.py
from ..data.storage.forecast_repository import ForecastRepository  # src/backend/data/storage/forecast_repository.py
from .data_api import DataAPI  # src/backend/api/data_api.py
from .model_api import ModelAPI  # src/backend/api/model_api.py
from ..config.schema import InferenceConfig  # src/backend/config/schema.py
from ..utils.type_definitions import DataFrameType, SeriesType, ModelType, ThresholdValue, NodeID, PathType  # src/backend/utils/type_definitions.py
from ..utils.logging import get_logger, log_execution_time  # src/backend/utils/logging.py
from ..utils.error_handling import handle_errors, InferenceError  # src/backend/utils/error_handling.py

# Initialize logger
logger = get_logger(__name__)

# Global constants
DEFAULT_FORECAST_HORIZON = 72
DEFAULT_FORECAST_PATH = os.environ.get('ERCOT_FORECAST_PATH', 'forecasts/')
DEFAULT_MODEL_PATH = os.environ.get('ERCOT_MODEL_PATH', 'models/')


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to generate forecast')
def generate_forecast(
    data_sources: Dict[str, DataFrameType],
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None,
    model_id: Optional[str] = None,
    output_path: Optional[PathType] = None,
    store_forecast: Optional[bool] = True
) -> DataFrameType:
    """
    Generates a forecast using the provided data sources and configuration

    Args:
        data_sources (Dict[str, DataFrameType]): Dictionary of data sources
        thresholds (Optional[List[ThresholdValue]]): List of threshold values
        nodes (Optional[List[NodeID]]): List of node IDs
        model_id (Optional[str]): Model ID
        output_path (Optional[PathType]): Output path
        store_forecast (Optional[bool]): Store forecast flag

    Returns:
        DataFrameType: Generated forecast DataFrame
    """
    # Create an InferenceConfig with the provided thresholds, nodes, and DEFAULT_FORECAST_HORIZON
    config = InferenceConfig(
        thresholds=thresholds,
        nodes=nodes,
        forecast_horizon=DEFAULT_FORECAST_HORIZON
    )

    # Initialize an InferenceEngine with the config and output_path
    engine = InferenceEngine(config=config, model_path=DEFAULT_MODEL_PATH, forecast_path=output_path)

    # If model_id is provided, load the specified model
    if model_id:
        engine.load_model(model_id=model_id)

    # Generate forecast using the engine's generate_forecast method with data_sources
    forecast = engine.generate_forecast(data_sources=data_sources)

    # If store_forecast is True, ensure the forecast is stored
    if store_forecast:
        # The actual storing is handled within the InferenceEngine.generate_forecast method
        pass

    # Log successful forecast generation
    logger.info("Successfully generated forecast")

    # Return the generated forecast DataFrame
    return forecast


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to retrieve latest forecast', default_return=(None, None, None))
def get_latest_forecast(
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None,
    repository_path: Optional[PathType] = None
) -> Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime]]:
    """
    Retrieves the latest forecast from the repository

    Args:
        thresholds (Optional[List[ThresholdValue]]): List of threshold values
        nodes (Optional[List[NodeID]]): List of node IDs
        repository_path (Optional[PathType]): Repository path

    Returns:
        Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime.datetime]]: Tuple of forecast DataFrame, metadata, and timestamp
    """
    # Initialize ForecastRepository with repository_path or DEFAULT_FORECAST_PATH
    forecast_repository = ForecastRepository(forecast_root=repository_path or DEFAULT_FORECAST_PATH)

    # Call get_latest_forecast method with thresholds and nodes
    forecast_df, metadata, timestamp = forecast_repository.get_latest_forecast(thresholds=thresholds, nodes=nodes)

    # Log information about the retrieved forecast
    if forecast_df is not None:
        logger.info(f"Retrieved latest forecast from {timestamp}")
    else:
        logger.warning("No latest forecast found")

    # Return the forecast DataFrame, metadata, and timestamp
    return forecast_df, metadata, timestamp


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to retrieve forecast by date', default_return=(None, None))
def get_forecast_by_date(
    forecast_date: datetime,
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None,
    repository_path: Optional[PathType] = None
) -> Tuple[Optional[DataFrameType], Optional[Dict[str, Any]]]:
    """
    Retrieves a forecast for a specific date from the repository

    Args:
        forecast_date (datetime.datetime): Forecast date
        thresholds (Optional[List[ThresholdValue]]): List of threshold values
        nodes (Optional[List[NodeID]]): List of node IDs
        repository_path (Optional[PathType]): Repository path

    Returns:
        Tuple[Optional[DataFrameType], Optional[Dict[str, Any]]]: Tuple of forecast DataFrame and metadata
    """
    # Initialize ForecastRepository with repository_path or DEFAULT_FORECAST_PATH
    forecast_repository = ForecastRepository(forecast_root=repository_path or DEFAULT_FORECAST_PATH)

    # Call get_forecast_by_date method with forecast_date, thresholds, and nodes
    forecast_df, metadata = forecast_repository.get_forecast_by_date(forecast_date=forecast_date, thresholds=thresholds, nodes=nodes)

    # Log information about the retrieved forecast
    if forecast_df is not None:
        logger.info(f"Retrieved forecast for {forecast_date}")
    else:
        logger.warning(f"No forecast found for {forecast_date}")

    # Return the forecast DataFrame and metadata
    return forecast_df, metadata


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to compare forecasts', default_return={})
def compare_forecasts(
    forecast1: DataFrameType,
    forecast2: DataFrameType,
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None
) -> Dict[str, Any]:
    """
    Compares two forecasts and calculates differences

    Args:
        forecast1 (DataFrameType): First forecast DataFrame
        forecast2 (DataFrameType): Second forecast DataFrame
        thresholds (Optional[List[ThresholdValue]]): List of threshold values
        nodes (Optional[List[NodeID]]): List of node IDs

    Returns:
        Dict[str, Any]: Dictionary with comparison metrics
    """
    # TODO: Implement forecast comparison logic
    return {}


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to list available models', default_return={})
def list_available_models(
    model_type: Optional[str] = None,
    model_path: Optional[PathType] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Lists all available models for inference

    Args:
        model_type (Optional[str]): Model type
        model_path (Optional[PathType]): Model path

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Dictionary of available models with their metadata
    """
    # Create an InferenceConfig with default values
    config = InferenceConfig()

    # Initialize an InferenceEngine with the config and model_path
    engine = InferenceEngine(config=config, model_path=model_path or DEFAULT_MODEL_PATH)

    # Call list_available_models method with the provided model_type
    models = engine.list_available_models(model_type=model_type)

    # Return the dictionary of available models
    return models


class InferenceAPI:
    """
    Class that provides a unified interface for inference operations
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        forecast_path: Optional[PathType] = None,
        model_path: Optional[PathType] = None,
        data_api: Optional[DataAPI] = None,
        model_api: Optional[ModelAPI] = None
    ):
        """
        Initializes the InferenceAPI with configuration

        Args:
            config (Optional[InferenceConfig]): Inference configuration
            forecast_path (Optional[PathType]): Forecast path
            model_path (Optional[PathType]): Model path
            data_api (Optional[DataAPI]): Data API
            model_api (Optional[ModelAPI]): Model API
        """
        # Set _forecast_path to forecast_path or DEFAULT_FORECAST_PATH
        self._forecast_path = forecast_path or DEFAULT_FORECAST_PATH

        # Set _model_path to model_path or DEFAULT_MODEL_PATH
        self._model_path = model_path or DEFAULT_MODEL_PATH

        # If config is provided, store it as _config
        if config:
            self._config = config
        # Otherwise, create a default InferenceConfig with DEFAULT_FORECAST_HORIZON
        else:
            self._config = InferenceConfig(forecast_horizon=DEFAULT_FORECAST_HORIZON)

        # Initialize _inference_engine with _config and _model_path
        self._inference_engine = InferenceEngine(config=self._config, model_path=self._model_path, forecast_path=self._forecast_path)

        # Initialize _forecast_repository with _forecast_path
        self._forecast_repository = ForecastRepository(forecast_root=self._forecast_path)

        # If data_api is provided, store it as _data_api
        if data_api:
            self._data_api = data_api
        # Otherwise, create a new DataAPI instance
        else:
            self._data_api = DataAPI(storage_path=self._forecast_path)

        # If model_api is provided, store it as _model_api
        if model_api:
            self._model_api = model_api
        # Otherwise, create a new ModelAPI instance with _model_path
        else:
            self._model_api = ModelAPI(model_path=self._model_path)

        # Log initialization of InferenceAPI
        logger.info("Initialized InferenceAPI")

    @log_execution_time(logger, 'INFO')
    def load_model(self, model_id: Optional[str] = None, model_version: Optional[str] = None) -> bool:
        """
        Loads a model for inference

        Args:
            model_id (Optional[str]): Model ID
            model_version (Optional[str]): Model version

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # If model_id is provided, attempt to get the model using _model_api
        # Call _inference_engine.load_model with the model_id and model_version
        success = self._inference_engine.load_model(model_id=model_id, model_version=model_version)

        # Log the result of the model loading operation
        if success:
            logger.info(f"Successfully loaded model {model_id} version {model_version}")
        else:
            logger.error(f"Failed to load model {model_id} version {model_version}")

        # Return True if successful, False otherwise
        return success

    @log_execution_time(logger, 'INFO')
    def initialize_calibrator(self, historical_predictions: DataFrameType, historical_actuals: DataFrameType, method: Optional[str] = None) -> bool:
        """
        Initializes the probability calibrator with historical data

        Args:
            historical_predictions (DataFrameType): Historical predictions
            historical_actuals (DataFrameType): Historical actuals
            method (Optional[str]): Method

        Returns:
            bool: True if calibrator initialized successfully, False otherwise
        """
        # Call _inference_engine.initialize_calibrator with the provided parameters
        success = self._inference_engine.initialize_calibrator(historical_predictions=historical_predictions, historical_actuals=historical_actuals, method=method)

        # Log the result of the calibrator initialization
        if success:
            logger.info("Successfully initialized probability calibrator")
        else:
            logger.error("Failed to initialize probability calibrator")

        # Return True if successful, False otherwise
        return success

    @log_execution_time(logger, 'INFO')
    def generate_forecast(self, data_sources: Optional[Dict[str, DataFrameType]] = None, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None, model_id: Optional[str] = None, feature_config: Optional[Dict[str, Any]] = None, store_forecast: Optional[bool] = True) -> DataFrameType:
        """
        Generates a forecast using the provided data sources

        Args:
            data_sources (Optional[Dict[str, DataFrameType]]): Data sources
            thresholds (Optional[List[ThresholdValue]]): Thresholds
            nodes (Optional[List[NodeID]]): Nodes
            model_id (Optional[str]): Model ID
            feature_config (Optional[Dict[str, Any]]): Feature config
            store_forecast (Optional[bool]): Store forecast

        Returns:
            DataFrameType: Generated forecast DataFrame
        """
        # If data_sources is None, fetch required data using _data_api
        # If model_id is provided and different from current model, load the model
        # Update _config with provided thresholds and nodes if specified
        # Call _inference_engine.generate_forecast with data_sources and feature_config
        # If store_forecast is True, ensure the forecast is stored in _forecast_repository
        # Return the generated forecast DataFrame
        if not data_sources:
            # Fetch data using the DataAPI
            now = datetime.now()
            data_sources = {
                "rtlmp": self._data_api.get_historical_rtlmp(start_date=now - timedelta(days=2), end_date=now, nodes=nodes),
                "weather": self._data_api.get_weather_forecast(forecast_date=now, horizon=72),
                "grid": self._data_api.get_historical_grid_conditions(start_date=now - timedelta(days=2), end_date=now)
            }

        if model_id and self._inference_engine.get_model_info().get("model_id") != model_id:
            self.load_model(model_id=model_id)

        if thresholds or nodes:
            self.update_config(thresholds=thresholds, nodes=nodes)

        forecast = self._inference_engine.generate_forecast(data_sources=data_sources, feature_config=feature_config)

        if store_forecast:
            self._forecast_repository.store_forecast(forecast, self._inference_engine.get_model_info())

        return forecast

    @log_execution_time(logger, 'INFO')
    def get_latest_forecast(self, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None) -> Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime]]:
        """
        Retrieves the latest forecast from the repository

        Args:
            thresholds (Optional[List[ThresholdValue]]): Thresholds
            nodes (Optional[List[NodeID]]): Nodes

        Returns:
            Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime.datetime]]: Tuple of forecast DataFrame, metadata, and timestamp
        """
        # Call _inference_engine.get_latest_forecast with thresholds and nodes
        forecast, metadata, timestamp = self._inference_engine.get_latest_forecast(thresholds=thresholds, nodes=nodes)

        # Return the forecast DataFrame, metadata, and timestamp
        return forecast, metadata, timestamp

    @log_execution_time(logger, 'INFO')
    def get_forecast_by_date(self, forecast_date: datetime, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None) -> Tuple[Optional[DataFrameType], Optional[Dict[str, Any]]]:
        """
        Retrieves a forecast for a specific date

        Args:
            forecast_date (datetime.datetime): Forecast date
            thresholds (Optional[List[ThresholdValue]]): Thresholds
            nodes (Optional[List[NodeID]]): Nodes

        Returns:
            Tuple[Optional[DataFrameType], Optional[Dict[str, Any]]]: Tuple of forecast DataFrame and metadata
        """
        # Call _forecast_repository.get_forecast_by_date with forecast_date, thresholds, and nodes
        forecast, metadata = self._forecast_repository.get_forecast_by_date(forecast_date=forecast_date, thresholds=thresholds, nodes=nodes)

        # Return the forecast DataFrame and metadata
        return forecast, metadata

    @log_execution_time(logger, 'INFO')
    def compare_with_previous_forecast(self, new_forecast: DataFrameType, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None) -> Dict[str, Any]:
        """
        Compares a new forecast with the previous one

        Args:
            new_forecast (DataFrameType): New forecast
            thresholds (Optional[List[ThresholdValue]]): Thresholds
            nodes (Optional[List[NodeID]]): Nodes

        Returns:
            Dict[str, Any]: Dictionary with comparison metrics
        """
        # Call _inference_engine.compare_with_previous_forecast with new_forecast, thresholds, and nodes
        comparison_metrics = self._inference_engine.compare_with_previous_forecast(new_forecast=new_forecast, thresholds=thresholds, nodes=nodes)

        # Return the comparison metrics dictionary
        return comparison_metrics

    @log_execution_time(logger, 'INFO')
    def list_available_models(self, model_type: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Lists all available models for inference

        Args:
            model_type (Optional[str]): Model type

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Dictionary of available models with their metadata
        """
        # Call _inference_engine.list_available_models with model_type
        models = self._inference_engine.list_available_models(model_type=model_type)

        # Return the dictionary of available models
        return models

    @log_execution_time(logger, 'INFO')
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gets information about the currently loaded model

        Returns:
            Dict[str, Any]: Dictionary with model information
        """
        # Call _inference_engine.get_model_info
        model_info = self._inference_engine.get_model_info()

        # Return the model information dictionary
        return model_info

    def update_config(self, forecast_horizon: Optional[int] = None, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None, additional_config: Optional[Dict[str, Any]] = None) -> InferenceConfig:
        """
        Updates the inference configuration

        Args:
            forecast_horizon (Optional[int]): Forecast horizon
            thresholds (Optional[List[ThresholdValue]]): Thresholds
            nodes (Optional[List[NodeID]]): Nodes
            additional_config (Optional[Dict[str, Any]]): Additional config

        Returns:
            InferenceConfig: Updated configuration
        """
        # Create a new InferenceConfig with updated parameters
        config_params = {}
        if forecast_horizon:
            config_params['forecast_horizon'] = forecast_horizon
        if thresholds:
            config_params['thresholds'] = thresholds
        if nodes:
            config_params['nodes'] = nodes
        if additional_config:
            config_params.update(additional_config)

        new_config = InferenceConfig(**config_params)

        # Update _config with the new configuration
        self._config = new_config

        # Reinitialize _inference_engine with the updated configuration
        self._inference_engine = InferenceEngine(config=self._config, model_path=self._model_path, forecast_path=self._forecast_path)

        # Return the updated configuration
        return self._config