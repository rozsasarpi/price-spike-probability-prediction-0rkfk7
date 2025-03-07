"""
Core inference engine for the ERCOT RTLMP spike prediction system. This module implements the main
inference functionality that generates probability forecasts for price spikes in the RTLMP market,
orchestrating the entire prediction process from model loading to forecast generation.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from datetime import datetime  # Standard
from typing import Dict, List, Optional, Union, Tuple, Any, Callable  # Standard
from pathlib import Path  # Standard

# Internal imports
from .threshold_config import ThresholdConfig  # src/backend/inference/threshold_config.py
from .thresholds import ThresholdApplier  # src/backend/inference/thresholds.py
from .calibration import ProbabilityCalibrator  # src/backend/inference/calibration.py
from .prediction_pipeline import PredictionPipeline  # src/backend/inference/prediction_pipeline.py
from ..models.persistence import ModelPersistence  # src/backend/models/persistence.py
from ..data.storage.forecast_repository import ForecastRepository  # src/backend/data/storage/forecast_repository.py
from ..features.feature_pipeline import FeaturePipeline  # src/backend/features/feature_pipeline.py
from ..utils.type_definitions import (  # src/backend/utils/type_definitions.py
    DataFrameType,
    SeriesType,
    ModelType,
    ThresholdValue,
    NodeID,
    PathType,
    InferenceEngineProtocol,
)
from ..utils.logging import get_logger, log_execution_time  # src/backend/utils/logging.py
from ..utils.error_handling import (  # src/backend/utils/error_handling.py
    retry_with_backoff,
    handle_errors,
    InferenceError,
)
from ..config.schema import InferenceConfig  # src/backend/config/schema.py

# Initialize logger
logger = get_logger(__name__)

# Global constants
DEFAULT_FORECAST_HORIZON = 72
DEFAULT_CONFIDENCE_LEVEL = 0.95


@retry_with_backoff(
    exceptions=(OSError, IOError), max_retries=3, initial_delay=1.0
)
@log_execution_time(logger, "INFO")
def load_model_for_inference(
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    model_path: Optional[PathType] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads a model for inference with appropriate error handling and validation

    Args:
        model_id (Optional[str]): The ID of the model to load.
        model_version (Optional[str]): The version of the model to load.
        model_path (Optional[PathType]): The path to the model file.

    Returns:
        Tuple[Any, Dict[str, Any]]: Tuple of model object and metadata
    """
    # Initialize ModelPersistence instance
    model_persistence = ModelPersistence()

    # If model_path is provided, load model from the specific path
    if model_path:
        model, metadata = model_persistence.load_model(
            model_id=model_id, model_type='xgboost', version=model_version, custom_path=model_path
        )
    # If model_id and model_version are provided, load that specific model
    elif model_id and model_version:
        model, metadata = model_persistence.load_model(
            model_id=model_id, model_type='xgboost', version=model_version
        )
    # If only model_id is provided, load the latest version of that model
    elif model_id:
        model, metadata = model_persistence.load_model(model_id=model_id, model_type='xgboost')
    # If no parameters are provided, load the latest model of any type
    else:
        model, metadata = model_persistence.load_model(model_id='any', model_type='xgboost')

    # Validate that the loaded model is suitable for inference
    if not validate_model_for_inference(model, metadata):
        raise InferenceError("Loaded model is not valid for inference")

    # Log information about the loaded model
    logger.info(
        f"Loaded model: {metadata.get('model_id')} version {metadata.get('version')}"
    )

    return model, metadata


def validate_model_for_inference(model: Any, metadata: Dict[str, Any]) -> bool:
    """
    Validates that a model is suitable for inference

    Args:
        model (Any): The model to validate.
        metadata (Dict[str, Any]): The metadata of the model.

    Returns:
        bool: True if model is valid for inference, False otherwise
    """
    # Check if model has predict_proba method
    if not hasattr(model, "predict_proba") or not callable(getattr(model, "predict_proba")):
        logger.error("Model does not have predict_proba method")
        return False

    # Check if metadata contains required fields (model_id, model_type, version)
    required_metadata_fields = ["model_id", "model_type", "version"]
    if not all(field in metadata for field in required_metadata_fields):
        logger.error(f"Metadata is missing required fields: {required_metadata_fields}")
        return False

    # Check if model has is_trained method and if it returns True
    if hasattr(model, "is_trained") and callable(getattr(model, "is_trained")):
        if not model.is_trained():
            logger.error("Model is not trained")
            return False

    logger.debug("Model validation passed")
    return True


@log_execution_time(logger, "INFO")
def generate_forecast(
    data_sources: Dict[str, DataFrameType],
    config: InferenceConfig,
    model_id: Optional[str] = None,
    output_path: Optional[PathType] = None,
) -> DataFrameType:
    """
    Generates a forecast using the provided data sources and configuration

    Args:
        data_sources (Dict[str, DataFrameType]): Dictionary of data sources.
        config (InferenceConfig): Inference configuration.
        model_id (Optional[str]): The ID of the model to use.
        output_path (Optional[PathType]): The path to store the forecast.

    Returns:
        DataFrameType: Generated forecast DataFrame
    """
    # Validate data sources contain required data (rtlmp, weather, grid_conditions)
    if "rtlmp" not in data_sources:
        raise ValueError("RTLMP data source is required")
    if "weather" not in data_sources:
        logger.warning("Weather data source is missing, predictions may be less accurate")
    if "grid_conditions" not in data_sources:
        logger.warning("Grid condition data source is missing, predictions may be less accurate")

    # Create a PredictionPipeline instance with the provided configuration
    pipeline = PredictionPipeline(config)

    # If model_id is provided, load the specified model
    if model_id:
        pipeline.load_model(model_id=model_id)
    # Otherwise, load the latest suitable model
    else:
        pipeline.load_model()

    # Run the prediction pipeline with the data sources
    forecast_df = pipeline.run(data_sources)

    # If output_path is provided, store the forecast at that location
    if output_path:
        # TODO: Implement forecast storage
        pass

    # Log successful forecast generation
    logger.info("Successfully generated forecast")

    return forecast_df


@handle_errors(
    exceptions=Exception,
    error_message="Failed to retrieve latest forecast",
    default_return=(None, None, None),
)
def get_latest_forecast(
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None,
    repository_path: Optional[PathType] = None,
) -> Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime]]:
    """
    Retrieves the latest forecast from the repository

    Args:
        thresholds (Optional[List[ThresholdValue]]): List of threshold values to filter by.
        nodes (Optional[List[NodeID]]): List of node IDs to filter by.
        repository_path (Optional[PathType]): Path to the forecast repository.

    Returns:
        Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime.datetime]]:
        Tuple of forecast DataFrame, metadata, and timestamp
    """
    # Initialize ForecastRepository with repository_path
    forecast_repository = ForecastRepository(forecast_root=repository_path)

    # Call get_latest_forecast method with thresholds and nodes
    forecast_df, metadata, timestamp = forecast_repository.get_latest_forecast(
        thresholds=thresholds, nodes=nodes
    )

    # Log information about the retrieved forecast
    if forecast_df is not None:
        logger.info(f"Retrieved latest forecast from {timestamp}")
    else:
        logger.warning("No latest forecast found")

    return forecast_df, metadata, timestamp


def compare_forecasts(
    forecast1: DataFrameType,
    forecast2: DataFrameType,
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None,
) -> Dict[str, Any]:
    """
    Compares two forecasts and calculates differences

    Args:
        forecast1 (DataFrameType): First forecast DataFrame.
        forecast2 (DataFrameType): Second forecast DataFrame.
        thresholds (Optional[List[ThresholdValue]]): List of threshold values to filter by.
        nodes (Optional[List[NodeID]]): List of node IDs to filter by.

    Returns:
        Dict[str, Any]: Dictionary with comparison metrics
    """
    # TODO: Implement forecast comparison logic
    return {}


class InferenceEngine(InferenceEngineProtocol):
    """
    Main class that implements the inference engine for RTLMP spike prediction
    """

    def __init__(self, config: InferenceConfig, model_path: Optional[PathType] = None, forecast_path: Optional[PathType] = None):
        """
        Initializes the InferenceEngine with configuration

        Args:
            config (InferenceConfig): Inference configuration.
            model_path (Optional[PathType]): Path to the model directory.
            forecast_path (Optional[PathType]): Path to the forecast repository.
        """
        # Store the configuration as _config
        self._config = config

        # Initialize _model_persistence with model_path
        self._model_persistence = ModelPersistence()

        # Initialize _forecast_repository with forecast_path
        self._forecast_repository = ForecastRepository(forecast_root=forecast_path)

        # Create ThresholdConfig from config.thresholds
        self._threshold_config = ThresholdConfig(default_thresholds=config.thresholds)

        # Set _model and _model_metadata to None initially
        self._model: Optional[Any] = None
        self._model_metadata: Optional[Dict[str, Any]] = None

        # Set _calibrator to None initially
        self._calibrator: Optional[ProbabilityCalibrator] = None

        # Initialize logger
        self.logger = get_logger(__name__)

    def load_model(self, model_id: Optional[str] = None, model_version: Optional[str] = None) -> bool:
        """
        Loads a model for inference

        Args:
            model_id (Optional[str]): The ID of the model to load.
            model_version (Optional[str]): The version of the model to load.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self._model, self._model_metadata = load_model_for_inference(model_id, model_version)
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def initialize_calibrator(self, historical_predictions: DataFrameType, historical_actuals: DataFrameType, method: Optional[str] = None) -> bool:
        """
        Initializes the probability calibrator with historical data

        Args:
            historical_predictions (DataFrameType): DataFrame of historical predictions.
            historical_actuals (DataFrameType): DataFrame of historical actuals.
            method (Optional[str]): Calibration method to use.

        Returns:
            bool: True if calibrator initialized successfully, False otherwise
        """
        # TODO: Implement calibrator initialization
        return True

    def generate_forecast(self, data_sources: Dict[str, DataFrameType], feature_config: Optional[Dict[str, Any]] = None, additional_metadata: Optional[Dict[str, Any]] = None) -> DataFrameType:
        """
        Generates a forecast using the provided data sources

        Args:
            data_sources (Dict[str, DataFrameType]): Dictionary of data sources.
            feature_config (Optional[Dict[str, Any]]): Feature configuration.
            additional_metadata (Optional[Dict[str, Any]]): Additional metadata to store with the forecast.

        Returns:
            DataFrameType: Generated forecast DataFrame
        """
        # Check if model is loaded, if not try to load latest model
        if self._model is None:
            self.load_model()

        # Create a PredictionPipeline with _config
        pipeline = PredictionPipeline(self._config)

        # Load the model into the pipeline
        pipeline.load_model()

        # If _calibrator is not None, initialize the pipeline's calibrator
        if self._calibrator is not None:
            # TODO: Implement calibrator initialization in pipeline
            pass

        # Run the prediction pipeline with the data sources and feature_config
        forecast_df = pipeline.run(data_sources)

        # Store the forecast using _forecast_repository
        self._forecast_repository.store_forecast(forecast_df, self._model_metadata)

        # Log successful forecast generation
        self.logger.info("Successfully generated forecast")

        return forecast_df

    def get_latest_forecast(self, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None) -> Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime]]:
        """
        Retrieves the latest forecast from the repository

        Args:
            thresholds (Optional[List[ThresholdValue]]): List of threshold values to filter by.
            nodes (Optional[List[NodeID]]): List of node IDs to filter by.

        Returns:
            Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime.datetime]]:
            Tuple of forecast DataFrame, metadata, and timestamp
        """
        # Call get_latest_forecast function with thresholds, nodes, and _forecast_repository.base_path
        forecast_df, metadata, timestamp = get_latest_forecast(thresholds, nodes, self._forecast_repository._forecast_root)

        return forecast_df, metadata, timestamp

    def list_available_models(self, model_type: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Lists all available models for inference

        Args:
            model_type (Optional[str]): The type of model to list.

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Dictionary of available models with their metadata
        """
        # Call _model_persistence.list_models with the provided model_type
        models = self._model_persistence.list_models(model_type)

        # Filter models to include only those suitable for inference
        # TODO: Implement model filtering

        return models

    def run_inference(self, data_sources: Dict[str, DataFrameType], model_id: Optional[str] = None, feature_config: Optional[Dict[str, Any]] = None) -> DataFrameType:
        """
        Runs the complete inference process from data to forecast

        Args:
            data_sources (Dict[str, DataFrameType]): Dictionary of data sources.
            model_id (Optional[str]): The ID of the model to use.
            feature_config (Optional[Dict[str, Any]]): Feature configuration.

        Returns:
            DataFrameType: Generated forecast DataFrame
        """
        # If model_id is provided, load the specified model
        if model_id:
            self.load_model(model_id=model_id)
        # Otherwise, use the currently loaded model or load the latest
        elif self._model is None:
            self.load_model()

        # Generate forecast using the loaded model and provided data sources
        forecast_df = self.generate_forecast(data_sources, feature_config)

        return forecast_df

    def compare_with_previous_forecast(self, new_forecast: DataFrameType, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None) -> Dict[str, Any]:
        """
        Compares a new forecast with the previous one

        Args:
            new_forecast (DataFrameType): The new forecast DataFrame.
            thresholds (Optional[List[ThresholdValue]]): List of threshold values to filter by.
            nodes (Optional[List[NodeID]]): List of node IDs to filter by.

        Returns:
            Dict[str, Any]: Dictionary with comparison metrics
        """
        # Retrieve the latest forecast using get_latest_forecast
        previous_forecast, _, _ = self.get_latest_forecast(thresholds, nodes)

        # If no previous forecast exists, return empty comparison
        if previous_forecast is None:
            return {}

        # Call compare_forecasts with the new and previous forecasts
        comparison_metrics = compare_forecasts(new_forecast, previous_forecast, thresholds, nodes)

        return comparison_metrics

    def get_model_info(self) -> Dict[str, Any]:
        """
        Gets information about the currently loaded model

        Returns:
            Dict[str, Any]: Dictionary with model information
        """
        # Check if model is loaded
        if self._model is None:
            logger.warning("No model loaded")
            return {}

        # If model is loaded, return _model_metadata
        return self._model_metadata