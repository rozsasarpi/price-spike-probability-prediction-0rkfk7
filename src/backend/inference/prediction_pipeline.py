"""
Implements the prediction pipeline for the ERCOT RTLMP spike prediction system.
This module orchestrates the end-to-end process of generating probability forecasts,
from feature preparation to model inference and calibration.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from datetime import datetime  # Standard
from typing import Dict, List, Optional, Union, Tuple, Any, Callable  # Standard

# Internal imports
from .threshold_config import ThresholdConfig  # src/backend/inference/threshold_config.py
from .thresholds import ThresholdApplier  # src/backend/inference/thresholds.py
from .calibration import ProbabilityCalibrator  # src/backend/inference/calibration.py
from ..features.feature_pipeline import FeaturePipeline  # src/backend/features/feature_pipeline.py
from ..models.persistence import ModelPersistence  # src/backend/models/persistence.py
from ..data.storage.forecast_repository import ForecastRepository  # src/backend/data/storage/forecast_repository.py
from ..utils.type_definitions import DataFrameType, SeriesType, ModelType, ThresholdValue, NodeID, PathType  # src/backend/utils/type_definitions.py
from ..utils.logging import get_logger, log_execution_time  # src/backend/utils/logging.py
from ..utils.error_handling import retry_with_backoff, handle_errors, InferenceError  # src/backend/utils/error_handling.py
from ..config.schema import InferenceConfig  # src/backend/config/schema.py

# Initialize logger
logger = get_logger(__name__)

# Global constants
DEFAULT_FORECAST_HORIZON = 72
DEFAULT_CONFIDENCE_LEVEL = 0.95


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=Exception, error_message='Failed to prepare features', default_return=None)
def prepare_features(
    data_sources: Dict[str, DataFrameType],
    feature_config: Optional[Dict[str, Any]] = None
) -> DataFrameType:
    """
    Prepares features for prediction using the feature pipeline

    Args:
        data_sources: Dictionary of data sources
        feature_config: Optional feature configuration

    Returns:
        DataFrame: DataFrame with prepared features
    """
    # Validate that required data sources are provided (rtlmp, weather, grid_conditions)
    if 'rtlmp_df' not in data_sources:
        raise ValueError("RTLMP data source is required")
    if 'weather_df' not in data_sources:
        logger.warning("Weather data source is missing, predictions may be less accurate")
    if 'grid_df' not in data_sources:
        logger.warning("Grid condition data source is missing, predictions may be less accurate")

    # Create a FeaturePipeline instance with the provided feature_config
    feature_pipeline = FeaturePipeline(feature_config=feature_config)

    # Add each data source to the feature pipeline
    for source_name, df in data_sources.items():
        feature_pipeline.add_data_source(source_name, df)

    # Generate features using the feature pipeline
    features_df = feature_pipeline.create_features()

    # Validate that the feature set is consistent and complete
    is_valid, inconsistent_features = feature_pipeline.validate_feature_set()
    if not is_valid:
        logger.warning(f"Inconsistent features found: {inconsistent_features}")

    # Log the number of features created
    num_features = len(feature_pipeline.get_feature_names())
    logger.info(f"Created {num_features} features")

    # Return the DataFrame with prepared features
    return features_df


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=Exception, error_message='Failed to generate predictions', default_return=None)
def generate_predictions(
    features_df: DataFrameType,
    model: Any,
    feature_names: List[str]
) -> SeriesType:
    """
    Generates raw probability predictions using a trained model

    Args:
        features_df: DataFrame with prepared features
        model: Trained machine learning model
        feature_names: List of feature names used for training the model

    Returns:
        SeriesType: Series with probability predictions
    """
    # Validate that features_df contains all required feature_names
    missing_features = [f for f in feature_names if f not in features_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Extract the feature columns from features_df
    X = features_df[feature_names]

    # Call model.predict_proba() to get raw probability predictions
    y_prob = model.predict_proba(X)[:, 1]

    # Create a Series with the probabilities and the same index as features_df
    probabilities = pd.Series(y_prob, index=features_df.index)

    # Log the prediction generation
    logger.debug("Generated probability predictions")

    # Return the Series of probability predictions
    return probabilities


@handle_errors(exceptions=Exception, error_message='Failed to calculate confidence intervals', default_return=(None, None))
def calculate_confidence_intervals(
    probabilities: SeriesType,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
) -> Tuple[SeriesType, SeriesType]:
    """
    Calculates confidence intervals for probability predictions

    Args:
        probabilities: Series of probability predictions
        confidence_level: Confidence level for the intervals

    Returns:
        Tuple[SeriesType, SeriesType]: Tuple of lower and upper confidence bounds
    """
    # Validate that probabilities contains valid probability values (0-1)
    validate_probability_values(probabilities.to_frame(name='probabilities'), ['probabilities'], raise_error=True)

    # Calculate the standard error for each probability
    std_error = np.sqrt(probabilities * (1 - probabilities) / len(probabilities))

    # Calculate the margin of error using the confidence level
    from scipy.stats import norm  # version: Standard
    margin_of_error = norm.ppf((1 + confidence_level) / 2) * std_error

    # Calculate lower bound as probability - margin of error
    lower_bound = probabilities - margin_of_error

    # Calculate upper bound as probability + margin of error
    upper_bound = probabilities + margin_of_error

    # Clip bounds to ensure they stay within 0-1 range
    lower_bound = np.clip(lower_bound, 0, 1)
    upper_bound = np.clip(upper_bound, 0, 1)

    # Return tuple of (lower_bound, upper_bound) Series
    return lower_bound, upper_bound


@handle_errors(exceptions=Exception, error_message='Failed to format forecast output', default_return=None)
def format_forecast_output(
    probabilities: SeriesType,
    confidence_intervals: Tuple[SeriesType, SeriesType],
    threshold: ThresholdValue,
    node_id: Optional[NodeID] = None,
    model_version: Optional[str] = None
) -> DataFrameType:
    """
    Formats the prediction results into a standardized forecast DataFrame

    Args:
        probabilities: Series of probability predictions
        confidence_intervals: Tuple of lower and upper confidence bounds
        threshold: Price threshold for spike definition
        node_id: Optional node identifier
        model_version: Optional model version

    Returns:
        DataFrameType: Formatted forecast DataFrame
    """
    # Create a new DataFrame with the probability values
    forecast_df = probabilities.to_frame(name='spike_probability')

    # Add lower and upper confidence interval columns
    lower_bound, upper_bound = confidence_intervals
    forecast_df['confidence_interval_lower'] = lower_bound
    forecast_df['confidence_interval_upper'] = upper_bound

    # Add threshold_value column with the provided threshold
    forecast_df['threshold_value'] = threshold

    # Add node_id column if provided
    if node_id:
        forecast_df['node_id'] = node_id

    # Add model_version column if provided
    if model_version:
        forecast_df['model_version'] = model_version

    # Add forecast_timestamp column with current datetime
    forecast_df['forecast_timestamp'] = datetime.now()

    # Rename the index to target_timestamp
    forecast_df.index.name = 'target_timestamp'

    # Ensure all required columns are present and correctly typed
    required_columns = [
        'spike_probability',
        'confidence_interval_lower',
        'confidence_interval_upper',
        'threshold_value',
        'forecast_timestamp'
    ]
    if node_id:
        required_columns.append('node_id')
    if model_version:
        required_columns.append('model_version')

    # Return the formatted forecast DataFrame
    return forecast_df


class PredictionPipeline:
    """
    Class that implements the end-to-end prediction pipeline for RTLMP spike prediction
    """

    def __init__(self, config: InferenceConfig, forecast_path: Optional[PathType] = None):
        """
        Initializes the PredictionPipeline with configuration

        Args:
            config: Inference configuration
            forecast_path: Optional path to store forecasts
        """
        # Store the configuration as _config
        self._config = config

        # Create ThresholdConfig from config.thresholds
        self._threshold_config = ThresholdConfig(default_thresholds=config.thresholds)

        # Initialize _model and _model_metadata as None
        self._model: Optional[Any] = None
        self._model_metadata: Optional[Dict[str, Any]] = None

        # Initialize _calibrator as None
        self._calibrator: Optional[ProbabilityCalibrator] = None

        # If forecast_path is provided, initialize _forecast_repository
        if forecast_path:
            self._forecast_repository = ForecastRepository(forecast_root=forecast_path)
        # Otherwise, set _forecast_repository to None
        else:
            self._forecast_repository = None

        # Initialize logger
        self.logger = get_logger(__name__)

    def load_model(self, model_id: Optional[str] = None, model_version: Optional[str] = None, model_path: Optional[PathType] = None) -> bool:
        """
        Loads a model for inference

        Args:
            model_id: Optional model identifier
            model_version: Optional model version
            model_path: Optional path to the model file

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Initialize ModelPersistence instance
        model_persistence = ModelPersistence()

        # If model_path is provided, load model from the specific path
        if model_path:
            try:
                self._model, self._model_metadata = model_persistence.load_model(model_id=model_id, model_type=self._model_metadata.get('model_type'), version=model_version, custom_path=model_path)
            except ModelLoadError as e:
                self.logger.error(f"Failed to load model from custom path: {e}")
                return False
        # If model_id and model_version are provided, load that specific model
        elif model_id and model_version:
            self._model, self._model_metadata = model_persistence.load_model(model_id=model_id, model_type=self._model_metadata.get('model_type'), version=model_version)
        # If only model_id is provided, load the latest version of that model
        elif model_id:
            self._model, self._model_metadata = model_persistence.load_model(model_id=model_id, model_type=self._model_metadata.get('model_type'))
        # If no parameters are provided, load the latest model of any type
        else:
            self._model, self._model_metadata = model_persistence.load_model(model_id=self._model_metadata.get('model_id'), model_type=self._model_metadata.get('model_type'))

        # Validate that the model has predict_proba method
        if not hasattr(self._model, 'predict_proba') or not callable(getattr(self._model, 'predict_proba')):
            self.logger.error("Loaded model does not have a predict_proba method")
            return False

        # Log information about the loaded model
        self.logger.info(f"Loaded model: {self._model_metadata.get('model_id')} version {self._model_metadata.get('version')}")

        # Return True if successful, False otherwise
        return True

    def set_calibrator(self, calibrator: ProbabilityCalibrator) -> None:
        """
        Sets the probability calibrator for the pipeline

        Args:
            calibrator: ProbabilityCalibrator instance
        """
        # Validate that calibrator is a ProbabilityCalibrator instance
        if not isinstance(calibrator, ProbabilityCalibrator):
            raise TypeError("calibrator must be an instance of ProbabilityCalibrator")

        # Store the calibrator in _calibrator
        self._calibrator = calibrator

        # Log the calibrator configuration
        self.logger.info(f"Set probability calibrator: {calibrator.get_calibration_method()}")

    def run(self, data_sources: Dict[str, DataFrameType], feature_config: Optional[Dict[str, Any]] = None, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None, store_forecast: Optional[bool] = True) -> DataFrameType:
        """
        Runs the complete prediction pipeline

        Args:
            data_sources: Dictionary of data sources
            feature_config: Optional feature configuration
            thresholds: Optional list of threshold values
            nodes: Optional list of node identifiers
            store_forecast: Optional flag to store the forecast

        Returns:
            DataFrameType: Generated forecast DataFrame
        """
        # Check if model is loaded, raise error if not
        if self._model is None:
            raise ValueError("Model is not loaded. Call load_model() before run()")

        # If thresholds is None, get thresholds from _threshold_config
        if thresholds is None:
            thresholds = self._threshold_config.get_thresholds()

        # Prepare features using prepare_features function
        features_df = prepare_features(data_sources, feature_config)

        # Extract feature names from _model_metadata
        feature_names = self._model_metadata.get('feature_names')

        # Initialize empty list for forecast DataFrames
        forecast_dataframes = []

        # For each threshold in thresholds:
        for threshold in thresholds:
            # Generate predictions using generate_predictions function
            probabilities = generate_predictions(features_df, self._model, feature_names)

            # If _calibrator is not None, calibrate the probabilities
            if self._calibrator is not None:
                probabilities = self._calibrator.calibrate(probabilities)

            # Calculate confidence intervals using calculate_confidence_intervals
            confidence_intervals = calculate_confidence_intervals(probabilities)

            # For each node in nodes (or default node if none provided):
            if nodes is None:
                nodes = ['HB_NORTH']  # Default node
            for node in nodes:
                # Format forecast output using format_forecast_output
                forecast_df = format_forecast_output(probabilities, confidence_intervals, threshold, node, self._model_metadata.get('version'))

                # Append to forecast DataFrames list
                forecast_dataframes.append(forecast_df)

        # Concatenate all forecast DataFrames
        complete_forecast = pd.concat(forecast_dataframes)

        # If store_forecast is True and _forecast_repository is not None, store the forecast
        if store_forecast and self._forecast_repository is not None:
            self._forecast_repository.store_forecast(complete_forecast, self._model_metadata)

        # Log successful forecast generation
        self.logger.info("Successfully generated forecast")

        # Return the complete forecast DataFrame
        return complete_forecast

    def validate_data_sources(self, data_sources: Dict[str, DataFrameType]) -> bool:
        """
        Validates that required data sources are available

        Args:
            data_sources: Dictionary of data sources

        Returns:
            bool: True if valid, False otherwise
        """
        # Check that 'rtlmp' is in data_sources
        if 'rtlmp' not in data_sources:
            self.logger.error("RTLMP data source is missing")
            return False

        # Validate that rtlmp DataFrame has required columns
        rtlmp_df = data_sources['rtlmp']
        required_columns = ['timestamp', 'node_id', 'price']
        for col in required_columns:
            if col not in rtlmp_df.columns:
                self.logger.error(f"RTLMP DataFrame is missing column: {col}")
                return False

        # If 'weather' in data_sources, validate its structure
        if 'weather' in data_sources:
            weather_df = data_sources['weather']
            if not isinstance(weather_df, pd.DataFrame):
                self.logger.error("Weather data source is not a DataFrame")
                return False
            # Add more specific checks here if needed

        # If 'grid_conditions' in data_sources, validate its structure
        if 'grid_conditions' in data_sources:
            grid_df = data_sources['grid_conditions']
            if not isinstance(grid_df, pd.DataFrame):
                self.logger.error("Grid conditions data source is not a DataFrame")
                return False
            # Add more specific checks here if needed

        # Return True if all validations pass, False otherwise
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the currently loaded model

        Returns:
            Dict[str, Any]: Dictionary with model information
        """
        # Check if model is loaded
        if self._model is None:
            self.logger.warning("No model loaded")
            return {}

        # If model is loaded, return a copy of _model_metadata
        return self._model_metadata.copy()


class MultiThresholdPredictionPipeline(PredictionPipeline):
    """
    Extended prediction pipeline that handles multiple thresholds efficiently
    """

    def __init__(self, config: InferenceConfig, forecast_path: Optional[PathType] = None):
        """
        Initializes the MultiThresholdPredictionPipeline

        Args:
            config: Inference configuration
            forecast_path: Optional path to store forecasts
        """
        # Call parent class constructor with config and forecast_path
        super().__init__(config, forecast_path)

        # Initialize empty dictionaries for threshold-specific models and calibrators
        self._threshold_models: Dict[ThresholdValue, Any] = {}
        self._threshold_calibrators: Dict[ThresholdValue, ProbabilityCalibrator] = {}

    def load_threshold_model(self, threshold: ThresholdValue, model_id: Optional[str] = None, model_version: Optional[str] = None) -> bool:
        """
        Loads a model specific to a threshold value

        Args:
            threshold: Threshold value
            model_id: Optional model identifier
            model_version: Optional model version

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Initialize ModelPersistence instance
        model_persistence = ModelPersistence()

        # Attempt to load a model specific to the threshold
        try:
            self._threshold_models[threshold], self._model_metadata = model_persistence.load_model(model_id=model_id, model_type=self._model_metadata.get('model_type'), version=model_version)
        except ModelLoadError as e:
            self.logger.error(f"Failed to load threshold-specific model for threshold {threshold}: {e}")
            return False

        # Log information about the loaded threshold-specific model
        self.logger.info(f"Loaded threshold-specific model for threshold {threshold}: {self._model_metadata.get('model_id')} version {self._model_metadata.get('version')}")

        # Return True if successful, False otherwise
        return True

    def set_threshold_calibrator(self, threshold: ThresholdValue, calibrator: ProbabilityCalibrator) -> None:
        """
        Sets a calibrator for a specific threshold

        Args:
            threshold: Threshold value
            calibrator: ProbabilityCalibrator instance
        """
        # Validate that calibrator is a ProbabilityCalibrator instance
        if not isinstance(calibrator, ProbabilityCalibrator):
            raise TypeError("calibrator must be an instance of ProbabilityCalibrator")

        # Store the calibrator in _threshold_calibrators with threshold as key
        self._threshold_calibrators[threshold] = calibrator

        # Log the threshold-specific calibrator configuration
        self.logger.info(f"Set threshold-specific calibrator for threshold {threshold}: {calibrator.get_calibration_method()}")

    def run(self, data_sources: Dict[str, DataFrameType], feature_config: Optional[Dict[str, Any]] = None, thresholds: Optional[List[ThresholdValue]] = None, nodes: Optional[List[NodeID]] = None, store_forecast: Optional[bool] = True) -> DataFrameType:
        """
        Runs the multi-threshold prediction pipeline

        Args:
            data_sources: Dictionary of data sources
            feature_config: Optional feature configuration
            thresholds: Optional list of threshold values
            nodes: Optional list of node identifiers
            store_forecast: Optional flag to store the forecast

        Returns:
            DataFrameType: Generated forecast DataFrame
        """
        # Override parent class run method to use threshold-specific models

        # Check if general model or threshold models are loaded
        if self._model is None and not self._threshold_models:
            raise ValueError("No models loaded. Call load_model() or load_threshold_model() before run()")

        # If thresholds is None, get thresholds from _threshold_config
        if thresholds is None:
            thresholds = self._threshold_config.get_thresholds()

        # Prepare features using prepare_features function
        features_df = prepare_features(data_sources, feature_config)

        # Initialize empty list for forecast DataFrames
        forecast_dataframes = []

        # For each threshold in thresholds:
        for threshold in thresholds:
            # If threshold in _threshold_models, use that model
            if threshold in self._threshold_models:
                model = self._threshold_models[threshold]
                self.logger.debug(f"Using threshold-specific model for threshold {threshold}")
            # Otherwise, use the general model
            else:
                model = self._model
                self.logger.debug("Using general model")

            # Extract feature names from model metadata
            feature_names = self._model_metadata.get('feature_names')

            # Generate predictions using generate_predictions function
            probabilities = generate_predictions(features_df, model, feature_names)

            # If threshold in _threshold_calibrators, use that calibrator
            if threshold in self._threshold_calibrators:
                calibrator = self._threshold_calibrators[threshold]
                probabilities = calibrator.calibrate(probabilities)
                self.logger.debug(f"Using threshold-specific calibrator for threshold {threshold}")
            # Otherwise, use general calibrator if available
            elif self._calibrator is not None:
                probabilities = self._calibrator.calibrate(probabilities)
                self.logger.debug("Using general calibrator")

            # Calculate confidence intervals using calculate_confidence_intervals
            confidence_intervals = calculate_confidence_intervals(probabilities)

            # For each node in nodes (or default node if none provided):
            if nodes is None:
                nodes = ['HB_NORTH']  # Default node
            for node in nodes:
                # Format forecast output using format_forecast_output
                forecast_df = format_forecast_output(probabilities, confidence_intervals, threshold, node, self._model_metadata.get('version'))

                # Append to forecast DataFrames list
                forecast_dataframes.append(forecast_df)

        # Concatenate all forecast DataFrames
        complete_forecast = pd.concat(forecast_dataframes)

        # If store_forecast is True and _forecast_repository is not None, store the forecast
        if store_forecast and self._forecast_repository is not None:
            self._forecast_repository.store_forecast(complete_forecast, self._model_metadata)

        # Log successful forecast generation
        self.logger.info("Successfully generated forecast")

        # Return the complete forecast DataFrame
        return complete_forecast