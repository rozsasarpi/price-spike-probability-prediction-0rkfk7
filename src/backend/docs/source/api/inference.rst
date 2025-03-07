Inference API
============

The inference module provides functionality for generating probability forecasts for RTLMP price spikes in the ERCOT market. It includes components for model loading, feature preparation, prediction generation, probability calibration, and threshold handling.

Inference Engine
---------------

The core inference engine that orchestrates the entire prediction process.

.. py:module:: inference.engine

.. py:class:: InferenceEngine

   The main class responsible for orchestrating the inference process for generating RTLMP spike probability forecasts.

   .. py:method:: __init__(model_path=None, threshold_config=None, forecast_horizon=None)

      Initialize the inference engine.

      :param str model_path: Optional path to a specific model file. If None, the latest model will be used.
      :param ThresholdConfig threshold_config: Configuration for price thresholds. If None, default thresholds will be used.
      :param int forecast_horizon: Number of hours to forecast. If None, DEFAULT_FORECAST_HORIZON will be used.

   .. py:method:: load_model(model_path=None, model_version=None)

      Load a model for inference.

      :param str model_path: Path to the model file. If None, uses the instance's model_path.
      :param str model_version: Specific model version to load. If provided, overrides model_path.
      :return: True if model was successfully loaded, False otherwise.
      :rtype: bool
      :raises ModelLoadError: If the model cannot be loaded.

   .. py:method:: prepare_features(latest_data=None)

      Prepare features required for inference.

      :param DataFrame latest_data: Optional DataFrame containing the latest data. If None, data will be fetched.
      :return: DataFrame containing prepared features for prediction.
      :rtype: pandas.DataFrame
      :raises FeatureEngineeringError: If feature preparation fails.

   .. py:method:: generate_predictions(features, thresholds=None)

      Generate raw probability predictions.

      :param DataFrame features: Features to use for prediction.
      :param list thresholds: List of price thresholds. If None, uses configured thresholds.
      :return: DataFrame with raw probability predictions.
      :rtype: pandas.DataFrame
      :raises PredictionError: If prediction generation fails.

   .. py:method:: calibrate_predictions(predictions)

      Apply probability calibration to raw predictions.

      :param DataFrame predictions: Raw probability predictions.
      :return: DataFrame with calibrated probabilities.
      :rtype: pandas.DataFrame
      :raises CalibrationError: If calibration fails.

   .. py:method:: format_output(predictions)

      Format predictions into the standard output format.

      :param DataFrame predictions: Calibrated probability predictions.
      :return: DataFrame with formatted predictions including confidence intervals.
      :rtype: pandas.DataFrame

   .. py:method:: run_inference(data=None, thresholds=None)

      Run the complete inference pipeline.

      :param DataFrame data: Optional data to use. If None, data will be fetched.
      :param list thresholds: Optional list of price thresholds. If None, uses configured thresholds.
      :return: DataFrame with forecast probabilities.
      :rtype: pandas.DataFrame
      :raises InferenceError: If the inference process fails.

   .. py:method:: save_forecast(forecast, metadata=None)

      Save the generated forecast.

      :param DataFrame forecast: Forecast to save.
      :param dict metadata: Optional metadata to save with the forecast.
      :return: Path where the forecast was saved.
      :rtype: str
      :raises StorageError: If saving the forecast fails.

.. py:function:: load_model_for_inference(model_path=None, model_version=None)

   Utility function to load a model for inference.

   :param str model_path: Path to the model file.
   :param str model_version: Specific model version to load.
   :return: Loaded model object.
   :raises ModelLoadError: If the model cannot be loaded.

.. py:function:: generate_forecast(model=None, data=None, thresholds=None, forecast_horizon=None)

   Generate a forecast using the provided model and data.

   :param model: Model object to use for prediction.
   :param DataFrame data: Data to use for prediction.
   :param list thresholds: List of price thresholds.
   :param int forecast_horizon: Number of hours to forecast.
   :return: DataFrame with forecast probabilities.
   :rtype: pandas.DataFrame
   :raises InferenceError: If forecast generation fails.

.. py:function:: get_latest_forecast()

   Retrieve the most recently generated forecast.

   :return: Tuple containing (forecast DataFrame, metadata dict, timestamp).
   :rtype: tuple
   :raises StorageError: If retrieving the forecast fails.

.. py:function:: compare_forecasts(forecast1, forecast2)

   Compare two forecasts and compute difference metrics.

   :param DataFrame forecast1: First forecast.
   :param DataFrame forecast2: Second forecast.
   :return: DataFrame with comparison metrics.
   :rtype: pandas.DataFrame

.. py:data:: DEFAULT_FORECAST_HORIZON

   Default number of hours to forecast (72).

.. py:data:: DEFAULT_CONFIDENCE_LEVEL

   Default confidence level for prediction intervals (0.95).

Prediction Pipeline
------------------

Components for the end-to-end prediction pipeline from feature preparation to forecast generation.

.. py:module:: inference.prediction_pipeline

.. py:class:: PredictionPipeline

   A pipeline that manages the end-to-end process of generating predictions.

   .. py:method:: __init__(model, threshold_config=None, forecast_horizon=None)

      Initialize the prediction pipeline.

      :param model: Model object to use for prediction.
      :param ThresholdConfig threshold_config: Configuration for price thresholds.
      :param int forecast_horizon: Number of hours to forecast.

   .. py:method:: prepare_input_features(data)

      Prepare input features for the model.

      :param DataFrame data: Raw data.
      :return: Prepared features.
      :rtype: pandas.DataFrame
      :raises FeatureEngineeringError: If feature preparation fails.

   .. py:method:: predict(features)

      Generate predictions using the model.

      :param DataFrame features: Prepared features.
      :return: Raw predictions.
      :rtype: pandas.DataFrame
      :raises PredictionError: If prediction fails.

   .. py:method:: postprocess_predictions(predictions)

      Apply post-processing to the raw predictions.

      :param DataFrame predictions: Raw predictions.
      :return: Post-processed predictions.
      :rtype: pandas.DataFrame

   .. py:method:: add_confidence_intervals(predictions, confidence_level=None)

      Add confidence intervals to the predictions.

      :param DataFrame predictions: Predictions.
      :param float confidence_level: Confidence level (between 0 and 1).
      :return: Predictions with confidence intervals.
      :rtype: pandas.DataFrame

   .. py:method:: run(data=None)

      Run the complete prediction pipeline.

      :param DataFrame data: Optional data to use.
      :return: Final predictions with confidence intervals.
      :rtype: pandas.DataFrame
      :raises PipelineError: If the pipeline execution fails.

.. py:class:: MultiThresholdPredictionPipeline

   A prediction pipeline that handles multiple price thresholds.

   .. py:method:: __init__(model, thresholds, forecast_horizon=None)

      Initialize the multi-threshold prediction pipeline.

      :param model: Model object to use for prediction.
      :param list thresholds: List of price thresholds.
      :param int forecast_horizon: Number of hours to forecast.

   .. py:method:: predict_for_thresholds(features)

      Generate predictions for each threshold.

      :param DataFrame features: Prepared features.
      :return: Dictionary mapping thresholds to predictions.
      :rtype: dict
      :raises PredictionError: If prediction fails.

   .. py:method:: combine_threshold_predictions(predictions_dict)

      Combine predictions for different thresholds into a single DataFrame.

      :param dict predictions_dict: Dictionary mapping thresholds to predictions.
      :return: Combined predictions.
      :rtype: pandas.DataFrame

   .. py:method:: run(data=None)

      Run the complete multi-threshold prediction pipeline.

      :param DataFrame data: Optional data to use.
      :return: Final predictions for all thresholds with confidence intervals.
      :rtype: pandas.DataFrame
      :raises PipelineError: If the pipeline execution fails.

.. py:function:: prepare_features(data, feature_config=None)

   Prepare features for prediction.

   :param DataFrame data: Raw data.
   :param dict feature_config: Feature configuration.
   :return: Prepared features.
   :rtype: pandas.DataFrame
   :raises FeatureEngineeringError: If feature preparation fails.

.. py:function:: generate_predictions(model, features, threshold=None)

   Generate predictions using a model.

   :param model: Model object.
   :param DataFrame features: Prepared features.
   :param float threshold: Price threshold.
   :return: Predictions.
   :rtype: pandas.DataFrame
   :raises PredictionError: If prediction fails.

.. py:function:: calculate_confidence_intervals(predictions, confidence_level=0.95)

   Calculate confidence intervals for predictions.

   :param DataFrame predictions: Predictions.
   :param float confidence_level: Confidence level (between 0 and 1).
   :return: DataFrame with added confidence interval columns.
   :rtype: pandas.DataFrame

.. py:function:: format_forecast_output(predictions, metadata=None)

   Format predictions into the standard output format.

   :param DataFrame predictions: Predictions with confidence intervals.
   :param dict metadata: Optional metadata to include.
   :return: Formatted forecast.
   :rtype: pandas.DataFrame

.. py:data:: DEFAULT_FORECAST_HORIZON

   Default number of hours to forecast (72).

.. py:data:: DEFAULT_CONFIDENCE_LEVEL

   Default confidence level for prediction intervals (0.95).

Calibration
----------

Tools for calibrating probability predictions to ensure they accurately reflect true probabilities.

.. py:module:: inference.calibration

.. py:class:: ProbabilityCalibrator

   Class for calibrating probability predictions.

   .. py:method:: __init__(method='isotonic', params=None)

      Initialize the calibrator.

      :param str method: Calibration method ('isotonic', 'platt', or 'beta').
      :param dict params: Parameters for the calibration method.

   .. py:method:: fit(y_true, y_pred)

      Fit the calibrator to the training data.

      :param array y_true: True binary labels.
      :param array y_pred: Predicted probabilities.
      :return: Self.
      :rtype: ProbabilityCalibrator

   .. py:method:: calibrate(y_pred)

      Calibrate predicted probabilities.

      :param array y_pred: Predicted probabilities.
      :return: Calibrated probabilities.
      :rtype: array
      :raises CalibrationError: If calibration fails.

   .. py:method:: save(path)

      Save the calibrator to a file.

      :param str path: Path to save the calibrator.
      :return: True if successful.
      :rtype: bool
      :raises StorageError: If saving fails.

   .. py:method:: load(path)

      Load a calibrator from a file.

      :param str path: Path to the calibrator file.
      :return: Self.
      :rtype: ProbabilityCalibrator
      :raises StorageError: If loading fails.

.. py:class:: CalibrationEvaluator

   Class for evaluating the quality of probability calibration.

   .. py:method:: __init__(n_bins=10)

      Initialize the evaluator.

      :param int n_bins: Number of bins for reliability diagram.

   .. py:method:: compute_reliability_diagram(y_true, y_pred)

      Compute reliability diagram data.

      :param array y_true: True binary labels.
      :param array y_pred: Predicted probabilities.
      :return: Tuple of (mean_predicted_probs, true_probs) per bin.
      :rtype: tuple

   .. py:method:: compute_calibration_metrics(y_true, y_pred)

      Compute calibration metrics.

      :param array y_true: True binary labels.
      :param array y_pred: Predicted probabilities.
      :return: Dictionary of calibration metrics.
      :rtype: dict

   .. py:method:: plot_reliability_diagram(y_true, y_pred, ax=None)

      Plot a reliability diagram.

      :param array y_true: True binary labels.
      :param array y_pred: Predicted probabilities.
      :param ax: Matplotlib axis.
      :return: Matplotlib axis.

.. py:function:: calibrate_probabilities(y_pred, calibrator)

   Calibrate predicted probabilities using a calibrator.

   :param array y_pred: Predicted probabilities.
   :param ProbabilityCalibrator calibrator: Calibrator object.
   :return: Calibrated probabilities.
   :rtype: array
   :raises CalibrationError: If calibration fails.

.. py:function:: evaluate_calibration(y_true, y_pred, n_bins=10)

   Evaluate the calibration of predicted probabilities.

   :param array y_true: True binary labels.
   :param array y_pred: Predicted probabilities.
   :param int n_bins: Number of bins for reliability diagram.
   :return: Dictionary of calibration metrics.
   :rtype: dict

.. py:function:: plot_calibration_curve(y_true, y_pred, n_bins=10, ax=None)

   Plot a calibration curve (reliability diagram).

   :param array y_true: True binary labels.
   :param array y_pred: Predicted probabilities.
   :param int n_bins: Number of bins for reliability diagram.
   :param ax: Matplotlib axis.
   :return: Matplotlib axis.

.. py:function:: calculate_expected_calibration_error(y_true, y_pred, n_bins=10)

   Calculate the expected calibration error.

   :param array y_true: True binary labels.
   :param array y_pred: Predicted probabilities.
   :param int n_bins: Number of bins for reliability diagram.
   :return: Expected calibration error.
   :rtype: float

.. py:function:: calculate_maximum_calibration_error(y_true, y_pred, n_bins=10)

   Calculate the maximum calibration error.

   :param array y_true: True binary labels.
   :param array y_pred: Predicted probabilities.
   :param int n_bins: Number of bins for reliability diagram.
   :return: Maximum calibration error.
   :rtype: float

.. py:data:: CALIBRATION_METHODS

   Dictionary of supported calibration methods.

.. py:data:: DEFAULT_CALIBRATION_METHOD

   Default calibration method ('isotonic').

Thresholds
---------

Components for managing and applying price threshold values to identify price spikes.

.. py:module:: inference.threshold_config

.. py:class:: ThresholdConfig

   Configuration for price thresholds.

   .. py:method:: __init__(thresholds=None, default_threshold=None)

      Initialize threshold configuration.

      :param list thresholds: List of threshold values.
      :param float default_threshold: Default threshold value.

   .. py:method:: get_thresholds()

      Get the list of thresholds.

      :return: List of threshold values.
      :rtype: list

   .. py:method:: get_default_threshold()

      Get the default threshold.

      :return: Default threshold value.
      :rtype: float

   .. py:method:: add_threshold(threshold)

      Add a threshold value.

      :param float threshold: Threshold value to add.
      :return: Self.
      :rtype: ThresholdConfig
      :raises ValueError: If threshold is invalid.

   .. py:method:: remove_threshold(threshold)

      Remove a threshold value.

      :param float threshold: Threshold value to remove.
      :return: Self.
      :rtype: ThresholdConfig
      :raises ValueError: If threshold not found.

.. py:class:: DynamicThresholdConfig

   Configuration for dynamically calculated thresholds.

   .. py:method:: __init__(base_threshold=None, percentile=None, lookback_days=None)

      Initialize dynamic threshold configuration.

      :param float base_threshold: Base threshold value.
      :param float percentile: Percentile for dynamic threshold calculation.
      :param int lookback_days: Number of days to look back for dynamic threshold.

   .. py:method:: calculate_thresholds(price_data)

      Calculate thresholds based on historical price data.

      :param DataFrame price_data: Historical price data.
      :return: List of calculated threshold values.
      :rtype: list
      :raises ValueError: If price_data is invalid.

   .. py:method:: update_thresholds(price_data)

      Update thresholds based on new price data.

      :param DataFrame price_data: New price data.
      :return: Self.
      :rtype: DynamicThresholdConfig
      :raises ValueError: If price_data is invalid.

.. py:function:: validate_thresholds(thresholds)

   Validate a list of threshold values.

   :param list thresholds: List of threshold values.
   :return: True if valid, False otherwise.
   :rtype: bool

.. py:function:: get_default_thresholds()

   Get the default threshold values.

   :return: List of default threshold values.
   :rtype: list

.. py:data:: DEFAULT_THRESHOLDS

   List of default threshold values ([50.0, 100.0, 200.0, 500.0, 1000.0]).

.. py:module:: inference.thresholds

.. py:class:: ThresholdApplier

   Class for applying thresholds to price data.

   .. py:method:: __init__(thresholds=None)

      Initialize the threshold applier.

      :param list thresholds: List of threshold values.

   .. py:method:: apply_thresholds(price_data)

      Apply thresholds to price data.

      :param DataFrame price_data: Price data.
      :return: DataFrame with threshold indicators.
      :rtype: pandas.DataFrame
      :raises ValueError: If price_data is invalid.

   .. py:method:: calculate_spike_statistics(price_data, threshold)

      Calculate statistics about price spikes.

      :param DataFrame price_data: Price data.
      :param float threshold: Threshold value.
      :return: Dictionary of spike statistics.
      :rtype: dict
      :raises ValueError: If price_data is invalid.

.. py:class:: RollingThresholdAnalyzer

   Class for analyzing price data with rolling thresholds.

   .. py:method:: __init__(window_size=12, step_size=1)

      Initialize the analyzer.

      :param int window_size: Size of the rolling window in hours.
      :param int step_size: Step size for the rolling window.

   .. py:method:: find_max_in_window(price_data)

      Find the maximum price in each rolling window.

      :param DataFrame price_data: Price data.
      :return: DataFrame with maximum prices.
      :rtype: pandas.DataFrame
      :raises ValueError: If price_data is invalid.

   .. py:method:: calculate_spike_frequency(price_data, threshold)

      Calculate frequency of spikes in rolling windows.

      :param DataFrame price_data: Price data.
      :param float threshold: Threshold value.
      :return: DataFrame with spike frequencies.
      :rtype: pandas.DataFrame
      :raises ValueError: If price_data is invalid.

.. py:function:: apply_threshold(price_data, threshold)

   Apply a single threshold to price data.

   :param DataFrame price_data: Price data.
   :param float threshold: Threshold value.
   :return: Series with boolean indicators where price exceeds threshold.
   :rtype: pandas.Series
   :raises ValueError: If price_data is invalid.

.. py:function:: apply_thresholds(price_data, thresholds)

   Apply multiple thresholds to price data.

   :param DataFrame price_data: Price data.
   :param list thresholds: List of threshold values.
   :return: DataFrame with boolean indicators for each threshold.
   :rtype: pandas.DataFrame
   :raises ValueError: If price_data is invalid.

.. py:function:: create_spike_indicator(price_data, threshold)

   Create a binary indicator for price spikes.

   :param DataFrame price_data: Price data.
   :param float threshold: Threshold value.
   :return: Series with binary indicators (0/1).
   :rtype: pandas.Series
   :raises ValueError: If price_data is invalid.

.. py:function:: create_multi_threshold_indicators(price_data, thresholds)

   Create binary indicators for multiple thresholds.

   :param DataFrame price_data: Price data.
   :param list thresholds: List of threshold values.
   :return: DataFrame with binary indicators for each threshold.
   :rtype: pandas.DataFrame
   :raises ValueError: If price_data is invalid.

.. py:function:: find_max_price_in_window(price_data, window_size=12)

   Find the maximum price in each rolling window.

   :param DataFrame price_data: Price data.
   :param int window_size: Size of the rolling window in hours.
   :return: Series with maximum prices.
   :rtype: pandas.Series
   :raises ValueError: If price_data is invalid.

.. py:function:: hourly_spike_occurrence(price_data, threshold, resample_rule='H')

   Determine if a spike occurred within each hour.

   :param DataFrame price_data: Price data.
   :param float threshold: Threshold value.
   :param str resample_rule: Rule for resampling to hourly data.
   :return: Series with binary indicators (0/1) for each hour.
   :rtype: pandas.Series
   :raises ValueError: If price_data is invalid.

Inference API
-----------

High-level API for inference operations that provides a unified interface for generating forecasts.

.. py:module:: api.inference_api

.. py:class:: InferenceAPI

   High-level API for inference operations.

   .. py:method:: __init__(model_path=None, threshold_config=None)

      Initialize the API.

      :param str model_path: Path to the model file.
      :param ThresholdConfig threshold_config: Threshold configuration.

   .. py:method:: load_model(model_path=None, model_version=None)

      Load a model for inference.

      :param str model_path: Path to the model file.
      :param str model_version: Model version to load.
      :return: True if successful.
      :rtype: bool
      :raises ModelLoadError: If loading fails.

   .. py:method:: generate_forecast(data=None, thresholds=None, forecast_horizon=None)

      Generate a forecast.

      :param DataFrame data: Data to use for prediction.
      :param list thresholds: List of threshold values.
      :param int forecast_horizon: Number of hours to forecast.
      :return: Forecast DataFrame.
      :rtype: pandas.DataFrame
      :raises InferenceError: If forecast generation fails.

   .. py:method:: get_latest_forecast()

      Get the latest generated forecast.

      :return: Tuple of (forecast, metadata, timestamp).
      :rtype: tuple
      :raises StorageError: If retrieval fails.

   .. py:method:: get_forecast_by_date(date)

      Get a forecast generated on a specific date.

      :param datetime date: Date of the forecast.
      :return: Tuple of (forecast, metadata).
      :rtype: tuple
      :raises StorageError: If retrieval fails.

   .. py:method:: compare_forecasts(forecast1, forecast2)

      Compare two forecasts.

      :param DataFrame forecast1: First forecast.
      :param DataFrame forecast2: Second forecast.
      :return: Comparison DataFrame.
      :rtype: pandas.DataFrame

   .. py:method:: list_available_models()

      List all available models.

      :return: List of model information dictionaries.
      :rtype: list

.. py:function:: generate_forecast(model_path=None, data=None, thresholds=None, forecast_horizon=None)

   Generate a forecast using the specified model.

   :param str model_path: Path to the model file.
   :param DataFrame data: Data to use for prediction.
   :param list thresholds: List of threshold values.
   :param int forecast_horizon: Number of hours to forecast.
   :return: Forecast DataFrame.
   :rtype: pandas.DataFrame
   :raises InferenceError: If forecast generation fails.

.. py:function:: get_latest_forecast()

   Get the latest generated forecast.

   :return: Tuple of (forecast, metadata, timestamp).
   :rtype: tuple
   :raises StorageError: If retrieval fails.

.. py:function:: get_forecast_by_date(date)

   Get a forecast generated on a specific date.

   :param datetime date: Date of the forecast.
   :return: Tuple of (forecast, metadata).
   :rtype: tuple
   :raises StorageError: If retrieval fails.

.. py:function:: compare_forecasts(forecast1, forecast2)

   Compare two forecasts.

   :param DataFrame forecast1: First forecast.
   :param DataFrame forecast2: Second forecast.
   :return: Comparison DataFrame.
   :rtype: pandas.DataFrame

.. py:function:: list_available_models()

   List all available models.

   :return: List of model information dictionaries.
   :rtype: list

.. py:data:: DEFAULT_FORECAST_HORIZON

   Default number of hours to forecast (72).

Examples
--------

Basic Forecast Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from backend.api.inference_api import InferenceAPI

   # Initialize the API
   inference_api = InferenceAPI()

   # Load the latest model
   inference_api.load_model()

   # Generate a forecast
   forecast_df = inference_api.generate_forecast()

   print(f"Generated forecast with shape: {forecast_df.shape}")

Custom Threshold Forecast
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from backend.api.inference_api import InferenceAPI

   # Initialize the API
   inference_api = InferenceAPI()

   # Generate a forecast with custom thresholds
   thresholds = [100.0, 200.0, 500.0]
   forecast_df = inference_api.generate_forecast(thresholds=thresholds)

   print(f"Generated forecast for thresholds: {thresholds}")

Retrieving Previous Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from backend.api.inference_api import InferenceAPI
   import datetime

   # Initialize the API
   inference_api = InferenceAPI()

   # Get the latest forecast
   latest_forecast, metadata, timestamp = inference_api.get_latest_forecast()

   # Get a forecast from a specific date
   forecast_date = datetime.datetime(2023, 7, 15)
   historical_forecast, hist_metadata = inference_api.get_forecast_by_date(forecast_date)

   print(f"Latest forecast generated at: {timestamp}")
   print(f"Historical forecast from: {forecast_date}")