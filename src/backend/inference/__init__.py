"""
Initialization module for the ERCOT RTLMP spike prediction system's inference package.
This module exposes the key components and interfaces for generating probability forecasts of price spikes in the RTLMP market.
"""

from .threshold_config import (  # src/backend/inference/threshold_config.py
    ThresholdConfig,
    DynamicThresholdConfig,
    validate_thresholds,
    get_default_thresholds,
    DEFAULT_THRESHOLDS,
)
from .thresholds import (  # src/backend/inference/thresholds.py
    ThresholdApplier,
    RollingThresholdAnalyzer,
    apply_threshold,
    apply_thresholds,
    create_spike_indicator,
    create_multi_threshold_indicators,
    find_max_price_in_window,
    hourly_spike_occurrence,
)
from .calibration import (  # src/backend/inference/calibration.py
    ProbabilityCalibrator,
    CalibrationEvaluator,
    calibrate_probabilities,
    evaluate_calibration,
    plot_calibration_curve,
    calculate_expected_calibration_error,
    calculate_maximum_calibration_error,
    CALIBRATION_METHODS,
    DEFAULT_CALIBRATION_METHOD,
)
from .engine import (  # src/backend/inference/engine.py
    InferenceEngine,
    load_model_for_inference,
    generate_forecast,
    get_latest_forecast,
    compare_forecasts,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_CONFIDENCE_LEVEL,
)
from .prediction_pipeline import (  # src/backend/inference/prediction_pipeline.py
    PredictionPipeline,
    MultiThresholdPredictionPipeline,
    prepare_features,
    generate_predictions,
    calculate_confidence_intervals,
    format_forecast_output,
)
from ..utils.logging import get_logger  # src/backend/utils/logging.py

logger = get_logger(__name__)

__version__ = "0.1.0"

__all__ = [
    "ThresholdConfig",
    "DynamicThresholdConfig",
    "ThresholdApplier",
    "RollingThresholdAnalyzer",
    "ProbabilityCalibrator",
    "CalibrationEvaluator",
    "InferenceEngine",
    "PredictionPipeline",
    "MultiThresholdPredictionPipeline",
    "validate_thresholds",
    "get_default_thresholds",
    "apply_threshold",
    "apply_thresholds",
    "create_spike_indicator",
    "create_multi_threshold_indicators",
    "find_max_price_in_window",
    "hourly_spike_occurrence",
    "calibrate_probabilities",
    "evaluate_calibration",
    "plot_calibration_curve",
    "load_model_for_inference",
    "generate_forecast",
    "get_latest_forecast",
    "compare_forecasts",
    "prepare_features",
    "generate_predictions",
    "calculate_confidence_intervals",
    "format_forecast_output",
    "DEFAULT_THRESHOLDS",
    "CALIBRATION_METHODS",
    "DEFAULT_CALIBRATION_METHOD",
    "DEFAULT_FORECAST_HORIZON",
    "DEFAULT_CONFIDENCE_LEVEL",
    "__version__",
]