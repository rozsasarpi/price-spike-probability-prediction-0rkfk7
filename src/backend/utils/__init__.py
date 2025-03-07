"""
Utility package for the ERCOT RTLMP spike prediction system.

This package provides common utility functions, classes, constants, and type definitions
that are used throughout the system to ensure consistent implementation of core
functionality such as data validation, error handling, logging, date/time operations,
and statistical calculations.
"""

# Import all items from utility modules
from .type_definitions import *
from .date_utils import *
from .statistics import *
from .validation import *
from .error_handling import *
from .logging import *

# Package version
__version__ = '0.1.0'

# Explicitly define all public symbols exported by the package
__all__ = [
    # Type definitions
    'DataFrameType', 'SeriesType', 'ArrayType', 'PathType', 'ThresholdValue', 'NodeID', 'ModelType',
    'FeatureGroupType', 'RTLMPDataDict', 'WeatherDataDict', 'GridConditionDict', 'ModelConfigDict',
    'ForecastResultDict', 'DataFetcherProtocol', 'ModelProtocol', 'FeatureEngineerProtocol',
    'InferenceEngineProtocol', 'BacktestingProtocol', 'StorageProtocol',

    # Date utilities
    'ERCOT_TIMEZONE', 'UTC_TIMEZONE', 'DEFAULT_DATETIME_FORMAT', 'ERCOT_API_DATETIME_FORMAT',
    'DAY_AHEAD_MARKET_CLOSURE_HOUR', 'localize_datetime', 'convert_to_utc', 'format_datetime',
    'parse_datetime', 'get_current_time', 'is_dst', 'create_date_range', 'round_datetime',
    'floor_datetime', 'ceil_datetime', 'validate_date_range', 'get_day_ahead_market_closure',
    'is_before_day_ahead_market_closure', 'get_forecast_horizon_end', 'get_datetime_components',

    # Statistics utilities
    'calculate_rolling_statistics', 'calculate_price_volatility', 'calculate_spike_frequency',
    'calculate_quantiles', 'calculate_distribution_metrics', 'calculate_autocorrelation',
    'calculate_crosscorrelation', 'calculate_hourly_statistics', 'calculate_binary_classification_metrics',
    'calculate_probability_metrics', 'calculate_confidence_intervals', 'DEFAULT_ROLLING_WINDOWS',
    'DEFAULT_QUANTILES', 'DEFAULT_STATISTICS',

    # Validation utilities
    'validate_dataframe_schema', 'validate_data_completeness', 'validate_value_ranges',
    'validate_temporal_consistency', 'validate_required_columns', 'validate_unique_values',
    'validate_data_types', 'validate_probability_values', 'validate_input_data',
    'validate_output_data', 'detect_outliers', 'validate_consistency', 'DataValidator',
    'ValidationResult', 'DEFAULT_COMPLETENESS_THRESHOLD', 'DEFAULT_OUTLIER_THRESHOLD',

    # Error handling utilities
    'retry_with_backoff', 'handle_errors', 'format_exception', 'get_error_context',
    'is_retryable_error', 'circuit_breaker', 'log_error', 'BaseError', 'DataError',
    'DataFormatError', 'MissingDataError', 'ConnectionError', 'RateLimitError',
    'ModelError', 'ModelLoadError', 'ModelTrainingError', 'InferenceError',
    'CircuitOpenError', 'ErrorHandler', 'RetryContext',

    # Logging utilities
    'setup_logging', 'get_logger', 'log_execution_time', 'log_function_call',
    'format_log_message', 'sanitize_log_data', 'configure_component_logger',
    'JsonFormatter', 'ContextAdapter', 'LoggingContext', 'PerformanceLogger',
    'DEFAULT_LOG_LEVEL', 'DEFAULT_LOG_FORMAT', 'DEFAULT_DATE_FORMAT', 'LOG_LEVELS'
]