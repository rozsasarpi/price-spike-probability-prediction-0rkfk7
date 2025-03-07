Utilities API Reference
======================

Comprehensive documentation of utility modules that provide core functionality for the ERCOT RTLMP spike prediction system.

Overview
--------

The utils package provides a collection of utility functions, classes, and constants that support the core functionality of the ERCOT RTLMP spike prediction system. These utilities are organized into several modules, each focusing on a specific area of functionality:

- **Type Definitions**: Type aliases, protocols, and type definitions for consistent typing across the system
- **Date Utilities**: Functions for date/time handling, timezone conversions, and temporal operations
- **Statistics**: Statistical functions for time series analysis and model evaluation
- **Validation**: Data validation utilities for ensuring data quality and integrity
- **Error Handling**: Error classes, decorators, and utilities for robust error management
- **Logging**: Logging configuration, formatters, and utilities for system monitoring

Type Definitions
---------------

The type_definitions module provides type aliases, protocols, and type definitions that ensure type consistency across the system and enable static type checking.

Type Aliases
~~~~~~~~~~~

.. data:: DataFrameType
   :annotation: = pandas.DataFrame

   Type alias for pandas DataFrame

.. data:: SeriesType
   :annotation: = pandas.Series

   Type alias for pandas Series

.. data:: ArrayType
   :annotation: = numpy.ndarray

   Type alias for numpy ndarray

.. data:: PathType
   :annotation: = Union[str, pathlib.Path]

   Type alias for file paths

.. data:: ThresholdValue
   :annotation: = float

   Type alias for price threshold values

.. data:: NodeID
   :annotation: = str

   Type alias for ERCOT node identifiers

.. data:: ModelType
   :annotation: = TypeVar('ModelType')

   Generic type variable for model objects

.. data:: FeatureGroupType
   :annotation: = Literal['time', 'statistical', 'weather', 'market']

   Literal type for feature group categories

Type Definitions
~~~~~~~~~~~~~~~

.. class:: RTLMPDataDict(TypedDict)

   Type definition for RTLMP data structure

   .. attribute:: timestamp
      :type: datetime

   .. attribute:: node_id
      :type: str

   .. attribute:: price
      :type: float

   .. attribute:: congestion_price
      :type: float

   .. attribute:: loss_price
      :type: float

   .. attribute:: energy_price
      :type: float

.. class:: WeatherDataDict(TypedDict)

   Type definition for weather data structure

   .. attribute:: timestamp
      :type: datetime

   .. attribute:: location_id
      :type: str

   .. attribute:: temperature
      :type: float

   .. attribute:: wind_speed
      :type: float

   .. attribute:: solar_irradiance
      :type: float

   .. attribute:: humidity
      :type: float

.. class:: GridConditionDict(TypedDict)

   Type definition for grid condition data structure

   .. attribute:: timestamp
      :type: datetime

   .. attribute:: total_load
      :type: float

   .. attribute:: available_capacity
      :type: float

   .. attribute:: wind_generation
      :type: float

   .. attribute:: solar_generation
      :type: float

.. class:: ModelConfigDict(TypedDict)

   Type definition for model configuration

   .. attribute:: model_id
      :type: str

   .. attribute:: model_type
      :type: str

   .. attribute:: version
      :type: str

   .. attribute:: hyperparameters
      :type: Dict[str, Any]

   .. attribute:: performance_metrics
      :type: Optional[Dict[str, float]]

   .. attribute:: training_date
      :type: Optional[datetime]

   .. attribute:: feature_names
      :type: Optional[List[str]]

.. class:: ForecastResultDict(TypedDict)

   Type definition for forecast result structure

   .. attribute:: forecast_timestamp
      :type: datetime

   .. attribute:: target_timestamp
      :type: datetime

   .. attribute:: threshold_value
      :type: float

   .. attribute:: spike_probability
      :type: float

   .. attribute:: confidence_interval_lower
      :type: float

   .. attribute:: confidence_interval_upper
      :type: float

   .. attribute:: model_version
      :type: str

Protocols
~~~~~~~~~

.. class:: DataFetcherProtocol(Protocol)

   Protocol defining the interface for data fetchers

   .. method:: fetch_data(params: Dict[str, Any]) -> DataFrameType
      
      Retrieve data according to the specified parameters

   .. method:: fetch_historical_data(params: Dict[str, Any]) -> DataFrameType
      
      Retrieve historical data according to the specified parameters

   .. method:: fetch_forecast_data(params: Dict[str, Any]) -> DataFrameType
      
      Retrieve forecast data according to the specified parameters

   .. method:: validate_data(data: DataFrameType) -> bool
      
      Validate that data meets expected format and quality requirements

.. class:: ModelProtocol(Protocol)

   Protocol defining the interface for prediction models

   .. method:: train(X: DataFrameType, y: SeriesType) -> None
      
      Train the model on the provided features and target

   .. method:: predict(X: DataFrameType) -> ArrayType
      
      Generate predictions for the provided features

   .. method:: predict_proba(X: DataFrameType) -> ArrayType
      
      Generate probability predictions for the provided features

   .. method:: save(path: PathType) -> None
      
      Save the model to the specified path

   .. method:: load(path: PathType) -> None
      
      Load the model from the specified path

   .. method:: get_feature_importance(self) -> Dict[str, float]
      
      Return feature importance scores as a dictionary

   .. method:: get_model_config(self) -> ModelConfigDict
      
      Return the model configuration details

.. class:: FeatureEngineerProtocol(Protocol)

   Protocol defining the interface for feature engineering

   .. method:: create_features(data: DataFrameType) -> DataFrameType
      
      Transform raw data into engineered features

   .. method:: get_feature_names() -> List[str]
      
      Return the list of feature names produced by this engineer

   .. method:: validate_feature_set(features: DataFrameType) -> bool
      
      Validate that a feature set contains all required features with correct types

.. class:: InferenceEngineProtocol(Protocol)

   Protocol defining the interface for inference engine

   .. method:: load_model(model_path: PathType) -> None
      
      Load a model from the specified path

   .. method:: generate_forecast(features: DataFrameType, thresholds: List[float]) -> DataFrameType
      
      Generate forecasts for the provided features and thresholds

   .. method:: store_forecast(forecast: DataFrameType, path: PathType) -> None
      
      Store the forecast results at the specified path

   .. method:: run_inference(config: Dict[str, Any]) -> DataFrameType
      
      Run the complete inference process according to configuration

.. class:: BacktestingProtocol(Protocol)

   Protocol defining the interface for backtesting framework

   .. method:: run_backtest(config: Dict[str, Any]) -> Dict[str, Any]
      
      Run a backtesting simulation with the specified configuration

   .. method:: calculate_metrics(predictions: DataFrameType, actuals: DataFrameType) -> Dict[str, float]
      
      Calculate performance metrics comparing predictions to actuals

   .. method:: generate_report(results: Dict[str, Any]) -> Dict[str, Any]
      
      Generate a report summarizing backtesting results

.. class:: StorageProtocol(Protocol)

   Protocol defining the interface for storage components

   .. method:: store(data: Any, path: PathType) -> None
      
      Store data at the specified path

   .. method:: retrieve(path: PathType) -> Any
      
      Retrieve data from the specified path

   .. method:: delete(path: PathType) -> None
      
      Delete data at the specified path

   .. method:: list(path: PathType) -> List[PathType]
      
      List contents at the specified path

Date Utilities
-------------

The date_utils module provides functions for date and time operations, timezone conversions, and temporal utilities to ensure consistent datetime processing across the application.

Constants
~~~~~~~~~

.. data:: ERCOT_TIMEZONE
   :annotation: = pytz.timezone('US/Central')

   Standard timezone for ERCOT operations

.. data:: UTC_TIMEZONE
   :annotation: = pytz.UTC

   UTC timezone constant

.. data:: DEFAULT_DATETIME_FORMAT
   :annotation: = '%Y-%m-%d %H:%M:%S'

   Default format for datetime strings

.. data:: ERCOT_API_DATETIME_FORMAT
   :annotation: = '%Y-%m-%dT%H:%M:%S'

   Datetime format used by ERCOT API

.. data:: DAY_AHEAD_MARKET_CLOSURE_HOUR
   :annotation: = 10

   Hour when the day-ahead market closes

Timezone Functions
~~~~~~~~~~~~~~~~~

.. function:: localize_datetime(dt: datetime.datetime, is_dst: bool = False) -> datetime.datetime

   Localizes a naive datetime object to the ERCOT timezone

   :param dt: The datetime object to localize
   :param is_dst: Flag indicating whether Daylight Saving Time should be used
   :return: A timezone-aware datetime object

.. function:: convert_to_utc(dt: datetime.datetime) -> datetime.datetime

   Converts a datetime object to UTC timezone

   :param dt: The datetime object to convert
   :return: A datetime object in UTC timezone

.. function:: is_dst(dt: datetime.datetime) -> bool

   Checks if a given datetime is during Daylight Saving Time in the ERCOT timezone

   :param dt: The datetime object to check
   :return: True if the datetime is during DST, False otherwise

Datetime Formatting
~~~~~~~~~~~~~~~~~~

.. function:: format_datetime(dt: datetime.datetime, format_string: str = DEFAULT_DATETIME_FORMAT) -> str

   Formats a datetime object as a string using the specified format

   :param dt: The datetime object to format
   :param format_string: The format string to use
   :return: A formatted datetime string

.. function:: parse_datetime(datetime_str: str, format_string: str = DEFAULT_DATETIME_FORMAT, localize: bool = True) -> datetime.datetime

   Parses a datetime string into a datetime object

   :param datetime_str: The string to parse
   :param format_string: The format string to use
   :param localize: Whether to localize the result to ERCOT timezone
   :return: A datetime object

.. function:: get_current_time() -> datetime.datetime

   Returns the current time in the ERCOT timezone

   :return: The current datetime in ERCOT timezone

Date Range Functions
~~~~~~~~~~~~~~~~~~~

.. function:: create_date_range(start_date: datetime.datetime, end_date: datetime.datetime, freq: str = 'H') -> pandas.DatetimeIndex

   Creates a pandas DatetimeIndex with the specified frequency between start and end dates

   :param start_date: The start date of the range
   :param end_date: The end date of the range
   :param freq: Frequency string (e.g., 'H' for hourly, '5T' for 5-minute)
   :return: A DatetimeIndex object

.. function:: validate_date_range(start_date: datetime.datetime, end_date: datetime.datetime, max_days: Optional[int] = None) -> bool

   Validates that a date range is valid and within acceptable bounds

   :param start_date: The start date to validate
   :param end_date: The end date to validate
   :param max_days: Maximum allowed number of days in the range
   :return: True if the range is valid, False otherwise

Datetime Rounding
~~~~~~~~~~~~~~~~

.. function:: round_datetime(dt: datetime.datetime, freq: str = 'H') -> datetime.datetime

   Rounds a datetime to the nearest specified frequency

   :param dt: The datetime to round
   :param freq: Frequency string (e.g., 'H' for hourly, '5T' for 5-minute)
   :return: A rounded datetime object

.. function:: floor_datetime(dt: datetime.datetime, freq: str = 'H') -> datetime.datetime

   Floors a datetime to the specified frequency

   :param dt: The datetime to floor
   :param freq: Frequency string (e.g., 'H' for hourly, '5T' for 5-minute)
   :return: A floored datetime object

.. function:: ceil_datetime(dt: datetime.datetime, freq: str = 'H') -> datetime.datetime

   Ceils a datetime to the specified frequency

   :param dt: The datetime to ceil
   :param freq: Frequency string (e.g., 'H' for hourly, '5T' for 5-minute)
   :return: A ceiling datetime object

Market-Specific Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: get_day_ahead_market_closure(target_date: datetime.datetime) -> datetime.datetime

   Returns the day-ahead market closure datetime for a given date

   :param target_date: The target date to get the closure time for
   :return: A datetime representing the DAM closure time

.. function:: is_before_day_ahead_market_closure(dt: datetime.datetime, reference_date: Optional[datetime.datetime] = None) -> bool

   Checks if a given datetime is before the day-ahead market closure

   :param dt: The datetime to check
   :param reference_date: Optional reference date (defaults to today)
   :return: True if dt is before DAM closure, False otherwise

.. function:: get_forecast_horizon_end(reference_date: datetime.datetime, horizon_hours: int = 72) -> datetime.datetime

   Calculates the end datetime for a forecast horizon starting from a reference date

   :param reference_date: The reference date to start from
   :param horizon_hours: The number of hours in the forecast horizon
   :return: A datetime representing the end of the forecast horizon

Feature Engineering Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: get_datetime_components(dt: datetime.datetime) -> Dict[str, int]

   Extracts various components from a datetime object for feature engineering

   :param dt: The datetime to extract components from
   :return: A dictionary of datetime components (hour, day, month, etc.)

Statistics
---------

The statistics module provides statistical functions for time series data analysis, including rolling statistics, volatility metrics, quantile analysis, and spike frequency calculations.

Constants
~~~~~~~~~

.. data:: DEFAULT_ROLLING_WINDOWS
   :annotation: = [1, 6, 12, 24, 48, 72, 168]

   Default window sizes for rolling calculations

.. data:: DEFAULT_QUANTILES
   :annotation: = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

   Default quantile values for distribution analysis

.. data:: DEFAULT_STATISTICS
   :annotation: = ['mean', 'std', 'min', 'max', 'median']

   Default statistical measures to calculate

Rolling Statistics
~~~~~~~~~~~~~~~~

.. function:: calculate_rolling_statistics(series: SeriesType, windows: Optional[List[int]] = None, statistics: Optional[List[str]] = None) -> DataFrameType

   Calculate rolling statistical measures for a time series

   :param series: The time series to analyze
   :param windows: List of window sizes in periods
   :param statistics: List of statistics to calculate
   :return: DataFrame of rolling statistics

.. function:: calculate_price_volatility(series: SeriesType, windows: Optional[List[int]] = None) -> DataFrameType

   Calculate price volatility metrics for a time series

   :param series: The price series to analyze
   :param windows: List of window sizes in periods
   :return: DataFrame of volatility metrics

.. function:: calculate_spike_frequency(series: SeriesType, thresholds: List[float], windows: Optional[List[int]] = None) -> DataFrameType

   Calculate frequency of price spikes above specified thresholds

   :param series: The price series to analyze
   :param thresholds: List of price thresholds to check
   :param windows: List of window sizes in periods
   :return: DataFrame of spike frequencies

Distribution Analysis
~~~~~~~~~~~~~~~~~~~

.. function:: calculate_quantiles(series: SeriesType, quantiles: Optional[List[float]] = None, windows: Optional[List[int]] = None) -> DataFrameType

   Calculate quantile values for a series over rolling windows

   :param series: The time series to analyze
   :param quantiles: List of quantiles to calculate
   :param windows: List of window sizes in periods
   :return: DataFrame of quantile values

.. function:: calculate_distribution_metrics(series: SeriesType, windows: Optional[List[int]] = None) -> DataFrameType

   Calculate distribution metrics (skewness, kurtosis) for a series

   :param series: The time series to analyze
   :param windows: List of window sizes in periods
   :return: DataFrame of distribution metrics

Correlation Analysis
~~~~~~~~~~~~~~~~~~

.. function:: calculate_autocorrelation(series: SeriesType, lags: Optional[List[int]] = None) -> DataFrameType

   Calculate autocorrelation of a series at specified lags

   :param series: The time series to analyze
   :param lags: List of lag periods
   :return: DataFrame of autocorrelation values

.. function:: calculate_crosscorrelation(series1: SeriesType, series2: SeriesType, lags: Optional[List[int]] = None) -> DataFrameType

   Calculate cross-correlation between two series at specified lags

   :param series1: First time series
   :param series2: Second time series
   :param lags: List of lag periods
   :return: DataFrame of cross-correlation values

Data Aggregation
~~~~~~~~~~~~~~

.. function:: calculate_hourly_statistics(df: DataFrameType, value_column: str, timestamp_column: str) -> DataFrameType

   Aggregate 5-minute data to hourly statistics

   :param df: DataFrame with 5-minute data
   :param value_column: Name of the value column to aggregate
   :param timestamp_column: Name of the timestamp column
   :return: DataFrame with hourly statistics

Model Evaluation
~~~~~~~~~~~~~~

.. function:: calculate_binary_classification_metrics(y_true: ArrayType, y_pred: ArrayType) -> Dict[str, float]

   Calculate binary classification metrics from actual and predicted values

   :param y_true: True binary labels
   :param y_pred: Predicted binary labels
   :return: Dictionary of classification metrics

.. function:: calculate_probability_metrics(y_true: ArrayType, y_prob: ArrayType) -> Dict[str, float]

   Calculate metrics for probability predictions

   :param y_true: True binary labels
   :param y_prob: Predicted probabilities
   :return: Dictionary of probability metrics

.. function:: calculate_confidence_intervals(series: SeriesType, confidence_level: float = 0.95) -> Tuple[float, float]

   Calculate confidence intervals for a series of values

   :param series: Series of values
   :param confidence_level: Confidence level (0-1)
   :return: Tuple of (lower_bound, upper_bound)

Validation
---------

The validation module provides data validation utilities to ensure data quality and integrity throughout the system.

Constants
~~~~~~~~~

.. data:: DEFAULT_COMPLETENESS_THRESHOLD
   :annotation: = 0.95

   Default threshold for data completeness checks

.. data:: DEFAULT_OUTLIER_THRESHOLD
   :annotation: = 3.0

   Default threshold for outlier detection

Data Validation Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: validate_dataframe_schema(df: DataFrameType, schema: Any, strict: bool = True) -> DataFrameType

   Validates a pandas DataFrame against a schema definition

   :param df: DataFrame to validate
   :param schema: Schema to validate against
   :param strict: Whether to enforce strict validation
   :return: Validated DataFrame
   :raises: SchemaError if validation fails

.. function:: validate_data_completeness(df: DataFrameType, threshold: float = DEFAULT_COMPLETENESS_THRESHOLD, raise_error: bool = True) -> dict

   Checks the completeness of data and identifies missing values

   :param df: DataFrame to check
   :param threshold: Minimum required completeness ratio
   :param raise_error: Whether to raise an error on validation failure
   :return: Validation results dictionary

.. function:: validate_value_ranges(df: DataFrameType, value_ranges: dict, raise_error: bool = True) -> dict

   Validates that values in specified columns fall within expected ranges

   :param df: DataFrame to validate
   :param value_ranges: Dictionary mapping column names to (min, max) tuples
   :param raise_error: Whether to raise an error on validation failure
   :return: Validation results dictionary

.. function:: validate_temporal_consistency(df: DataFrameType, timestamp_column: str, expected_frequency: str, raise_error: bool = True) -> dict

   Validates the temporal consistency of time series data

   :param df: DataFrame to validate
   :param timestamp_column: Name of the timestamp column
   :param expected_frequency: Expected time frequency
   :param raise_error: Whether to raise an error on validation failure
   :return: Validation results dictionary

Column Validation
~~~~~~~~~~~~~~~

.. function:: validate_required_columns(df: DataFrameType, required_columns: list, raise_error: bool = True) -> bool

   Validates that a DataFrame contains all required columns

   :param df: DataFrame to validate
   :param required_columns: List of required column names
   :param raise_error: Whether to raise an error on validation failure
   :return: True if validation passes, False otherwise

.. function:: validate_unique_values(df: DataFrameType, columns: list, raise_error: bool = True) -> dict

   Validates that specified columns contain unique values

   :param df: DataFrame to validate
   :param columns: List of column names to check
   :param raise_error: Whether to raise an error on validation failure
   :return: Validation results dictionary

.. function:: validate_data_types(df: DataFrameType, expected_types: dict, raise_error: bool = True) -> dict

   Validates that columns have the expected data types

   :param df: DataFrame to validate
   :param expected_types: Dictionary mapping column names to expected types
   :param raise_error: Whether to raise an error on validation failure
   :return: Validation results dictionary

.. function:: validate_probability_values(df: DataFrameType, probability_columns: list, raise_error: bool = True) -> dict

   Validates that values in specified columns are valid probabilities (between 0 and 1)

   :param df: DataFrame to validate
   :param probability_columns: List of column names containing probabilities
   :param raise_error: Whether to raise an error on validation failure
   :return: Validation results dictionary

Validation Decorators
~~~~~~~~~~~~~~~~~~~

.. function:: validate_input_data(validation_rules: dict) -> Callable

   Decorator that validates function input data against specified criteria

   :param validation_rules: Dictionary of validation rules
   :return: Decorator function

.. function:: validate_output_data(validation_rules: dict) -> Callable

   Decorator that validates function output data against specified criteria

   :param validation_rules: Dictionary of validation rules
   :return: Decorator function

Advanced Validation
~~~~~~~~~~~~~~~~~

.. function:: detect_outliers(series: SeriesType, method: str = 'zscore', threshold: float = DEFAULT_OUTLIER_THRESHOLD) -> SeriesType

   Detects outliers in a DataFrame column using specified method

   :param series: Series to check for outliers
   :param method: Detection method ('zscore', 'iqr', 'mad')
   :param threshold: Threshold for outlier detection
   :return: Boolean Series indicating outliers

.. function:: validate_consistency(df: DataFrameType, consistency_rules: dict, raise_error: bool = True) -> dict

   Validates data consistency based on custom rules

   :param df: DataFrame to validate
   :param consistency_rules: Dictionary of consistency rules
   :param raise_error: Whether to raise an error on validation failure
   :return: Validation results dictionary

Validation Classes
~~~~~~~~~~~~~~~~

.. class:: DataValidator

   Class that provides methods for comprehensive data validation

   .. method:: __init__(strict_mode: bool = True)
      
      Initialize the validator
      
      :param strict_mode: Whether to operate in strict mode

   .. method:: add_validation_rule(data_type: str, rule_name: str, rule_function: Callable, rule_params: dict) -> None
      
      Add a validation rule
      
      :param data_type: Type of data this rule applies to
      :param rule_name: Unique name for the rule
      :param rule_function: Function implementing the rule
      :param rule_params: Parameters for the rule function

   .. method:: add_schema_validator(data_type: str, schema: Any) -> None
      
      Add a schema validator for a data type
      
      :param data_type: Type of data this schema applies to
      :param schema: Schema definition

   .. method:: validate(df: DataFrameType, data_type: str) -> dict
      
      Validate data against all rules for its type
      
      :param df: DataFrame to validate
      :param data_type: Type of data being validated
      :return: Validation results

   .. method:: validate_with_rules(df: DataFrameType, rule_names: list) -> dict
      
      Validate data against specific rules
      
      :param df: DataFrame to validate
      :param rule_names: List of rule names to apply
      :return: Validation results

   .. method:: set_strict_mode(strict_mode: bool) -> None
      
      Set the strict mode setting
      
      :param strict_mode: Whether to operate in strict mode

.. class:: ValidationResult

   Class that represents the result of a validation operation

   .. method:: __init__(is_valid: bool = True, errors: list = None, warnings: list = None, details: dict = None)
      
      Initialize the validation result
      
      :param is_valid: Whether validation passed
      :param errors: List of error messages
      :param warnings: List of warning messages
      :param details: Details about the validation

   .. method:: add_error(error_message: str, error_details: dict = None) -> None
      
      Add an error to the validation result
      
      :param error_message: Error message
      :param error_details: Additional error details

   .. method:: add_warning(warning_message: str, warning_details: dict = None) -> None
      
      Add a warning to the validation result
      
      :param warning_message: Warning message
      :param warning_details: Additional warning details

   .. method:: to_dict() -> dict
      
      Convert the validation result to a dictionary
      
      :return: Dictionary representation

   .. method:: merge(other: 'ValidationResult') -> 'ValidationResult'
      
      Merge with another validation result
      
      :param other: Another ValidationResult to merge with
      :return: Merged ValidationResult

Error Handling
-------------

The error_handling module provides standardized error classes, error handling decorators, retry mechanisms, and context managers to ensure consistent error management across all system components.

Constants
~~~~~~~~~

.. data:: DEFAULT_MAX_RETRIES
   :annotation: = 3

   Default maximum number of retry attempts

.. data:: DEFAULT_BACKOFF_FACTOR
   :annotation: = 2.0

   Default factor for exponential backoff calculation

.. data:: DEFAULT_JITTER_FACTOR
   :annotation: = 0.1

   Default jitter factor to add randomness to retry delays

Error Handling Decorators
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: retry_with_backoff(exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]], max_retries: int = DEFAULT_MAX_RETRIES, backoff_factor: float = DEFAULT_BACKOFF_FACTOR, jitter_factor: float = DEFAULT_JITTER_FACTOR, log_retries: bool = True) -> Callable

   Decorator that retries a function with exponential backoff on specified exceptions

   :param exceptions: Exception type(s) to retry on
   :param max_retries: Maximum number of retry attempts
   :param backoff_factor: Base factor for exponential backoff
   :param jitter_factor: Randomization factor for delays
   :param log_retries: Whether to log retry attempts
   :return: Decorator function

.. function:: handle_errors(exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]], default_return: Any = None, reraise: bool = False, error_message: Optional[str] = None) -> Callable

   Decorator that handles specified exceptions with custom error handling

   :param exceptions: Exception type(s) to handle
   :param default_return: Value to return on exception
   :param reraise: Whether to re-raise the exception after handling
   :param error_message: Custom error message
   :return: Decorator function

.. function:: circuit_breaker(failure_threshold: int = 3, reset_timeout: float = 60.0) -> Callable

   Implements the circuit breaker pattern to prevent repeated calls to failing services

   :param failure_threshold: Number of failures before opening circuit
   :param reset_timeout: Seconds before attempting to close circuit
   :return: Decorator function

Error Utility Functions
~~~~~~~~~~~~~~~~~~~~~

.. function:: format_exception(exc: Exception, include_traceback: bool = True) -> str

   Formats an exception with traceback information for logging

   :param exc: The exception to format
   :param include_traceback: Whether to include traceback information
   :return: Formatted exception string

.. function:: get_error_context(exc: Exception, additional_context: Dict[str, Any] = None) -> Dict[str, Any]

   Extracts contextual information from an exception for error reporting

   :param exc: The exception to extract context from
   :param additional_context: Additional context to include
   :return: Dictionary of error context

.. function:: is_retryable_error(exc: Exception) -> bool

   Determines if an error is retryable based on its type and attributes

   :param exc: The exception to check
   :return: True if the error is retryable, False otherwise

.. function:: log_error(exc: Exception, message: str = '', level: str = 'ERROR', context: Dict[str, Any] = None) -> None

   Logs an error with appropriate severity and context

   :param exc: The exception to log
   :param message: Additional message
   :param level: Log level
   :param context: Additional context
   :return: None

Exception Classes
~~~~~~~~~~~~~~~

.. class:: BaseError(Exception)

   Base exception class for all custom errors in the system

   .. method:: __init__(message: str, context: Dict[str, Any] = None, retryable: bool = False)
      
      Initialize the error
      
      :param message: Error message
      :param context: Error context
      :param retryable: Whether the error is retryable

   .. method:: to_dict() -> Dict[str, Any]
      
      Convert the error to a dictionary
      
      :return: Dictionary representation of the error

.. class:: DataError(BaseError)

   Base class for data-related errors

.. class:: DataFormatError(DataError)

   Error raised when data has an invalid format

.. class:: MissingDataError(DataError)

   Error raised when required data is missing

.. class:: ConnectionError(BaseError)

   Error raised when a connection to an external service fails

.. class:: RateLimitError(BaseError)

   Error raised when an API rate limit is exceeded

   .. method:: __init__(message: str, context: Dict[str, Any] = None, retry_after: Optional[float] = None)
      
      Initialize the rate limit error
      
      :param message: Error message
      :param context: Error context
      :param retry_after: Seconds to wait before retrying

.. class:: ModelError(BaseError)

   Base class for model-related errors

.. class:: ModelLoadError(ModelError)

   Error raised when a model cannot be loaded

.. class:: ModelTrainingError(ModelError)

   Error raised when model training fails

.. class:: InferenceError(BaseError)

   Error raised when inference fails

.. class:: CircuitOpenError(BaseError)

   Error raised when a circuit breaker is open

   .. method:: __init__(message: str, reset_time: float, context: Dict[str, Any] = None)
      
      Initialize the circuit open error
      
      :param message: Error message
      :param reset_time: Time when the circuit will reset
      :param context: Error context

Context Managers
~~~~~~~~~~~~~~

.. class:: ErrorHandler

   Context manager for handling errors with custom error handling

   .. method:: __init__(exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]], default_return: Any = None, reraise: bool = False, error_message: Optional[str] = None, handler: Optional[Callable[[Exception], Any]] = None)
      
      Initialize the error handler
      
      :param exceptions: Exception type(s) to handle
      :param default_return: Value to return on exception
      :param reraise: Whether to re-raise the exception after handling
      :param error_message: Custom error message
      :param handler: Custom handler function

   .. method:: __enter__() -> 'ErrorHandler'
      
      Enter the context
      
      :return: The error handler instance

   .. method:: __exit__(exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[traceback]) -> bool
      
      Exit the context and handle any exceptions
      
      :param exc_type: Exception type
      :param exc_val: Exception value
      :param exc_tb: Exception traceback
      :return: True if exception was handled, False otherwise

.. class:: RetryContext

   Context manager for retrying operations with exponential backoff

   .. method:: __init__(exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]], max_retries: int = DEFAULT_MAX_RETRIES, backoff_factor: float = DEFAULT_BACKOFF_FACTOR, jitter_factor: float = DEFAULT_JITTER_FACTOR, log_retries: bool = True)
      
      Initialize the retry context
      
      :param exceptions: Exception type(s) to retry on
      :param max_retries: Maximum number of retry attempts
      :param backoff_factor: Base factor for exponential backoff
      :param jitter_factor: Randomization factor for delays
      :param log_retries: Whether to log retry attempts

   .. method:: __enter__() -> 'RetryContext'
      
      Enter the context
      
      :return: The retry context instance

   .. method:: __exit__(exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[traceback]) -> bool
      
      Exit the context and handle any exceptions with retries
      
      :param exc_type: Exception type
      :param exc_val: Exception value
      :param exc_tb: Exception traceback
      :return: True if exception was handled, False otherwise

   .. method:: reset() -> None
      
      Reset the retry counter
      
      :return: None

   .. method:: get_retry_stats() -> Dict[str, Any]
      
      Get statistics about retry attempts
      
      :return: Dictionary of retry statistics

Logging
------

The logging module provides standardized logging configuration, formatters, context managers, and decorators to ensure consistent logging across all system components.

Constants
~~~~~~~~~

.. data:: DEFAULT_LOG_LEVEL
   :annotation: = logging.INFO

   Default logging level (INFO)

.. data:: DEFAULT_LOG_FORMAT
   :annotation: = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

   Default log message format string

.. data:: DEFAULT_DATE_FORMAT
   :annotation: = "%Y-%m-%d %H:%M:%S"

   Default date format for log timestamps

.. data:: DEFAULT_LOG_FILE
   :annotation: = "ercot_rtlmp_prediction.log"

   Default log file name

.. data:: LOG_LEVELS
   :annotation: = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}

   Mapping of level names to logging level constants

Logging Configuration
~~~~~~~~~~~~~~~~~~~

.. function:: setup_logging(config: Optional[Dict[str, Any]] = None, log_file: Optional[str] = None, log_level: Optional[str] = None, console_output: bool = True, file_output: bool = True, json_format: bool = False) -> logging.Logger

   Configures the logging system with appropriate handlers and formatters

   :param config: Optional configuration dictionary
   :param log_file: Path to log file
   :param log_level: Logging level
   :param console_output: Whether to output logs to console
   :param file_output: Whether to output logs to file
   :param json_format: Whether to use JSON format for logs
   :return: Configured root logger

.. function:: get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger

   Gets a logger for a specific module with consistent configuration

   :param name: Logger name (typically __name__)
   :param log_level: Optional logging level override
   :return: Configured logger

.. function:: configure_component_logger(component_name: str, log_level: Optional[str] = None, log_file: Optional[str] = None, propagate: bool = True) -> logging.Logger

   Configures a logger for a specific component with custom settings

   :param component_name: Name of the component
   :param log_level: Logging level for this component
   :param log_file: Optional separate log file for this component
   :param propagate: Whether to propagate logs to parent loggers
   :return: Configured component logger

Logging Decorators
~~~~~~~~~~~~~~~~

.. function:: log_execution_time(logger: Optional[logging.Logger] = None, level: str = 'INFO', message: str = 'Execution time: {duration:.4f} seconds') -> Callable

   Decorator for logging function execution time

   :param logger: Logger to use (defaults to module logger)
   :param level: Log level
   :param message: Message template
   :return: Decorator function

.. function:: log_function_call(logger: Optional[logging.Logger] = None, level: str = 'DEBUG', log_args: bool = True, log_result: bool = False) -> Callable

   Decorator for logging function calls with parameters and return values

   :param logger: Logger to use (defaults to module logger)
   :param level: Log level
   :param log_args: Whether to log function arguments
   :param log_result: Whether to log function result
   :return: Decorator function

Logging Utilities
~~~~~~~~~~~~~~~

.. function:: format_log_message(message: str, context: Optional[Dict[str, Any]] = None) -> str

   Formats log messages with consistent structure and context

   :param message: The log message
   :param context: Additional context
   :return: Formatted message

.. function:: sanitize_log_data(data: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]

   Sanitizes sensitive data from log messages

   :param data: Data to sanitize
   :param sensitive_keys: Keys to sanitize
   :return: Sanitized data

Logging Classes
~~~~~~~~~~~~~

.. class:: JsonFormatter(logging.Formatter)

   Custom formatter that outputs log records as JSON objects

   .. method:: __init__(include_timestamp: bool = True, include_level: bool = True, include_name: bool = True, include_path: bool = False, include_function: bool = False, include_process: bool = False)
      
      Initialize the formatter
      
      :param include_timestamp: Whether to include timestamp
      :param include_level: Whether to include log level
      :param include_name: Whether to include logger name
      :param include_path: Whether to include file path
      :param include_function: Whether to include function name
      :param include_process: Whether to include process ID

   .. method:: format(record: logging.LogRecord) -> str
      
      Format the log record as JSON
      
      :param record: Log record to format
      :return: JSON-formatted string

.. class:: ContextAdapter

   Adapter for adding contextual information to logs

   .. method:: __init__(logger: logging.Logger, context: Dict[str, Any])
      
      Initialize the adapter
      
      :param logger: Logger to adapt
      :param context: Context to add to logs

   .. method:: debug(msg: str, *args, **kwargs) -> None
      
      Log a debug message with context
      
      :param msg: Message to log
      :param args: Positional arguments
      :param kwargs: Keyword arguments

   .. method:: info(msg: str, *args, **kwargs) -> None
      
      Log an info message with context
      
      :param msg: Message to log
      :param args: Positional arguments
      :param kwargs: Keyword arguments

   .. method:: warning(msg: str, *args, **kwargs) -> None
      
      Log a warning message with context
      
      :param msg: Message to log
      :param args: Positional arguments
      :param kwargs: Keyword arguments

   .. method:: error(msg: str, *args, **kwargs) -> None
      
      Log an error message with context
      
      :param msg: Message to log
      :param args: Positional arguments
      :param kwargs: Keyword arguments

   .. method:: critical(msg: str, *args, **kwargs) -> None
      
      Log a critical message with context
      
      :param msg: Message to log
      :param args: Positional arguments
      :param kwargs: Keyword arguments

   .. method:: exception(msg: str, *args, **kwargs) -> None
      
      Log an exception message with context
      
      :param msg: Message to log
      :param args: Positional arguments
      :param kwargs: Keyword arguments

   .. method:: log(level: int, msg: str, *args, **kwargs) -> None
      
      Log a message with the specified level and context
      
      :param level: Log level
      :param msg: Message to log
      :param args: Positional arguments
      :param kwargs: Keyword arguments

.. class:: LoggingContext

   Context manager for temporarily adding context to logs

   .. method:: __init__(logger: logging.Logger, context: Dict[str, Any])
      
      Initialize the context
      
      :param logger: Logger to adapt
      :param context: Context to add to logs

   .. method:: __enter__() -> ContextAdapter
      
      Enter the context
      
      :return: Context adapter

   .. method:: __exit__(exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[traceback]) -> bool
      
      Exit the context
      
      :param exc_type: Exception type
      :param exc_val: Exception value
      :param exc_tb: Exception traceback
      :return: Always False to propagate exceptions

.. class:: PerformanceLogger

   Utility for logging performance metrics and execution times

   .. method:: __init__(logger: Optional[logging.Logger] = None)
      
      Initialize the performance logger
      
      :param logger: Logger to use (defaults to module logger)

   .. method:: start_timer(operation: str, category: Optional[str] = None) -> None
      
      Start timing an operation
      
      :param operation: Name of the operation
      :param category: Optional category
      :return: None

   .. method:: stop_timer(operation: str, category: Optional[str] = None, log_result: bool = True) -> float
      
      Stop timing an operation and return duration
      
      :param operation: Name of the operation
      :param category: Optional category
      :param log_result: Whether to log the result
      :return: Duration in seconds

   .. method:: log_metric(metric_name: str, value: float, category: Optional[str] = None) -> None
      
      Log a performance metric
      
      :param metric_name: Name of the metric
      :param value: Metric value
      :param category: Optional category
      :return: None

   .. method:: get_metrics(category: Optional[str] = None) -> Dict[str, Dict[str, List[float]]]
      
      Get recorded metrics
      
      :param category: Optional category to filter by
      :return: Dictionary of metrics

   .. method:: get_average_metrics(category: Optional[str] = None) -> Dict[str, Dict[str, float]]
      
      Get average values of recorded metrics
      
      :param category: Optional category to filter by
      :return: Dictionary of average metrics

   .. method:: reset_metrics(category: Optional[str] = None) -> None
      
      Reset recorded metrics
      
      :param category: Optional category to reset
      :return: None

   .. method:: time_operation(operation: str, category: Optional[str] = None, log_result: bool = True) -> contextlib._GeneratorContextManager
      
      Context manager for timing an operation
      
      :param operation: Name of the operation
      :param category: Optional category
      :param log_result: Whether to log the result
      :return: Context manager

Usage Examples
------------

Type Definitions Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.backend.utils import DataFrameType, SeriesType, ModelProtocol

   # Define a function with type annotations
   def process_data(df: DataFrameType, target_column: str) -> SeriesType:
       return df[target_column]

   # Use a protocol to define expected interface
   class MyModel(ModelProtocol):
       def train(self, X, y):
           # Implementation
           pass
       
       def predict(self, X):
           # Implementation
           pass
       
       # Implement other required methods

Date Utilities Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.backend.utils import localize_datetime, get_day_ahead_market_closure, is_before_day_ahead_market_closure
   from datetime import datetime

   # Get current time in ERCOT timezone
   current_time = get_current_time()

   # Check if current time is before DAM closure
   if is_before_day_ahead_market_closure(current_time):
       print("Still time to submit bids for tomorrow")
   else:
       print("Day-ahead market is closed for tomorrow")

   # Get DAM closure time for a specific date
   target_date = datetime(2023, 7, 15)
   dam_closure = get_day_ahead_market_closure(target_date)
   print(f"DAM closes at {dam_closure}")

Statistics Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from src.backend.utils import calculate_rolling_statistics, calculate_spike_frequency

   # Create a sample price series
   prices = pd.Series([50, 55, 60, 58, 120, 200, 70, 65, 60, 55], 
                     index=pd.date_range('2023-07-01', periods=10, freq='H'))

   # Calculate rolling statistics
   stats_df = calculate_rolling_statistics(prices, windows=[2, 3], 
                                         statistics=['mean', 'max'])

   # Calculate spike frequency above thresholds
   spike_df = calculate_spike_frequency(prices, thresholds=[100, 150], 
                                      windows=[3, 5])

   print(stats_df)
   print(spike_df)

Validation Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from src.backend.utils import validate_dataframe_schema, validate_data_completeness, validate_required_columns

   # Create a sample DataFrame
   df = pd.DataFrame({
       'timestamp': pd.date_range('2023-07-01', periods=5, freq='H'),
       'price': [50, 55, None, 58, 60],
       'node_id': ['HB_NORTH', 'HB_NORTH', 'HB_NORTH', 'HB_SOUTH', 'HB_SOUTH']
   })

   # Validate required columns
   valid_columns = validate_required_columns(df, 
                                           ['timestamp', 'price', 'node_id'], 
                                           raise_error=False)

   # Check data completeness
   completeness = validate_data_completeness(df, threshold=0.8, raise_error=False)

   print(f"Valid columns: {valid_columns}")
   print(f"Completeness: {completeness}")

Error Handling Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.backend.utils import retry_with_backoff, handle_errors, ErrorHandler

   # Use retry decorator for functions that might fail transiently
   @retry_with_backoff(exceptions=(ConnectionError, TimeoutError), max_retries=3)
   def fetch_external_data(url):
       # Implementation that might raise connection errors
       pass

   # Use handle_errors decorator for graceful error handling
   @handle_errors(exceptions=ValueError, default_return=None)
   def parse_data(data):
       # Implementation that might raise ValueError
       pass

   # Use context manager for error handling
   def process_data():
       with ErrorHandler(exceptions=Exception, error_message="Data processing failed"):
           # Code that might raise exceptions
           pass

Logging Example
~~~~~~~~~~~~~

.. code-block:: python

   from src.backend.utils import get_logger, log_execution_time, LoggingContext

   # Get a logger for the current module
   logger = get_logger(__name__)

   # Use execution time decorator
   @log_execution_time(logger, level='INFO')
   def process_large_dataset(data):
       # Implementation
       pass

   # Use logging context for adding context to logs
   def process_node_data(node_id, date):
       context = {'node_id': node_id, 'date': str(date)}
       with LoggingContext(logger, context) as log:
           log.info("Starting data processing")
           # Implementation
           log.info("Processing complete")