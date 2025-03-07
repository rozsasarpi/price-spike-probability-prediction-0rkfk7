"""
Entry point for the data module of the ERCOT RTLMP spike prediction system.
Exposes key components and functionality from the data submodules including fetchers, validators, and storage.
"""

# Version information
__version__ = '0.1.0'

# Import and re-export from parquet_store.py
from .fetchers import (
    BaseDataFetcher,
    ERCOTDataFetcher,
    WeatherDataFetcher,
    GridConditionsFetcher,
    MockDataFetcher,
    generate_cache_key,
    DEFAULT_NODES,
    DEFAULT_LOCATIONS
)

# Import and re-export from model_registry.py
from .validators import (
    RTLMP_SCHEMA, WEATHER_SCHEMA, GRID_CONDITION_SCHEMA, FEATURE_SCHEMA, FORECAST_SCHEMA,
    validate_json_schema, validate_rtlmp_json, validate_weather_json, validate_grid_condition_json,
    validate_feature_json, validate_forecast_json, JSONSchemaValidator, ValidationError,
    convert_to_json_schema, datetime_to_str, str_to_datetime,
    RTLMPSchema, WeatherSchema, GridConditionSchema, FeatureSchema, ForecastSchema,
    SchemaRegistry, validate_with_rtlmp_schema, validate_with_weather_schema,
    validate_with_grid_condition_schema, validate_with_feature_schema, validate_with_forecast_schema,
    get_schema_for_data_type, RTLMP_VALUE_RANGES, WEATHER_VALUE_RANGES, GRID_VALUE_RANGES
)

# Import and re-export from forecast_repository.py
from .storage import (
    ParquetStore,
    write_dataframe_to_parquet,
    read_dataframe_from_parquet,
    find_parquet_files,
    read_dataframes_from_partitions,
    get_partition_path,
    DEFAULT_STORAGE_ROOT,
    DEFAULT_COMPRESSION,
    DEFAULT_ROW_GROUP_SIZE,
    ModelRegistry,
    save_model,
    load_model,
    save_metadata,
    load_metadata,
    list_models,
    get_latest_model_version,
    increment_version,
    ForecastRepository,
    get_forecast_partition_path,
    generate_forecast_file_name,
    save_forecast_metadata,
    load_forecast_metadata,
    find_latest_forecast_directory,
    DEFAULT_FORECAST_ROOT,
)

# Define public API
__all__ = [
    # Data Fetchers
    'BaseDataFetcher',
    'ERCOTDataFetcher',
    'WeatherDataFetcher',
    'GridConditionsFetcher',
    'MockDataFetcher',
    'generate_cache_key',
    'DEFAULT_NODES',
    'DEFAULT_LOCATIONS',

    # Validators
    'RTLMP_SCHEMA',
    'WEATHER_SCHEMA',
    'GRID_CONDITION_SCHEMA',
    'FEATURE_SCHEMA',
    'FORECAST_SCHEMA',
    'validate_json_schema',
    'validate_rtlmp_json',
    'validate_weather_json',
    'validate_grid_condition_json',
    'validate_feature_json',
    'validate_forecast_json',
    'JSONSchemaValidator',
    'ValidationError',
    'convert_to_json_schema',
    'datetime_to_str',
    'str_to_datetime',
    'RTLMPSchema',
    'WeatherSchema',
    'GridConditionSchema',
    'FeatureSchema',
    'ForecastSchema',
    'SchemaRegistry',
    'validate_with_rtlmp_schema',
    'validate_with_weather_schema',
    'validate_with_grid_condition_schema',
    'validate_with_feature_schema',
    'validate_with_forecast_schema',
    'get_schema_for_data_type',
    'RTLMP_VALUE_RANGES',
    'WEATHER_VALUE_RANGES',
    'GRID_VALUE_RANGES',

    # Storage
    'ParquetStore',
    'write_dataframe_to_parquet',
    'read_dataframe_from_parquet',
    'find_parquet_files',
    'read_dataframes_from_partitions',
    'get_partition_path',
    'DEFAULT_STORAGE_ROOT',
    'DEFAULT_COMPRESSION',
    'DEFAULT_ROW_GROUP_SIZE',
    'ModelRegistry',
    'save_model',
    'load_model',
    'save_metadata',
    'load_metadata',
    'list_models',
    'get_latest_model_version',
    'increment_version',
    'ForecastRepository',
    'get_forecast_partition_path',
    'generate_forecast_file_name',
    'save_forecast_metadata',
    'load_forecast_metadata',
    'find_latest_forecast_directory',
    'DEFAULT_FORECAST_ROOT',
]