"""
Entry point for the storage module in the ERCOT RTLMP spike prediction system.

This module exposes the key classes and functions for data persistence, including Parquet 
storage for time series data, model registry for trained models, and forecast repository 
for prediction results.
"""

# Version information
__version__ = '0.1.0'

# Import and re-export from parquet_store.py
from .parquet_store import (
    ParquetStore,
    write_dataframe_to_parquet,
    read_dataframe_from_parquet,
    find_parquet_files,
    read_dataframes_from_partitions,
    get_partition_path,
    DEFAULT_STORAGE_ROOT,
    DEFAULT_COMPRESSION,
    DEFAULT_ROW_GROUP_SIZE,
)

# Import and re-export from model_registry.py
from .model_registry import (
    ModelRegistry,
    save_model,
    load_model,
    save_metadata,
    load_metadata,
    list_models,
    get_latest_model_version,
    increment_version,
)

# Import and re-export from forecast_repository.py
from .forecast_repository import (
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
    # ParquetStore and utilities
    'ParquetStore',
    'write_dataframe_to_parquet',
    'read_dataframe_from_parquet',
    'find_parquet_files',
    'read_dataframes_from_partitions',
    'get_partition_path',
    'DEFAULT_STORAGE_ROOT',
    'DEFAULT_COMPRESSION',
    'DEFAULT_ROW_GROUP_SIZE',
    
    # ModelRegistry and utilities
    'ModelRegistry',
    'save_model',
    'load_model',
    'save_metadata',
    'load_metadata',
    'list_models',
    'get_latest_model_version',
    'increment_version',
    
    # ForecastRepository and utilities
    'ForecastRepository',
    'get_forecast_partition_path',
    'generate_forecast_file_name',
    'save_forecast_metadata',
    'load_forecast_metadata',
    'find_latest_forecast_directory',
    'DEFAULT_FORECAST_ROOT',
]