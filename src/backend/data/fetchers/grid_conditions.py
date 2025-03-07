"""
Implementation of a specialized data fetcher for retrieving ERCOT grid condition data including load, 
generation mix, and capacity metrics. This module extends the base data fetcher functionality with 
grid-specific data processing and validation.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, cast

from .base import BaseDataFetcher
from .ercot_api import ERCOTDataFetcher
from ...utils.type_definitions import DataFrameType, GridConditionDict
from ...utils.logging import get_logger, log_execution_time
from ...utils.error_handling import retry_with_backoff, handle_errors, ConnectionError, DataFormatError, MissingDataError
from ..validators.schemas import validate_grid_condition_data
from ...utils.date_utils import ERCOT_TIMEZONE, localize_datetime

# Set up logger
logger = get_logger(__name__)

# Constants
GRID_DATA_FREQUENCY = '1H'
DEFAULT_FORECAST_HORIZON = 72
MAX_HISTORICAL_DAYS = 365
GENERATION_TYPES = ['wind', 'solar', 'thermal', 'hydro', 'nuclear', 'other']


@handle_errors(exceptions=Exception, default_return=0.0)
def calculate_reserve_margin(total_load: float, available_capacity: float) -> float:
    """
    Calculates the reserve margin from total load and available capacity.
    
    Args:
        total_load: Total load in MW
        available_capacity: Available generation capacity in MW
        
    Returns:
        Reserve margin as a percentage
    """
    # Check if inputs are valid numbers
    if not isinstance(total_load, (int, float)) or not isinstance(available_capacity, (int, float)):
        logger.warning(f"Invalid inputs for reserve margin calculation: total_load={total_load}, available_capacity={available_capacity}")
        return 0.0
    
    # If total_load is zero, return a large value to avoid division by zero
    if total_load == 0:
        logger.warning("Total load is zero, returning maximum reserve margin")
        return 100.0
    
    # Calculate reserve margin as (available_capacity - total_load) / total_load * 100
    reserve_margin = (available_capacity - total_load) / total_load * 100
    
    return reserve_margin


@handle_errors(exceptions=Exception, error_message='Error enriching grid data')
def enrich_grid_data(grid_data: DataFrameType) -> DataFrameType:
    """
    Enriches grid condition data with additional calculated metrics.
    
    Args:
        grid_data: DataFrame containing grid condition data
        
    Returns:
        Enriched grid condition data
    """
    # Check if the DataFrame is empty or missing required columns
    if grid_data.empty:
        logger.warning("Empty DataFrame provided for enrichment")
        return grid_data
    
    required_columns = ['timestamp', 'total_load', 'available_capacity']
    missing_columns = [col for col in required_columns if col not in grid_data.columns]
    if missing_columns:
        logger.warning(f"Missing required columns for grid data enrichment: {missing_columns}")
        return grid_data
    
    # Create a copy of the DataFrame to avoid modifying the original
    enriched_data = grid_data.copy()
    
    # Calculate reserve margin if not already present
    if 'reserve_margin' not in enriched_data.columns:
        logger.debug("Calculating reserve margin")
        enriched_data['reserve_margin'] = enriched_data.apply(
            lambda row: calculate_reserve_margin(row['total_load'], row['available_capacity']),
            axis=1
        )
    
    # Calculate generation mix percentages
    generation_columns = [col for col in enriched_data.columns if col.endswith('_generation')]
    if len(generation_columns) > 0:
        logger.debug("Calculating generation mix percentages")
        total_generation = enriched_data[generation_columns].sum(axis=1)
        
        for col in generation_columns:
            generation_type = col.replace('_generation', '')
            mix_column = f'generation_mix_{generation_type}'
            enriched_data[mix_column] = enriched_data[col] / total_generation * 100
            enriched_data[mix_column] = enriched_data[mix_column].fillna(0)
    
    # Calculate load factor (ratio of average to peak load)
    if 'total_load' in enriched_data.columns:
        logger.debug("Calculating load factor")
        # Group by date to calculate daily peak
        enriched_data['date'] = pd.to_datetime(enriched_data['timestamp']).dt.date
        daily_peak = enriched_data.groupby('date')['total_load'].transform('max')
        enriched_data['load_factor'] = enriched_data['total_load'] / daily_peak * 100
        enriched_data['load_factor'] = enriched_data['load_factor'].fillna(0)
        enriched_data = enriched_data.drop('date', axis=1)
    
    # Calculate renewable penetration (wind + solar as percentage of total generation)
    if 'wind_generation' in enriched_data.columns and 'solar_generation' in enriched_data.columns:
        logger.debug("Calculating renewable penetration")
        renewable_generation = enriched_data['wind_generation'] + enriched_data['solar_generation']
        total_generation = enriched_data[[col for col in enriched_data.columns if col.endswith('_generation')]].sum(axis=1)
        enriched_data['renewable_penetration'] = renewable_generation / total_generation * 100
        enriched_data['renewable_penetration'] = enriched_data['renewable_penetration'].fillna(0)
    
    return enriched_data


@handle_errors(exceptions=Exception, error_message='Error resampling grid data')
def resample_grid_data(grid_data: DataFrameType, frequency: str) -> DataFrameType:
    """
    Resamples grid condition data to the specified frequency.
    
    Args:
        grid_data: DataFrame containing grid condition data
        frequency: Frequency to resample to (e.g., '1H', '15min')
        
    Returns:
        Resampled grid condition data
    """
    # Check if the DataFrame is empty or missing timestamp index
    if grid_data.empty:
        logger.warning("Empty DataFrame provided for resampling")
        return grid_data
    
    if 'timestamp' not in grid_data.columns:
        logger.warning("Missing timestamp column for resampling")
        return grid_data
    
    # Ensure timestamp column is set as index and is datetime type
    if not pd.api.types.is_datetime64_any_dtype(grid_data['timestamp']):
        logger.debug("Converting timestamp column to datetime")
        grid_data = grid_data.copy()
        grid_data['timestamp'] = pd.to_datetime(grid_data['timestamp'])
    
    # Set timestamp as index if it's not already
    if grid_data.index.name != 'timestamp':
        logger.debug("Setting timestamp as index for resampling")
        resampled_data = grid_data.set_index('timestamp').copy()
    else:
        resampled_data = grid_data.copy()
    
    # Identify column types for appropriate resampling methods
    load_columns = ['total_load'] + [col for col in resampled_data.columns if col.endswith('_generation')]
    capacity_columns = ['available_capacity']
    derived_columns = ['reserve_margin'] + [col for col in resampled_data.columns if col.startswith('generation_mix_') or col in ['load_factor', 'renewable_penetration']]
    
    # Create dictionary of resampling methods for different column types
    resampling_dict = {}
    for col in load_columns:
        if col in resampled_data.columns:
            resampling_dict[col] = 'mean'
    
    for col in capacity_columns:
        if col in resampled_data.columns:
            resampling_dict[col] = 'min'  # Conservative approach for capacity
    
    # Perform resampling with the specified frequency
    logger.debug(f"Resampling grid data to {frequency} frequency")
    try:
        resampled = resampled_data.resample(frequency).agg(resampling_dict)
        
        # Reset index to get timestamp back as a column
        resampled = resampled.reset_index()
        
        # Recalculate derived metrics on the resampled data
        if set(derived_columns).intersection(set(resampled_data.columns)):
            logger.debug("Recalculating derived metrics after resampling")
            resampled = enrich_grid_data(resampled)
        
        return resampled
    except Exception as e:
        logger.error(f"Error during grid data resampling: {str(e)}")
        raise


class GridConditionsFetcher(BaseDataFetcher):
    """
    Specialized data fetcher for retrieving and processing ERCOT grid condition data.
    """
    
    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 3,
        cache_ttl: int = 3600,
        cache_dir: str = None,
        data_frequency: str = GRID_DATA_FREQUENCY
    ):
        """
        Initialize the grid conditions fetcher with configuration options.
        
        Args:
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            cache_ttl: Time-to-live for cached data in seconds
            cache_dir: Directory to store cached data files
            data_frequency: Frequency of grid data (default: GRID_DATA_FREQUENCY)
        """
        # Call the parent BaseDataFetcher constructor with timeout, max_retries, cache_ttl, and cache_dir
        super().__init__(timeout, max_retries, cache_ttl, cache_dir)
        
        # Initialize the ERCOTDataFetcher instance for accessing ERCOT API
        self._ercot_fetcher = ERCOTDataFetcher(timeout, max_retries, cache_ttl, cache_dir)
        
        # Set data_frequency (default to GRID_DATA_FREQUENCY if not provided)
        self.data_frequency = data_frequency
        
        logger.info(f"Initialized grid conditions fetcher with data_frequency={data_frequency}")
    
    @log_execution_time(logger, 'INFO')
    def fetch_data(self, params: Dict[str, Any]) -> DataFrameType:
        """
        Generic method to fetch grid condition data based on parameters.
        
        Args:
            params: Dictionary of parameters for the data fetch
            
        Returns:
            DataFrame with grid condition data
        """
        # Check if data is available in cache using _get_from_cache(params)
        cached_data = self._get_from_cache(params)
        if cached_data is not None:
            logger.info("Returning cached grid condition data")
            return cached_data
        
        # Extract start_date and end_date from params
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        
        # Extract forecast_date and horizon if this is a forecast request
        forecast_date = params.get('forecast_date')
        horizon = params.get('horizon', DEFAULT_FORECAST_HORIZON)
        
        # Extract frequency from params (default to self.data_frequency)
        frequency = params.get('frequency', self.data_frequency)
        
        # Extract identifiers (node IDs) if provided
        identifiers = params.get('identifiers', [])
        
        # Call fetch_historical_data or fetch_forecast_data based on params
        if forecast_date:
            result = self.fetch_forecast_data(forecast_date, horizon, identifiers)
        elif start_date and end_date:
            result = self.fetch_historical_data(start_date, end_date, identifiers)
        else:
            raise ValueError("Invalid parameters: must provide either forecast_date or start_date and end_date")
        
        # Validate the fetched data using validate_data()
        self.validate_data(result)
        
        # Store the result in cache using _store_in_cache(params, result)
        self._store_in_cache(params, result)
        
        return result
    
    @log_execution_time(logger, 'INFO')
    def fetch_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        identifiers: List[str]
    ) -> DataFrameType:
        """
        Fetch historical grid condition data for a specific date range.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            identifiers: List of identifiers (optional, ignored for grid conditions)
            
        Returns:
            DataFrame with historical grid condition data
        """
        # Validate the date range using _validate_date_range(start_date, end_date, MAX_HISTORICAL_DAYS)
        if not self._validate_date_range(start_date, end_date, MAX_HISTORICAL_DAYS):
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")
        
        # Localize start_date and end_date to ERCOT_TIMEZONE if not already localized
        start_date = localize_datetime(start_date) if start_date.tzinfo is None else start_date
        end_date = localize_datetime(end_date) if end_date.tzinfo is None else end_date
        
        # If date range exceeds MAX_HISTORICAL_DAYS, split into smaller chunks
        days_diff = (end_date - start_date).days
        if days_diff > MAX_HISTORICAL_DAYS:
            logger.info(f"Date range exceeds MAX_HISTORICAL_DAYS ({MAX_HISTORICAL_DAYS}), splitting into smaller chunks")
            
            result_dfs = []
            current_start = start_date
            
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=MAX_HISTORICAL_DAYS), end_date)
                
                # Fetch grid condition data for the chunk
                chunk_data = self._ercot_fetcher.fetch_grid_conditions(current_start, current_end)
                
                # Append to result list
                result_dfs.append(chunk_data)
                
                # Update current_start for next chunk
                current_start = current_end
            
            # Concatenate all chunks
            result = pd.concat(result_dfs, ignore_index=True)
        else:
            # Fetch grid condition data directly using ERCOT fetcher
            result = self._ercot_fetcher.fetch_grid_conditions(start_date, end_date)
        
        # Enrich the data with additional metrics using enrich_grid_data()
        result = enrich_grid_data(result)
        
        # Resample the data to the requested frequency if different from raw data
        if self.data_frequency != GRID_DATA_FREQUENCY:
            result = resample_grid_data(result, self.data_frequency)
        
        # Validate the final data using validate_data()
        self.validate_data(result)
        
        return result
    
    @log_execution_time(logger, 'INFO')
    def fetch_forecast_data(
        self,
        forecast_date: datetime,
        horizon: int = DEFAULT_FORECAST_HORIZON,
        identifiers: List[str] = None
    ) -> DataFrameType:
        """
        Fetch forecast grid condition data for a specific date and horizon.
        
        Args:
            forecast_date: Date for which to retrieve forecasts
            horizon: Forecast horizon in hours (default: DEFAULT_FORECAST_HORIZON)
            identifiers: List of identifiers (optional, ignored for grid conditions)
            
        Returns:
            DataFrame with forecast grid condition data
        """
        # Set horizon to DEFAULT_FORECAST_HORIZON if not provided
        if horizon is None:
            horizon = DEFAULT_FORECAST_HORIZON
        
        # Calculate end_date as forecast_date + horizon hours
        end_date = forecast_date + timedelta(hours=horizon)
        
        # Localize forecast_date and end_date to ERCOT_TIMEZONE if not already localized
        forecast_date = localize_datetime(forecast_date) if forecast_date.tzinfo is None else forecast_date
        end_date = localize_datetime(end_date) if end_date.tzinfo is None else end_date
        
        logger.info(f"Fetching grid condition forecast from {forecast_date} to {end_date} (horizon: {horizon} hours)")
        
        # Fetch grid condition forecast using the ERCOT fetcher
        result = self._ercot_fetcher.fetch_grid_conditions(forecast_date, end_date)
        
        # Enrich the data with additional metrics using enrich_grid_data()
        result = enrich_grid_data(result)
        
        # Resample the data to the requested frequency if different from raw data
        if self.data_frequency != GRID_DATA_FREQUENCY:
            result = resample_grid_data(result, self.data_frequency)
        
        # Validate the data using validate_data()
        self.validate_data(result)
        
        return result
    
    @retry_with_backoff(exceptions=(ConnectionError, DataFormatError), max_retries=3)
    def get_generation_mix(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> DataFrameType:
        """
        Extract generation mix data from grid conditions.
        
        Args:
            start_date: Start date for the data
            end_date: End date for the data
            
        Returns:
            DataFrame with generation mix data
        """
        # Fetch grid condition data for the specified date range
        grid_data = self.fetch_historical_data(start_date, end_date, [])
        
        # Extract generation by type columns
        generation_columns = [col for col in grid_data.columns if col.startswith('generation_mix_')]
        
        if not generation_columns:
            logger.warning("No generation mix columns found in grid data")
            # If no generation mix columns exist, we might need to calculate them
            grid_data = enrich_grid_data(grid_data)
            generation_columns = [col for col in grid_data.columns if col.startswith('generation_mix_')]
            
            if not generation_columns:
                raise DataFormatError("Unable to extract generation mix data")
        
        # Select timestamp and generation mix columns
        mix_data = grid_data[['timestamp'] + generation_columns].copy()
        
        # Ensure timestamp is a datetime column
        mix_data['timestamp'] = pd.to_datetime(mix_data['timestamp'])
        
        return mix_data
    
    @retry_with_backoff(exceptions=(ConnectionError, DataFormatError), max_retries=3)
    def get_load_forecast(
        self,
        forecast_date: datetime,
        horizon: int = DEFAULT_FORECAST_HORIZON
    ) -> DataFrameType:
        """
        Get load forecast data for the specified horizon.
        
        Args:
            forecast_date: Date for which to retrieve forecast
            horizon: Forecast horizon in hours (default: DEFAULT_FORECAST_HORIZON)
            
        Returns:
            DataFrame with load forecast data
        """
        # Fetch forecast grid condition data for the specified date and horizon
        forecast_data = self.fetch_forecast_data(forecast_date, horizon, [])
        
        # Extract the total_load column
        if 'total_load' not in forecast_data.columns:
            raise DataFormatError("Total load data not found in forecast")
        
        # Select timestamp and total_load columns
        load_forecast = forecast_data[['timestamp', 'total_load']].copy()
        
        # Ensure timestamp is a datetime column
        load_forecast['timestamp'] = pd.to_datetime(load_forecast['timestamp'])
        
        return load_forecast
    
    @retry_with_backoff(exceptions=(ConnectionError, DataFormatError), max_retries=3)
    def get_renewable_forecast(
        self,
        forecast_date: datetime,
        horizon: int = DEFAULT_FORECAST_HORIZON
    ) -> DataFrameType:
        """
        Get renewable generation forecast data for the specified horizon.
        
        Args:
            forecast_date: Date for which to retrieve forecast
            horizon: Forecast horizon in hours (default: DEFAULT_FORECAST_HORIZON)
            
        Returns:
            DataFrame with renewable generation forecast data
        """
        # Fetch forecast grid condition data for the specified date and horizon
        forecast_data = self.fetch_forecast_data(forecast_date, horizon, [])
        
        # Check if wind_generation and solar_generation columns exist
        if 'wind_generation' not in forecast_data.columns or 'solar_generation' not in forecast_data.columns:
            raise DataFormatError("Renewable generation data not found in forecast")
        
        # Select timestamp, wind_generation, and solar_generation columns
        renewable_forecast = forecast_data[['timestamp', 'wind_generation', 'solar_generation']].copy()
        
        # Calculate total renewable generation
        renewable_forecast['total_renewable'] = renewable_forecast['wind_generation'] + renewable_forecast['solar_generation']
        
        # Ensure timestamp is a datetime column
        renewable_forecast['timestamp'] = pd.to_datetime(renewable_forecast['timestamp'])
        
        return renewable_forecast
    
    def validate_data(self, data: DataFrameType) -> bool:
        """
        Validate the structure and content of grid condition data.
        
        Args:
            data: DataFrame with grid condition data
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if the DataFrame is empty
        if data.empty:
            logger.warning("Empty DataFrame, validation failed")
            return False
        
        # Verify that all required columns are present
        required_columns = ['timestamp', 'total_load', 'available_capacity']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Validate data types of key columns
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            logger.warning("timestamp column is not datetime type")
            return False
        
        numeric_columns = ['total_load', 'available_capacity']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                logger.warning(f"{col} column is not numeric type")
                return False
        
        # Check for missing values in critical columns
        for col in numeric_columns:
            if col in data.columns and data[col].isna().any():
                missing_count = data[col].isna().sum()
                logger.warning(f"{missing_count} missing values in {col} column")
                # We don't fail validation for this, just warn
        
        # Validate value ranges (e.g., generation values should be non-negative)
        for col in numeric_columns:
            if col in data.columns and (data[col] < 0).any():
                negative_count = (data[col] < 0).sum()
                logger.warning(f"{negative_count} negative values in {col} column")
                # We don't fail validation for this, just warn
        
        # Use validate_grid_condition_data() for schema validation
        try:
            schema_valid = validate_grid_condition_data(data)
            if not schema_valid:
                logger.warning("Schema validation failed")
                return False
        except Exception as e:
            logger.error(f"Error during schema validation: {str(e)}")
            return False
        
        # Log validation results
        logger.debug("Grid condition data validation successful")
        return True