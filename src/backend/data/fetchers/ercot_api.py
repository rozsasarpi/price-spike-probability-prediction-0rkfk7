"""
Implementation of the data fetcher interface for retrieving ERCOT market data including RTLMP values,
grid conditions, and related metrics through the ERCOT API.

This module handles authentication, rate limiting, data formatting, and error recovery for ERCOT API
requests to ensure reliable data retrieval for the RTLMP spike prediction system.
"""

import os
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, cast

import pandas as pd  # version 2.0+
import requests  # version 2.28+
import pytz  # version 2023.3+

from .base import BaseDataFetcher
from ...utils.type_definitions import RTLMPDataDict, GridConditionDict, DataFrameType
from ...utils.logging import get_logger, log_execution_time
from ...utils.error_handling import (
    retry_with_backoff, handle_errors, 
    ConnectionError, RateLimitError, DataFormatError, MissingDataError
)
from ../validators/schemas import validate_rtlmp_data, validate_grid_condition_data
from ...utils.date_utils import ERCOT_TIMEZONE, localize_datetime

# Set up logger
logger = get_logger(__name__)

# API endpoints and configuration
ERCOT_API_BASE_URL = 'https://api.ercot.com/'
ERCOT_API_VERSION = 'v1'
RTLMP_ENDPOINT = f'{ERCOT_API_BASE_URL}{ERCOT_API_VERSION}/rtlmp'
GRID_CONDITIONS_ENDPOINT = f'{ERCOT_API_BASE_URL}{ERCOT_API_VERSION}/gridconditions'
DEFAULT_NODES = ['HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_HOUSTON']
MAX_DATE_RANGE_DAYS = 365
RTLMP_FREQUENCY = '5min'
GRID_CONDITIONS_FREQUENCY = '1H'


@handle_errors(exceptions=DataFormatError, error_message='Error parsing RTLMP response')
def parse_rtlmp_response(response_data: Dict[str, Any]) -> List[RTLMPDataDict]:
    """
    Parses the RTLMP API response into a standardized format.
    
    Args:
        response_data: JSON response data from the ERCOT API
        
    Returns:
        List of standardized RTLMP data dictionaries
    """
    # Validate that response_data contains expected fields
    if not isinstance(response_data, dict) or 'data' not in response_data:
        raise DataFormatError("Invalid RTLMP response format: missing 'data' field")
    
    # Extract the RTLMP records from the response
    rtlmp_records = response_data.get('data', [])
    
    if not isinstance(rtlmp_records, list):
        raise DataFormatError("Invalid RTLMP response format: 'data' field is not a list")
    
    # Initialize empty list to store parsed data
    result = []
    
    # For each record in the response
    for record in rtlmp_records:
        try:
            # Extract timestamp and convert to datetime
            timestamp_str = record.get('timestamp')
            if not timestamp_str:
                logger.warning(f"Skipping RTLMP record without timestamp: {record}")
                continue
            
            timestamp = datetime.fromisoformat(timestamp_str)
            
            # Extract node_id
            node_id = record.get('node_id')
            if not node_id:
                logger.warning(f"Skipping RTLMP record without node_id: {record}")
                continue
            
            # Extract price components (total, congestion, loss, energy)
            price = float(record.get('price', 0))
            congestion_price = float(record.get('congestion_price', 0))
            loss_price = float(record.get('loss_price', 0))
            energy_price = float(record.get('energy_price', 0))
            
            # Create RTLMPDataDict with extracted values
            rtlmp_data: RTLMPDataDict = {
                'timestamp': timestamp,
                'node_id': node_id,
                'price': price,
                'congestion_price': congestion_price,
                'loss_price': loss_price,
                'energy_price': energy_price
            }
            
            # Append to result list
            result.append(rtlmp_data)
        except Exception as e:
            logger.warning(f"Error parsing RTLMP record: {str(e)}, record: {record}")
            continue
    
    if not result:
        raise DataFormatError("No valid RTLMP records found in response")
    
    return result


@handle_errors(exceptions=DataFormatError, error_message='Error parsing grid conditions response')
def parse_grid_conditions_response(response_data: Dict[str, Any]) -> List[GridConditionDict]:
    """
    Parses the grid conditions API response into a standardized format.
    
    Args:
        response_data: JSON response data from the ERCOT API
        
    Returns:
        List of standardized grid condition dictionaries
    """
    # Validate that response_data contains expected fields
    if not isinstance(response_data, dict) or 'data' not in response_data:
        raise DataFormatError("Invalid grid conditions response format: missing 'data' field")
    
    # Extract the grid condition records from the response
    grid_records = response_data.get('data', [])
    
    if not isinstance(grid_records, list):
        raise DataFormatError("Invalid grid conditions response format: 'data' field is not a list")
    
    # Initialize empty list to store parsed data
    result = []
    
    # For each record in the response
    for record in grid_records:
        try:
            # Extract timestamp and convert to datetime
            timestamp_str = record.get('timestamp')
            if not timestamp_str:
                logger.warning(f"Skipping grid condition record without timestamp: {record}")
                continue
            
            timestamp = datetime.fromisoformat(timestamp_str)
            
            # Extract total_load
            total_load = float(record.get('total_load', 0))
            
            # Extract available_capacity
            available_capacity = float(record.get('available_capacity', 0))
            
            # Extract wind_generation
            wind_generation = float(record.get('wind_generation', 0))
            
            # Extract solar_generation
            solar_generation = float(record.get('solar_generation', 0))
            
            # Calculate reserve_margin if not provided
            reserve_margin = record.get('reserve_margin')
            if reserve_margin is None and available_capacity > 0:
                reserve_margin = (available_capacity - total_load) / available_capacity
            
            # Create GridConditionDict with extracted values
            grid_data: GridConditionDict = {
                'timestamp': timestamp,
                'total_load': total_load,
                'available_capacity': available_capacity,
                'wind_generation': wind_generation,
                'solar_generation': solar_generation
            }
            
            # Add reserve_margin if calculated or provided
            if reserve_margin is not None:
                grid_data['reserve_margin'] = float(reserve_margin)
            
            # Append to result list
            result.append(grid_data)
        except Exception as e:
            logger.warning(f"Error parsing grid condition record: {str(e)}, record: {record}")
            continue
    
    if not result:
        raise DataFormatError("No valid grid condition records found in response")
    
    return result


@handle_errors(exceptions=Exception, error_message='Error building authentication headers')
def build_auth_headers() -> Dict[str, str]:
    """
    Builds authentication headers for ERCOT API requests.
    
    Returns:
        Dictionary of authentication headers
    """
    # Get API key and secret from environment variables
    api_key = os.environ.get('ERCOT_API_KEY')
    api_secret = os.environ.get('ERCOT_API_SECRET')
    
    # If credentials are not found, log warning and return empty headers
    if not api_key or not api_secret:
        logger.warning("ERCOT API credentials not found in environment variables")
        return {}
    
    # Encode credentials using base64
    credentials = f"{api_key}:{api_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    # Create authorization header with encoded credentials
    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/json'
    }
    
    return headers


class ERCOTDataFetcher(BaseDataFetcher):
    """
    Implementation of the data fetcher interface for retrieving data from the ERCOT API.
    
    This class extends BaseDataFetcher to provide specific functionality for fetching
    RTLMP data, grid condition data, and related metrics from the ERCOT API.
    """
    
    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 3,
        cache_ttl: int = 3600,
        cache_dir: str = None,
        max_date_range_days: int = MAX_DATE_RANGE_DAYS
    ):
        """
        Initialize the ERCOT data fetcher with configuration options.
        
        Args:
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            cache_ttl: Time-to-live for cached data in seconds
            cache_dir: Directory to store cached data files
            max_date_range_days: Maximum number of days for a single date range query
        """
        # Call the parent BaseDataFetcher constructor with timeout, max_retries, cache_ttl, and cache_dir
        super().__init__(timeout, max_retries, cache_ttl, cache_dir)
        
        # Set max_date_range_days (default to MAX_DATE_RANGE_DAYS if not provided)
        self.max_date_range_days = max_date_range_days
        
        # Initialize auth_headers by calling build_auth_headers()
        self.auth_headers = build_auth_headers()
        
        logger.info(f"Initialized ERCOT data fetcher with max_date_range_days={max_date_range_days}")
    
    @log_execution_time(logger, 'INFO')
    def fetch_data(self, params: Dict[str, Any]) -> DataFrameType:
        """
        Generic method to fetch data from ERCOT API based on parameters.
        
        Args:
            params: Dictionary of parameters for the data fetch
            
        Returns:
            DataFrame with fetched data
        """
        # Check if data is available in cache using _get_from_cache(params)
        cached_data = self._get_from_cache(params)
        if cached_data is not None:
            logger.info("Returning cached data")
            return cached_data
        
        # Extract data_type from params ('rtlmp' or 'grid')
        data_type = params.get('data_type')
        if not data_type:
            raise ValueError("Missing 'data_type' in params")
        
        # Extract start_date and end_date from params
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        
        if not start_date or not end_date:
            raise ValueError("Missing 'start_date' or 'end_date' in params")
        
        # Extract nodes from params if applicable
        nodes = params.get('nodes', DEFAULT_NODES)
        
        # Based on data_type, call appropriate method to fetch data
        if data_type == 'rtlmp':
            result = self.fetch_rtlmp_data(start_date, end_date, nodes)
        elif data_type == 'grid':
            result = self.fetch_grid_conditions(start_date, end_date)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
        
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
        Fetch historical data from ERCOT API for a specific date range.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            identifiers: List of identifiers (node IDs for RTLMP, empty for grid conditions)
            
        Returns:
            DataFrame with historical data
        """
        # Validate the date range using _validate_date_range(start_date, end_date, max_date_range_days)
        if not self._validate_date_range(start_date, end_date, self.max_date_range_days):
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")
        
        # Determine data type based on identifiers (nodes for RTLMP, empty for grid conditions)
        data_type = 'rtlmp' if identifiers else 'grid'
        
        # If date range exceeds max_date_range_days, split into smaller chunks
        days_diff = (end_date - start_date).days
        if days_diff > self.max_date_range_days:
            logger.info(f"Date range exceeds max_date_range_days ({self.max_date_range_days}), " 
                       f"splitting into smaller chunks")
            
            result_chunks = []
            current_start = start_date
            
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=self.max_date_range_days), end_date)
                
                # Fetch data for the chunk date range
                if data_type == 'rtlmp':
                    chunk_data = self.fetch_rtlmp_data(current_start, current_end, identifiers)
                else:
                    chunk_data = self.fetch_grid_conditions(current_start, current_end)
                
                result_chunks.append(chunk_data)
                current_start = current_end
            
            # Combine chunks
            result = pd.concat(result_chunks, ignore_index=True)
            
            # Remove any duplicates
            result = result.drop_duplicates()
            
            # Sort by timestamp
            if 'timestamp' in result.columns:
                result = result.sort_values('timestamp')
        else:
            # Date range is within limits, fetch directly
            if data_type == 'rtlmp':
                result = self.fetch_rtlmp_data(start_date, end_date, identifiers)
            else:
                result = self.fetch_grid_conditions(start_date, end_date)
        
        # Validate the combined data using validate_data()
        self.validate_data(result)
        
        return result
    
    @log_execution_time(logger, 'INFO')
    def fetch_forecast_data(
        self,
        forecast_date: datetime,
        horizon: int,
        identifiers: List[str]
    ) -> DataFrameType:
        """
        Fetch forecast data from ERCOT API for a specific date and horizon.
        
        Args:
            forecast_date: Date for which to retrieve forecasts
            horizon: Forecast horizon in hours
            identifiers: List of identifiers (node IDs for RTLMP, empty for grid conditions)
            
        Returns:
            DataFrame with forecast data
        """
        # Calculate end_date as forecast_date + horizon hours
        end_date = forecast_date + timedelta(hours=horizon)
        
        # Validate the date range using _validate_date_range(forecast_date, end_date)
        if not self._validate_date_range(forecast_date, end_date):
            raise ValueError(f"Invalid forecast range: {forecast_date} to {end_date}")
        
        # Determine data type based on identifiers (nodes for RTLMP, empty for grid conditions)
        data_type = 'rtlmp' if identifiers else 'grid'
        
        # Construct API request parameters for forecast data
        params = {
            'data_type': data_type,
            'start_date': forecast_date,
            'end_date': end_date,
            'forecast': True
        }
        
        if data_type == 'rtlmp':
            params['nodes'] = identifiers
        
        # Make API request to appropriate endpoint
        return self.fetch_data(params)
    
    @retry_with_backoff(exceptions=(ConnectionError, RateLimitError), max_retries=None)
    def fetch_rtlmp_data(
        self,
        start_date: datetime,
        end_date: datetime,
        nodes: List[str]
    ) -> DataFrameType:
        """
        Fetch RTLMP data from ERCOT API for a specific date range and nodes.
        
        Args:
            start_date: Start date for RTLMP data
            end_date: End date for RTLMP data
            nodes: List of node IDs to fetch data for
            
        Returns:
            DataFrame with RTLMP data
        """
        # Validate input parameters (start_date before end_date, valid nodes)
        if start_date >= end_date:
            raise ValueError(f"start_date must be before end_date: {start_date} >= {end_date}")
        
        # Ensure nodes is not empty (use DEFAULT_NODES if empty)
        if not nodes:
            logger.warning("No nodes specified, using DEFAULT_NODES")
            nodes = DEFAULT_NODES
        
        # Localize start_date and end_date to ERCOT_TIMEZONE if not already localized
        start_date = localize_datetime(start_date) if start_date.tzinfo is None else start_date
        end_date = localize_datetime(end_date) if end_date.tzinfo is None else end_date
        
        # Construct API request parameters for RTLMP data
        params = {
            'startDate': start_date.isoformat(),
            'endDate': end_date.isoformat(),
            'nodes': ','.join(nodes),
            'frequency': RTLMP_FREQUENCY
        }
        
        logger.info(f"Fetching RTLMP data for {len(nodes)} nodes from {start_date} to {end_date}")
        
        # Make API request to RTLMP endpoint using _make_request()
        response = self._make_request(
            url=RTLMP_ENDPOINT,
            method='GET',
            params=params,
            headers=self.auth_headers
        )
        
        # Parse the response using parse_rtlmp_response()
        response_data = response.json()
        rtlmp_data = parse_rtlmp_response(response_data)
        
        # Convert parsed data to DataFrame using pandas.DataFrame()
        df = pd.DataFrame(rtlmp_data)
        
        # Ensure timestamp column is datetime type with correct timezone
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(ERCOT_TIMEZONE, ambiguous='infer')
        
        # Sort DataFrame by timestamp and node_id
        df = df.sort_values(['timestamp', 'node_id'])
        
        logger.info(f"Fetched {len(df)} RTLMP records")
        
        return df
    
    @retry_with_backoff(exceptions=(ConnectionError, RateLimitError), max_retries=None)
    def fetch_grid_conditions(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> DataFrameType:
        """
        Fetch grid condition data from ERCOT API for a specific date range.
        
        Args:
            start_date: Start date for grid condition data
            end_date: End date for grid condition data
            
        Returns:
            DataFrame with grid condition data
        """
        # Validate input parameters (start_date before end_date)
        if start_date >= end_date:
            raise ValueError(f"start_date must be before end_date: {start_date} >= {end_date}")
        
        # Localize start_date and end_date to ERCOT_TIMEZONE if not already localized
        start_date = localize_datetime(start_date) if start_date.tzinfo is None else start_date
        end_date = localize_datetime(end_date) if end_date.tzinfo is None else end_date
        
        # Construct API request parameters for grid condition data
        params = {
            'startDate': start_date.isoformat(),
            'endDate': end_date.isoformat(),
            'frequency': GRID_CONDITIONS_FREQUENCY
        }
        
        logger.info(f"Fetching grid condition data from {start_date} to {end_date}")
        
        # Make API request to GRID_CONDITIONS_ENDPOINT using _make_request()
        response = self._make_request(
            url=GRID_CONDITIONS_ENDPOINT,
            method='GET',
            params=params,
            headers=self.auth_headers
        )
        
        # Parse the response using parse_grid_conditions_response()
        response_data = response.json()
        grid_data = parse_grid_conditions_response(response_data)
        
        # Convert parsed data to DataFrame using pandas.DataFrame()
        df = pd.DataFrame(grid_data)
        
        # Ensure timestamp column is datetime type with correct timezone
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(ERCOT_TIMEZONE, ambiguous='infer')
        
        # Sort DataFrame by timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Fetched {len(df)} grid condition records")
        
        return df
    
    def validate_data(self, data: DataFrameType) -> bool:
        """
        Validate the structure and content of fetched data.
        
        Args:
            data: DataFrame with data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if the DataFrame is empty
        if data.empty:
            logger.warning("Empty DataFrame, validation failed")
            return False
        
        # Determine data type based on columns (RTLMP or grid conditions)
        try:
            if 'node_id' in data.columns and 'price' in data.columns:
                # RTLMP data
                logger.debug("Validating RTLMP data")
                is_valid = validate_rtlmp_data(data)
            elif 'total_load' in data.columns and 'available_capacity' in data.columns:
                # Grid condition data
                logger.debug("Validating grid condition data")
                is_valid = validate_grid_condition_data(data)
            else:
                logger.warning("Unknown data type, cannot validate")
                return False
            
            # Log validation results
            if is_valid:
                logger.debug("Data validation successful")
            else:
                logger.warning("Data validation failed")
            
            return is_valid
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False
    
    def refresh_auth_headers(self) -> Dict[str, str]:
        """
        Refresh the authentication headers for API requests.
        
        Returns:
            Updated authentication headers
        """
        # Call build_auth_headers() to generate fresh headers
        self.auth_headers = build_auth_headers()
        
        # Log the refresh operation
        logger.info("Refreshed authentication headers")
        
        return self.auth_headers