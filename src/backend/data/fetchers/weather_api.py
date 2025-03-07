"""
Implementation of a weather data fetcher for the ERCOT RTLMP spike prediction system.

This module provides functionality to retrieve weather forecast data from external weather APIs,
which is an important input for predicting price spikes in the electricity market.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd  # version 2.0+
import requests  # version 2.28+

from .base import BaseDataFetcher
from ...utils.type_definitions import DataFrameType, WeatherDataDict
from ...utils.error_handling import ConnectionError, RateLimitError, DataFormatError
from ...utils.logging import get_logger, log_execution_time
from ..validators.schemas import validate_weather_data
from ..validators.pandera_schemas import WeatherSchema

# Set up logger
logger = get_logger(__name__)

# Default configuration
DEFAULT_WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', '')
DEFAULT_WEATHER_API_URL = 'https://api.weatherapi.com/v1'
DEFAULT_LOCATIONS = [
    {'id': 'HB_NORTH', 'lat': '32.7767', 'lon': '-96.7970'},
    {'id': 'HB_SOUTH', 'lat': '29.7604', 'lon': '-95.3698'},
    {'id': 'HB_WEST', 'lat': '31.5604', 'lon': '-97.1861'},
    {'id': 'HB_HOUSTON', 'lat': '29.7604', 'lon': '-95.3698'}
]
MAX_FORECAST_DAYS = 14
MAX_HISTORY_DAYS = 30


def parse_weather_response(response_data: Dict[str, Any], location_id: str) -> List[WeatherDataDict]:
    """
    Parses the JSON response from the weather API into a standardized format.
    
    Args:
        response_data: Raw JSON response from the weather API
        location_id: ID of the location for which data was retrieved
        
    Returns:
        List of weather data dictionaries with standardized format
    """
    results = []
    
    # Check if the response contains forecast or history data
    if 'forecast' in response_data:
        # Extract forecast data
        forecast_days = response_data['forecast']['forecastday']
        
        for day in forecast_days:
            date = day['date']
            
            # Process hourly data
            if 'hour' in day:
                for hour_data in day['hour']:
                    weather_data = WeatherDataDict(
                        timestamp=datetime.fromisoformat(hour_data['time']),
                        location_id=location_id,
                        temperature=float(hour_data['temp_c']),
                        wind_speed=float(hour_data['wind_kph']) / 3.6,  # Convert km/h to m/s
                        solar_irradiance=float(hour_data.get('uv', 0)) * 100,  # Approximate conversion
                        humidity=float(hour_data['humidity'])
                    )
                    results.append(weather_data)
    
    elif 'history' in response_data:
        # Extract historical data
        history_days = response_data['history']['forecastday']
        
        for day in history_days:
            date = day['date']
            
            # Process hourly data
            if 'hour' in day:
                for hour_data in day['hour']:
                    weather_data = WeatherDataDict(
                        timestamp=datetime.fromisoformat(hour_data['time']),
                        location_id=location_id,
                        temperature=float(hour_data['temp_c']),
                        wind_speed=float(hour_data['wind_kph']) / 3.6,  # Convert km/h to m/s
                        solar_irradiance=float(hour_data.get('uv', 0)) * 100,  # Approximate conversion
                        humidity=float(hour_data['humidity'])
                    )
                    results.append(weather_data)
    
    logger.debug(f"Parsed {len(results)} weather data points for location {location_id}")
    return results


def convert_location_format(location_id: str) -> Dict[str, str]:
    """
    Converts between different location identifier formats.
    
    Args:
        location_id: Location identifier to convert
        
    Returns:
        Dictionary with latitude and longitude
        
    Raises:
        ValueError: If location_id is not found in DEFAULT_LOCATIONS
    """
    for location in DEFAULT_LOCATIONS:
        if location['id'] == location_id:
            return {'lat': location['lat'], 'lon': location['lon']}
    
    raise ValueError(f"Location ID '{location_id}' not found in configured locations")


class WeatherAPIFetcher(BaseDataFetcher):
    """
    Fetcher for retrieving weather data from external weather API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        locations: Optional[List[Dict[str, str]]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        cache_ttl: int = 3600,
        cache_dir: str = None
    ):
        """
        Initialize the weather API fetcher with API credentials and configuration.
        
        Args:
            api_key: API key for the weather service (defaults to environment variable)
            api_url: Base URL for the weather API
            locations: List of location dictionaries with id, lat, and lon
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            cache_ttl: Time-to-live for cached data in seconds
            cache_dir: Directory to store cached data files
        """
        super().__init__(timeout, max_retries, cache_ttl, cache_dir)
        
        self.api_key = api_key or DEFAULT_WEATHER_API_KEY
        self.api_url = api_url or DEFAULT_WEATHER_API_URL
        self.locations = locations or DEFAULT_LOCATIONS
        
        if not self.api_key:
            raise ValueError(
                "Weather API key is required. Set the WEATHER_API_KEY environment "
                "variable or provide the api_key parameter."
            )
        
        logger.info(
            f"Initialized WeatherAPIFetcher with {len(self.locations)} locations, "
            f"timeout={timeout}s, retries={max_retries}"
        )
    
    @log_execution_time
    def fetch_data(self, params: Dict[str, Any]) -> DataFrameType:
        """
        Generic method to fetch weather data based on parameters.
        
        This method handles both historical and forecast data retrieval based
        on the parameters provided.
        
        Args:
            params: Dictionary of parameters for the data fetch
                - 'start_date': Start date for historical data
                - 'end_date': End date for historical data
                - 'forecast_date': Date for which to retrieve forecasts
                - 'horizon': Forecast horizon in hours
                - 'identifiers': List of location identifiers
                
        Returns:
            DataFrame with weather data
        """
        # Check if data is available in cache
        cache_data = self._get_from_cache(params)
        if cache_data is not None:
            logger.debug("Returning cached weather data")
            return cache_data
        
        # Determine if this is a historical or forecast request
        is_historical = 'start_date' in params and 'end_date' in params
        is_forecast = 'forecast_date' in params and 'horizon' in params
        
        if not (is_historical or is_forecast):
            raise ValueError(
                "Invalid parameters: must include either 'start_date' and 'end_date' "
                "for historical data, or 'forecast_date' and 'horizon' for forecast data"
            )
        
        # Get location identifiers
        identifiers = params.get('identifiers', [loc['id'] for loc in self.locations])
        
        # Initialize results list
        all_weather_data = []
        
        # Process each location
        for location_id in identifiers:
            try:
                # Convert location_id to lat/lon
                location = convert_location_format(location_id)
                
                # Prepare API request parameters
                if is_historical:
                    endpoint = '/history.json'
                    api_params = {
                        'q': f"{location['lat']},{location['lon']}",
                        'dt': params['start_date'].strftime('%Y-%m-%d'),
                        'end_dt': params['end_date'].strftime('%Y-%m-%d'),
                        'key': self.api_key
                    }
                else:  # is_forecast
                    endpoint = '/forecast.json'
                    days = min(MAX_FORECAST_DAYS, (params['horizon'] + 23) // 24)  # Convert hours to days, rounding up
                    api_params = {
                        'q': f"{location['lat']},{location['lon']}",
                        'days': days,
                        'key': self.api_key
                    }
                
                # Make the API request
                url = self._build_api_url(endpoint)
                response = self._make_request(url, params=api_params)
                
                if response and response.status_code == 200:
                    data = response.json()
                    weather_data = parse_weather_response(data, location_id)
                    all_weather_data.extend(weather_data)
                else:
                    logger.warning(f"Failed to retrieve weather data for location {location_id}")
            
            except (ConnectionError, RateLimitError) as e:
                # These errors are already logged and handled by _make_request
                raise
            
            except Exception as e:
                logger.error(f"Error fetching weather data for location {location_id}: {str(e)}")
                raise DataFormatError(f"Error fetching weather data: {str(e)}")
        
        # Convert to DataFrame
        if all_weather_data:
            df = pd.DataFrame(all_weather_data)
            
            # Filter to relevant time range for forecast data
            if is_forecast:
                forecast_date = params['forecast_date']
                end_date = forecast_date + timedelta(hours=params['horizon'])
                df = df[(df['timestamp'] >= forecast_date) & (df['timestamp'] < end_date)]
            
            # Validate the data
            if self.validate_data(df):
                # Store in cache for future use
                self._store_in_cache(params, df)
                return df
            else:
                raise DataFormatError("Weather data validation failed")
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['timestamp', 'location_id', 'temperature', 
                                        'wind_speed', 'solar_irradiance', 'humidity'])
    
    @log_execution_time
    def fetch_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        identifiers: List[str]
    ) -> DataFrameType:
        """
        Fetch historical weather data for a specific date range.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            identifiers: List of location identifiers
            
        Returns:
            DataFrame with historical weather data
        """
        # Validate date range
        if not self._validate_date_range(start_date, end_date, MAX_HISTORY_DAYS):
            raise ValueError(
                f"Invalid date range: must be within {MAX_HISTORY_DAYS} days and not in the future"
            )
        
        # Prepare parameters for fetch_data
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'identifiers': identifiers
        }
        
        return self.fetch_data(params)
    
    @log_execution_time
    def fetch_forecast_data(
        self,
        forecast_date: datetime,
        horizon: int,
        identifiers: List[str]
    ) -> DataFrameType:
        """
        Fetch weather forecast data for a specific date and horizon.
        
        Args:
            forecast_date: Date for which to retrieve forecasts
            horizon: Forecast horizon in hours
            identifiers: List of location identifiers
            
        Returns:
            DataFrame with forecast weather data
        """
        # Validate horizon
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        
        # Convert horizon from hours to days, rounding up
        days_needed = (horizon + 23) // 24
        
        if days_needed > MAX_FORECAST_DAYS:
            raise ValueError(f"Forecast horizon exceeds maximum of {MAX_FORECAST_DAYS} days")
        
        # Prepare parameters for fetch_data
        params = {
            'forecast_date': forecast_date,
            'horizon': horizon,
            'identifiers': identifiers
        }
        
        return self.fetch_data(params)
    
    def validate_data(self, data: DataFrameType) -> bool:
        """
        Validate the structure and content of fetched weather data.
        
        Args:
            data: DataFrame with weather data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Use schema validation for weather data
            validate_weather_data(data)
            return True
        except Exception as e:
            logger.error(f"Weather data validation failed: {str(e)}")
            return False
    
    def _build_api_url(self, endpoint: str, params: Dict[str, Any] = None) -> str:
        """
        Build the API URL for a specific endpoint and parameters.
        
        Args:
            endpoint: API endpoint path
            params: Dictionary of query parameters
            
        Returns:
            Complete API URL
        """
        base_url = self.api_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        url = f"{base_url}/{endpoint}"
        
        return url