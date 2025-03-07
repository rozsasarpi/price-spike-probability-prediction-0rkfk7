"""
Mock data fetcher implementation for the ERCOT RTLMP spike prediction system.

Provides synthetic data for RTLMP values, weather forecasts, and grid conditions
for testing and development purposes without requiring access to external APIs.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import random

from .base import BaseDataFetcher
from ...utils.type_definitions import (
    RTLMPDataDict,
    WeatherDataDict,
    GridConditionDict
)
from ...utils.logging import get_logger, log_execution_time
from ...utils.validation import validate_dataframe_schema
from ...utils.date_utils import ERCOT_TIMEZONE, localize_datetime

# Set up logger
logger = get_logger(__name__)

# Default nodes and locations
DEFAULT_NODES = ['HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_HOUSTON']
DEFAULT_LOCATIONS = ['NORTH_TX', 'SOUTH_TX', 'WEST_TX', 'HOUSTON']

# Parameters for synthetic data generation
BASE_PRICE = 30.0
PRICE_VOLATILITY = 10.0
SPIKE_PROBABILITY = 0.05
SPIKE_MAGNITUDE = 100.0
BASE_TEMPERATURE = 75.0
TEMPERATURE_VARIATION = 15.0
BASE_LOAD = 40000.0
LOAD_VARIATION = 10000.0


def generate_mock_rtlmp_data(start_date: datetime, end_date: datetime, nodes: List[str]) -> List[RTLMPDataDict]:
    """
    Generates synthetic RTLMP data for a given time range and nodes.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        nodes: List of node IDs
        
    Returns:
        List of synthetic RTLMP data dictionaries
    """
    # Validate input parameters
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    
    if not nodes:
        raise ValueError("Node list cannot be empty")
    
    # Calculate the number of 5-minute intervals between start_date and end_date
    time_delta = end_date - start_date
    total_minutes = time_delta.total_seconds() / 60
    intervals = int(total_minutes / 5)
    
    # Initialize empty list to store RTLMP data
    data = []
    
    # For each 5-minute interval and each node
    for i in range(intervals + 1):
        current_time = start_date + timedelta(minutes=5 * i)
        
        # Generate base price with daily and hourly patterns
        hour_factor = 1.0 + 0.2 * np.sin(np.pi * current_time.hour / 12)  # Higher in midday
        day_factor = 1.0 + 0.1 * np.sin(np.pi * current_time.weekday() / 3.5)  # Higher mid-week
        
        for node in nodes:
            # Generate base price with patterns
            base_price = BASE_PRICE * hour_factor * day_factor
            
            # Add random noise to simulate price volatility
            price_noise = np.random.normal(0, PRICE_VOLATILITY)
            price = max(1.0, base_price + price_noise)  # Ensure price is at least 1.0
            
            # Randomly generate price spikes based on SPIKE_PROBABILITY
            if random.random() < SPIKE_PROBABILITY:
                spike_multiplier = 1.0 + random.random() * 5.0  # Random spike between 1-6x
                price += SPIKE_MAGNITUDE * spike_multiplier
            
            # Calculate congestion, loss, and energy components of the price
            congestion_component = price * 0.3 * random.random()
            loss_component = price * 0.1 * random.random()
            energy_component = price - congestion_component - loss_component
            
            # Create RTLMPDataDict record and append to data list
            data_entry: RTLMPDataDict = {
                "timestamp": current_time,
                "node_id": node,
                "price": price,
                "congestion_price": congestion_component,
                "loss_price": loss_component,
                "energy_price": energy_component
            }
            
            data.append(data_entry)
    
    return data


def generate_mock_weather_data(start_date: datetime, end_date: datetime, locations: List[str]) -> List[WeatherDataDict]:
    """
    Generates synthetic weather data for a given time range and locations.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        locations: List of location IDs
        
    Returns:
        List of synthetic weather data dictionaries
    """
    # Validate input parameters
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    
    if not locations:
        raise ValueError("Location list cannot be empty")
    
    # Calculate the number of hourly intervals between start_date and end_date
    time_delta = end_date - start_date
    total_hours = int(time_delta.total_seconds() / 3600) + 1
    
    # Initialize empty list to store weather data
    data = []
    
    # For each hourly interval and each location
    for i in range(total_hours):
        current_time = start_date + timedelta(hours=i)
        
        # Apply seasonal and daily temperature patterns
        day_of_year = current_time.timetuple().tm_yday
        season_factor = np.sin(np.pi * day_of_year / 182.5)  # Seasonal variation
        hour_factor = np.sin(np.pi * current_time.hour / 12)  # Daily variation
        
        for location in locations:
            # Base temperature varies by location
            loc_index = locations.index(location)
            loc_temp_adjustment = (loc_index - len(locations) / 2) * 5.0
            
            # Calculate temperature with daily and seasonal patterns
            temperature = BASE_TEMPERATURE + loc_temp_adjustment + \
                          TEMPERATURE_VARIATION * season_factor + \
                          5.0 * hour_factor + \
                          np.random.normal(0, 3.0)  # Random noise
            
            # Generate wind speed with random variations
            wind_speed = 5.0 + 8.0 * random.random() + \
                         3.0 * np.sin(np.pi * current_time.hour / 12)  # Higher in afternoon
            
            # Generate solar irradiance based on time of day
            if 6 <= current_time.hour <= 18:
                solar_factor = np.sin(np.pi * (current_time.hour - 6) / 12)
                solar_irradiance = 800.0 * solar_factor * (0.7 + 0.3 * random.random())
            else:
                solar_irradiance = 0.0
            
            # Generate humidity with random variations
            humidity = 30.0 + 50.0 * random.random()
            
            # Create WeatherDataDict record and append to data list
            data_entry: WeatherDataDict = {
                "timestamp": current_time,
                "location_id": location,
                "temperature": temperature,
                "wind_speed": wind_speed,
                "solar_irradiance": solar_irradiance,
                "humidity": humidity
            }
            
            data.append(data_entry)
    
    return data


def generate_mock_grid_conditions(start_date: datetime, end_date: datetime) -> List[GridConditionDict]:
    """
    Generates synthetic grid condition data for a given time range.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        
    Returns:
        List of synthetic grid condition dictionaries
    """
    # Validate input parameters
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    
    # Calculate the number of hourly intervals between start_date and end_date
    time_delta = end_date - start_date
    total_hours = int(time_delta.total_seconds() / 3600) + 1
    
    # Initialize empty list to store grid condition data
    data = []
    
    # For each hourly interval
    for i in range(total_hours):
        current_time = start_date + timedelta(hours=i)
        
        # Apply load patterns based on time
        day_of_year = current_time.timetuple().tm_yday
        season_factor = 1.0 + 0.2 * np.sin(np.pi * day_of_year / 182.5)  # Seasonal variation
        hour_factor = 1.0 + 0.3 * np.sin(np.pi * current_time.hour / 12)  # Daily variation
        day_factor = 1.0 - 0.2 * (1 if current_time.weekday() >= 5 else 0)  # Weekend reduction
        
        # Generate total load with daily and seasonal patterns
        total_load = BASE_LOAD * season_factor * hour_factor * day_factor + \
                     np.random.normal(0, LOAD_VARIATION * 0.05)  # Random noise
        
        # Generate available capacity with random variations
        capacity_factor = 1.1 + 0.1 * random.random()  # Capacity is typically more than load
        available_capacity = total_load * capacity_factor
        
        # Generate wind generation based on time patterns
        time_of_day = current_time.hour
        wind_factor = 1.2 - 0.4 * np.sin(np.pi * time_of_day / 12)  # More at night
        wind_generation = 5000.0 * wind_factor * (0.7 + 0.6 * random.random())
        
        # Generate solar generation based on time of day
        if 6 <= time_of_day <= 18:
            solar_factor = np.sin(np.pi * (time_of_day - 6) / 12)
            solar_generation = 3000.0 * solar_factor * (0.7 + 0.3 * random.random())
        else:
            solar_generation = 0.0
        
        # Calculate reserve margin as (available_capacity - total_load) / total_load
        reserve_margin = (available_capacity - total_load) / total_load
        
        # Create GridConditionDict record and append to data list
        data_entry: GridConditionDict = {
            "timestamp": current_time,
            "total_load": total_load,
            "available_capacity": available_capacity,
            "wind_generation": wind_generation,
            "solar_generation": solar_generation,
            "reserve_margin": reserve_margin
        }
        
        data.append(data_entry)
    
    return data


def convert_to_dataframe(data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts a list of dictionaries to a pandas DataFrame.
    
    Args:
        data_list: List of dictionaries containing data
        
    Returns:
        DataFrame representation of the data
    """
    # Convert list of dictionaries to pandas DataFrame
    df = pd.DataFrame(data_list)
    
    # Ensure timestamp column is datetime type
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort DataFrame by timestamp and any ID columns
    sort_columns = ['timestamp']
    if 'node_id' in df.columns:
        sort_columns.append('node_id')
    elif 'location_id' in df.columns:
        sort_columns.append('location_id')
    
    df = df.sort_values(sort_columns)
    
    return df


class MockDataFetcher(BaseDataFetcher):
    """
    Implementation of the data fetcher interface that provides synthetic data for testing and development.
    """
    
    def __init__(
        self,
        add_noise: bool = True,
        spike_probability: float = SPIKE_PROBABILITY,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the mock data fetcher with configuration options.
        
        Args:
            add_noise: Whether to add random noise to synthetic data
            spike_probability: Probability of generating price spikes
            config: Additional configuration parameters
            timeout: HTTP request timeout (used by parent, not relevant for mock)
            max_retries: Maximum retry attempts (used by parent, not relevant for mock)
        """
        # Call the parent BaseDataFetcher constructor with timeout and max_retries
        super().__init__(timeout=timeout, max_retries=max_retries)
        
        # Set add_noise flag (default to True if not provided)
        self.add_noise = add_noise
        
        # Set spike_probability (default to SPIKE_PROBABILITY if not provided)
        self.spike_probability = spike_probability
        
        # Initialize config dictionary with default values
        self.config = {
            'base_price': BASE_PRICE,
            'price_volatility': PRICE_VOLATILITY,
            'spike_magnitude': SPIKE_MAGNITUDE,
            'base_temperature': BASE_TEMPERATURE,
            'temperature_variation': TEMPERATURE_VARIATION,
            'base_load': BASE_LOAD,
            'load_variation': LOAD_VARIATION
        }
        
        # Update config with provided values if any
        if config:
            self.config.update(config)
        
        # Log initialization of the mock data fetcher
        logger.info(f"Initialized MockDataFetcher with spike_probability={spike_probability}, add_noise={add_noise}")
    
    @log_execution_time(logger, 'INFO')
    def fetch_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generic method to fetch mock data based on parameters.
        
        Args:
            params: Dictionary of parameters for the data fetch
            
        Returns:
            DataFrame with mock data
        """
        # Check if data is available in cache
        cached_data = self._get_from_cache(params)
        if cached_data is not None:
            logger.info("Returning cached data")
            return cached_data
        
        # Extract data_type from params ('rtlmp', 'weather', or 'grid')
        data_type = params.get('data_type', 'rtlmp')
        
        # Extract start_date and end_date or forecast_date
        if 'forecast_date' in params:
            forecast_date = params['forecast_date']
            horizon = params.get('horizon', 72)  # Default to 72 hours
            end_date = forecast_date + timedelta(hours=horizon)
            start_date = forecast_date
        else:
            start_date = params.get('start_date', datetime.now() - timedelta(days=7))
            end_date = params.get('end_date', datetime.now())
        
        # Extract nodes or locations from params
        if data_type == 'rtlmp':
            nodes = params.get('nodes', DEFAULT_NODES)
            data_list = generate_mock_rtlmp_data(start_date, end_date, nodes)
        elif data_type == 'weather':
            locations = params.get('locations', DEFAULT_LOCATIONS)
            data_list = generate_mock_weather_data(start_date, end_date, locations)
        elif data_type == 'grid':
            data_list = generate_mock_grid_conditions(start_date, end_date)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Convert the generated data to a DataFrame
        df = convert_to_dataframe(data_list)
        
        # Validate the data structure using validate_data()
        self.validate_data(df)
        
        # Store the result in cache
        self._store_in_cache(params, df)
        
        return df
    
    @log_execution_time(logger, 'INFO')
    def fetch_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        identifiers: List[str]
    ) -> pd.DataFrame:
        """
        Fetch historical mock data for a specific date range.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            identifiers: List of identifiers (node IDs or location IDs)
            
        Returns:
            DataFrame with historical mock data
        """
        # Validate the date range (start_date before end_date)
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Determine data type based on identifiers (nodes or locations)
        if any(node in identifiers for node in DEFAULT_NODES):
            # If nodes, generate mock RTLMP data
            data_list = generate_mock_rtlmp_data(start_date, end_date, identifiers)
        elif any(location in identifiers for location in DEFAULT_LOCATIONS):
            # If locations, generate mock weather data
            data_list = generate_mock_weather_data(start_date, end_date, identifiers)
        else:
            # If neither, generate mock grid conditions
            data_list = generate_mock_grid_conditions(start_date, end_date)
        
        # Convert the generated data to a DataFrame
        df = convert_to_dataframe(data_list)
        
        # Validate the data structure
        self.validate_data(df)
        
        return df
    
    @log_execution_time(logger, 'INFO')
    def fetch_forecast_data(
        self,
        forecast_date: datetime,
        horizon: int,
        identifiers: List[str]
    ) -> pd.DataFrame:
        """
        Fetch forecast mock data for a specific date and horizon.
        
        Args:
            forecast_date: Date for which to retrieve forecasts
            horizon: Forecast horizon in hours
            identifiers: List of identifiers (node IDs or location IDs)
            
        Returns:
            DataFrame with forecast mock data
        """
        # Calculate end_date as forecast_date + horizon hours
        end_date = forecast_date + timedelta(hours=horizon)
        
        # Determine data type based on identifiers (nodes or locations)
        if any(node in identifiers for node in DEFAULT_NODES):
            # If nodes, generate mock RTLMP data
            data_list = generate_mock_rtlmp_data(forecast_date, end_date, identifiers)
        elif any(location in identifiers for location in DEFAULT_LOCATIONS):
            # If locations, generate mock weather data
            data_list = generate_mock_weather_data(forecast_date, end_date, identifiers)
        else:
            # If neither, generate mock grid conditions
            data_list = generate_mock_grid_conditions(forecast_date, end_date)
        
        # Convert the generated data to a DataFrame
        df = convert_to_dataframe(data_list)
        
        # Validate the data structure
        self.validate_data(df)
        
        return df
    
    @log_execution_time(logger, 'INFO')
    def fetch_grid_conditions(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch mock grid condition data for a specific date range.
        
        Args:
            start_date: Start date for grid condition data
            end_date: End date for grid condition data
            
        Returns:
            DataFrame with mock grid condition data
        """
        # Generate mock grid conditions
        data_list = generate_mock_grid_conditions(start_date, end_date)
        
        # Convert the generated data to a DataFrame
        df = convert_to_dataframe(data_list)
        
        # Validate the data structure
        self.validate_data(df)
        
        return df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the structure and content of mock data.
        
        Args:
            data: DataFrame with data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if the DataFrame is empty
        if data.empty:
            logger.warning("Data validation failed: DataFrame is empty")
            return False
        
        # Determine data type based on columns
        if 'node_id' in data.columns and 'price' in data.columns:
            # RTLMP data
            required_columns = ['timestamp', 'node_id', 'price', 'congestion_price', 'loss_price', 'energy_price']
        elif 'location_id' in data.columns and 'temperature' in data.columns:
            # Weather data
            required_columns = ['timestamp', 'location_id', 'temperature', 'wind_speed', 'solar_irradiance', 'humidity']
        elif 'total_load' in data.columns and 'available_capacity' in data.columns:
            # Grid condition data
            required_columns = ['timestamp', 'total_load', 'available_capacity', 'wind_generation', 'solar_generation', 'reserve_margin']
        else:
            logger.warning("Data validation failed: Unable to determine data type")
            return False
        
        # Verify that required columns are present
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Data validation failed: Missing columns {missing_columns}")
            return False
        
        # Check for missing values in critical columns
        for col in required_columns:
            if data[col].isnull().any():
                logger.warning(f"Data validation failed: Null values found in column {col}")
                return False
        
        # Validate data types of columns
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            logger.warning("Data validation failed: 'timestamp' column is not datetime type")
            return False
        
        # Verify that timestamps are in chronological order
        if not data['timestamp'].equals(data['timestamp'].sort_values()):
            logger.warning("Data validation failed: Timestamps are not in chronological order")
            return False
        
        # Log validation results
        logger.debug("Data validation passed")
        return True
    
    def set_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration for mock data generation.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        # Update the config dictionary with new values
        self.config.update(new_config)
        logger.info(f"Updated MockDataFetcher configuration: {new_config}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration for mock data generation.
        
        Returns:
            Current configuration dictionary
        """
        # Return a copy of the current config dictionary
        return self.config.copy()