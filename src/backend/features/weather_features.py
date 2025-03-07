"""
Module for creating weather-related features from weather data for the ERCOT RTLMP spike prediction system.

This module transforms raw weather data into predictive features that can be used to forecast 
price spikes in the electricity market.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from typing import Dict, List, Optional, Union, Tuple, Any

from ..utils.type_definitions import DataFrameType, WeatherDataDict
from ..utils.validation import validate_feature_data
from ..utils.logging import get_logger, log_execution_time

# Setup logger
logger = get_logger(__name__)

# Default column mappings between DataFrame columns and weather variables
DEFAULT_WEATHER_COLUMNS = {
    'temperature': 'temperature',
    'wind_speed': 'wind_speed',
    'solar_irradiance': 'solar_irradiance',
    'humidity': 'humidity'
}

# Default window sizes for rolling calculations (in hours)
DEFAULT_ROLLING_WINDOWS = [24, 48, 72, 168]  # 24h, 48h, 72h, 1 week

# Registry to track created features and their metadata
WEATHER_FEATURE_REGISTRY = []

@log_execution_time(logger, 'INFO')
def create_all_weather_features(
    df: DataFrameType,
    column_mapping: Optional[Dict[str, str]] = None,
    windows: Optional[List[int]] = None,
    include_interactions: bool = True,
    include_location_aggregation: bool = False
) -> DataFrameType:
    """
    Creates all weather-related features from weather data.
    
    Args:
        df: DataFrame containing weather data
        column_mapping: Optional mapping of DataFrame columns to weather variables
        windows: Optional list of window sizes for rolling calculations (in hours)
        include_interactions: Whether to include interaction features between weather variables
        include_location_aggregation: Whether to create features aggregated across locations
        
    Returns:
        DataFrame with added weather features
    """
    # Validate input DataFrame
    if df.empty:
        logger.warning("Empty DataFrame provided to create_all_weather_features")
        return df
    
    # Use default column mapping if none provided
    if column_mapping is None:
        column_mapping = DEFAULT_WEATHER_COLUMNS
    
    # Use default windows if none provided
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Validate that required columns exist
    for weather_var, column_name in column_mapping.items():
        if column_name not in result_df.columns:
            logger.warning(f"Required column '{column_name}' for '{weather_var}' not found in DataFrame")
    
    # Create temperature features
    if column_mapping.get('temperature') in result_df.columns:
        result_df = create_temperature_features(
            result_df, 
            column_mapping['temperature'],
            windows
        )
    
    # Create wind features
    if column_mapping.get('wind_speed') in result_df.columns:
        result_df = create_wind_features(
            result_df,
            column_mapping['wind_speed'],
            windows
        )
    
    # Create solar features (requires both solar_irradiance and humidity)
    if (column_mapping.get('solar_irradiance') in result_df.columns and 
        column_mapping.get('humidity') in result_df.columns):
        result_df = create_solar_features(
            result_df,
            column_mapping['solar_irradiance'],
            column_mapping['humidity'],
            windows
        )
    
    # Create interaction features if requested
    if include_interactions:
        # Check if we have at least two weather variables to create interactions
        available_vars = [var for var, col in column_mapping.items() if col in result_df.columns]
        if len(available_vars) >= 2:
            result_df = create_weather_interaction_features(result_df, column_mapping)
        else:
            logger.warning("Not enough weather variables available to create interaction features")
    
    # Create location-aggregated features if requested
    if include_location_aggregation:
        # Check if location column exists
        location_col = 'location_id'  # Default location column name
        if location_col in result_df.columns:
            result_df = create_location_aggregated_features(result_df, location_col, column_mapping)
        else:
            logger.warning(f"Location column '{location_col}' not found, skipping location aggregation")
    
    # Validate the resulting feature data
    try:
        validate_feature_data(result_df)
    except Exception as e:
        logger.warning(f"Feature validation warning: {str(e)}")
    
    return result_df

@log_execution_time(logger, 'DEBUG')
def create_temperature_features(
    df: DataFrameType,
    temperature_column: str,
    windows: List[int]
) -> DataFrameType:
    """
    Creates temperature-related features from weather data.
    
    Args:
        df: DataFrame containing weather data
        temperature_column: Name of the column containing temperature data
        windows: List of window sizes for rolling calculations (in hours)
        
    Returns:
        DataFrame with added temperature features
    """
    # Validate that the temperature column exists
    if temperature_column not in df.columns:
        logger.warning(f"Temperature column '{temperature_column}' not found in DataFrame")
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Create basic temperature features
    
    # Temperature squared (for non-linear effects)
    feature_name = f"{temperature_column}_squared"
    result_df[feature_name] = result_df[temperature_column] ** 2
    register_weather_feature(
        feature_name,
        "Temperature squared",
        "float",
        {"source_column": temperature_column},
        [temperature_column]
    )
    
    # Temperature deviation from daily average (assumes hourly data)
    if pd.api.types.is_datetime64_any_dtype(result_df.index):
        # Group by day and calculate daily average
        daily_avg = result_df.groupby(result_df.index.date)[temperature_column].transform('mean')
        feature_name = f"{temperature_column}_dev_daily_avg"
        result_df[feature_name] = result_df[temperature_column] - daily_avg
        register_weather_feature(
            feature_name,
            "Temperature deviation from daily average",
            "float",
            {"source_column": temperature_column, "transformation": "deviation_from_daily_avg"},
            [temperature_column]
        )
    
    # Create rolling statistics for each window size
    for window in windows:
        # Rolling mean
        feature_name = f"{temperature_column}_rolling_mean_{window}h"
        result_df[feature_name] = result_df[temperature_column].rolling(window=window, min_periods=1).mean()
        register_weather_feature(
            feature_name,
            f"Temperature rolling mean over {window} hours",
            "float",
            {"source_column": temperature_column, "window": window, "statistic": "mean"},
            [temperature_column]
        )
        
        # Rolling standard deviation (volatility)
        feature_name = f"{temperature_column}_rolling_std_{window}h"
        result_df[feature_name] = result_df[temperature_column].rolling(window=window, min_periods=2).std()
        register_weather_feature(
            feature_name,
            f"Temperature rolling standard deviation over {window} hours",
            "float",
            {"source_column": temperature_column, "window": window, "statistic": "std"},
            [temperature_column]
        )
        
        # Rolling min
        feature_name = f"{temperature_column}_rolling_min_{window}h"
        result_df[feature_name] = result_df[temperature_column].rolling(window=window, min_periods=1).min()
        register_weather_feature(
            feature_name,
            f"Temperature rolling minimum over {window} hours",
            "float",
            {"source_column": temperature_column, "window": window, "statistic": "min"},
            [temperature_column]
        )
        
        # Rolling max
        feature_name = f"{temperature_column}_rolling_max_{window}h"
        result_df[feature_name] = result_df[temperature_column].rolling(window=window, min_periods=1).max()
        register_weather_feature(
            feature_name,
            f"Temperature rolling maximum over {window} hours",
            "float",
            {"source_column": temperature_column, "window": window, "statistic": "max"},
            [temperature_column]
        )
    
    # Create temperature change rate features
    # Temperature change from 24 hours ago
    if len(result_df) > 24:
        feature_name = f"{temperature_column}_delta_24h"
        result_df[feature_name] = result_df[temperature_column].diff(periods=24)
        register_weather_feature(
            feature_name,
            "Temperature change from 24 hours ago",
            "float",
            {"source_column": temperature_column, "transformation": "diff_24h"},
            [temperature_column]
        )
    
    # Hourly temperature change rate
    feature_name = f"{temperature_column}_hourly_change"
    result_df[feature_name] = result_df[temperature_column].diff()
    register_weather_feature(
        feature_name,
        "Hourly temperature change",
        "float",
        {"source_column": temperature_column, "transformation": "diff_1h"},
        [temperature_column]
    )
    
    return result_df

@log_execution_time(logger, 'DEBUG')
def create_wind_features(
    df: DataFrameType,
    wind_speed_column: str,
    windows: List[int]
) -> DataFrameType:
    """
    Creates wind-related features from weather data.
    
    Args:
        df: DataFrame containing weather data
        wind_speed_column: Name of the column containing wind speed data
        windows: List of window sizes for rolling calculations (in hours)
        
    Returns:
        DataFrame with added wind features
    """
    # Validate that the wind speed column exists
    if wind_speed_column not in df.columns:
        logger.warning(f"Wind speed column '{wind_speed_column}' not found in DataFrame")
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Create basic wind features
    
    # Wind power potential (cubic relationship with wind speed)
    feature_name = f"{wind_speed_column}_power_potential"
    result_df[feature_name] = result_df[wind_speed_column] ** 3
    register_weather_feature(
        feature_name,
        "Wind power potential (cubic wind speed)",
        "float",
        {"source_column": wind_speed_column, "transformation": "cubic"},
        [wind_speed_column]
    )
    
    # Wind variability (absolute change from previous hour)
    feature_name = f"{wind_speed_column}_hourly_variability"
    result_df[feature_name] = result_df[wind_speed_column].diff().abs()
    register_weather_feature(
        feature_name,
        "Absolute hourly wind speed change",
        "float",
        {"source_column": wind_speed_column, "transformation": "abs_diff_1h"},
        [wind_speed_column]
    )
    
    # Create rolling statistics for each window size
    for window in windows:
        # Rolling mean
        feature_name = f"{wind_speed_column}_rolling_mean_{window}h"
        result_df[feature_name] = result_df[wind_speed_column].rolling(window=window, min_periods=1).mean()
        register_weather_feature(
            feature_name,
            f"Wind speed rolling mean over {window} hours",
            "float",
            {"source_column": wind_speed_column, "window": window, "statistic": "mean"},
            [wind_speed_column]
        )
        
        # Rolling standard deviation (volatility)
        feature_name = f"{wind_speed_column}_rolling_std_{window}h"
        result_df[feature_name] = result_df[wind_speed_column].rolling(window=window, min_periods=2).std()
        register_weather_feature(
            feature_name,
            f"Wind speed rolling standard deviation over {window} hours",
            "float",
            {"source_column": wind_speed_column, "window": window, "statistic": "std"},
            [wind_speed_column]
        )
        
        # Rolling min
        feature_name = f"{wind_speed_column}_rolling_min_{window}h"
        result_df[feature_name] = result_df[wind_speed_column].rolling(window=window, min_periods=1).min()
        register_weather_feature(
            feature_name,
            f"Wind speed rolling minimum over {window} hours",
            "float",
            {"source_column": wind_speed_column, "window": window, "statistic": "min"},
            [wind_speed_column]
        )
        
        # Rolling max
        feature_name = f"{wind_speed_column}_rolling_max_{window}h"
        result_df[feature_name] = result_df[wind_speed_column].rolling(window=window, min_periods=1).max()
        register_weather_feature(
            feature_name,
            f"Wind speed rolling maximum over {window} hours",
            "float",
            {"source_column": wind_speed_column, "window": window, "statistic": "max"},
            [wind_speed_column]
        )
    
    # Create wind ramp features (rapid changes in wind speed)
    # Wind ramp: maximum change over 3-hour window
    if len(result_df) >= 3:
        feature_name = f"{wind_speed_column}_ramp_3h"
        result_df[feature_name] = result_df[wind_speed_column].rolling(window=3).max() - result_df[wind_speed_column].rolling(window=3).min()
        register_weather_feature(
            feature_name,
            "Maximum wind speed change over 3 hours (ramp)",
            "float",
            {"source_column": wind_speed_column, "window": 3, "transformation": "max_min_diff"},
            [wind_speed_column]
        )
    
    # Wind change from 24 hours ago
    if len(result_df) > 24:
        feature_name = f"{wind_speed_column}_delta_24h"
        result_df[feature_name] = result_df[wind_speed_column].diff(periods=24)
        register_weather_feature(
            feature_name,
            "Wind speed change from 24 hours ago",
            "float",
            {"source_column": wind_speed_column, "transformation": "diff_24h"},
            [wind_speed_column]
        )
    
    return result_df

@log_execution_time(logger, 'DEBUG')
def create_solar_features(
    df: DataFrameType,
    solar_irradiance_column: str,
    humidity_column: str,
    windows: List[int]
) -> DataFrameType:
    """
    Creates solar-related features from weather data.
    
    Args:
        df: DataFrame containing weather data
        solar_irradiance_column: Name of the column containing solar irradiance data
        humidity_column: Name of the column containing humidity data
        windows: List of window sizes for rolling calculations (in hours)
        
    Returns:
        DataFrame with added solar features
    """
    # Validate that required columns exist
    required_columns = [solar_irradiance_column, humidity_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Required columns {missing_columns} not found in DataFrame")
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Create basic solar features
    
    # Solar power potential (considering humidity impact)
    feature_name = f"{solar_irradiance_column}_power_potential"
    # Simplified model: solar power decreases with humidity (humidity as a percentage)
    humidity_factor = (100 - result_df[humidity_column]) / 100
    result_df[feature_name] = result_df[solar_irradiance_column] * humidity_factor
    register_weather_feature(
        feature_name,
        "Solar power potential (accounting for humidity)",
        "float",
        {"source_columns": [solar_irradiance_column, humidity_column], "transformation": "humidity_adjusted"},
        [solar_irradiance_column, humidity_column]
    )
    
    # Solar variability (absolute change from previous hour)
    feature_name = f"{solar_irradiance_column}_hourly_variability"
    result_df[feature_name] = result_df[solar_irradiance_column].diff().abs()
    register_weather_feature(
        feature_name,
        "Absolute hourly solar irradiance change",
        "float",
        {"source_column": solar_irradiance_column, "transformation": "abs_diff_1h"},
        [solar_irradiance_column]
    )
    
    # Humidity impact on solar efficiency
    feature_name = f"humidity_solar_impact"
    # Higher humidity reduces solar panel efficiency
    result_df[feature_name] = result_df[humidity_column] / 100
    register_weather_feature(
        feature_name,
        "Humidity impact on solar efficiency (higher is worse)",
        "float",
        {"source_column": humidity_column, "transformation": "normalized"},
        [humidity_column]
    )
    
    # Create rolling statistics for solar irradiance
    for window in windows:
        # Rolling mean
        feature_name = f"{solar_irradiance_column}_rolling_mean_{window}h"
        result_df[feature_name] = result_df[solar_irradiance_column].rolling(window=window, min_periods=1).mean()
        register_weather_feature(
            feature_name,
            f"Solar irradiance rolling mean over {window} hours",
            "float",
            {"source_column": solar_irradiance_column, "window": window, "statistic": "mean"},
            [solar_irradiance_column]
        )
        
        # Rolling standard deviation (volatility)
        feature_name = f"{solar_irradiance_column}_rolling_std_{window}h"
        result_df[feature_name] = result_df[solar_irradiance_column].rolling(window=window, min_periods=2).std()
        register_weather_feature(
            feature_name,
            f"Solar irradiance rolling standard deviation over {window} hours",
            "float",
            {"source_column": solar_irradiance_column, "window": window, "statistic": "std"},
            [solar_irradiance_column]
        )
        
        # Rolling max
        feature_name = f"{solar_irradiance_column}_rolling_max_{window}h"
        result_df[feature_name] = result_df[solar_irradiance_column].rolling(window=window, min_periods=1).max()
        register_weather_feature(
            feature_name,
            f"Solar irradiance rolling maximum over {window} hours",
            "float",
            {"source_column": solar_irradiance_column, "window": window, "statistic": "max"},
            [solar_irradiance_column]
        )
    
    # Create solar ramp features (rapid changes in solar irradiance)
    if len(result_df) >= 3:
        feature_name = f"{solar_irradiance_column}_ramp_3h"
        result_df[feature_name] = result_df[solar_irradiance_column].rolling(window=3).max() - result_df[solar_irradiance_column].rolling(window=3).min()
        register_weather_feature(
            feature_name,
            "Maximum solar irradiance change over 3 hours (ramp)",
            "float",
            {"source_column": solar_irradiance_column, "window": 3, "transformation": "max_min_diff"},
            [solar_irradiance_column]
        )
    
    # Solar change from 24 hours ago (day-to-day comparison)
    if len(result_df) > 24:
        feature_name = f"{solar_irradiance_column}_delta_24h"
        result_df[feature_name] = result_df[solar_irradiance_column].diff(periods=24)
        register_weather_feature(
            feature_name,
            "Solar irradiance change from 24 hours ago",
            "float",
            {"source_column": solar_irradiance_column, "transformation": "diff_24h"},
            [solar_irradiance_column]
        )
    
    return result_df

@log_execution_time(logger, 'DEBUG')
def create_weather_interaction_features(
    df: DataFrameType,
    column_mapping: Dict[str, str]
) -> DataFrameType:
    """
    Creates interaction features between different weather variables.
    
    Args:
        df: DataFrame containing weather data
        column_mapping: Mapping of weather variables to column names
        
    Returns:
        DataFrame with added interaction features
    """
    # Validate that required columns exist
    available_columns = [col for var, col in column_mapping.items() if col in df.columns]
    if len(available_columns) < 2:
        logger.warning("At least two weather variables required for interaction features")
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Extract column names for readability
    temp_col = column_mapping.get('temperature')
    wind_col = column_mapping.get('wind_speed')
    solar_col = column_mapping.get('solar_irradiance')
    humidity_col = column_mapping.get('humidity')
    
    # Create temperature-wind interaction features
    if temp_col in result_df.columns and wind_col in result_df.columns:
        # Wind chill effect (simplified)
        feature_name = "wind_chill_effect"
        result_df[feature_name] = result_df[temp_col] - (0.5 * result_df[wind_col])
        register_weather_feature(
            feature_name,
            "Wind chill effect (simplified)",
            "float",
            {"source_columns": [temp_col, wind_col], "transformation": "wind_chill"},
            [temp_col, wind_col]
        )
        
        # Temperature and wind combined effect on load
        feature_name = "temp_wind_load_effect"
        result_df[feature_name] = result_df[temp_col] * result_df[wind_col]
        register_weather_feature(
            feature_name,
            "Temperature and wind combined effect on load",
            "float",
            {"source_columns": [temp_col, wind_col], "transformation": "product"},
            [temp_col, wind_col]
        )
    
    # Create temperature-solar interaction features
    if temp_col in result_df.columns and solar_col in result_df.columns:
        # Temperature and solar combined effect on AC load
        feature_name = "temp_solar_ac_load"
        result_df[feature_name] = result_df[temp_col] * result_df[solar_col]
        register_weather_feature(
            feature_name,
            "Temperature and solar combined effect on AC load",
            "float",
            {"source_columns": [temp_col, solar_col], "transformation": "product"},
            [temp_col, solar_col]
        )
    
    # Create wind-solar interaction features
    if wind_col in result_df.columns and solar_col in result_df.columns:
        # Renewable generation potential (simplified)
        feature_name = "renewable_generation_potential"
        # Normalize and combine (equally weighted for simplicity)
        wind_norm = result_df[wind_col] / result_df[wind_col].max() if result_df[wind_col].max() > 0 else 0
        solar_norm = result_df[solar_col] / result_df[solar_col].max() if result_df[solar_col].max() > 0 else 0
        result_df[feature_name] = 0.5 * wind_norm + 0.5 * solar_norm
        register_weather_feature(
            feature_name,
            "Combined renewable generation potential (wind and solar)",
            "float",
            {"source_columns": [wind_col, solar_col], "transformation": "normalized_average"},
            [wind_col, solar_col]
        )
        
        # Wind and solar variability interaction
        feature_name = "wind_solar_variability"
        wind_var = result_df[wind_col].diff().abs()
        solar_var = result_df[solar_col].diff().abs()
        # Normalize and combine
        wind_var_norm = wind_var / wind_var.max() if wind_var.max() > 0 else 0
        solar_var_norm = solar_var / solar_var.max() if solar_var.max() > 0 else 0
        result_df[feature_name] = wind_var_norm * solar_var_norm
        register_weather_feature(
            feature_name,
            "Combined wind and solar variability",
            "float",
            {"source_columns": [wind_col, solar_col], "transformation": "variability_product"},
            [wind_col, solar_col]
        )
    
    # Create humidity interaction features
    if humidity_col in result_df.columns:
        # Humidity and temperature interaction (heat index effect)
        if temp_col in result_df.columns:
            feature_name = "heat_index_effect"
            # Simplified heat index calculation
            result_df[feature_name] = result_df[temp_col] + (0.05 * result_df[humidity_col])
            register_weather_feature(
                feature_name,
                "Heat index effect (simplified)",
                "float",
                {"source_columns": [temp_col, humidity_col], "transformation": "heat_index"},
                [temp_col, humidity_col]
            )
        
        # Humidity and solar interaction (solar efficiency reduction)
        if solar_col in result_df.columns:
            feature_name = "humidity_solar_efficiency"
            result_df[feature_name] = result_df[solar_col] * (1 - (result_df[humidity_col] / 100))
            register_weather_feature(
                feature_name,
                "Humidity-adjusted solar efficiency",
                "float",
                {"source_columns": [solar_col, humidity_col], "transformation": "efficiency_reduction"},
                [solar_col, humidity_col]
            )
    
    return result_df

@log_execution_time(logger, 'DEBUG')
def create_location_aggregated_features(
    df: DataFrameType,
    location_column: str,
    column_mapping: Dict[str, str]
) -> DataFrameType:
    """
    Creates aggregated weather features across different locations.
    
    Args:
        df: DataFrame containing weather data
        location_column: Name of the column containing location identifiers
        column_mapping: Mapping of weather variables to column names
        
    Returns:
        DataFrame with added location-aggregated features
    """
    # Validate that required columns exist
    if location_column not in df.columns:
        logger.warning(f"Location column '{location_column}' not found in DataFrame")
        return df
    
    # Check if we have multiple locations
    unique_locations = df[location_column].nunique()
    if unique_locations <= 1:
        logger.warning(f"Only one location found in '{location_column}', skipping location aggregation")
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Get timestamp column (assuming DataFrame is indexed by timestamp)
    if pd.api.types.is_datetime64_any_dtype(result_df.index):
        # Use the index as timestamp
        timestamps = result_df.index
    else:
        # Try to find a timestamp column
        timestamp_col = next((col for col in result_df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
        if timestamp_col is None:
            logger.warning("No timestamp column found for location aggregation")
            return df
        timestamps = result_df[timestamp_col]
    
    # Process each weather variable
    for weather_var, column_name in column_mapping.items():
        if column_name not in result_df.columns:
            continue
        
        # Group by timestamp and calculate aggregated statistics
        if pd.api.types.is_datetime64_any_dtype(result_df.index):
            # Group by index
            grouped = result_df.groupby(result_df.index)
        else:
            # Group by timestamp column
            grouped = result_df.groupby(timestamps)
        
        # Calculate mean across locations
        feature_name = f"{column_name}_location_mean"
        location_mean = grouped[column_name].transform('mean')
        result_df[feature_name] = location_mean
        register_weather_feature(
            feature_name,
            f"Mean {weather_var} across all locations",
            "float",
            {"source_column": column_name, "aggregation": "mean", "dimension": "location"},
            [column_name]
        )
        
        # Calculate standard deviation across locations
        feature_name = f"{column_name}_location_std"
        location_std = grouped[column_name].transform('std')
        result_df[feature_name] = location_std
        register_weather_feature(
            feature_name,
            f"Standard deviation of {weather_var} across locations",
            "float",
            {"source_column": column_name, "aggregation": "std", "dimension": "location"},
            [column_name]
        )
        
        # Calculate min and max across locations
        feature_name = f"{column_name}_location_min"
        location_min = grouped[column_name].transform('min')
        result_df[feature_name] = location_min
        register_weather_feature(
            feature_name,
            f"Minimum {weather_var} across locations",
            "float",
            {"source_column": column_name, "aggregation": "min", "dimension": "location"},
            [column_name]
        )
        
        feature_name = f"{column_name}_location_max"
        location_max = grouped[column_name].transform('max')
        result_df[feature_name] = location_max
        register_weather_feature(
            feature_name,
            f"Maximum {weather_var} across locations",
            "float",
            {"source_column": column_name, "aggregation": "max", "dimension": "location"},
            [column_name]
        )
        
        # Calculate the range (max-min) across locations
        feature_name = f"{column_name}_location_range"
        result_df[feature_name] = location_max - location_min
        register_weather_feature(
            feature_name,
            f"Range of {weather_var} across locations",
            "float",
            {"source_column": column_name, "aggregation": "range", "dimension": "location"},
            [column_name]
        )
    
    return result_df

def get_weather_feature_names() -> List[str]:
    """
    Returns a list of all available weather feature names.
    
    Returns:
        List of weather feature names
    """
    return [feature["feature_id"] for feature in WEATHER_FEATURE_REGISTRY]

def get_weather_feature_registry() -> List[Dict[str, Any]]:
    """
    Returns the complete weather feature registry with metadata.
    
    Returns:
        List of weather feature metadata dictionaries
    """
    return WEATHER_FEATURE_REGISTRY.copy()

def register_weather_feature(
    feature_id: str,
    feature_name: str,
    data_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    dependencies: Optional[List[str]] = None
) -> bool:
    """
    Helper function to register a weather feature in the local registry.
    
    Args:
        feature_id: Unique identifier for the feature
        feature_name: Human-readable name for the feature
        data_type: Data type of the feature (e.g., 'float', 'int', 'boolean')
        metadata: Optional metadata about the feature
        dependencies: Optional list of column dependencies
        
    Returns:
        True if registration was successful
    """
    # Check if feature already exists
    for feature in WEATHER_FEATURE_REGISTRY:
        if feature["feature_id"] == feature_id:
            # Update existing feature
            feature.update({
                "feature_name": feature_name,
                "data_type": data_type,
                "metadata": metadata or {},
                "dependencies": dependencies or []
            })
            return True
    
    # Create new feature entry
    feature_entry = {
        "feature_id": feature_id,
        "feature_name": feature_name,
        "feature_group": "weather",
        "data_type": data_type,
        "metadata": metadata or {},
        "dependencies": dependencies or []
    }
    
    # Add to registry
    WEATHER_FEATURE_REGISTRY.append(feature_entry)
    logger.debug(f"Registered weather feature: {feature_id}")
    
    return True