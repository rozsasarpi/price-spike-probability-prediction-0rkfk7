"""
Module for generating time-based features from timestamp data in the ERCOT RTLMP spike prediction system.

This module provides functions to extract temporal patterns such as hour of day, day of week, 
month, season, and holiday indicators that are critical for predicting price spikes.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from typing import List, Dict, Optional, Union, Tuple, Any
import datetime
import holidays  # version 0.25+

from ..utils.type_definitions import DataFrameType
from ..utils.logging import get_logger, log_execution_time
from ..utils.date_utils import ERCOT_TIMEZONE, get_datetime_components

# Set up logger
logger = get_logger(__name__)

# Initialize US holidays
US_HOLIDAYS = holidays.US()

# Define season mapping
SEASON_MAPPING = {
    1: 'winter',
    2: 'winter',
    3: 'spring',
    4: 'spring',
    5: 'spring',
    6: 'summer',
    7: 'summer',
    8: 'summer',
    9: 'fall',
    10: 'fall',
    11: 'fall',
    12: 'winter'
}

# List of available time features
TIME_FEATURE_NAMES = [
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'month',
    'season',
    'is_holiday'
]

# Metadata for time features
TIME_FEATURE_METADATA = {
    'hour_of_day': {
        'feature_name': 'Hour of Day',
        'data_type': 'int',
        'range': [0, 23],
        'description': 'Hour of day (0-23)'
    },
    'day_of_week': {
        'feature_name': 'Day of Week',
        'data_type': 'int',
        'range': [0, 6],
        'description': 'Day of week (0=Monday, 6=Sunday)'
    },
    'is_weekend': {
        'feature_name': 'Is Weekend',
        'data_type': 'bool',
        'range': [0, 1],
        'description': 'Binary indicator for weekend days'
    },
    'month': {
        'feature_name': 'Month',
        'data_type': 'int',
        'range': [1, 12],
        'description': 'Month of year (1-12)'
    },
    'season': {
        'feature_name': 'Season',
        'data_type': 'category',
        'categories': ['winter', 'spring', 'summer', 'fall'],
        'description': 'Season of year'
    },
    'is_holiday': {
        'feature_name': 'Is Holiday',
        'data_type': 'bool',
        'range': [0, 1],
        'description': 'Binary indicator for US holidays'
    }
}


def extract_hour_of_day(df: DataFrameType, timestamp_column: str) -> DataFrameType:
    """
    Extracts the hour of day (0-23) from a timestamp column.

    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column

    Returns:
        DataFrame with added hour_of_day column
    """
    # Validate that timestamp_column exists in the DataFrame
    if timestamp_column not in df.columns:
        logger.error(f"Column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        logger.warning(f"Column '{timestamp_column}' is not datetime type. Attempting conversion.")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Extract hour component from timestamp
    df['hour_of_day'] = df[timestamp_column].dt.hour
    
    logger.debug(f"Added 'hour_of_day' feature with range [{df['hour_of_day'].min()}, {df['hour_of_day'].max()}]")
    
    return df


def extract_day_of_week(df: DataFrameType, timestamp_column: str) -> DataFrameType:
    """
    Extracts the day of week (0=Monday, 6=Sunday) from a timestamp column.

    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column

    Returns:
        DataFrame with added day_of_week column
    """
    # Validate that timestamp_column exists in the DataFrame
    if timestamp_column not in df.columns:
        logger.error(f"Column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        logger.warning(f"Column '{timestamp_column}' is not datetime type. Attempting conversion.")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Extract day of week component from timestamp
    df['day_of_week'] = df[timestamp_column].dt.dayofweek
    
    logger.debug(f"Added 'day_of_week' feature with range [{df['day_of_week'].min()}, {df['day_of_week'].max()}]")
    
    return df


def extract_is_weekend(df: DataFrameType, timestamp_column: str) -> DataFrameType:
    """
    Creates a binary indicator for weekends (1=weekend, 0=weekday) from a timestamp column.

    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column

    Returns:
        DataFrame with added is_weekend column
    """
    # Validate that timestamp_column exists in the DataFrame
    if timestamp_column not in df.columns:
        logger.error(f"Column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        logger.warning(f"Column '{timestamp_column}' is not datetime type. Attempting conversion.")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Create binary indicator where Saturday and Sunday are 1, other days are 0
    df['is_weekend'] = (df[timestamp_column].dt.dayofweek >= 5).astype(int)
    
    logger.debug(f"Added 'is_weekend' feature with distribution: {df['is_weekend'].value_counts()}")
    
    return df


def extract_month(df: DataFrameType, timestamp_column: str) -> DataFrameType:
    """
    Extracts the month (1-12) from a timestamp column.

    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column

    Returns:
        DataFrame with added month column
    """
    # Validate that timestamp_column exists in the DataFrame
    if timestamp_column not in df.columns:
        logger.error(f"Column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        logger.warning(f"Column '{timestamp_column}' is not datetime type. Attempting conversion.")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Extract month component from timestamp
    df['month'] = df[timestamp_column].dt.month
    
    logger.debug(f"Added 'month' feature with range [{df['month'].min()}, {df['month'].max()}]")
    
    return df


def extract_season(df: DataFrameType, timestamp_column: str) -> DataFrameType:
    """
    Creates a categorical season feature (winter, spring, summer, fall) from a timestamp column.

    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column

    Returns:
        DataFrame with added season column
    """
    # Validate that timestamp_column exists in the DataFrame
    if timestamp_column not in df.columns:
        logger.error(f"Column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        logger.warning(f"Column '{timestamp_column}' is not datetime type. Attempting conversion.")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Extract month component from timestamp and map to season
    months = df[timestamp_column].dt.month
    df['season'] = months.map(SEASON_MAPPING)
    
    # Convert to categorical type
    df['season'] = df['season'].astype('category')
    
    logger.debug(f"Added 'season' feature with categories: {df['season'].cat.categories.tolist()}")
    
    return df


def extract_is_holiday(df: DataFrameType, timestamp_column: str) -> DataFrameType:
    """
    Creates a binary indicator for US holidays (1=holiday, 0=non-holiday) from a timestamp column.

    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column

    Returns:
        DataFrame with added is_holiday column
    """
    # Validate that timestamp_column exists in the DataFrame
    if timestamp_column not in df.columns:
        logger.error(f"Column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        logger.warning(f"Column '{timestamp_column}' is not datetime type. Attempting conversion.")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Extract date component from timestamp and check against US_HOLIDAYS
    df['is_holiday'] = 0  # Initialize all days as non-holidays
    
    # Apply holiday check to each date
    holiday_dates = df[timestamp_column].dt.date.apply(lambda x: x in US_HOLIDAYS)
    df.loc[holiday_dates, 'is_holiday'] = 1
    
    logger.debug(f"Added 'is_holiday' feature with distribution: {df['is_holiday'].value_counts()}")
    
    return df


@log_execution_time(logger, 'INFO')
def create_all_time_features(df: DataFrameType, timestamp_column: str, features: Optional[List[str]] = None) -> DataFrameType:
    """
    Creates all time-based features from a timestamp column.

    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column
        features: List of specific time features to create (default: all available features)

    Returns:
        DataFrame with all requested time features
    """
    # Validate that timestamp_column exists in the DataFrame
    if timestamp_column not in df.columns:
        logger.error(f"Column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        logger.warning(f"Column '{timestamp_column}' is not datetime type. Attempting conversion.")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # If features is None, use all available time features
    if features is None:
        features = TIME_FEATURE_NAMES
    else:
        # Validate requested features
        invalid_features = [f for f in features if f not in TIME_FEATURE_NAMES]
        if invalid_features:
            logger.warning(f"Invalid feature names requested: {invalid_features}. Using available features only.")
            features = [f for f in features if f in TIME_FEATURE_NAMES]

    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # For each requested feature, call the corresponding extraction function
    if 'hour_of_day' in features:
        result_df = extract_hour_of_day(result_df, timestamp_column)
    
    if 'day_of_week' in features:
        result_df = extract_day_of_week(result_df, timestamp_column)
    
    if 'is_weekend' in features:
        result_df = extract_is_weekend(result_df, timestamp_column)
    
    if 'month' in features:
        result_df = extract_month(result_df, timestamp_column)
    
    if 'season' in features:
        result_df = extract_season(result_df, timestamp_column)
    
    if 'is_holiday' in features:
        result_df = extract_is_holiday(result_df, timestamp_column)
    
    logger.info(f"Created {len(features)} time features from column '{timestamp_column}'")
    
    return result_df


def get_time_feature_names() -> List[str]:
    """
    Returns a list of all available time feature names.

    Returns:
        List of time feature names
    """
    return TIME_FEATURE_NAMES


def get_time_feature_metadata(feature_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Returns metadata for all time features or a specific time feature.

    Args:
        feature_name: Name of the specific feature to get metadata for (default: None for all features)

    Returns:
        Dictionary with feature metadata
    """
    if feature_name is None:
        return TIME_FEATURE_METADATA
    
    if feature_name in TIME_FEATURE_METADATA:
        return {feature_name: TIME_FEATURE_METADATA[feature_name]}
    
    logger.warning(f"Feature '{feature_name}' not found in time features metadata")
    return {}