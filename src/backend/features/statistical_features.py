"""
Module for generating statistical features from RTLMP time series data in the ERCOT RTLMP spike prediction system.

Implements various rolling statistics, volatility metrics, and price spike indicators that are critical for 
predicting price spikes in the RTLMP market.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from typing import List, Dict, Optional, Union, Tuple, Any

from ...utils.type_definitions import DataFrameType, SeriesType
from ...utils.logging import get_logger, log_execution_time
from ...utils.validation import validate_dataframe_schema
from ...utils.statistics import (
    calculate_rolling_statistics,
    calculate_price_volatility,
    calculate_spike_frequency,
    DEFAULT_ROLLING_WINDOWS
)

# Set up logger
logger = get_logger(__name__)

# Default thresholds for price spike detection
PRICE_SPIKE_THRESHOLDS = [50.0, 100.0, 200.0, 500.0, 1000.0]

# Custom window sizes for rolling calculations
CUSTOM_ROLLING_WINDOWS = [1, 6, 12, 24, 48, 72, 168]

# Global list of feature names
STATISTICAL_FEATURE_NAMES = []

# Global list of feature metadata
STATISTICAL_FEATURE_METADATA = []


@log_execution_time(logger, 'INFO')
def create_rolling_statistics_features(
    df: DataFrameType,
    price_column: str,
    windows: Optional[List[int]] = None,
    statistics: Optional[List[str]] = None
) -> DataFrameType:
    """
    Creates rolling statistical features from RTLMP price data.
    
    Args:
        df: DataFrame containing RTLMP price data
        price_column: Name of the column containing price values
        windows: List of window sizes (periods) for rolling calculations, defaults to CUSTOM_ROLLING_WINDOWS
        statistics: List of statistics to calculate, defaults to ['mean', 'std', 'min', 'max']
        
    Returns:
        DataFrame with added rolling statistics features
    """
    # Validate that price_column exists in the DataFrame
    if price_column not in df.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame")
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    # Use default values if not provided
    if windows is None:
        windows = CUSTOM_ROLLING_WINDOWS
    
    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max']
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Extract the price series from the DataFrame
    price_series = df[price_column]
    
    # Calculate rolling statistics using the utility function
    rolling_stats = calculate_rolling_statistics(price_series, windows, statistics)
    
    # Add rolling statistics to the result DataFrame
    for column in rolling_stats.columns:
        result_df[column] = rolling_stats[column]
    
    logger.info(f"Created {len(rolling_stats.columns)} rolling statistics features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_price_volatility_features(
    df: DataFrameType,
    price_column: str,
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Creates price volatility features from RTLMP price data.
    
    Args:
        df: DataFrame containing RTLMP price data
        price_column: Name of the column containing price values
        windows: List of window sizes (periods) for calculating volatility, defaults to CUSTOM_ROLLING_WINDOWS
        
    Returns:
        DataFrame with added price volatility features
    """
    # Validate that price_column exists in the DataFrame
    if price_column not in df.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame")
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    # Use default windows if not provided
    if windows is None:
        windows = CUSTOM_ROLLING_WINDOWS
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Extract the price series from the DataFrame
    price_series = df[price_column]
    
    # Calculate price volatility using the utility function
    volatility_features = calculate_price_volatility(price_series, windows)
    
    # Add volatility features to the result DataFrame
    for column in volatility_features.columns:
        result_df[column] = volatility_features[column]
    
    logger.info(f"Created {len(volatility_features.columns)} price volatility features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_price_spike_indicator_features(
    df: DataFrameType,
    price_column: str,
    thresholds: Optional[List[float]] = None
) -> DataFrameType:
    """
    Creates binary indicators for price spikes above specified thresholds.
    
    Args:
        df: DataFrame containing RTLMP price data
        price_column: Name of the column containing price values
        thresholds: List of threshold values for defining price spikes, defaults to PRICE_SPIKE_THRESHOLDS
        
    Returns:
        DataFrame with added price spike indicator features
    """
    # Validate that price_column exists in the DataFrame
    if price_column not in df.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame")
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    # Use default thresholds if not provided
    if thresholds is None:
        thresholds = PRICE_SPIKE_THRESHOLDS
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Create binary indicator for each threshold
    for threshold in thresholds:
        column_name = f'price_spike_{threshold}'
        result_df[column_name] = (df[price_column] > threshold).astype(int)
    
    logger.info(f"Created {len(thresholds)} price spike indicator features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_price_spike_frequency_features(
    df: DataFrameType,
    price_column: str,
    thresholds: Optional[List[float]] = None,
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Creates features for the frequency of price spikes over rolling windows.
    
    Args:
        df: DataFrame containing RTLMP price data
        price_column: Name of the column containing price values
        thresholds: List of threshold values for defining price spikes, defaults to PRICE_SPIKE_THRESHOLDS
        windows: List of window sizes (periods) for calculating frequencies, defaults to CUSTOM_ROLLING_WINDOWS
        
    Returns:
        DataFrame with added price spike frequency features
    """
    # Validate that price_column exists in the DataFrame
    if price_column not in df.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame")
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    # Use default values if not provided
    if thresholds is None:
        thresholds = PRICE_SPIKE_THRESHOLDS
    
    if windows is None:
        windows = CUSTOM_ROLLING_WINDOWS
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Extract the price series from the DataFrame
    price_series = df[price_column]
    
    # Calculate spike frequency using the utility function
    spike_freq_features = calculate_spike_frequency(price_series, thresholds, windows)
    
    # Add spike frequency features to the result DataFrame
    for column in spike_freq_features.columns:
        result_df[column] = spike_freq_features[column]
    
    logger.info(f"Created {len(spike_freq_features.columns)} price spike frequency features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_hourly_price_features(
    df: DataFrameType,
    price_column: str,
    timestamp_column: str
) -> DataFrameType:
    """
    Aggregates 5-minute RTLMP data to create hourly price features.
    
    Args:
        df: DataFrame containing 5-minute RTLMP data
        price_column: Name of the column containing price values
        timestamp_column: Name of the column containing timestamps
        
    Returns:
        DataFrame with hourly price features
    """
    # Validate that required columns exist in the DataFrame
    if price_column not in df.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame")
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    if timestamp_column not in df.columns:
        logger.error(f"Timestamp column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_column]):
        logger.info(f"Converting {timestamp_column} to datetime type")
        result_df[timestamp_column] = pd.to_datetime(result_df[timestamp_column])
    
    # Create hour start timestamps by flooring to the hour
    result_df['hour_start'] = result_df[timestamp_column].dt.floor('H')
    
    # Group data by hour start timestamps
    grouped = result_df.groupby('hour_start')
    
    # Calculate hourly statistics
    hourly_stats = pd.DataFrame({
        'hourly_mean': grouped[price_column].mean(),
        'hourly_min': grouped[price_column].min(),
        'hourly_max': grouped[price_column].max(),
        'hourly_std': grouped[price_column].std(),
        'hourly_range': grouped[price_column].max() - grouped[price_column].min(),
        'hourly_count': grouped[price_column].count()
    })
    
    # Reset index to convert groupby result back to DataFrame
    hourly_stats.reset_index(inplace=True)
    
    # Merge hourly statistics back to the original DataFrame
    result_df = result_df.merge(hourly_stats, on='hour_start', how='left')
    
    # Drop the temporary column
    result_df.drop('hour_start', axis=1, inplace=True)
    
    logger.info(f"Created 6 hourly price features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_price_difference_features(
    df: DataFrameType,
    price_column: str,
    periods: Optional[List[int]] = None
) -> DataFrameType:
    """
    Creates features based on price differences between consecutive periods.
    
    Args:
        df: DataFrame containing RTLMP price data
        price_column: Name of the column containing price values
        periods: List of periods (lags) for calculating differences, defaults to [1, 6, 12, 24]
        
    Returns:
        DataFrame with added price difference features
    """
    # Validate that price_column exists in the DataFrame
    if price_column not in df.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame")
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    # Use default periods if not provided
    if periods is None:
        periods = [1, 6, 12, 24]
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Calculate price differences for each period
    for period in periods:
        # Calculate absolute difference
        diff_col = f'price_diff_{period}'
        result_df[diff_col] = df[price_column].diff(periods=period)
        
        # Calculate percentage change
        pct_col = f'price_pct_change_{period}'
        result_df[pct_col] = df[price_column].pct_change(periods=period)
    
    # Handle NaN values created by diff and pct_change
    result_df.fillna(0, inplace=True)
    
    logger.info(f"Created {len(periods) * 2} price difference features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_all_statistical_features(
    df: DataFrameType,
    price_column: str,
    timestamp_column: str,
    feature_types: Optional[List[str]] = None
) -> DataFrameType:
    """
    Creates all statistical features from RTLMP price data.
    
    Args:
        df: DataFrame containing RTLMP price data
        price_column: Name of the column containing price values
        timestamp_column: Name of the column containing timestamps
        feature_types: List of feature types to create, defaults to all available types
        
    Returns:
        DataFrame with all requested statistical features
    """
    # Validate that required columns exist in the DataFrame
    if price_column not in df.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame")
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    if timestamp_column not in df.columns:
        logger.error(f"Timestamp column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
    
    # If feature_types is None, include all available feature types
    if feature_types is None:
        feature_types = [
            'rolling_statistics',
            'volatility',
            'spike_indicators',
            'spike_frequency',
            'hourly_price',
            'price_difference'
        ]
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Create features for each requested type
    feature_count = 0
    
    if 'rolling_statistics' in feature_types:
        result_df = create_rolling_statistics_features(result_df, price_column)
        feature_count += len(result_df.columns) - len(df.columns)
    
    if 'volatility' in feature_types:
        result_df = create_price_volatility_features(result_df, price_column)
        feature_count += len(result_df.columns) - (len(df.columns) + feature_count)
    
    if 'spike_indicators' in feature_types:
        result_df = create_price_spike_indicator_features(result_df, price_column)
        feature_count += len(result_df.columns) - (len(df.columns) + feature_count)
    
    if 'spike_frequency' in feature_types:
        result_df = create_price_spike_frequency_features(result_df, price_column)
        feature_count += len(result_df.columns) - (len(df.columns) + feature_count)
    
    if 'hourly_price' in feature_types:
        result_df = create_hourly_price_features(result_df, price_column, timestamp_column)
        feature_count += len(result_df.columns) - (len(df.columns) + feature_count)
    
    if 'price_difference' in feature_types:
        result_df = create_price_difference_features(result_df, price_column)
        feature_count += len(result_df.columns) - (len(df.columns) + feature_count)
    
    logger.info(f"Created a total of {feature_count} statistical features")
    
    return result_df


def get_statistical_feature_names() -> List[str]:
    """
    Returns a list of all available statistical feature names.
    
    Returns:
        List of statistical feature names
    """
    return STATISTICAL_FEATURE_NAMES


def get_statistical_feature_metadata() -> List[Dict[str, Any]]:
    """
    Returns metadata for all statistical features for registration purposes.
    
    Returns:
        List of feature metadata dictionaries
    """
    # Initialize metadata list if empty
    if not STATISTICAL_FEATURE_METADATA:
        initialize_statistical_features()
    
    return STATISTICAL_FEATURE_METADATA


@log_execution_time(logger, 'INFO')
def initialize_statistical_features() -> int:
    """
    Initializes statistical feature definitions and metadata for use by other modules.
    
    Returns:
        Number of features initialized
    """
    global STATISTICAL_FEATURE_NAMES, STATISTICAL_FEATURE_METADATA
    
    # Clear existing lists if they contain data
    STATISTICAL_FEATURE_NAMES = []
    STATISTICAL_FEATURE_METADATA = []
    
    # Add rolling statistics feature metadata
    for window in CUSTOM_ROLLING_WINDOWS:
        for stat in ['mean', 'std', 'min', 'max', 'median']:
            feature_name = f'rolling_{stat}_{window}h'
            STATISTICAL_FEATURE_METADATA.append({
                'name': feature_name,
                'type': 'numeric',
                'group': 'statistical',
                'subgroup': 'rolling_statistics',
                'description': f'Rolling {stat} of RTLMP over {window}h window',
                'window': window,
                'statistic': stat
            })
    
    # Add price volatility feature metadata
    for window in CUSTOM_ROLLING_WINDOWS:
        for metric in ['volatility', 'normalized_range', 'cv']:
            feature_name = f'{metric}_{window}h'
            STATISTICAL_FEATURE_METADATA.append({
                'name': feature_name,
                'type': 'numeric',
                'group': 'statistical',
                'subgroup': 'volatility',
                'description': f'{metric.replace("_", " ")} over {window}h window',
                'window': window
            })
    
    # Add price spike indicator feature metadata
    for threshold in PRICE_SPIKE_THRESHOLDS:
        feature_name = f'price_spike_{threshold}'
        STATISTICAL_FEATURE_METADATA.append({
            'name': feature_name,
            'type': 'binary',
            'group': 'statistical',
            'subgroup': 'spike_indicators',
            'description': f'Binary indicator for price > {threshold}',
            'threshold': threshold
        })
    
    # Add price spike frequency feature metadata
    for threshold in PRICE_SPIKE_THRESHOLDS:
        for window in CUSTOM_ROLLING_WINDOWS:
            feature_name = f'spike_freq_{threshold}_{window}h'
            STATISTICAL_FEATURE_METADATA.append({
                'name': feature_name,
                'type': 'numeric',
                'group': 'statistical',
                'subgroup': 'spike_frequency',
                'description': f'Frequency of prices > {threshold} in {window}h window',
                'threshold': threshold,
                'window': window
            })
    
    # Add hourly price feature metadata
    for stat in ['mean', 'min', 'max', 'std', 'range', 'count']:
        feature_name = f'hourly_{stat}'
        STATISTICAL_FEATURE_METADATA.append({
            'name': feature_name,
            'type': 'numeric',
            'group': 'statistical',
            'subgroup': 'hourly_price',
            'description': f'Hourly {stat} of 5-minute RTLMP values',
            'statistic': stat
        })
    
    # Add price difference feature metadata
    for period in [1, 6, 12, 24]:
        # Absolute difference
        feature_name = f'price_diff_{period}'
        STATISTICAL_FEATURE_METADATA.append({
            'name': feature_name,
            'type': 'numeric',
            'group': 'statistical',
            'subgroup': 'price_difference',
            'description': f'Price difference with {period} periods lag',
            'period': period
        })
        
        # Percentage change
        feature_name = f'price_pct_change_{period}'
        STATISTICAL_FEATURE_METADATA.append({
            'name': feature_name,
            'type': 'numeric',
            'group': 'statistical',
            'subgroup': 'price_difference',
            'description': f'Price percentage change with {period} periods lag',
            'period': period
        })
    
    # Update feature names list from metadata
    STATISTICAL_FEATURE_NAMES = [feature['name'] for feature in STATISTICAL_FEATURE_METADATA]
    
    logger.info(f"Initialized {len(STATISTICAL_FEATURE_NAMES)} statistical features")
    
    return len(STATISTICAL_FEATURE_NAMES)