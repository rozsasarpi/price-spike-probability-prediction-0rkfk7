"""
Module for generating market-related features from ERCOT RTLMP and grid condition data.

This module implements features based on congestion components, grid conditions, reserve margins,
and generation mix that are critical for predicting price spikes in the RTLMP market.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from typing import List, Dict, Optional, Union, Tuple, Any

from ..utils.type_definitions import DataFrameType, SeriesType
from ..utils.logging import get_logger, log_execution_time
from ..utils.validation import validate_dataframe_schema
from ..utils.statistics import calculate_rolling_statistics, calculate_crosscorrelation, DEFAULT_ROLLING_WINDOWS

# Set up logger
logger = get_logger(__name__)

# Default window sizes for rolling calculations
DEFAULT_WINDOWS = [1, 6, 12, 24, 48, 72, 168]  # hours

# Global list to store market feature names and metadata
MARKET_FEATURE_NAMES = []
MARKET_FEATURE_METADATA = []


@log_execution_time(logger, 'INFO')
def create_congestion_features(
    rtlmp_df: DataFrameType,
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Creates features based on congestion price components from RTLMP data.
    
    Args:
        rtlmp_df: DataFrame containing RTLMP data with price and congestion components
        windows: List of window sizes for rolling calculations, defaults to DEFAULT_WINDOWS
        
    Returns:
        DataFrame with added congestion-related features
    """
    # Validate required columns
    required_columns = ['price', 'congestion_price', 'loss_price', 'energy_price']
    if not all(col in rtlmp_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in rtlmp_df.columns]
        logger.warning(f"Missing required columns for congestion features: {missing}")
        return rtlmp_df.copy()
    
    # Use default windows if not provided
    if windows is None:
        windows = DEFAULT_WINDOWS
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = rtlmp_df.copy()
    
    # Calculate congestion, loss, and energy ratios
    result_df['congestion_ratio'] = result_df['congestion_price'] / result_df['price'].replace(0, np.nan)
    result_df['loss_ratio'] = result_df['loss_price'] / result_df['price'].replace(0, np.nan)
    result_df['energy_ratio'] = result_df['energy_price'] / result_df['price'].replace(0, np.nan)
    
    # Fill NaN values with 0 for better handling
    result_df[['congestion_ratio', 'loss_ratio', 'energy_ratio']] = result_df[
        ['congestion_ratio', 'loss_ratio', 'energy_ratio']
    ].fillna(0)
    
    # Calculate rolling statistics for congestion ratio
    for window in windows:
        # Rolling mean of congestion ratio
        result_df[f'congestion_ratio_mean_{window}h'] = (
            result_df['congestion_ratio'].rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling standard deviation of congestion ratio (volatility)
        result_df[f'congestion_ratio_std_{window}h'] = (
            result_df['congestion_ratio'].rolling(window=window, min_periods=1).std()
        )
        
        # Rolling max of congestion ratio
        result_df[f'congestion_ratio_max_{window}h'] = (
            result_df['congestion_ratio'].rolling(window=window, min_periods=1).max()
        )
        
        # Similar statistics for loss ratio
        result_df[f'loss_ratio_mean_{window}h'] = (
            result_df['loss_ratio'].rolling(window=window, min_periods=1).mean()
        )
        
        # Similar statistics for energy ratio
        result_df[f'energy_ratio_mean_{window}h'] = (
            result_df['energy_ratio'].rolling(window=window, min_periods=1).mean()
        )
    
    # Calculate congestion volatility (24-hour rolling std of congestion ratio)
    result_df['congestion_volatility_24h'] = (
        result_df['congestion_ratio'].rolling(window=24, min_periods=1).std()
    )
    
    feature_count = len(result_df.columns) - len(rtlmp_df.columns)
    logger.info(f"Created {feature_count} congestion-related features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_grid_condition_features(
    grid_df: DataFrameType,
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Creates features based on grid condition data including load, capacity, and generation mix.
    
    Args:
        grid_df: DataFrame containing grid condition data
        windows: List of window sizes for rolling calculations, defaults to DEFAULT_WINDOWS
        
    Returns:
        DataFrame with added grid condition features
    """
    # Validate required columns
    required_columns = ['total_load', 'available_capacity', 'wind_generation', 'solar_generation']
    if not all(col in grid_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in grid_df.columns]
        logger.warning(f"Missing required columns for grid condition features: {missing}")
        return grid_df.copy()
    
    # Use default windows if not provided
    if windows is None:
        windows = DEFAULT_WINDOWS
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = grid_df.copy()
    
    # Calculate reserve margin (as a percentage)
    result_df['reserve_margin'] = (
        (result_df['available_capacity'] - result_df['total_load']) / 
        result_df['total_load'].replace(0, np.nan) * 100
    )
    
    # Calculate renewable penetration (as a percentage)
    result_df['renewable_penetration'] = (
        (result_df['wind_generation'] + result_df['solar_generation']) / 
        result_df['total_load'].replace(0, np.nan) * 100
    )
    
    # Fill NaN values with 0 for better handling
    result_df[['reserve_margin', 'renewable_penetration']] = result_df[
        ['reserve_margin', 'renewable_penetration']
    ].fillna(0)
    
    # Calculate rolling statistics for key metrics
    for window in windows:
        # Reserve margin statistics
        result_df[f'reserve_margin_mean_{window}h'] = (
            result_df['reserve_margin'].rolling(window=window, min_periods=1).mean()
        )
        
        result_df[f'reserve_margin_min_{window}h'] = (
            result_df['reserve_margin'].rolling(window=window, min_periods=1).min()
        )
        
        # Renewable penetration statistics
        result_df[f'renewable_penetration_mean_{window}h'] = (
            result_df['renewable_penetration'].rolling(window=window, min_periods=1).mean()
        )
        
        result_df[f'renewable_penetration_std_{window}h'] = (
            result_df['renewable_penetration'].rolling(window=window, min_periods=1).std()
        )
        
        # Load statistics
        result_df[f'load_mean_{window}h'] = (
            result_df['total_load'].rolling(window=window, min_periods=1).mean()
        )
        
        result_df[f'load_max_{window}h'] = (
            result_df['total_load'].rolling(window=window, min_periods=1).max()
        )
    
    # Calculate load ramp rate (hourly rate of change)
    result_df['load_ramp_rate'] = result_df['total_load'].diff() / result_df['total_load'].shift(1) * 100
    
    # Calculate renewable variability (24-hour rolling std of renewable penetration)
    result_df['renewable_variability_24h'] = (
        result_df['renewable_penetration'].rolling(window=24, min_periods=1).std()
    )
    
    feature_count = len(result_df.columns) - len(grid_df.columns)
    logger.info(f"Created {feature_count} grid condition features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_market_correlation_features(
    rtlmp_df: DataFrameType,
    grid_df: DataFrameType,
    lags: Optional[List[int]] = None
) -> DataFrameType:
    """
    Creates features based on correlations between RTLMP prices and grid conditions.
    
    Args:
        rtlmp_df: DataFrame containing RTLMP price data
        grid_df: DataFrame containing grid condition data
        lags: List of lag values for correlation calculations, defaults to 
              [-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24]
              
    Returns:
        DataFrame with added correlation features
    """
    # Validate required columns
    if 'price' not in rtlmp_df.columns:
        logger.warning("Missing 'price' column in RTLMP data for correlation features")
        return rtlmp_df.copy()
    
    grid_required_columns = ['total_load', 'reserve_margin', 'renewable_penetration']
    missing_grid_columns = [
        col for col in grid_required_columns 
        if col not in grid_df.columns and f"{col}_mean_24h" not in grid_df.columns
    ]
    
    if missing_grid_columns:
        logger.warning(f"Missing required grid columns for correlation features: {missing_grid_columns}")
        # If reserve_margin or renewable_penetration are missing, try to calculate them
        grid_df = grid_df.copy()
        
        if ('reserve_margin' not in grid_df.columns and 
            'total_load' in grid_df.columns and 
            'available_capacity' in grid_df.columns):
            grid_df['reserve_margin'] = (
                (grid_df['available_capacity'] - grid_df['total_load']) / 
                grid_df['total_load'].replace(0, np.nan) * 100
            ).fillna(0)
        
        if ('renewable_penetration' not in grid_df.columns and 
            'total_load' in grid_df.columns and 
            'wind_generation' in grid_df.columns and 
            'solar_generation' in grid_df.columns):
            grid_df['renewable_penetration'] = (
                (grid_df['wind_generation'] + grid_df['solar_generation']) / 
                grid_df['total_load'].replace(0, np.nan) * 100
            ).fillna(0)
    
    # Use default lags if not provided
    if lags is None:
        lags = [-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24]
    
    # Create a copy of the RTLMP DataFrame to avoid modifying the original
    result_df = rtlmp_df.copy()
    
    # Ensure both DataFrames have the same index or can be aligned
    if not rtlmp_df.index.equals(grid_df.index):
        logger.warning("RTLMP and grid DataFrames have different indices, aligning by index")
        # Get the intersection of indices
        common_index = rtlmp_df.index.intersection(grid_df.index)
        if len(common_index) == 0:
            logger.error("No common timestamps found between RTLMP and grid data")
            return result_df
        
        # Filter both DataFrames to common indices
        price_series = rtlmp_df.loc[common_index, 'price']
        
        # Extract relevant grid series, handle missing columns
        if 'total_load' in grid_df.columns:
            load_series = grid_df.loc[common_index, 'total_load']
        elif 'load_mean_24h' in grid_df.columns:
            load_series = grid_df.loc[common_index, 'load_mean_24h']
        else:
            load_series = None
        
        if 'reserve_margin' in grid_df.columns:
            reserve_series = grid_df.loc[common_index, 'reserve_margin']
        elif 'reserve_margin_mean_24h' in grid_df.columns:
            reserve_series = grid_df.loc[common_index, 'reserve_margin_mean_24h']
        else:
            reserve_series = None
        
        if 'renewable_penetration' in grid_df.columns:
            renewable_series = grid_df.loc[common_index, 'renewable_penetration']
        elif 'renewable_penetration_mean_24h' in grid_df.columns:
            renewable_series = grid_df.loc[common_index, 'renewable_penetration_mean_24h']
        else:
            renewable_series = None
    else:
        # Extract relevant series from the DataFrames
        price_series = rtlmp_df['price']
        
        # Handle potentially missing columns
        load_series = (
            grid_df['total_load'] if 'total_load' in grid_df.columns 
            else grid_df.get('load_mean_24h', None)
        )
        
        reserve_series = (
            grid_df['reserve_margin'] if 'reserve_margin' in grid_df.columns 
            else grid_df.get('reserve_margin_mean_24h', None)
        )
        
        renewable_series = (
            grid_df['renewable_penetration'] if 'renewable_penetration' in grid_df.columns 
            else grid_df.get('renewable_penetration_mean_24h', None)
        )
    
    # Calculate cross-correlations if series are available
    feature_count = 0
    
    if load_series is not None:
        # Calculate cross-correlation between price and load at different lags
        for lag in lags:
            col_name = f'price_load_corr_lag{lag}'
            shifted_load = load_series.shift(lag)
            valid_indices = price_series.notna() & shifted_load.notna()
            
            if valid_indices.sum() > 1:  # Need at least 2 points for correlation
                result_df[col_name] = (
                    price_series.loc[valid_indices].corr(shifted_load.loc[valid_indices])
                )
                feature_count += 1
            else:
                result_df[col_name] = 0
    
    if reserve_series is not None:
        # Calculate cross-correlation between price and reserve margin at different lags
        for lag in lags:
            col_name = f'price_reserve_corr_lag{lag}'
            shifted_reserve = reserve_series.shift(lag)
            valid_indices = price_series.notna() & shifted_reserve.notna()
            
            if valid_indices.sum() > 1:  # Need at least 2 points for correlation
                result_df[col_name] = (
                    price_series.loc[valid_indices].corr(shifted_reserve.loc[valid_indices])
                )
                feature_count += 1
            else:
                result_df[col_name] = 0
    
    if renewable_series is not None:
        # Calculate cross-correlation between price and renewable penetration at different lags
        for lag in lags:
            col_name = f'price_renewable_corr_lag{lag}'
            shifted_renewable = renewable_series.shift(lag)
            valid_indices = price_series.notna() & shifted_renewable.notna()
            
            if valid_indices.sum() > 1:  # Need at least 2 points for correlation
                result_df[col_name] = (
                    price_series.loc[valid_indices].corr(shifted_renewable.loc[valid_indices])
                )
                feature_count += 1
            else:
                result_df[col_name] = 0
    
    logger.info(f"Created {feature_count} market correlation features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_day_ahead_features(
    rtlmp_df: DataFrameType,
    dam_df: DataFrameType
) -> DataFrameType:
    """
    Creates features based on day-ahead market prices and their relationship to real-time prices.
    
    Args:
        rtlmp_df: DataFrame containing RTLMP price data
        dam_df: DataFrame containing day-ahead market price data
        
    Returns:
        DataFrame with added day-ahead market features
    """
    # Validate required columns
    if 'price' not in rtlmp_df.columns:
        logger.warning("Missing 'price' column in RTLMP data for day-ahead features")
        return rtlmp_df.copy()
    
    if 'price' not in dam_df.columns and 'dam_price' not in dam_df.columns:
        logger.warning("Missing price column in day-ahead data")
        return rtlmp_df.copy()
    
    # Create a copy of the RTLMP DataFrame to avoid modifying the original
    result_df = rtlmp_df.copy()
    
    # Identify the day-ahead price column name
    dam_price_col = 'dam_price' if 'dam_price' in dam_df.columns else 'price'
    
    # Ensure both DataFrames have the same index or can be aligned
    if not rtlmp_df.index.equals(dam_df.index):
        logger.warning("RTLMP and day-ahead DataFrames have different indices, aligning by index")
        # Get the intersection of indices
        common_index = rtlmp_df.index.intersection(dam_df.index)
        if len(common_index) == 0:
            logger.error("No common timestamps found between RTLMP and day-ahead data")
            return result_df
        
        # Filter both DataFrames to common indices
        rtlmp_price = rtlmp_df.loc[common_index, 'price']
        dam_price = dam_df.loc[common_index, dam_price_col]
    else:
        # Extract price series from the DataFrames
        rtlmp_price = rtlmp_df['price']
        dam_price = dam_df[dam_price_col]
    
    # Calculate price spread (RTLMP - DAM)
    result_df['price_spread'] = rtlmp_price - dam_price
    
    # Calculate price ratio (RTLMP / DAM)
    result_df['price_ratio'] = rtlmp_price / dam_price.replace(0, np.nan)
    result_df['price_ratio'] = result_df['price_ratio'].fillna(1)  # Use 1 as default when dam_price is 0
    
    # Calculate statistical features for price spread over different windows
    windows = [6, 12, 24, 48, 72, 168]  # hours
    
    for window in windows:
        # Rolling mean of price spread
        result_df[f'price_spread_mean_{window}h'] = (
            result_df['price_spread'].rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling standard deviation of price spread
        result_df[f'price_spread_std_{window}h'] = (
            result_df['price_spread'].rolling(window=window, min_periods=1).std()
        )
        
        # Rolling min and max of price spread
        result_df[f'price_spread_min_{window}h'] = (
            result_df['price_spread'].rolling(window=window, min_periods=1).min()
        )
        
        result_df[f'price_spread_max_{window}h'] = (
            result_df['price_spread'].rolling(window=window, min_periods=1).max()
        )
        
        # Rolling mean of price ratio
        result_df[f'price_ratio_mean_{window}h'] = (
            result_df['price_ratio'].rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling standard deviation of price ratio
        result_df[f'price_ratio_std_{window}h'] = (
            result_df['price_ratio'].rolling(window=window, min_periods=1).std()
        )
    
    # Create binary indicators for when RTLMP exceeds DAM by various thresholds
    thresholds = [10, 25, 50, 100, 200]  # $/MWh
    
    for threshold in thresholds:
        result_df[f'rtlmp_exceeds_dam_by_{threshold}'] = (
            (result_df['price_spread'] > threshold).astype(int)
        )
    
    feature_count = len(result_df.columns) - len(rtlmp_df.columns)
    logger.info(f"Created {feature_count} day-ahead market features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_all_market_features(
    rtlmp_df: DataFrameType,
    grid_df: DataFrameType,
    dam_df: Optional[DataFrameType] = None,
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Creates all market-related features by combining congestion, grid condition, 
    correlation, and day-ahead features.
    
    Args:
        rtlmp_df: DataFrame containing RTLMP price data
        grid_df: DataFrame containing grid condition data
        dam_df: Optional DataFrame containing day-ahead market price data
        windows: List of window sizes for rolling calculations, defaults to DEFAULT_WINDOWS
        
    Returns:
        DataFrame with all market-related features
    """
    # Validate required columns
    if 'price' not in rtlmp_df.columns:
        logger.warning("Missing 'price' column in RTLMP data")
        return rtlmp_df.copy()
    
    grid_required_columns = ['total_load', 'available_capacity']
    if not all(col in grid_df.columns for col in grid_required_columns):
        missing = [col for col in grid_required_columns if col not in grid_df.columns]
        logger.warning(f"Missing required grid columns: {missing}")
    
    # Use default windows if not provided
    if windows is None:
        windows = DEFAULT_WINDOWS
    
    # Start with a copy of the RTLMP DataFrame
    result_df = rtlmp_df.copy()
    
    # 1. Create congestion features if congestion data is available
    if all(col in rtlmp_df.columns for col in ['congestion_price', 'loss_price', 'energy_price']):
        congestion_features = create_congestion_features(rtlmp_df, windows)
        # Add new columns to result_df
        for col in congestion_features.columns:
            if col not in rtlmp_df.columns:
                result_df[col] = congestion_features[col]
    else:
        logger.warning("Congestion components not available, skipping congestion features")
    
    # 2. Create grid condition features
    grid_features = create_grid_condition_features(grid_df, windows)
    # Add grid condition features to result_df
    for col in grid_features.columns:
        if col not in grid_df.columns:
            # Use index alignment to handle potential index differences
            result_df[col] = grid_features[col].reindex(result_df.index, method='ffill')
    
    # 3. Create market correlation features
    correlation_features = create_market_correlation_features(result_df, grid_features)
    # Add correlation features to result_df
    for col in correlation_features.columns:
        if col not in result_df.columns:
            result_df[col] = correlation_features[col]
    
    # 4. Create day-ahead features if day-ahead data is available
    if dam_df is not None:
        day_ahead_features = create_day_ahead_features(result_df, dam_df)
        # Add day-ahead features to result_df
        for col in day_ahead_features.columns:
            if col not in result_df.columns:
                result_df[col] = day_ahead_features[col]
    else:
        logger.info("Day-ahead market data not provided, skipping day-ahead features")
    
    feature_count = len(result_df.columns) - len(rtlmp_df.columns)
    logger.info(f"Created a total of {feature_count} market-related features")
    
    return result_df


def get_market_feature_names() -> List[str]:
    """
    Returns a list of all available market feature names.
    
    Returns:
        List of market feature names
    """
    return MARKET_FEATURE_NAMES


def get_market_feature_metadata() -> List[Dict[str, Any]]:
    """
    Returns metadata for all market features for registration purposes.
    
    Returns:
        List of feature metadata dictionaries
    """
    if not MARKET_FEATURE_METADATA:
        # Initialize feature metadata if empty
        initialize_market_features()
    
    return MARKET_FEATURE_METADATA


@log_execution_time(logger, 'INFO')
def initialize_market_features() -> int:
    """
    Initializes market feature definitions and metadata for use by other modules.
    
    Returns:
        Number of features initialized
    """
    global MARKET_FEATURE_NAMES, MARKET_FEATURE_METADATA
    
    # Only initialize if not already done
    if MARKET_FEATURE_NAMES and MARKET_FEATURE_METADATA:
        logger.debug("Market features already initialized")
        return len(MARKET_FEATURE_NAMES)
    
    # Initialize empty lists if they don't exist
    if not MARKET_FEATURE_NAMES:
        MARKET_FEATURE_NAMES = []
    
    if not MARKET_FEATURE_METADATA:
        MARKET_FEATURE_METADATA = []
    
    # Add congestion feature metadata
    congestion_features = [
        {
            "name": "congestion_ratio",
            "description": "Ratio of congestion price to total price",
            "group": "market",
            "subgroup": "congestion",
            "data_type": "float",
            "value_range": [0, 1],
            "importance": "high"
        },
        {
            "name": "congestion_volatility_24h",
            "description": "24-hour rolling standard deviation of congestion ratio",
            "group": "market",
            "subgroup": "congestion",
            "data_type": "float",
            "value_range": [0, float('inf')],
            "importance": "high"
        }
    ]
    
    # Add dynamic congestion features based on window sizes
    for window in DEFAULT_WINDOWS:
        congestion_features.extend([
            {
                "name": f"congestion_ratio_mean_{window}h",
                "description": f"{window}-hour rolling mean of congestion ratio",
                "group": "market",
                "subgroup": "congestion",
                "data_type": "float",
                "value_range": [0, 1],
                "importance": "medium"
            },
            {
                "name": f"congestion_ratio_std_{window}h",
                "description": f"{window}-hour rolling standard deviation of congestion ratio",
                "group": "market",
                "subgroup": "congestion",
                "data_type": "float",
                "value_range": [0, float('inf')],
                "importance": "medium"
            }
        ])
    
    # Add grid condition feature metadata
    grid_features = [
        {
            "name": "reserve_margin",
            "description": "Percentage of available capacity above current load",
            "group": "market",
            "subgroup": "grid_condition",
            "data_type": "float",
            "value_range": [-100, float('inf')],
            "importance": "high"
        },
        {
            "name": "renewable_penetration",
            "description": "Percentage of load served by renewable generation",
            "group": "market",
            "subgroup": "grid_condition",
            "data_type": "float",
            "value_range": [0, 100],
            "importance": "high"
        },
        {
            "name": "load_ramp_rate",
            "description": "Rate of change in total load (percentage)",
            "group": "market",
            "subgroup": "grid_condition",
            "data_type": "float",
            "value_range": [-100, 100],
            "importance": "medium"
        },
        {
            "name": "renewable_variability_24h",
            "description": "24-hour rolling standard deviation of renewable penetration",
            "group": "market",
            "subgroup": "grid_condition",
            "data_type": "float",
            "value_range": [0, 100],
            "importance": "medium"
        }
    ]
    
    # Add dynamic grid condition features based on window sizes
    for window in DEFAULT_WINDOWS:
        grid_features.extend([
            {
                "name": f"reserve_margin_mean_{window}h",
                "description": f"{window}-hour rolling mean of reserve margin",
                "group": "market",
                "subgroup": "grid_condition",
                "data_type": "float",
                "value_range": [-100, float('inf')],
                "importance": "medium"
            },
            {
                "name": f"renewable_penetration_mean_{window}h",
                "description": f"{window}-hour rolling mean of renewable penetration",
                "group": "market",
                "subgroup": "grid_condition",
                "data_type": "float",
                "value_range": [0, 100],
                "importance": "medium"
            }
        ])
    
    # Add market correlation feature metadata
    correlation_features = []
    
    # Add dynamic correlation features based on lag values
    lags = [-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24]
    for lag in lags:
        correlation_features.extend([
            {
                "name": f"price_load_corr_lag{lag}",
                "description": f"Correlation between price and load with {lag}-hour lag",
                "group": "market",
                "subgroup": "correlation",
                "data_type": "float",
                "value_range": [-1, 1],
                "importance": "medium"
            },
            {
                "name": f"price_reserve_corr_lag{lag}",
                "description": f"Correlation between price and reserve margin with {lag}-hour lag",
                "group": "market",
                "subgroup": "correlation",
                "data_type": "float",
                "value_range": [-1, 1],
                "importance": "medium"
            },
            {
                "name": f"price_renewable_corr_lag{lag}",
                "description": f"Correlation between price and renewable penetration with {lag}-hour lag",
                "group": "market",
                "subgroup": "correlation",
                "data_type": "float",
                "value_range": [-1, 1],
                "importance": "medium"
            }
        ])
    
    # Add day-ahead feature metadata
    day_ahead_features = [
        {
            "name": "price_spread",
            "description": "Difference between RTLMP and day-ahead price",
            "group": "market",
            "subgroup": "day_ahead",
            "data_type": "float",
            "value_range": [-float('inf'), float('inf')],
            "importance": "high"
        },
        {
            "name": "price_ratio",
            "description": "Ratio of RTLMP to day-ahead price",
            "group": "market",
            "subgroup": "day_ahead",
            "data_type": "float",
            "value_range": [0, float('inf')],
            "importance": "high"
        }
    ]
    
    # Add dynamic day-ahead features based on window sizes
    windows = [6, 12, 24, 48, 72, 168]
    for window in windows:
        day_ahead_features.extend([
            {
                "name": f"price_spread_mean_{window}h",
                "description": f"{window}-hour rolling mean of price spread",
                "group": "market",
                "subgroup": "day_ahead",
                "data_type": "float",
                "value_range": [-float('inf'), float('inf')],
                "importance": "medium"
            },
            {
                "name": f"price_ratio_mean_{window}h",
                "description": f"{window}-hour rolling mean of price ratio",
                "group": "market",
                "subgroup": "day_ahead",
                "data_type": "float",
                "value_range": [0, float('inf')],
                "importance": "medium"
            }
        ])
    
    # Add threshold-based day-ahead features
    thresholds = [10, 25, 50, 100, 200]
    for threshold in thresholds:
        day_ahead_features.append({
            "name": f"rtlmp_exceeds_dam_by_{threshold}",
            "description": f"Binary indicator for RTLMP exceeding DAM by {threshold} $/MWh",
            "group": "market",
            "subgroup": "day_ahead",
            "data_type": "int",
            "value_range": [0, 1],
            "importance": "high"
        })
    
    # Combine all feature metadata
    MARKET_FEATURE_METADATA.extend(congestion_features)
    MARKET_FEATURE_METADATA.extend(grid_features)
    MARKET_FEATURE_METADATA.extend(correlation_features)
    MARKET_FEATURE_METADATA.extend(day_ahead_features)
    
    # Update feature names
    MARKET_FEATURE_NAMES = [feature["name"] for feature in MARKET_FEATURE_METADATA]
    
    logger.info(f"Initialized {len(MARKET_FEATURE_NAMES)} market features")
    return len(MARKET_FEATURE_NAMES)