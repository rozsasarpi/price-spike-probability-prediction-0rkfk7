"""
Implements functionality for applying price threshold values to RTLMP data to identify price spikes.

This module provides tools for determining when price values exceed specified thresholds,
which is essential for the spike prediction system.
"""

from typing import Dict, List, Optional, Union, Callable, Any

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+

# Internal imports
from .threshold_config import ThresholdConfig, get_default_threshold
from ..utils.type_definitions import DataFrameType, SeriesType, ThresholdValue, NodeID
from ..utils.logging import get_logger
from ..utils.validation import validate_value_ranges

# Set up logger
logger = get_logger(__name__)


def apply_threshold(price_series: SeriesType, threshold: ThresholdValue) -> SeriesType:
    """
    Applies a single threshold value to a price series and returns a boolean mask of where prices exceed the threshold.
    
    Args:
        price_series: Series containing price values
        threshold: Threshold value to apply
        
    Returns:
        Boolean mask where True indicates price exceeds threshold
    
    Raises:
        ValueError: If price_series is not a pandas Series or threshold is not a positive number
    """
    # Validate inputs
    if not isinstance(price_series, pd.Series):
        raise ValueError(f"Expected price_series to be a pandas Series, got {type(price_series)}")
    
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError(f"Threshold must be a positive number, got {threshold}")
    
    # Create boolean mask where price exceeds threshold
    mask = price_series > threshold
    
    logger.debug(f"Applied threshold {threshold} to price series, {mask.sum()} values exceeded threshold")
    return mask


def apply_thresholds(price_series: SeriesType, thresholds: List[ThresholdValue]) -> Dict[ThresholdValue, SeriesType]:
    """
    Applies multiple threshold values to a price series and returns a dictionary of boolean masks for each threshold.
    
    Args:
        price_series: Series containing price values
        thresholds: List of threshold values to apply
        
    Returns:
        Dictionary mapping thresholds to boolean masks
        
    Raises:
        ValueError: If price_series is not a pandas Series or thresholds is empty or contains invalid values
    """
    # Validate inputs
    if not isinstance(price_series, pd.Series):
        raise ValueError(f"Expected price_series to be a pandas Series, got {type(price_series)}")
    
    if not thresholds:
        raise ValueError("Thresholds list cannot be empty")
    
    # Apply each threshold and store results in a dictionary
    result = {}
    for threshold in thresholds:
        result[threshold] = apply_threshold(price_series, threshold)
    
    logger.debug(f"Applied {len(thresholds)} thresholds to price series")
    return result


def create_spike_indicator(df: DataFrameType, price_column: str, threshold: ThresholdValue, 
                          output_column: Optional[str] = None) -> DataFrameType:
    """
    Creates a binary indicator column for whether a price spike occurred based on a threshold.
    
    Args:
        df: DataFrame containing price data
        price_column: Name of the column containing price values
        threshold: Threshold value to define a price spike
        output_column: Optional name for the output column, defaults to 'spike_{threshold}'
        
    Returns:
        DataFrame with added spike indicator column
        
    Raises:
        ValueError: If df is not a DataFrame or price_column does not exist
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected df to be a pandas DataFrame, got {type(df)}")
    
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    # Set default output column name if not provided
    if output_column is None:
        output_column = f"spike_{threshold}"
    
    # Apply threshold to create spike indicator
    df = df.copy()
    df[output_column] = apply_threshold(df[price_column], threshold)
    
    logger.debug(f"Created spike indicator '{output_column}' for threshold {threshold}")
    return df


def create_multi_threshold_indicators(df: DataFrameType, price_column: str, 
                                     thresholds: List[ThresholdValue],
                                     prefix: Optional[str] = None) -> DataFrameType:
    """
    Creates multiple binary indicator columns for different threshold values.
    
    Args:
        df: DataFrame containing price data
        price_column: Name of the column containing price values
        thresholds: List of threshold values to apply
        prefix: Optional prefix for column names, defaults to 'spike'
        
    Returns:
        DataFrame with added spike indicator columns for each threshold
        
    Raises:
        ValueError: If df is not a DataFrame or price_column does not exist
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected df to be a pandas DataFrame, got {type(df)}")
    
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    # Set default prefix if not provided
    if prefix is None:
        prefix = "spike"
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Apply each threshold and create indicator columns
    for threshold in thresholds:
        column_name = f"{prefix}_{threshold}"
        result_df = create_spike_indicator(result_df, price_column, threshold, column_name)
    
    logger.debug(f"Created {len(thresholds)} spike indicators with prefix '{prefix}'")
    return result_df


def find_max_price_in_window(df: DataFrameType, price_column: str, window: str,
                            output_column: Optional[str] = None) -> DataFrameType:
    """
    Finds the maximum price within a rolling time window for each timestamp.
    
    Args:
        df: DataFrame containing price data with DatetimeIndex
        price_column: Name of the column containing price values
        window: Rolling window size as pandas window string (e.g., '1H', '24H')
        output_column: Optional name for the output column, defaults to '{price_column}_max_{window}'
        
    Returns:
        DataFrame with added column for maximum price in window
        
    Raises:
        ValueError: If df is not a DataFrame with DatetimeIndex or price_column does not exist
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected df to be a pandas DataFrame, got {type(df)}")
    
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Set default output column name if not provided
    if output_column is None:
        output_column = f"{price_column}_max_{window}"
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Calculate rolling maximum
    result_df[output_column] = df[price_column].rolling(window=window).max()
    
    logger.debug(f"Calculated rolling maximum '{output_column}' with window {window}")
    return result_df


def hourly_spike_occurrence(df_5min: DataFrameType, price_column: str, threshold: ThresholdValue,
                           output_column: Optional[str] = None) -> DataFrameType:
    """
    Determines if a price spike occurred at least once within each hour based on 5-minute data.
    
    Args:
        df_5min: DataFrame containing 5-minute price data with DatetimeIndex
        price_column: Name of the column containing price values
        threshold: Threshold value to define a price spike
        output_column: Optional name for the output column, defaults to 'hourly_spike_{threshold}'
        
    Returns:
        Hourly DataFrame with spike occurrence indicator
        
    Raises:
        ValueError: If df_5min is not a DataFrame with 5-minute frequency or price_column does not exist
    """
    # Validate inputs
    if not isinstance(df_5min, pd.DataFrame):
        raise ValueError(f"Expected df_5min to be a pandas DataFrame, got {type(df_5min)}")
    
    if price_column not in df_5min.columns:
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    if not isinstance(df_5min.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Check if data is roughly 5-minute frequency (allow some tolerance)
    freq_check = df_5min.index.to_series().diff().median()
    if freq_check > pd.Timedelta(minutes=6) or freq_check < pd.Timedelta(minutes=4):
        logger.warning(f"Data frequency appears to be {freq_check} which is not close to 5-minutes")
    
    # Set default output column name if not provided
    if output_column is None:
        output_column = f"hourly_spike_{threshold}"
    
    # Create spike indicator for 5-minute data
    df_with_spike = create_spike_indicator(df_5min, price_column, threshold)
    
    # Resample to hourly frequency and check if any spike occurred in each hour
    spike_column = f"spike_{threshold}"
    hourly_df = df_with_spike.resample('H')[spike_column].any().to_frame(name=output_column)
    
    logger.debug(f"Created hourly spike occurrence indicator '{output_column}' for threshold {threshold}")
    return hourly_df


class ThresholdApplier:
    """
    Class for applying threshold operations to price data with configurable settings.
    """
    
    def __init__(self, threshold_config: ThresholdConfig):
        """
        Initializes ThresholdApplier with threshold configuration.
        
        Args:
            threshold_config: Configuration object for threshold values
        """
        self._threshold_config = threshold_config
        self._aggregation_functions = {
            "any": lambda x: x.any(),
            "all": lambda x: x.all(),
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
            "count": lambda x: x.sum(),
            "max": lambda x: x.max(),
        }
        logger.debug("Initialized ThresholdApplier")
    
    def apply_to_series(self, price_series: SeriesType, 
                       thresholds: Optional[List[ThresholdValue]] = None,
                       node_id: Optional[NodeID] = None) -> Dict[ThresholdValue, SeriesType]:
        """
        Applies thresholds to a price series.
        
        Args:
            price_series: Series containing price values
            thresholds: Optional list of threshold values to apply. If None, uses thresholds from config.
            node_id: Optional node identifier to get node-specific thresholds
            
        Returns:
            Dictionary mapping thresholds to boolean masks
        """
        # Get thresholds from config if not provided
        if thresholds is None:
            thresholds = self._threshold_config.get_thresholds(node_id)
        
        # Apply thresholds to price series
        result = apply_thresholds(price_series, thresholds)
        
        logger.debug(f"Applied {len(thresholds)} thresholds to price series")
        return result
    
    def apply_to_dataframe(self, df: DataFrameType, price_column: str,
                          thresholds: Optional[List[ThresholdValue]] = None,
                          node_id: Optional[NodeID] = None,
                          prefix: Optional[str] = None) -> DataFrameType:
        """
        Applies thresholds to a price column in a DataFrame.
        
        Args:
            df: DataFrame containing price data
            price_column: Name of the column containing price values
            thresholds: Optional list of threshold values to apply. If None, uses thresholds from config.
            node_id: Optional node identifier to get node-specific thresholds
            prefix: Optional prefix for column names
            
        Returns:
            DataFrame with added threshold indicator columns
        """
        # Get thresholds from config if not provided
        if thresholds is None:
            thresholds = self._threshold_config.get_thresholds(node_id)
        
        # Apply thresholds to DataFrame
        result_df = create_multi_threshold_indicators(df, price_column, thresholds, prefix)
        
        logger.debug(f"Applied {len(thresholds)} thresholds to DataFrame")
        return result_df
    
    def hourly_spike_indicators(self, df_5min: DataFrameType, price_column: str,
                               thresholds: Optional[List[ThresholdValue]] = None,
                               node_id: Optional[NodeID] = None,
                               prefix: Optional[str] = None,
                               aggregation: str = "any") -> DataFrameType:
        """
        Creates hourly spike indicators from 5-minute price data.
        
        Args:
            df_5min: DataFrame containing 5-minute price data with DatetimeIndex
            price_column: Name of the column containing price values
            thresholds: Optional list of threshold values to apply. If None, uses thresholds from config.
            node_id: Optional node identifier to get node-specific thresholds
            prefix: Optional prefix for column names
            aggregation: Aggregation function to use when resampling ('any', 'all', 'mean', etc.)
            
        Returns:
            Hourly DataFrame with spike indicators for each threshold
            
        Raises:
            ValueError: If aggregation function is not recognized
        """
        # Get thresholds from config if not provided
        if thresholds is None:
            thresholds = self._threshold_config.get_thresholds(node_id)
        
        # Validate aggregation function
        if aggregation not in self._aggregation_functions:
            raise ValueError(f"Unknown aggregation function: {aggregation}. " 
                            f"Available options: {list(self._aggregation_functions.keys())}")
        
        # Set default prefix if not provided
        if prefix is None:
            prefix = "spike"
        
        # Create indicator columns for each threshold
        df_with_spikes = create_multi_threshold_indicators(df_5min, price_column, thresholds, prefix)
        
        # Resample to hourly frequency
        spike_columns = [f"{prefix}_{threshold}" for threshold in thresholds]
        agg_func = self._aggregation_functions[aggregation]
        
        hourly_df = pd.DataFrame(index=pd.date_range(
            start=df_with_spikes.index.min().floor('H'),
            end=df_with_spikes.index.max().ceil('H'),
            freq='H'
        ))
        
        for col in spike_columns:
            if col in df_with_spikes.columns:
                hourly_df[f"hourly_{col}"] = df_with_spikes.resample('H')[col].apply(agg_func)
        
        logger.debug(f"Created hourly spike indicators for {len(thresholds)} thresholds using {aggregation} aggregation")
        return hourly_df
    
    def add_aggregation_function(self, name: str, function: Callable) -> None:
        """
        Adds a custom aggregation function for resampling.
        
        Args:
            name: Name to identify the function
            function: Callable function that takes a Series and returns a scalar
            
        Raises:
            ValueError: If function is not callable
        """
        if not callable(function):
            raise ValueError(f"Function must be callable, got {type(function)}")
        
        self._aggregation_functions[name] = function
        logger.debug(f"Added custom aggregation function: {name}")
    
    def get_available_aggregations(self) -> List[str]:
        """
        Returns a list of available aggregation function names.
        
        Returns:
            List of aggregation function names
        """
        return list(self._aggregation_functions.keys())


class RollingThresholdAnalyzer:
    """
    Class for analyzing price data with rolling windows and thresholds.
    """
    
    def __init__(self, threshold_config: ThresholdConfig, 
                window_settings: Optional[Dict[str, str]] = None):
        """
        Initializes RollingThresholdAnalyzer with threshold configuration.
        
        Args:
            threshold_config: Configuration object for threshold values
            window_settings: Optional dictionary mapping window names to window strings
        """
        self._threshold_config = threshold_config
        
        # Set default window settings if not provided
        if window_settings is None:
            self._window_settings = {
                "short": "1H",
                "medium": "24H",
                "long": "7D"
            }
        else:
            self._window_settings = window_settings
        
        logger.debug("Initialized RollingThresholdAnalyzer")
    
    def analyze_rolling_max(self, df: DataFrameType, price_column: str,
                           windows: Optional[List[str]] = None,
                           thresholds: Optional[List[ThresholdValue]] = None,
                           node_id: Optional[NodeID] = None) -> DataFrameType:
        """
        Analyzes maximum prices within rolling windows.
        
        Args:
            df: DataFrame containing price data with DatetimeIndex
            price_column: Name of the column containing price values
            windows: Optional list of window strings to use. If None, uses default windows.
            thresholds: Optional list of threshold values to apply. If None, uses thresholds from config.
            node_id: Optional node identifier to get node-specific thresholds
            
        Returns:
            DataFrame with added rolling max and threshold indicators
        """
        # Get windows from settings if not provided
        if windows is None:
            windows = list(self._window_settings.values())
        
        # Get thresholds from config if not provided
        if thresholds is None:
            thresholds = self._threshold_config.get_thresholds(node_id)
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Calculate rolling max for each window
        for window in windows:
            max_col = f"{price_column}_max_{window}"
            result_df = find_max_price_in_window(result_df, price_column, window, max_col)
            
            # Create indicator columns for whether rolling max exceeds thresholds
            for threshold in thresholds:
                indicator_col = f"{max_col}_exceeds_{threshold}"
                result_df[indicator_col] = result_df[max_col] > threshold
        
        logger.debug(f"Analyzed rolling max for {len(windows)} windows and {len(thresholds)} thresholds")
        return result_df
    
    def analyze_price_volatility(self, df: DataFrameType, price_column: str,
                                windows: Optional[List[str]] = None) -> DataFrameType:
        """
        Analyzes price volatility within rolling windows.
        
        Args:
            df: DataFrame containing price data with DatetimeIndex
            price_column: Name of the column containing price values
            windows: Optional list of window strings to use. If None, uses default windows.
            
        Returns:
            DataFrame with added volatility metrics
        """
        # Get windows from settings if not provided
        if windows is None:
            windows = list(self._window_settings.values())
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Calculate volatility metrics for each window
        for window in windows:
            # Standard deviation
            std_col = f"{price_column}_std_{window}"
            result_df[std_col] = df[price_column].rolling(window=window).std()
            
            # Range (max - min)
            range_col = f"{price_column}_range_{window}"
            result_df[range_col] = (
                df[price_column].rolling(window=window).max() - 
                df[price_column].rolling(window=window).min()
            )
            
            # Coefficient of variation (std/mean)
            cv_col = f"{price_column}_cv_{window}"
            mean = df[price_column].rolling(window=window).mean()
            
            # Avoid division by zero
            result_df[cv_col] = np.where(
                mean != 0,
                result_df[std_col] / mean,
                np.nan
            )
        
        logger.debug(f"Analyzed price volatility for {len(windows)} windows")
        return result_df
    
    def set_window_settings(self, window_settings: Dict[str, str]) -> None:
        """
        Updates the window settings for rolling analyses.
        
        Args:
            window_settings: Dictionary mapping window names to window strings
            
        Raises:
            ValueError: If window_settings is not a dictionary or contains invalid window strings
        """
        if not isinstance(window_settings, dict):
            raise ValueError(f"Window settings must be a dictionary, got {type(window_settings)}")
        
        # Validate window strings (simple validation for pandas window strings)
        for name, window in window_settings.items():
            if not isinstance(window, str) or not (window.endswith('D') or window.endswith('H') or 
                                                 window.endswith('min') or window.endswith('s')):
                raise ValueError(f"Invalid window string for {name}: {window}")
        
        self._window_settings = window_settings.copy()
        logger.debug(f"Updated window settings: {window_settings}")
    
    def get_window_settings(self) -> Dict[str, str]:
        """
        Returns the current window settings.
        
        Returns:
            Dictionary of window settings
        """
        return self._window_settings.copy()