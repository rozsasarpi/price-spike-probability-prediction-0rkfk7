"""
Core module that implements the complete feature engineering pipeline for the ERCOT RTLMP spike prediction system.

This module orchestrates the creation of time-based, statistical, weather, and market features,
providing a unified interface for transforming raw data into model-ready features with consistent formatting.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from typing import Dict, List, Optional, Union, Any, Tuple

from ..utils.type_definitions import DataFrameType, SeriesType, FeatureEngineerProtocol
from ..utils.logging import get_logger, log_execution_time
from .time_features import create_all_time_features, get_time_feature_names
from .statistical_features import create_all_statistical_features, PRICE_SPIKE_THRESHOLDS
from .weather_features import create_all_weather_features, get_weather_feature_names
from .market_features import create_all_market_features
from .feature_registry import register_feature, get_all_features, validate_feature_consistency
from .feature_selection import select_features_pipeline

# Set up logger
logger = get_logger(__name__)

# Default feature configuration
DEFAULT_FEATURE_CONFIG = {
    'time_features': {
        'enabled': True,
        'timestamp_column': 'timestamp'
    },
    'statistical_features': {
        'enabled': True,
        'price_column': 'price',
        'timestamp_column': 'timestamp'
    },
    'weather_features': {
        'enabled': True,
        'column_mapping': None,  # Use default mapping
        'include_interactions': True
    },
    'market_features': {
        'enabled': True
    },
    'feature_selection': {
        'enabled': False,
        'methods': ['importance', 'correlation'],
        'importance_threshold': 0.01,
        'correlation_threshold': 0.85
    }
}


@log_execution_time(logger, 'INFO')
def create_time_features(
    df: DataFrameType,
    timestamp_column: str,
    features: Optional[List[str]] = None
) -> DataFrameType:
    """
    Creates time-based features from timestamp data.

    Args:
        df: Input DataFrame
        timestamp_column: Name of the column containing timestamps
        features: List of specific time features to create (default: all available features)

    Returns:
        DataFrame with added time features
    """
    # Validate that timestamp_column exists in the DataFrame
    if timestamp_column not in df.columns:
        logger.error(f"Column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")
    
    # Call the create_all_time_features function from time_features module
    result_df = create_all_time_features(df, timestamp_column, features)
    
    # Log the number of time features created
    time_feature_count = len(result_df.columns) - len(df.columns)
    logger.info(f"Created {time_feature_count} time features from column '{timestamp_column}'")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_statistical_features(
    df: DataFrameType,
    price_column: str,
    timestamp_column: str,
    feature_types: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None
) -> DataFrameType:
    """
    Creates statistical features from RTLMP price data.

    Args:
        df: DataFrame containing RTLMP price data
        price_column: Name of the column containing price values
        timestamp_column: Name of the column containing timestamps
        feature_types: List of feature types to create, defaults to all available types
        thresholds: List of threshold values for defining price spikes, defaults to PRICE_SPIKE_THRESHOLDS

    Returns:
        DataFrame with added statistical features
    """
    # Validate that required columns exist in the DataFrame
    if price_column not in df.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame")
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    if timestamp_column not in df.columns:
        logger.error(f"Timestamp column '{timestamp_column}' not found in DataFrame")
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
    
    # Use default thresholds if not provided
    if thresholds is None:
        thresholds = PRICE_SPIKE_THRESHOLDS
    
    # Call the create_all_statistical_features function from statistical_features module
    result_df = create_all_statistical_features(df, price_column, timestamp_column, feature_types)
    
    # Log the number of statistical features created
    stat_feature_count = len(result_df.columns) - len(df.columns)
    logger.info(f"Created {stat_feature_count} statistical features from columns '{price_column}' and '{timestamp_column}'")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_weather_features(
    df: DataFrameType,
    column_mapping: Optional[Dict[str, str]] = None,
    include_interactions: bool = True
) -> DataFrameType:
    """
    Creates weather-related features from weather data.

    Args:
        df: DataFrame containing weather data
        column_mapping: Optional mapping of DataFrame columns to weather variables
        include_interactions: Whether to include interaction features between weather variables

    Returns:
        DataFrame with added weather features
    """
    # Call the create_all_weather_features function from weather_features module
    result_df = create_all_weather_features(df, column_mapping, include_interactions=include_interactions)
    
    # Log the number of weather features created
    weather_feature_count = len(result_df.columns) - len(df.columns)
    logger.info(f"Created {weather_feature_count} weather features")
    
    return result_df


@log_execution_time(logger, 'INFO')
def create_market_features(
    rtlmp_df: DataFrameType,
    grid_df: DataFrameType,
    dam_df: Optional[DataFrameType] = None
) -> DataFrameType:
    """
    Creates market-related features from RTLMP and grid data.

    Args:
        rtlmp_df: DataFrame containing RTLMP price data
        grid_df: DataFrame containing grid condition data
        dam_df: Optional DataFrame containing day-ahead market price data

    Returns:
        DataFrame with added market features
    """
    # Validate that rtlmp_df contains required columns
    if 'price' not in rtlmp_df.columns:
        logger.error("'price' column not found in RTLMP DataFrame")
        raise ValueError("'price' column not found in RTLMP DataFrame")
    
    # Validate that grid_df contains required columns
    required_grid_columns = ['total_load', 'available_capacity']
    if not all(col in grid_df.columns for col in required_grid_columns):
        missing = [col for col in required_grid_columns if col not in grid_df.columns]
        logger.warning(f"Missing required grid columns: {missing}")
    
    # Call the create_all_market_features function from market_features module
    result_df = create_all_market_features(rtlmp_df, grid_df, dam_df)
    
    # Log the number of market features created
    market_feature_count = len(result_df.columns) - len(rtlmp_df.columns)
    logger.info(f"Created {market_feature_count} market features")
    
    return result_df


def deep_update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep update dictionary d with values from dictionary u.
    
    Args:
        d: Dictionary to update
        u: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = deep_update_dict(d[k], v)
        else:
            d[k] = v
    return d


@log_execution_time(logger, 'INFO')
def create_feature_pipeline(
    data_sources: Dict[str, DataFrameType],
    feature_config: Optional[Dict[str, Any]] = None,
    target: Optional[SeriesType] = None
) -> DataFrameType:
    """
    Creates a complete feature engineering pipeline that processes all feature types.

    Args:
        data_sources: Dictionary of data source DataFrames
        feature_config: Configuration for the feature engineering process
        target: Optional target variable for feature selection

    Returns:
        DataFrame with all engineered features
    """
    # Validate that required data sources are provided
    if 'rtlmp_df' not in data_sources:
        logger.error("'rtlmp_df' not provided in data_sources")
        raise ValueError("'rtlmp_df' not provided in data_sources")
    
    # Use default config if not provided
    if feature_config is None:
        feature_config = DEFAULT_FEATURE_CONFIG
    
    # Create a copy of rtlmp_df to serve as the base for adding features
    rtlmp_df = data_sources['rtlmp_df']
    result_df = rtlmp_df.copy()
    initial_column_count = len(result_df.columns)
    
    # Create time features if enabled
    if feature_config.get('time_features', {}).get('enabled', True):
        timestamp_column = feature_config.get('time_features', {}).get('timestamp_column', 'timestamp')
        if timestamp_column in result_df.columns:
            time_features = create_time_features(result_df, timestamp_column)
            # If time_features is a new DataFrame, merge results
            if id(time_features) != id(result_df):
                for col in time_features.columns:
                    if col not in rtlmp_df.columns:
                        result_df[col] = time_features[col]
        else:
            logger.warning(f"Timestamp column '{timestamp_column}' not found, skipping time features")
    
    # Create statistical features if enabled
    if feature_config.get('statistical_features', {}).get('enabled', True):
        price_column = feature_config.get('statistical_features', {}).get('price_column', 'price')
        timestamp_column = feature_config.get('statistical_features', {}).get('timestamp_column', 'timestamp')
        feature_types = feature_config.get('statistical_features', {}).get('feature_types', None)
        thresholds = feature_config.get('statistical_features', {}).get('thresholds', None)
        
        if price_column in result_df.columns and timestamp_column in result_df.columns:
            stat_features = create_statistical_features(
                result_df, 
                price_column, 
                timestamp_column, 
                feature_types, 
                thresholds
            )
            # If stat_features is a new DataFrame, merge results
            if id(stat_features) != id(result_df):
                for col in stat_features.columns:
                    if col not in rtlmp_df.columns:
                        result_df[col] = stat_features[col]
        else:
            logger.warning(f"Required columns for statistical features not found, skipping")
    
    # Create weather features if enabled and weather_df is provided
    if feature_config.get('weather_features', {}).get('enabled', True) and 'weather_df' in data_sources:
        weather_df = data_sources['weather_df']
        column_mapping = feature_config.get('weather_features', {}).get('column_mapping', None)
        include_interactions = feature_config.get('weather_features', {}).get('include_interactions', True)
        
        weather_features = create_weather_features(
            weather_df, 
            column_mapping, 
            include_interactions
        )
        
        # Merge weather features into result_df (based on common index or timestamp)
        if len(weather_features.columns) > len(weather_df.columns):
            # Extract new columns only
            weather_feature_cols = [col for col in weather_features.columns if col not in weather_df.columns]
            
            # Reset index if indices don't align
            if not result_df.index.equals(weather_features.index):
                logger.warning("Weather features index doesn't match result_df index, aligning by index")
                if 'timestamp' in result_df.columns and 'timestamp' in weather_features.columns:
                    # Merge on timestamp column
                    result_df = pd.merge(
                        result_df,
                        weather_features[weather_feature_cols + ['timestamp']],
                        on='timestamp',
                        how='left'
                    )
                else:
                    # Try to align by index values if possible
                    for col in weather_feature_cols:
                        result_df[col] = weather_features[col].reindex(result_df.index)
            else:
                # Add columns directly if indices match
                for col in weather_feature_cols:
                    result_df[col] = weather_features[col]
        else:
            logger.warning("No new weather features created")
    else:
        if not feature_config.get('weather_features', {}).get('enabled', True):
            logger.info("Weather features disabled in configuration")
        if 'weather_df' not in data_sources:
            logger.info("No weather data provided, skipping weather features")
    
    # Create market features if enabled and grid_df is provided
    if feature_config.get('market_features', {}).get('enabled', True) and 'grid_df' in data_sources:
        grid_df = data_sources['grid_df']
        dam_df = data_sources.get('dam_df', None)  # Optional day-ahead market data
        
        market_features = create_market_features(
            result_df,  # Use current result_df as rtlmp_df to include already engineered features
            grid_df,
            dam_df
        )
        
        # Merge market features into result_df
        if len(market_features.columns) > len(result_df.columns):
            # Extract new columns only
            market_feature_cols = [col for col in market_features.columns if col not in result_df.columns]
            
            # Add columns directly (market_features should have the same index as result_df)
            for col in market_feature_cols:
                result_df[col] = market_features[col]
        else:
            logger.warning("No new market features created")
    else:
        if not feature_config.get('market_features', {}).get('enabled', True):
            logger.info("Market features disabled in configuration")
        if 'grid_df' not in data_sources:
            logger.info("No grid data provided, skipping market features")
    
    # Apply feature selection if enabled
    if feature_config.get('feature_selection', {}).get('enabled', False):
        selection_methods = feature_config.get('feature_selection', {}).get('methods', ['importance', 'correlation'])
        importance_threshold = feature_config.get('feature_selection', {}).get('importance_threshold', 0.01)
        correlation_threshold = feature_config.get('feature_selection', {}).get('correlation_threshold', 0.85)
        
        selection_params = {
            'methods': selection_methods,
            'importance_threshold': importance_threshold,
            'correlation_threshold': correlation_threshold
        }
        
        # Get all feature columns (excluding original columns from rtlmp_df)
        feature_cols = [col for col in result_df.columns if col not in rtlmp_df.columns]
        
        if feature_cols:
            logger.info(f"Applying feature selection to {len(feature_cols)} features")
            selected_features = select_features_pipeline(
                result_df[feature_cols],
                target,
                selection_params
            )
            
            # Keep only selected features and original columns
            original_cols = [col for col in rtlmp_df.columns]
            result_df = result_df[original_cols + selected_features]
            
            logger.info(f"Selected {len(selected_features)} features after feature selection")
        else:
            logger.warning("No features to select, skipping feature selection")
    
    # Validate feature consistency
    validate_feature_consistency(result_df)
    
    # Log the total number of features created
    feature_count = len(result_df.columns) - initial_column_count
    logger.info(f"Created a total of {feature_count} features in the pipeline")
    
    return result_df


class FeaturePipeline(FeatureEngineerProtocol):
    """
    Class-based interface for the feature engineering pipeline that implements
    the FeatureEngineerProtocol.
    """
    
    def __init__(self, feature_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new FeaturePipeline instance.
        
        Args:
            feature_config: Configuration for the feature engineering process
        """
        self._data_sources = {}
        self._feature_config = feature_config or DEFAULT_FEATURE_CONFIG
        self._features_df = None
        self._selected_features = []
    
    def add_data_source(self, source_name: str, df: DataFrameType) -> None:
        """
        Add a data source to the pipeline.
        
        Args:
            source_name: Name of the data source
            df: DataFrame containing the data
        """
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Data source '{source_name}' is not a pandas DataFrame")
            raise TypeError(f"Data source '{source_name}' is not a pandas DataFrame")
        
        self._data_sources[source_name] = df
        logger.info(f"Added data source '{source_name}' with {len(df)} rows and {len(df.columns)} columns")
    
    def create_features(self, target: Optional[SeriesType] = None) -> DataFrameType:
        """
        Create features using all configured feature types.
        
        Args:
            target: Optional target variable for feature selection
            
        Returns:
            DataFrame with all engineered features
        """
        # Validate that required data sources are available
        if 'rtlmp_df' not in self._data_sources:
            logger.error("'rtlmp_df' not provided in data sources")
            raise ValueError("'rtlmp_df' not provided in data sources")
        
        # Create features using the pipeline function
        self._features_df = create_feature_pipeline(
            self._data_sources,
            self._feature_config,
            target
        )
        
        # If feature selection is enabled, store the list of selected features
        if self._feature_config.get('feature_selection', {}).get('enabled', False):
            original_cols = list(self._data_sources['rtlmp_df'].columns)
            self._selected_features = [col for col in self._features_df.columns if col not in original_cols]
        
        return self._features_df
    
    def get_features(self) -> Optional[DataFrameType]:
        """
        Get the current features DataFrame.
        
        Returns:
            Features DataFrame or None if not created yet
        """
        return self._features_df
    
    def update_feature_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the feature configuration.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        # Deep merge config_updates into _feature_config
        deep_update_dict(self._feature_config, config_updates)
        logger.info("Updated feature configuration")
    
    def get_feature_names(self, selected_only: bool = False) -> List[str]:
        """
        Get the names of all features or selected features.
        
        Args:
            selected_only: Whether to return only selected features
            
        Returns:
            List of feature names
        """
        if self._features_df is None:
            return []
        
        if selected_only and self._selected_features:
            return self._selected_features
        
        # Return all feature columns (excluding original columns from rtlmp_df)
        if 'rtlmp_df' in self._data_sources:
            original_cols = list(self._data_sources['rtlmp_df'].columns)
            return [col for col in self._features_df.columns if col not in original_cols]
        
        return list(self._features_df.columns)
    
    def validate_feature_set(self) -> Tuple[bool, List[str]]:
        """
        Validate the feature set for consistency.
        
        Returns:
            Tuple of (is_valid, inconsistent_features)
        """
        if self._features_df is None:
            return False, []
        
        return validate_feature_consistency(self._features_df)