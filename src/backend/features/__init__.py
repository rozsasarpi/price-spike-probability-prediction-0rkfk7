"""
Entry point for the feature engineering module in the ERCOT RTLMP spike prediction system.

This module exposes key functionality from time-based, statistical, weather, market feature generators,
feature registry, and feature selection components to provide a unified interface for feature engineering.
It serves as the primary interface for transforming raw data into model-ready features with consistent
formatting for RTLMP spike prediction.
"""

# Time-based feature functions
from .time_features import (
    create_all_time_features,
    get_time_feature_names,
    get_time_feature_metadata,
    TIME_FEATURE_NAMES,
    TIME_FEATURE_METADATA
)

# Statistical feature functions
from .statistical_features import (
    create_all_statistical_features,
    get_statistical_feature_names,
    get_statistical_feature_metadata,
    initialize_statistical_features,
    PRICE_SPIKE_THRESHOLDS
)

# Weather feature functions
from .weather_features import (
    create_all_weather_features,
    get_weather_feature_names,
    get_weather_feature_registry,
    DEFAULT_WEATHER_COLUMNS
)

# Market feature functions
from .market_features import (
    create_all_market_features,
    get_market_feature_names,
    get_market_feature_metadata,
    initialize_market_features
)

# Feature registry functions
from .feature_registry import (
    register_feature,
    get_feature,
    get_features_by_group,
    get_all_features,
    initialize_default_features,
    validate_feature_consistency,
    FeatureRegistry
)

# Feature pipeline functions
from .feature_pipeline import (
    create_feature_pipeline,
    FeaturePipeline,
    DEFAULT_FEATURE_CONFIG
)

# Feature selection functions
from .feature_selection import (
    select_features_pipeline,
    FeatureSelector,
    get_feature_importance_dict
)

# Module version
__version__ = '0.1.0'

def initialize_features() -> bool:
    """
    Initialize all feature types and the feature registry.
    
    This function initializes statistical features, market features,
    and the feature registry with default features, ensuring that all
    feature components are properly set up before using the feature
    engineering pipeline.
    
    Returns:
        bool: True if initialization was successful
    """
    # Initialize statistical features
    initialize_statistical_features()
    
    # Initialize market features
    initialize_market_features()
    
    # Initialize the feature registry
    initialize_default_features()
    
    return True