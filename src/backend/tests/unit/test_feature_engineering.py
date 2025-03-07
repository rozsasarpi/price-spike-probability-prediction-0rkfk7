"""
Unit tests for the feature engineering modules in the ERCOT RTLMP spike prediction system.

Tests the creation of time-based, statistical, weather, and market features,
as well as the feature pipeline, registry, and selection functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Import sample data generators
from ...tests.fixtures.sample_data import (
    get_sample_rtlmp_data,
    get_sample_weather_data,
    get_sample_grid_condition_data,
    SAMPLE_NODES,
    SAMPLE_START_DATE,
    SAMPLE_END_DATE
)

# Import time feature modules
from ...features.time_features import (
    create_all_time_features,
    get_time_feature_names,
    TIME_FEATURE_NAMES
)

# Import statistical feature modules
from ...features.statistical_features import (
    create_all_statistical_features,
    PRICE_SPIKE_THRESHOLDS
)

# Import weather feature modules
from ...features.weather_features import (
    create_all_weather_features,
    get_weather_feature_names
)

# Import market feature modules
from ...features.market_features import (
    create_all_market_features
)

# Import feature pipeline modules
from ...features.feature_pipeline import (
    create_feature_pipeline,
    FeaturePipeline,
    DEFAULT_FEATURE_CONFIG
)

# Import feature registry modules
from ...features.feature_registry import (
    register_feature,
    get_feature,
    get_all_features,
    validate_feature_consistency
)

# Import feature selection modules
from ...features.feature_selection import (
    select_features_by_importance,
    select_features_by_correlation,
    select_features_pipeline,
    FeatureSelector
)

# Import validation utilities
from ...utils.validation import (
    validate_required_columns,
    validate_data_types
)

# Import date utilities
from ...utils.date_utils import ERCOT_TIMEZONE


@pytest.fixture
def setup_test_data():
    """
    Creates test data fixtures for feature engineering tests.
    
    Returns:
        dict: Dictionary containing test DataFrames for different data types
    """
    # Generate sample RTLMP data
    rtlmp_df = get_sample_rtlmp_data(
        start_date=SAMPLE_START_DATE,
        end_date=SAMPLE_END_DATE,
        nodes=SAMPLE_NODES[:2],  # Use only two nodes for faster tests
        random_seed=42,
        include_spikes=True
    )
    
    # Generate sample weather data
    weather_df = get_sample_weather_data(
        start_date=SAMPLE_START_DATE,
        end_date=SAMPLE_END_DATE,
        random_seed=42
    )
    
    # Generate sample grid condition data
    grid_df = get_sample_grid_condition_data(
        start_date=SAMPLE_START_DATE,
        end_date=SAMPLE_END_DATE,
        random_seed=42
    )
    
    # Return dictionary of test data
    return {
        'rtlmp_df': rtlmp_df,
        'weather_df': weather_df,
        'grid_df': grid_df
    }


def test_time_features_creation(setup_test_data):
    """
    Tests the creation of time-based features.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    rtlmp_df = setup_test_data['rtlmp_df']
    
    # Create time features
    result_df = create_all_time_features(rtlmp_df, 'timestamp')
    
    # Verify that all expected time features are created
    for feature in TIME_FEATURE_NAMES:
        assert feature in result_df.columns
    
    # Verify ranges of specific features
    assert result_df['hour_of_day'].min() >= 0
    assert result_df['hour_of_day'].max() <= 23
    assert result_df['day_of_week'].min() >= 0
    assert result_df['day_of_week'].max() <= 6
    assert set(result_df['is_weekend'].unique()).issubset({0, 1})
    assert result_df['month'].min() >= 1
    assert result_df['month'].max() <= 12
    assert set(result_df['season'].unique()).issubset({'winter', 'spring', 'summer', 'fall'})
    assert set(result_df['is_holiday'].unique()).issubset({0, 1})


def test_statistical_features_creation(setup_test_data):
    """
    Tests the creation of statistical features from RTLMP data.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    rtlmp_df = setup_test_data['rtlmp_df']
    
    # Create statistical features
    result_df = create_all_statistical_features(rtlmp_df, 'price', 'timestamp')
    
    # Verify that rolling statistics features are created
    assert any(col.startswith('rolling_mean_') for col in result_df.columns)
    assert any(col.startswith('rolling_std_') for col in result_df.columns)
    assert any(col.startswith('rolling_min_') for col in result_df.columns)
    assert any(col.startswith('rolling_max_') for col in result_df.columns)
    
    # Verify that price volatility features are created
    assert any(col.startswith('volatility_') for col in result_df.columns)
    
    # Verify that price spike indicator features are created
    for threshold in PRICE_SPIKE_THRESHOLDS:
        assert f'price_spike_{threshold}' in result_df.columns
    
    # Verify that price spike frequency features are created
    assert any(col.startswith('spike_freq_') for col in result_df.columns)
    
    # Verify that hourly price features are created
    assert 'hourly_mean' in result_df.columns
    assert 'hourly_max' in result_df.columns
    assert 'hourly_min' in result_df.columns
    
    # Verify that price difference features are created
    assert any(col.startswith('price_diff_') for col in result_df.columns)
    assert any(col.startswith('price_pct_change_') for col in result_df.columns)


def test_weather_features_creation(setup_test_data):
    """
    Tests the creation of weather-related features.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    weather_df = setup_test_data['weather_df']
    
    # Create weather features
    result_df = create_all_weather_features(weather_df)
    
    # Verify that temperature features are created
    assert any(col.startswith('temperature_') for col in result_df.columns)
    
    # Verify that wind features are created
    assert any(col.startswith('wind_speed_') for col in result_df.columns)
    
    # Verify that solar features are created
    assert any(col.startswith('solar_irradiance_') for col in result_df.columns)
    
    # Test creating features with interactions
    interact_df = create_all_weather_features(weather_df, include_interactions=True)
    assert any(col.startswith('temp_wind_') or col.startswith('wind_solar_') 
               for col in interact_df.columns)
    
    # Test creating features without interactions
    no_interact_df = create_all_weather_features(weather_df, include_interactions=False)
    assert not any(col.startswith('temp_wind_') or col.startswith('wind_solar_') 
                  for col in no_interact_df.columns)


def test_market_features_creation(setup_test_data):
    """
    Tests the creation of market-related features.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    rtlmp_df = setup_test_data['rtlmp_df']
    grid_df = setup_test_data['grid_df']
    
    # Create market features
    result_df = create_all_market_features(rtlmp_df, grid_df)
    
    # Verify that congestion features are created
    assert 'congestion_ratio' in result_df.columns
    assert any(col.startswith('congestion_ratio_') for col in result_df.columns)
    
    # Verify that grid condition features are created
    assert 'reserve_margin' in result_df.columns
    assert 'renewable_penetration' in result_df.columns
    assert any(col.startswith('reserve_margin_') for col in result_df.columns)
    
    # Verify that market correlation features are created
    assert any(col.startswith('price_load_corr_') for col in result_df.columns)
    assert any(col.startswith('price_reserve_corr_') for col in result_df.columns)


def test_feature_pipeline(setup_test_data):
    """
    Tests the complete feature engineering pipeline.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    rtlmp_df = setup_test_data['rtlmp_df']
    weather_df = setup_test_data['weather_df']
    grid_df = setup_test_data['grid_df']
    
    # Create data_sources dictionary
    data_sources = {
        'rtlmp_df': rtlmp_df,
        'weather_df': weather_df,
        'grid_df': grid_df
    }
    
    # Create feature pipeline
    result_df = create_feature_pipeline(data_sources, DEFAULT_FEATURE_CONFIG)
    
    # Verify that the pipeline created features from all categories
    assert any(col in result_df.columns for col in TIME_FEATURE_NAMES), "No time features created"
    assert any(col.startswith('rolling_') for col in result_df.columns), "No statistical features created"
    assert any(col.startswith('temperature_') or col.startswith('wind_speed_') 
               for col in result_df.columns), "No weather features created"
    assert any(col.startswith('reserve_margin') or col.startswith('renewable_penetration') 
               for col in result_df.columns), "No market features created"
    
    # Verify the number of features created
    original_column_count = len(rtlmp_df.columns)
    new_feature_count = len(result_df.columns) - original_column_count
    assert new_feature_count > 0, "No new features were created by the pipeline"
    
    # Verify that there are no missing values in critical features
    time_features = [col for col in result_df.columns if col in TIME_FEATURE_NAMES]
    assert not result_df[time_features].isnull().any().any(), "Missing values in time features"


def test_feature_pipeline_class(setup_test_data):
    """
    Tests the FeaturePipeline class interface.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    rtlmp_df = setup_test_data['rtlmp_df']
    weather_df = setup_test_data['weather_df']
    grid_df = setup_test_data['grid_df']
    
    # Create FeaturePipeline instance
    pipeline = FeaturePipeline(DEFAULT_FEATURE_CONFIG)
    
    # Add data sources
    pipeline.add_data_source('rtlmp_df', rtlmp_df)
    pipeline.add_data_source('weather_df', weather_df)
    pipeline.add_data_source('grid_df', grid_df)
    
    # Create features
    result_df = pipeline.create_features()
    
    # Verify that features were created
    assert result_df is not None
    assert len(result_df.columns) > len(rtlmp_df.columns)
    
    # Test get_features method
    features_df = pipeline.get_features()
    assert features_df is not None
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df.columns) == len(result_df.columns)


def test_feature_registry():
    """
    Tests the feature registry functionality.
    """
    # Create test feature metadata
    test_feature = {
        'name': 'Test Feature',
        'data_type': 'float',
        'group': 'statistical',
        'description': 'A test feature for unit testing',
        'valid_range': [0, 100],
        'dependencies': []
    }
    
    # Register the feature
    feature_id = 'test_feature'
    assert register_feature(feature_id, test_feature), "Failed to register feature"
    
    # Verify that the feature was registered correctly
    retrieved_feature = get_feature(feature_id)
    assert retrieved_feature is not None, "Failed to retrieve registered feature"
    assert retrieved_feature['name'] == test_feature['name']
    assert retrieved_feature['data_type'] == test_feature['data_type']
    
    # Verify that the feature appears in get_all_features
    all_features = get_all_features()
    assert feature_id in all_features, "Registered feature not found in all features"
    
    # Update feature metadata
    updated_metadata = {'description': 'Updated description for testing'}
    from ...features.feature_registry import update_feature_metadata
    assert update_feature_metadata(feature_id, updated_metadata), "Failed to update feature metadata"
    
    # Verify update
    updated_feature = get_feature(feature_id)
    assert updated_feature['description'] == updated_metadata['description']


def test_feature_consistency_validation(setup_test_data):
    """
    Tests the feature consistency validation functionality.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    rtlmp_df = setup_test_data['rtlmp_df']
    
    # Create time features
    df_with_features = create_all_time_features(rtlmp_df, 'timestamp')
    
    # Create metadata for validation
    hour_feature_id = 'hour_of_day'
    hour_metadata = {
        'name': 'Hour of Day',
        'data_type': 'int',
        'group': 'time',
        'valid_range': [0, 23]
    }
    
    # Register feature for validation
    register_feature(hour_feature_id, hour_metadata)
    
    # Validate feature consistency
    is_valid, inconsistent = validate_feature_consistency(df_with_features, [hour_feature_id])
    assert is_valid, f"Feature validation failed: {inconsistent}"
    
    # Modify feature to have incorrect data type and test validation
    df_invalid = df_with_features.copy()
    df_invalid['hour_of_day'] = df_invalid['hour_of_day'].astype(float) + 0.5  # No longer integers
    
    is_valid, inconsistent = validate_feature_consistency(df_invalid, [hour_feature_id])
    assert not is_valid, "Validation should fail for incorrect data type"
    assert hour_feature_id in ''.join(inconsistent), "hour_of_day should be flagged as inconsistent"
    
    # Modify feature to have out-of-range values and test validation
    df_invalid = df_with_features.copy()
    df_invalid.loc[0, 'hour_of_day'] = 25  # Out of valid range [0, 23]
    
    is_valid, inconsistent = validate_feature_consistency(df_invalid, [hour_feature_id])
    assert not is_valid, "Validation should fail for out-of-range values"


def test_feature_selection(setup_test_data):
    """
    Tests the feature selection functionality.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Create a complete feature set using the pipeline
    rtlmp_df = setup_test_data['rtlmp_df']
    weather_df = setup_test_data['weather_df']
    grid_df = setup_test_data['grid_df']
    
    data_sources = {
        'rtlmp_df': rtlmp_df,
        'weather_df': weather_df,
        'grid_df': grid_df
    }
    
    # Create features
    feature_df = create_feature_pipeline(data_sources, DEFAULT_FEATURE_CONFIG)
    
    # Extract only the engineered features
    original_cols = rtlmp_df.columns.tolist()
    feature_cols = [col for col in feature_df.columns if col not in original_cols]
    features_only_df = feature_df[feature_cols]
    
    # Create mock importance scores
    num_features = len(feature_cols)
    importances = {}
    for i, col in enumerate(feature_cols):
        importances[col] = (num_features - i) / num_features  # Decreasing importance
    
    # Test feature selection by importance
    importance_threshold = 0.5  # Select top 50%
    selected_by_importance = select_features_by_importance(
        features_only_df, importance_scores=importances, threshold=importance_threshold
    )
    
    assert len(selected_by_importance) < len(feature_cols), "Feature selection should reduce features"
    assert len(selected_by_importance) > 0, "No features were selected"
    
    # Test feature selection by correlation
    correlation_threshold = 0.95  # Very high correlation threshold for testing
    selected_by_correlation = select_features_by_correlation(
        features_only_df, threshold=correlation_threshold, importance_scores=importances
    )
    
    assert len(selected_by_correlation) <= len(feature_cols), "Correlation filtering should not increase features"
    
    # Test feature selection pipeline
    selection_params = {
        'methods': ['importance', 'correlation'],
        'importance_threshold': importance_threshold,
        'correlation_threshold': correlation_threshold,
        'importance_scores': importances
    }
    
    selected_combined = select_features_pipeline(features_only_df, selection_params=selection_params)
    assert len(selected_combined) > 0, "No features were selected by pipeline"
    assert len(selected_combined) <= len(feature_cols), "Feature selection should not increase features"


def test_feature_selector_class(setup_test_data):
    """
    Tests the FeatureSelector class interface.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Create a complete feature set using the pipeline
    rtlmp_df = setup_test_data['rtlmp_df']
    weather_df = setup_test_data['weather_df']
    grid_df = setup_test_data['grid_df']
    
    data_sources = {
        'rtlmp_df': rtlmp_df,
        'weather_df': weather_df,
        'grid_df': grid_df
    }
    
    # Create features
    feature_df = create_feature_pipeline(data_sources, DEFAULT_FEATURE_CONFIG)
    
    # Extract only the engineered features
    original_cols = rtlmp_df.columns.tolist()
    feature_cols = [col for col in feature_df.columns if col not in original_cols]
    features_only_df = feature_df[feature_cols]
    
    # Create mock importance scores
    num_features = len(feature_cols)
    importances = {}
    for i, col in enumerate(feature_cols):
        importances[col] = (num_features - i) / num_features  # Decreasing importance
    
    # Create a FeatureSelector instance
    selector = FeatureSelector({
        'methods': ['importance', 'correlation'],
        'importance_threshold': 0.5,
        'correlation_threshold': 0.95
    })
    
    # Set importance scores
    selector.set_importance_scores(importances)
    
    # Select features
    selected_features = selector.select_features(features_only_df)
    assert len(selected_features) > 0, "No features were selected"
    
    # Test filtered DataFrame
    filtered_df = selector.filter_dataframe(feature_df)
    assert len(filtered_df.columns) < len(feature_df.columns), "Feature filtering should reduce columns"
    assert all(col in filtered_df.columns for col in selected_features), "Not all selected features in filtered DataFrame"


def test_feature_engineering_with_missing_data(setup_test_data):
    """
    Tests feature engineering robustness with missing data.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    rtlmp_df = setup_test_data['rtlmp_df']
    
    # Create a modified DataFrame with missing values
    df_with_missing = rtlmp_df.copy()
    # Introduce missing values in price column
    missing_indices = np.random.choice(df_with_missing.index, size=len(df_with_missing) // 10, replace=False)
    df_with_missing.loc[missing_indices, 'price'] = np.nan
    
    # Test statistical features with missing data
    try:
        result_df = create_all_statistical_features(df_with_missing, 'price', 'timestamp')
        # Verify that the function doesn't crash with missing data
        assert True, "Function should handle missing data without crashing"
        
        # Verify that NaN values are handled appropriately
        # Check if features were created despite missing data
        assert 'rolling_mean_24h' in result_df.columns
        assert 'volatility_24h' in result_df.columns
        
    except Exception as e:
        assert False, f"Feature engineering should handle missing data but failed with: {str(e)}"


def test_feature_engineering_with_edge_cases(setup_test_data):
    """
    Tests feature engineering with edge cases like extreme values.
    
    Args:
        setup_test_data: Pytest fixture providing test data
    """
    # Extract test data
    rtlmp_df = setup_test_data['rtlmp_df']
    
    # Create a modified DataFrame with extreme values
    df_with_extremes = rtlmp_df.copy()
    # Add some extreme price spikes
    extreme_indices = np.random.choice(df_with_extremes.index, size=5, replace=False)
    df_with_extremes.loc[extreme_indices, 'price'] = 9999.0  # Very high price
    
    # Test statistical features with extreme values
    try:
        result_df = create_all_statistical_features(df_with_extremes, 'price', 'timestamp')
        # Verify that the function handles extreme values appropriately
        assert True, "Function should handle extreme values without crashing"
        
        # Verify that spike indicators correctly identify the extreme values
        for threshold in PRICE_SPIKE_THRESHOLDS:
            feature_name = f'price_spike_{threshold}'
            if threshold < 9999.0:
                # Extreme indices should be marked as spikes
                for idx in extreme_indices:
                    if idx in result_df.index:
                        assert result_df.loc[idx, feature_name] == 1, f"Extreme value should be detected as spike by {feature_name}"
        
        # Verify that rolling statistics include extreme values but aren't broken by them
        assert 'rolling_max_24h' in result_df.columns
        assert result_df['rolling_max_24h'].max() >= 9999.0, "Rolling max should capture extreme values"
        
    except Exception as e:
        assert False, f"Feature engineering should handle extreme values but failed with: {str(e)}"