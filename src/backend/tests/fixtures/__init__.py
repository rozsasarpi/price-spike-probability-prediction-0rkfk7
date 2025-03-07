"""
Initialization file for the fixtures package that exports test fixtures for the ERCOT RTLMP spike prediction system.

This module makes sample data generators and mock API responses available to test modules,
enabling consistent and reproducible testing across unit and integration tests.
"""

# Import sample data generators and constants
from .sample_data import (
    get_sample_rtlmp_data,
    get_sample_weather_data,
    get_sample_grid_condition_data,
    get_sample_feature_data,
    generate_spike_labels,
    generate_hourly_spike_labels,
    SAMPLE_NODES,
    SAMPLE_START_DATE,
    SAMPLE_END_DATE,
    PRICE_SPIKE_THRESHOLD,
    DEFAULT_LOCATIONS
)

# Import mock API response generators
from .mock_responses import (
    get_mock_rtlmp_response,
    get_mock_grid_conditions_response,
    get_mock_weather_response,
    generate_error_response
)