"""
Fixtures package for CLI tests.

This package provides sample configurations and mock responses for testing CLI components
of the ERCOT RTLMP spike prediction system. It centralizes access to test fixtures to ensure
consistent test data across test modules.
"""

# Import sample configuration fixtures
from .sample_configs import (
    SAMPLE_CLI_CONFIG,
    SAMPLE_FETCH_DATA_PARAMS,
    SAMPLE_TRAIN_PARAMS,
    SAMPLE_PREDICT_PARAMS,
    SAMPLE_BACKTEST_PARAMS,
    SAMPLE_EVALUATE_PARAMS,
    SAMPLE_VISUALIZE_PARAMS,
    SAMPLE_COMMAND_PARAMS,
    get_sample_cli_config,
    get_sample_command_params,
    create_test_config,
    create_test_command_params,
)

# Import mock response fixtures
from .mock_responses import (
    MOCK_RTLMP_DATA,
    MOCK_WEATHER_DATA,
    MOCK_GRID_CONDITIONS_DATA,
    MOCK_COMBINED_DATA,
    MOCK_MODEL_TRAINING_RESULT,
    MOCK_MODEL_LIST,
    MOCK_MODEL_COMPARISON,
    MOCK_FORECAST_RESULT,
    MOCK_FORECAST_DATA,
    MOCK_BACKTEST_RESULT,
    MOCK_EVALUATION_METRICS,
    MOCK_VISUALIZATION_DATA,
    MOCK_ERROR_RESPONSES,
    get_mock_response,
    create_mock_dataframe,
    create_mock_model_info,
    create_mock_forecast,
    create_mock_error,
)