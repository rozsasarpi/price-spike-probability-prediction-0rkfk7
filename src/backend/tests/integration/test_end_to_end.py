"""
Integration test module that verifies the end-to-end functionality of the ERCOT RTLMP spike prediction system.
This module tests the complete workflow from data fetching through feature engineering, model training, and inference to ensure all components work together correctly.
"""

import pytest  # pytest-7.3+
import pandas as pd  # pandas-2.0+
import numpy as np  # numpy-1.24+
import tempfile  # Standard
import pathlib  # Standard
from datetime import datetime, timedelta  # Standard
from typing import List, Dict  # Standard

from ..fixtures.sample_data import (  # src/backend/tests/fixtures/sample_data.py
    get_sample_rtlmp_data,
    get_sample_weather_data,
    get_sample_grid_condition_data,
    get_sample_feature_data,
    generate_spike_labels,
    SAMPLE_NODES,
    SAMPLE_START_DATE,
    SAMPLE_END_DATE,
    PRICE_SPIKE_THRESHOLD
)
from ...data.fetchers.mock import MockDataFetcher  # src/backend/data/fetchers/mock.py
from ...features.feature_pipeline import FeaturePipeline, DEFAULT_FEATURE_CONFIG  # src/backend/features/feature_pipeline.py
from ...models.training import train_model, MODEL_TYPES  # src/backend/models/training.py
from ...inference.engine import InferenceEngine  # src/backend/inference/engine.py
from ...backtesting.framework import BacktestingFramework  # src/backend/backtesting/framework.py
from ...backtesting.scenario_definitions import ScenarioConfig  # src/backend/backtesting/scenario_definitions.py
from ...config.schema import InferenceConfig  # src/backend/config/schema.py


def setup_test_data(start_date: datetime, end_date: datetime, nodes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Creates a set of test data for end-to-end testing

    Args:
        start_date (datetime): start_date
        end_date (datetime): end_date
        nodes (List[str]): nodes

    Returns:
        Dict[str, DataFrame]: Dictionary of test data sources
    """
    # Generate RTLMP data using get_sample_rtlmp_data
    rtlmp_df = get_sample_rtlmp_data(start_date=start_date, end_date=end_date, nodes=nodes)
    # Generate weather data using get_sample_weather_data
    weather_df = get_sample_weather_data(start_date=start_date, end_date=end_date)
    # Generate grid condition data using get_sample_grid_condition_data
    grid_df = get_sample_grid_condition_data(start_date=start_date, end_date=end_date)
    # Return dictionary with rtlmp_df, weather_df, and grid_df keys
    return {"rtlmp_df": rtlmp_df, "weather_df": weather_df, "grid_df": grid_df}


def setup_mock_data_fetcher() -> MockDataFetcher:
    """
    Creates and configures a MockDataFetcher for testing

    Args:

    Returns:
        MockDataFetcher: Configured mock data fetcher
    """
    # Create a MockDataFetcher instance
    fetcher = MockDataFetcher()
    # Configure the fetcher with appropriate test settings
    return fetcher


def create_temp_model_path() -> pathlib.Path:
    """
    Creates a temporary directory for model storage

    Args:

    Returns:
        Path: Path to temporary model directory
    """
    # Create a temporary directory using tempfile.TemporaryDirectory
    temp_dir = tempfile.TemporaryDirectory()
    # Return the path as a pathlib.Path object
    return pathlib.Path(temp_dir.name)


def test_end_to_end_data_flow():
    """
    Tests the complete data flow from fetching to feature engineering

    Args:

    """
    # Set up test data using setup_test_data
    test_data = setup_test_data(start_date=SAMPLE_START_DATE, end_date=SAMPLE_END_DATE, nodes=SAMPLE_NODES)
    # Create a MockDataFetcher using setup_mock_data_fetcher
    fetcher = setup_mock_data_fetcher()
    # Fetch historical data using the mock fetcher
    rtlmp_df = fetcher.fetch_historical_data(start_date=SAMPLE_START_DATE, end_date=SAMPLE_END_DATE, identifiers=SAMPLE_NODES)
    # Create a FeaturePipeline instance
    feature_pipeline = FeaturePipeline()
    # Add data sources to the pipeline
    feature_pipeline.add_data_source("rtlmp_df", rtlmp_df)
    # Generate features using the pipeline
    features = feature_pipeline.create_features()
    # Assert that features DataFrame has expected shape and columns
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    # Assert that feature values are within expected ranges
    assert (features["hour_of_day"] >= 0).all()
    assert (features["hour_of_day"] <= 23).all()


def test_end_to_end_model_training():
    """
    Tests the model training workflow with feature data

    Args:

    """
    # Set up test data using setup_test_data
    test_data = setup_test_data(start_date=SAMPLE_START_DATE, end_date=SAMPLE_END_DATE, nodes=SAMPLE_NODES)
    # Create a FeaturePipeline and generate features
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_data_source("rtlmp_df", test_data["rtlmp_df"])
    features = feature_pipeline.create_features()
    # Generate target labels using generate_spike_labels
    target = generate_spike_labels(test_data["rtlmp_df"])
    # Create a temporary model directory using create_temp_model_path
    temp_model_path = create_temp_model_path()
    # Train a model using train_model with features and targets
    model, metrics = train_model(
        model_type="xgboost", features=features, targets=target, model_path=temp_model_path
    )
    # Assert that model training completes successfully
    assert model is not None
    # Assert that model performance metrics are reasonable
    assert metrics["auc"] > 0.5
    # Assert that model file is created in the temporary directory
    assert (temp_model_path / "model.joblib").exists()


def test_end_to_end_inference():
    """
    Tests the inference workflow with a trained model

    Args:

    """
    # Set up test data using setup_test_data
    test_data = setup_test_data(start_date=SAMPLE_START_DATE, end_date=SAMPLE_END_DATE, nodes=SAMPLE_NODES)
    # Create a FeaturePipeline and generate features
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_data_source("rtlmp_df", test_data["rtlmp_df"])
    features = feature_pipeline.create_features()
    # Generate target labels using generate_spike_labels
    target = generate_spike_labels(test_data["rtlmp_df"])
    # Create a temporary model directory using create_temp_model_path
    temp_model_path = create_temp_model_path()
    # Train a model using train_model with features and targets
    model, metrics = train_model(
        model_type="xgboost", features=features, targets=target, model_path=temp_model_path
    )
    # Create an InferenceEngine instance with appropriate configuration
    inference_config = InferenceConfig(thresholds=[PRICE_SPIKE_THRESHOLD])
    inference_engine = InferenceEngine(config=inference_config)
    # Load the trained model into the inference engine
    inference_engine.load_model(model_path=temp_model_path)
    # Generate a forecast using the inference engine
    forecast = inference_engine.generate_forecast(data_sources={"rtlmp_df": features})
    # Assert that forecast has expected shape and structure
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) > 0
    # Assert that probability values are between 0 and 1
    assert (forecast["spike_probability"] >= 0).all()
    assert (forecast["spike_probability"] <= 1).all()
    # Assert that forecast covers the expected time horizon
    assert len(forecast) == 72


def test_end_to_end_backtesting():
    """
    Tests the backtesting workflow with historical data

    Args:

    """
    # Set up a MockDataFetcher using setup_mock_data_fetcher
    fetcher = setup_mock_data_fetcher()
    # Create a temporary model directory using create_temp_model_path
    temp_model_path = create_temp_model_path()
    # Create a ScenarioConfig for backtesting
    scenario_config = ScenarioConfig(
        name="Test Backtesting",
        start_date=SAMPLE_START_DATE,
        end_date=SAMPLE_END_DATE,
        thresholds=[PRICE_SPIKE_THRESHOLD],
        nodes=SAMPLE_NODES,
        forecast_horizon=72,
    )
    # Create a BacktestingFramework with the mock fetcher
    backtesting_framework = BacktestingFramework(data_fetcher=fetcher, model_path=temp_model_path)
    # Execute the backtesting scenario
    backtesting_framework.execute_scenario(scenario_config)
    # Get the backtesting results
    results = backtesting_framework.get_results(scenario_name="Test Backtesting")
    # Assert that results contain expected metrics
    assert "metrics" in results
    assert "window_results" in results
    # Assert that metrics are within reasonable ranges
    assert results["metrics"][PRICE_SPIKE_THRESHOLD]["auc"] > 0.5
    # Assert that results contain forecasts and actuals for comparison
    assert len(results["window_results"]) > 0


def test_complete_pipeline_execution():
    """
    Tests the complete pipeline from data to forecast

    Args:

    """
    # Set up test data using setup_test_data
    test_data = setup_test_data(start_date=SAMPLE_START_DATE, end_date=SAMPLE_END_DATE, nodes=SAMPLE_NODES)
    # Create a MockDataFetcher using setup_mock_data_fetcher
    fetcher = setup_mock_data_fetcher()
    # Create a temporary model directory using create_temp_model_path
    temp_model_path = create_temp_model_path()
    # Create a FeaturePipeline and generate features
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_data_source("rtlmp_df", test_data["rtlmp_df"])
    features = feature_pipeline.create_features()
    # Generate target labels using generate_spike_labels
    target = generate_spike_labels(test_data["rtlmp_df"])
    # Train a model using train_model with features and targets
    model, metrics = train_model(
        model_type="xgboost", features=features, targets=target, model_path=temp_model_path
    )
    # Create an InferenceEngine instance with appropriate configuration
    inference_config = InferenceConfig(thresholds=[PRICE_SPIKE_THRESHOLD])
    inference_engine = InferenceEngine(config=inference_config)
    # Load the trained model into the inference engine
    inference_engine.load_model(model_path=temp_model_path)
    # Generate a forecast using the inference engine
    forecast = inference_engine.generate_forecast(data_sources={"rtlmp_df": features})
    # Create a ScenarioConfig for backtesting
    scenario_config = ScenarioConfig(
        name="Test Complete Pipeline",
        start_date=SAMPLE_START_DATE,
        end_date=SAMPLE_END_DATE,
        thresholds=[PRICE_SPIKE_THRESHOLD],
        nodes=SAMPLE_NODES,
        forecast_horizon=72,
    )
    # Create a BacktestingFramework with the mock fetcher
    backtesting_framework = BacktestingFramework(data_fetcher=fetcher, model_path=temp_model_path)
    # Execute the backtesting scenario
    backtesting_framework.execute_scenario(scenario_config)
    # Assert that all components work together correctly
    assert True
    # Assert that the entire pipeline produces expected results
    results = backtesting_framework.get_results(scenario_name="Test Complete Pipeline")
    assert "metrics" in results
    assert "window_results" in results
    assert results["metrics"][PRICE_SPIKE_THRESHOLD]["auc"] > 0.5
    assert len(results["window_results"]) > 0