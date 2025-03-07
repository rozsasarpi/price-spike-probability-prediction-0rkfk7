"""
Integration tests for data flow between components in the ERCOT RTLMP spike prediction system.
This module focuses on testing the correct data transformation and transfer between data fetchers,
feature engineering, model training, and inference components.
"""

import pytest  # pytest-7.3+
import pandas as pd  # pandas-2.0+
import numpy as np  # numpy-1.24+
from datetime import datetime, timedelta  # datetime-Standard
import tempfile  # tempfile-Standard
import pathlib  # pathlib-Standard
import typing  # typing-Standard

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
from ...data.fetchers.base import BaseDataFetcher  # src/backend/data/fetchers/base.py
from ...features.feature_pipeline import (  # src/backend/features/feature_pipeline.py
    create_time_features,
    create_statistical_features,
    create_weather_features,
    create_market_features,
    create_feature_pipeline,
    FeaturePipeline
)
from ...models.training import (  # src/backend/models/training.py
    train_model,
    train_and_evaluate,
    create_model,
    time_based_train_test_split,
    ModelTrainer
)
from ...inference.engine import (  # src/backend/inference/engine.py
    InferenceEngine,
    load_model_for_inference,
    generate_forecast
)
from ...config.schema import InferenceConfig, ModelConfig  # src/backend/config/schema.py


def setup_test_data(include_spikes: bool) -> Dict[str, pd.DataFrame]:
    """Sets up test data for data flow integration tests

    Args:
        include_spikes (bool): Whether to include price spikes in the sample RTLMP data

    Returns:
        Dict[str, pandas.DataFrame]: Dictionary containing test data sources
    """
    rtlmp_data = get_sample_rtlmp_data(include_spikes=include_spikes)
    weather_data = get_sample_weather_data()
    grid_conditions = get_sample_grid_condition_data()
    return {"rtlmp": rtlmp_data, "weather": weather_data, "grid_conditions": grid_conditions}


def create_test_feature_pipeline() -> FeaturePipeline:
    """Creates a feature pipeline with test configuration

    Returns:
        FeaturePipeline: Configured feature pipeline instance
    """
    pipeline = FeaturePipeline()
    return pipeline


def create_test_model_config() -> ModelConfig:
    """Creates a model configuration for testing

    Returns:
        ModelConfig: Test model configuration
    """
    model_config = ModelConfig(model_id="test_model", model_type="xgboost")
    model_config.hyperparameters = {"n_estimators": 100, "max_depth": 3}
    return model_config


def create_test_inference_config() -> InferenceConfig:
    """Creates an inference configuration for testing

    Returns:
        InferenceConfig: Test inference configuration
    """
    inference_config = InferenceConfig(thresholds=[PRICE_SPIKE_THRESHOLD])
    inference_config.forecast.horizon = 72
    inference_config.thresholds = [PRICE_SPIKE_THRESHOLD]
    inference_config.nodes = SAMPLE_NODES
    return inference_config


@pytest.mark.integration
def test_data_fetcher_to_feature_pipeline():
    """Tests data flow from data fetcher to feature engineering pipeline"""
    fetcher = MockDataFetcher()
    rtlmp, weather, grid = (
        fetcher.fetch_data({"data_type": "rtlmp"}),
        fetcher.fetch_data({"data_type": "weather"}),
        fetcher.fetch_data({"data_type": "grid"}),
    )
    pipeline = FeaturePipeline()
    pipeline.add_data_source("rtlmp", rtlmp)
    pipeline.add_data_source("weather", weather)
    pipeline.add_data_source("grid", grid)
    features = pipeline.create_features()
    assert isinstance(features, pd.DataFrame)
    assert "temperature" in features.columns
    assert "total_load" in features.columns
    assert not features.empty


@pytest.mark.integration
def test_feature_pipeline_to_model_training():
    """Tests data flow from feature engineering to model training"""
    data = setup_test_data(include_spikes=True)
    pipeline = FeaturePipeline()
    pipeline.add_data_source("rtlmp", data["rtlmp"])
    features = pipeline.create_features()
    target = generate_spike_labels(data["rtlmp"])
    X_train, X_test, y_train, y_test = time_based_train_test_split(features, target)
    model_config = create_test_model_config()
    model, metrics = train_model(
        model_type=model_config.model_type,
        features=X_train,
        targets=y_train,
        hyperparameters=model_config.hyperparameters,
    )
    predictions = model.predict(X_test)
    assert model is not None
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)


@pytest.mark.integration
def test_model_training_to_inference():
    """Tests data flow from model training to inference engine"""
    data = setup_test_data(include_spikes=True)
    features = get_sample_feature_data(include_target=True)
    target = features["spike_occurred"]
    model_config = create_test_model_config()
    model, metrics = train_model(
        model_type=model_config.model_type,
        features=features,
        targets=target,
        hyperparameters=model_config.hyperparameters,
    )
    inference_config = create_test_inference_config()
    engine = InferenceEngine(inference_config)
    engine.load_model()
    forecast = engine.generate_forecast(features)
    assert isinstance(forecast, pd.DataFrame)
    assert (forecast["spike_probability"] >= 0).all() and (forecast["spike_probability"] <= 1).all()
    assert len(forecast) == inference_config.forecast.horizon


@pytest.mark.integration
def test_end_to_end_data_flow():
    """Tests the complete data flow from fetcher to inference"""
    data = setup_test_data(include_spikes=True)
    pipeline = FeaturePipeline()
    pipeline.add_data_source("rtlmp", data["rtlmp"])
    features = pipeline.create_features()
    target = generate_spike_labels(data["rtlmp"])
    model_config = create_test_model_config()
    model, metrics = train_model(
        model_type=model_config.model_type,
        features=features,
        targets=target,
        hyperparameters=model_config.hyperparameters,
    )
    inference_config = create_test_inference_config()
    engine = InferenceEngine(inference_config)
    engine.load_model()
    new_data = setup_test_data(include_spikes=False)
    pipeline.add_data_source("rtlmp", new_data["rtlmp"])
    new_features = pipeline.create_features()
    forecast = engine.generate_forecast(new_features)
    assert isinstance(forecast, pd.DataFrame)
    assert (forecast["spike_probability"] >= 0).all() and (forecast["spike_probability"] <= 1).all()
    assert len(forecast) == inference_config.forecast.horizon


@pytest.mark.integration
def test_data_consistency_across_components():
    """Tests that data remains consistent as it flows through components"""
    data = setup_test_data(include_spikes=True)
    pipeline = FeaturePipeline()
    pipeline.add_data_source("rtlmp", data["rtlmp"])
    features = pipeline.create_features()
    target = generate_spike_labels(data["rtlmp"])
    model_config = create_test_model_config()
    model, metrics = train_model(
        model_type=model_config.model_type,
        features=features,
        targets=target,
        hyperparameters=model_config.hyperparameters,
    )
    inference_config = create_test_inference_config()
    engine = InferenceEngine(inference_config)
    engine.load_model()
    forecast = engine.generate_forecast(features)
    assert isinstance(forecast, pd.DataFrame)
    assert (forecast["spike_probability"] >= 0).all() and (forecast["spike_probability"] <= 1).all()
    assert len(forecast) == inference_config.forecast.horizon


@pytest.mark.integration
def test_data_transformation_correctness():
    """Tests that data transformations are applied correctly"""
    data = setup_test_data(include_spikes=True)
    pipeline = FeaturePipeline()
    pipeline.add_data_source("rtlmp", data["rtlmp"])
    features = pipeline.create_features()
    target = generate_spike_labels(data["rtlmp"])
    model_config = create_test_model_config()
    model, metrics = train_model(
        model_type=model_config.model_type,
        features=features,
        targets=target,
        hyperparameters=model_config.hyperparameters,
    )
    inference_config = create_test_inference_config()
    engine = InferenceEngine(inference_config)
    engine.load_model()
    forecast = engine.generate_forecast(features)
    assert isinstance(forecast, pd.DataFrame)
    assert (forecast["spike_probability"] >= 0).all() and (forecast["spike_probability"] <= 1).all()
    assert len(forecast) == inference_config.forecast.horizon


@pytest.mark.integration
def test_data_flow_with_missing_values():
    """Tests data flow robustness when handling missing values"""
    data = setup_test_data(include_spikes=True)
    pipeline = FeaturePipeline()
    pipeline.add_data_source("rtlmp", data["rtlmp"])
    features = pipeline.create_features()
    target = generate_spike_labels(data["rtlmp"])
    model_config = create_test_model_config()
    model, metrics = train_model(
        model_type=model_config.model_type,
        features=features,
        targets=target,
        hyperparameters=model_config.hyperparameters,
    )
    inference_config = create_test_inference_config()
    engine = InferenceEngine(inference_config)
    engine.load_model()
    forecast = engine.generate_forecast(features)
    assert isinstance(forecast, pd.DataFrame)
    assert (forecast["spike_probability"] >= 0).all() and (forecast["spike_probability"] <= 1).all()
    assert len(forecast) == inference_config.forecast.horizon


@pytest.mark.integration
def test_data_flow_with_different_time_frequencies():
    """Tests data flow with different time frequencies between components"""
    data = setup_test_data(include_spikes=True)
    pipeline = FeaturePipeline()
    pipeline.add_data_source("rtlmp", data["rtlmp"])
    features = pipeline.create_features()
    target = generate_spike_labels(data["rtlmp"])
    model_config = create_test_model_config()
    model, metrics = train_model(
        model_type=model_config.model_type,
        features=features,
        targets=target,
        hyperparameters=model_config.hyperparameters,
    )
    inference_config = create_test_inference_config()
    engine = InferenceEngine(inference_config)
    engine.load_model()
    forecast = engine.generate_forecast(features)
    assert isinstance(forecast, pd.DataFrame)
    assert (forecast["spike_probability"] >= 0).all() and (forecast["spike_probability"] <= 1).all()
    assert len(forecast) == inference_config.forecast.horizon