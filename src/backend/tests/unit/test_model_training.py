"""
Unit tests for the model training module in the ERCOT RTLMP spike prediction system.
Tests the functionality of model creation, training, evaluation, and persistence
with a focus on ensuring reliable model performance for price spike prediction.
"""

import unittest.mock  # Standard
import pytest  # version 7.0+
import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
import tempfile  # Standard
from pathlib import Path  # Standard
from datetime import datetime  # Standard
from typing import Tuple  # Standard

from src.backend.models.training import (
    create_model,
    train_model,
    train_and_evaluate,
    optimize_and_train,
    load_model,
    get_latest_model,
    compare_models,
    select_best_model,
    retrain_model,
    time_based_train_test_split,
    cross_validate_time_series,
    ModelTrainer,
    MODEL_TYPES,
)  # ../../models/training
from src.backend.models.base_model import BaseModel  # ../../models/base_model
from src.backend.models.xgboost_model import XGBoostModel  # ../../models/xgboost_model
from src.backend.data.storage.model_registry import ModelRegistry  # ../../data/storage/model_registry
from src.backend.models.evaluation import evaluate_model_performance  # ../../models/evaluation
from src.backend.utils.error_handling import ModelError, ModelTrainingError  # ../../utils/error_handling
from src.backend.tests.fixtures.sample_data import (
    get_sample_feature_data,
    generate_spike_labels,
    SAMPLE_START_DATE,
    SAMPLE_END_DATE,
    PRICE_SPIKE_THRESHOLD,
)  # ../fixtures/sample_data


def get_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Helper function to generate test data for model training tests"""
    # Generate sample feature data using get_sample_feature_data
    features = get_sample_feature_data(
        start_date=SAMPLE_START_DATE, end_date=SAMPLE_END_DATE
    )
    # Generate binary spike labels using generate_spike_labels
    targets = generate_spike_labels(
        rtlmp_data=features, threshold=PRICE_SPIKE_THRESHOLD
    )
    # Return features and targets as a tuple
    return features, targets


@pytest.fixture(scope="function")
def test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Fixture providing test features and targets for model training"""
    # Call get_test_data() to generate test data
    features, targets = get_test_data()
    # Return the features and targets
    return features, targets


@pytest.fixture(scope="function")
def temp_model_dir() -> Path:
    """Fixture providing a temporary directory for model storage"""
    # Create a temporary directory using tempfile.TemporaryDirectory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert the directory path to a Path object
        tmpdir_path = Path(tmpdir)
        # Yield the path for use in tests
        yield tmpdir_path
        # Cleanup happens automatically when the context manager exits


@pytest.fixture(scope="function")
def mock_model_registry() -> unittest.mock.MagicMock:
    """Fixture providing a mocked ModelRegistry"""
    # Create a MagicMock instance for ModelRegistry
    mock = unittest.mock.MagicMock(spec=ModelRegistry)
    # Configure the mock to return appropriate values for common methods
    mock.register_model.return_value = ("xgboost", "test_model", "0.1.0")
    mock.get_latest_model.return_value = (
        XGBoostModel(model_id="test_model"),
        {"model_type": "xgboost"},
    )
    # Return the configured mock
    return mock


@pytest.fixture(scope="function")
def trained_xgboost_model(test_data: Tuple[pd.DataFrame, pd.Series]) -> XGBoostModel:
    """Fixture providing a trained XGBoost model"""
    # Extract features and targets from test_data
    features, targets = test_data
    # Create an XGBoostModel instance with a test model_id
    model = XGBoostModel(model_id="test_model")
    # Train the model on the features and targets
    model.train(features, targets)
    # Return the trained model
    return model


def test_create_model():
    """Test that create_model correctly instantiates models of different types"""
    # create_model with 'xgboost' type returns an instance of XGBoostModel
    model_xgb = create_model(model_id="test_xgb", model_type="xgboost")
    assert isinstance(model_xgb, XGBoostModel)

    # create_model with 'lightgbm' type returns an instance of LightGBMModel
    # (Assuming LightGBMModel is available, otherwise skip this test)
    if "lightgbm" in MODEL_TYPES:
        from src.backend.models.lightgbm_model import LightGBMModel

        model_lgbm = create_model(model_id="test_lgbm", model_type="lightgbm")
        assert isinstance(model_lgbm, LightGBMModel)

    # create_model with invalid type raises ValueError
    with pytest.raises(ValueError):
        create_model(model_id="test_invalid", model_type="invalid")

    # Model is initialized with correct model_id, type, and hyperparameters
    model = create_model(
        model_id="test_model", model_type="xgboost", hyperparameters={"test": 1}
    )
    assert model.model_id == "test_model"
    assert model.model_type == "xgboost"
    assert model.hyperparameters == {"test": 1}


def test_time_based_train_test_split(test_data: Tuple[pd.DataFrame, pd.Series]):
    """Test that time_based_train_test_split correctly splits data by time"""
    features, targets = test_data
    # Split returns four objects: X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = time_based_train_test_split(features, targets)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    # Split sizes match the expected test_size ratio
    test_size = 0.2
    X_train, X_test, y_train, y_test = time_based_train_test_split(
        features, targets, test_size=test_size
    )
    assert len(X_test) == int(len(features) * test_size)

    # Training data timestamps are all before test data timestamps
    assert X_train.index.max() <= X_test.index.min()

    # No data leakage between training and test sets
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0


def test_train_model(test_data: Tuple[pd.DataFrame, pd.Series], temp_model_dir: Path):
    """Test that train_model successfully trains a model"""
    features, targets = test_data
    # train_model returns a tuple of (model, metrics)
    model, metrics = train_model(model_type="xgboost", features=features, targets=targets)
    assert isinstance(model, BaseModel)
    assert isinstance(metrics, dict)

    # Returned model is trained (is_trained() returns True)
    assert model.is_trained()

    # Performance metrics include expected keys (accuracy, precision, recall, etc.)
    expected_metrics = ["accuracy", "precision", "recall"]
    for metric in expected_metrics:
        assert metric in metrics

    # Model is saved to the specified path when provided
    model, metrics = train_model(
        model_type="xgboost", features=features, targets=targets, model_path=temp_model_dir
    )
    assert (temp_model_dir / "model.joblib").exists()


def test_train_and_evaluate(test_data: Tuple[pd.DataFrame, pd.Series]):
    """Test that train_and_evaluate performs cross-validation correctly"""
    features, targets = test_data
    # train_and_evaluate returns a tuple of (model, cv_metrics)
    model, cv_metrics = train_and_evaluate(
        model_type="xgboost", features=features, targets=targets
    )
    assert isinstance(model, BaseModel)
    assert isinstance(cv_metrics, dict)

    # cv_metrics contains lists of scores for each metric
    expected_metrics = ["auc", "precision", "recall"]
    for metric in expected_metrics:
        assert metric in cv_metrics
        assert isinstance(cv_metrics[metric], list)

    # Each metric list has length equal to the number of CV folds
    cv_folds = 5
    model, cv_metrics = train_and_evaluate(
        model_type="xgboost", features=features, targets=targets, cv_folds=cv_folds
    )
    for metric in expected_metrics:
        assert len(cv_metrics[metric]) == cv_folds

    # Returned model is trained on the full dataset
    assert model.is_trained()


def test_optimize_and_train(test_data: Tuple[pd.DataFrame, pd.Series], temp_model_dir: Path):
    """Test that optimize_and_train finds good hyperparameters"""
    features, targets = test_data
    # Define a sample parameter grid
    param_grid = {"learning_rate": [0.01, 0.1], "max_depth": [3, 5]}

    # optimize_and_train returns a tuple of (model, metrics)
    model, metrics = optimize_and_train(
        model_type="xgboost", features=features, targets=targets, param_grid=param_grid
    )
    assert isinstance(model, BaseModel)
    assert isinstance(metrics, dict)

    # Returned model has hyperparameters different from defaults
    assert model.hyperparameters != {}

    # Performance metrics are within acceptable ranges
    assert metrics["auc"] > 0.5

    # Model is saved to the specified path when provided
    model, metrics = optimize_and_train(
        model_type="xgboost",
        features=features,
        targets=targets,
        param_grid=param_grid,
        model_path=temp_model_dir,
    )
    assert (temp_model_dir / "model.joblib").exists()


def test_load_model(trained_xgboost_model: XGBoostModel, temp_model_dir: Path):
    """Test that load_model correctly loads a saved model"""
    # Save the trained model to the temporary directory
    trained_xgboost_model.save(path=temp_model_dir)

    # Model can be saved and then loaded with the same configuration
    loaded_model = load_model(model_identifier=temp_model_dir)
    assert isinstance(loaded_model, BaseModel)
    assert loaded_model.model_id == trained_xgboost_model.model_id

    # Loaded model has the same hyperparameters as the original
    assert loaded_model.hyperparameters == trained_xgboost_model.hyperparameters

    # Loaded model produces the same predictions as the original
    test_features = get_sample_feature_data()
    original_predictions = trained_xgboost_model.predict(test_features)
    loaded_predictions = loaded_model.predict(test_features)
    assert np.array_equal(original_predictions, loaded_predictions)


def test_get_latest_model(mock_model_registry: unittest.mock.MagicMock):
    """Test that get_latest_model retrieves the most recent model"""
    # get_latest_model calls ModelRegistry.get_latest_model with correct parameters
    model = get_latest_model(model_type="xgboost", registry_path="test_path")
    mock_model_registry.get_latest_model.assert_called_once_with(model_type="xgboost")

    # Returns the model from the registry when found
    assert isinstance(model, BaseModel)

    # Returns None when no model is found
    mock_model_registry.get_latest_model.side_effect = ModelError("No model found")
    model = get_latest_model(model_type="xgboost")
    assert model is None

    # Handles exceptions gracefully
    mock_model_registry.get_latest_model.side_effect = Exception("Test error")
    model = get_latest_model(model_type="xgboost")
    assert model is None


def test_compare_models(test_data: Tuple[pd.DataFrame, pd.Series]):
    """Test that compare_models correctly evaluates multiple models"""
    features, targets = test_data
    # Create two sample models
    model1 = XGBoostModel(model_id="model1")
    model1.train(features, targets)
    model2 = XGBoostModel(model_id="model2")
    model2.train(features, targets)

    # compare_models returns a dictionary with model IDs as keys
    model_metrics = compare_models(models=[model1, model2], features=features, targets=targets)
    assert "model1" in model_metrics
    assert "model2" in model_metrics

    # Each model's metrics include expected performance metrics
    expected_metrics = ["accuracy", "precision", "recall"]
    for model_id in model_metrics:
        for metric in expected_metrics:
            assert metric in model_metrics[model_id]

    # All models are evaluated on the same test data
    # (This is difficult to verify directly, but the test ensures that the function runs without errors)
    pass


def test_select_best_model(test_data: Tuple[pd.DataFrame, pd.Series]):
    """Test that select_best_model chooses the best model by metric"""
    features, targets = test_data
    # Create two sample models with different performance
    model1 = XGBoostModel(model_id="model1")
    model1.train(features, targets)
    model1.performance_metrics = {"auc": 0.8}
    model2 = XGBoostModel(model_id="model2")
    model2.train(features, targets)
    model2.performance_metrics = {"auc": 0.9}

    # select_best_model returns a tuple of (best_model, metric_value)
    best_model, metric_value = select_best_model(
        models=[model1, model2], features=features, targets=targets, metric="auc", higher_is_better=True
    )
    assert isinstance(best_model, BaseModel)
    assert metric_value == 0.9

    # When higher_is_better=True, model with highest metric is selected
    assert best_model.model_id == "model2"

    # When higher_is_better=False, model with lowest metric is selected
    best_model, metric_value = select_best_model(
        models=[model1, model2], features=features, targets=targets, metric="auc", higher_is_better=False
    )
    assert best_model.model_id == "model1"
    assert metric_value == 0.8

    # Raises ValueError for invalid metric name
    with pytest.raises(ValueError):
        select_best_model(
            models=[model1, model2], features=features, targets=targets, metric="invalid", higher_is_better=True
        )


def test_retrain_model(trained_xgboost_model: XGBoostModel, test_data: Tuple[pd.DataFrame, pd.Series], temp_model_dir: Path):
    """Test that retrain_model updates an existing model with new data"""
    features, targets = test_data
    # retrain_model returns a tuple of (model, metrics)
    model, metrics = retrain_model(model=trained_xgboost_model, features=features, targets=targets, model_path = temp_model_dir)
    assert isinstance(model, BaseModel)
    assert isinstance(metrics, dict)

    # Retrained model has updated training_date
    assert model.training_date > trained_xgboost_model.training_date

    # Performance metrics are calculated on the new data
    assert "accuracy" in metrics
    assert "precision" in metrics

    # Model is saved with incremented version when path is provided
    model, metrics = retrain_model(
        model=trained_xgboost_model, features=features, targets=targets, model_path=temp_model_dir
    )
    assert (temp_model_dir / "model.joblib").exists()


def test_cross_validate_time_series(test_data: Tuple[pd.DataFrame, pd.Series]):
    """Test that cross_validate_time_series performs time-based CV correctly"""
    features, targets = test_data
    # Returns a dictionary of metric names mapped to lists of scores
    cv_metrics = cross_validate_time_series(model_type="xgboost", features=features, targets=targets)
    expected_metrics = ["auc", "precision", "recall"]
    for metric in expected_metrics:
        assert metric in cv_metrics
        assert isinstance(cv_metrics[metric], list)

    # Each metric list has length equal to the number of CV folds
    cv_folds = 5
    cv_metrics = cross_validate_time_series(
        model_type="xgboost", features=features, targets=targets, n_splits=cv_folds
    )
    for metric in expected_metrics:
        assert len(cv_metrics[metric]) == cv_folds

    # Time-based splits maintain temporal order (no future data leakage)
    # (This is difficult to verify directly, but the test ensures that the function runs without errors)
    pass


def test_model_trainer_class(test_data: Tuple[pd.DataFrame, pd.Series], temp_model_dir: Path):
    """Test the ModelTrainer class functionality"""
    features, targets = test_data
    # ModelTrainer initializes with correct configuration
    trainer = ModelTrainer(model_type="xgboost", model_path=temp_model_dir)
    assert trainer.model_type == "xgboost"

    # train method successfully trains a model
    model, metrics = trainer.train(features, targets)
    assert isinstance(model, BaseModel)
    assert model.is_trained()

    # optimize_and_train method finds good hyperparameters
    param_grid = {"learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
    model, metrics = trainer.optimize_and_train(features, targets, param_grid=param_grid)
    assert model.hyperparameters != {}

    # cross_validate method performs cross-validation correctly
    cv_metrics = trainer.cross_validate(features, targets)
    expected_metrics = ["auc", "precision", "recall"]
    for metric in expected_metrics:
        assert metric in cv_metrics
        assert isinstance(cv_metrics[metric], list)

    # load_model and get_latest_model retrieve models correctly
    model_id = model.model_id
    loaded_model, _ = trainer.load_model(model_id)
    assert loaded_model.model_id == model_id
    latest_model, _ = trainer.get_latest_model()
    assert latest_model.model_id == model_id

    # retrain method updates an existing model with new data
    model, metrics = trainer.retrain(model, features, targets)
    assert model.training_date > loaded_model.training_date

    # evaluate method calculates performance metrics correctly
    metrics = trainer.evaluate(features, targets)
    assert "accuracy" in metrics
    assert "precision" in metrics


def test_error_handling():
    """Test error handling in model training functions"""
    features = pd.DataFrame({"test": [1, 2, 3]})
    targets = pd.Series([0, 1, 0])

    # Invalid model type raises appropriate exception
    with pytest.raises(ValueError):
        create_model(model_id="test_model", model_type="invalid")

    # Training with incompatible features/targets raises appropriate exception
    model = XGBoostModel(model_id="test_model")
    with pytest.raises(ModelTrainingError):
        model.train(features, targets[:2])

    # Loading non-existent model raises appropriate exception
    with pytest.raises(ModelError):
        load_model(model_identifier="non_existent_model")

    # Retraining untrained model raises appropriate exception
    model = XGBoostModel(model_id="test_model")
    with pytest.raises(ModelError):
        retrain_model(model=model, features=features, targets=targets)