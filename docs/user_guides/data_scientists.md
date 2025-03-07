# Introduction

This guide provides comprehensive documentation for data scientists working with the ERCOT RTLMP spike prediction system. The system is designed to forecast the probability of price spikes in the Real-Time Locational Marginal Price (RTLMP) market before day-ahead market closure, enabling more informed bidding strategies for battery storage operators.

## Purpose and Scope

The ERCOT RTLMP spike prediction system helps battery storage operators optimize charging/discharging strategies by forecasting the probability of price spikes. As a data scientist, you'll be responsible for data preparation, feature engineering, model training, evaluation, and deployment of these forecasting models.

## System Overview

The system consists of several modular components:

- **Data Fetching Interface**: Retrieves ERCOT market data and weather forecasts
- **Feature Engineering Module**: Transforms raw data into model-ready features
- **Model Training Module**: Trains prediction models with cross-validation
- **Inference Engine**: Generates 72-hour probability forecasts
- **Backtesting Framework**: Simulates historical forecasts for model evaluation
- **Visualization and Metrics Tools**: Generates performance visualizations and metrics

As a data scientist, you'll primarily interact with the Feature Engineering, Model Training, and Backtesting components.

## Prerequisites

Before using the system, ensure you have:

- Python 3.10 or higher installed
- Required dependencies installed via `pip install -r requirements.txt`
- Access to ERCOT market data sources
- Basic understanding of machine learning for time series forecasting
- Familiarity with pandas, scikit-learn, and gradient boosting libraries

# Getting Started

This section covers how to set up your environment and run basic operations with the system.

## Installation

To install the ERCOT RTLMP spike prediction system:

```bash
# Clone the repository
git clone https://github.com/your-organization/ercot-rtlmp-prediction.git
cd ercot-rtlmp-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

This will install all required dependencies and make the CLI commands available.

## Configuration

The system uses YAML configuration files for various components. Key configuration files include:

1. **Data Configuration**: Specifies data sources and parameters
   ```yaml
   # config/hydra/data.yaml
   data_sources:
     ercot_api:
       base_url: "https://api.ercot.com/"
       api_key: "${env:ERCOT_API_KEY}"
     weather_api:
       base_url: "https://api.weather.com/"
       api_key: "${env:WEATHER_API_KEY}"
   ```

2. **Feature Configuration**: Defines feature engineering parameters
   ```yaml
   # config/hydra/features.yaml
   feature_engineering:
     time_features:
       enabled: true
       timestamp_column: "timestamp"
     statistical_features:
       enabled: true
       price_column: "price"
       timestamp_column: "timestamp"
     weather_features:
       enabled: true
       include_interactions: true
   ```

3. **Model Configuration**: Specifies model hyperparameters
   ```yaml
   # config/hydra/models.yaml
   model:
     type: "xgboost"
     hyperparameters:
       learning_rate: 0.05
       max_depth: 6
       min_child_weight: 1
       subsample: 0.8
       colsample_bytree: 0.8
       n_estimators: 200
   ```

You can create custom configuration files or modify the existing ones based on your requirements.

## Command Line Interface

The system provides a command-line interface for common operations:

```bash
# Fetch data from external sources
rtlmp_predict fetch-data --start-date 2023-01-01 --end-date 2023-06-30 --nodes HB_NORTH

# Train a new model
rtlmp_predict train --config config/training_config.yaml

# Generate forecasts
rtlmp_predict predict --threshold 100 --nodes HB_NORTH

# Run backtesting
rtlmp_predict backtest --config config/backtest_config.yaml

# Evaluate model performance
rtlmp_predict evaluate --model-id xgb_model_20230701 --test-data data/test_data.parquet

# Generate visualizations
rtlmp_predict visualize --forecast-path forecasts/latest.parquet --output-path visualizations
```

Use `rtlmp_predict [command] --help` for detailed information on each command.

## Python API

For more flexibility, you can use the Python API directly in your scripts or notebooks:

```python
# Example: Basic workflow using the Python API
from src.backend.data.fetchers.ercot_api import ERCOTDataFetcher
from src.backend.features.feature_pipeline import FeaturePipeline
from src.backend.models.training import ModelTrainer
from src.backend.inference.engine import InferenceEngine
from datetime import datetime, timedelta

# Initialize components
data_fetcher = ERCOTDataFetcher()
feature_pipeline = FeaturePipeline()
model_trainer = ModelTrainer(model_type="xgboost")

# Fetch data
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 6, 30)
rtlmp_data = data_fetcher.fetch_historical_data(
    start_date=start_date,
    end_date=end_date,
    nodes=["HB_NORTH"]
)
weather_data = data_fetcher.fetch_weather_data(
    start_date=start_date,
    end_date=end_date
)

# Create features
feature_pipeline.add_data_source("rtlmp", rtlmp_data)
feature_pipeline.add_data_source("weather", weather_data)
features = feature_pipeline.create_features()

# Prepare target variable
target = rtlmp_data["spike_indicator"]

# Train model
model, metrics = model_trainer.train(features, target)

# Generate forecast
inference_engine = InferenceEngine(config={
    "forecast_horizon": 72,
    "thresholds": [100.0],
    "nodes": ["HB_NORTH"]
})
inference_engine.load_model(model_id=model.model_id)
forecast = inference_engine.generate_forecast({
    "rtlmp": rtlmp_data,
    "weather": weather_data
})

print(f"Model performance: {metrics}")
print(f"Forecast generated for {len(forecast)} hours")
```

This example demonstrates a basic workflow from data fetching to forecast generation.

# Data Management

This section covers how to work with ERCOT market data and other data sources required for the prediction system.

## Data Sources

The system uses several data sources:

1. **RTLMP Data**: 5-minute Real-Time Locational Marginal Prices from ERCOT
   - Source: ERCOT API or historical data files
   - Key fields: timestamp, node_id, price, congestion_price, loss_price, energy_price

2. **Weather Data**: Historical and forecast weather information
   - Source: Weather API or historical data files
   - Key fields: timestamp, location_id, temperature, wind_speed, solar_irradiance, humidity

3. **Grid Condition Data**: ERCOT grid operational metrics
   - Source: ERCOT API or historical data files
   - Key fields: timestamp, total_load, available_capacity, wind_generation, solar_generation

The `DataFetcher` classes provide standardized interfaces for retrieving this data.

## Data Fetching

To fetch data using the Python API:

```python
from src.backend.data.fetchers.ercot_api import ERCOTDataFetcher
from src.backend.data.fetchers.weather_api import WeatherDataFetcher
from datetime import datetime

# Initialize data fetchers
ercot_fetcher = ERCOTDataFetcher()
weather_fetcher = WeatherDataFetcher()

# Fetch RTLMP data
rtlmp_data = ercot_fetcher.fetch_historical_data(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 6, 30),
    nodes=["HB_NORTH", "HB_SOUTH"],
    data_type="rtlmp"
)

# Fetch grid condition data
grid_data = ercot_fetcher.fetch_historical_data(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 6, 30),
    data_type="grid_conditions"
)

# Fetch weather data
weather_data = weather_fetcher.fetch_historical_data(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 6, 30),
    locations=["NORTH_CENTRAL", "SOUTH_CENTRAL"]
)

# Fetch forecast data
weather_forecast = weather_fetcher.fetch_forecast_data(
    forecast_date=datetime.now(),
    horizon=72,  # hours
    locations=["NORTH_CENTRAL", "SOUTH_CENTRAL"]
)
```

Alternatively, use the CLI command:

```bash
rtlmp_predict fetch-data \
    --start-date 2023-01-01 \
    --end-date 2023-06-30 \
    --nodes HB_NORTH HB_SOUTH \
    --data-types rtlmp grid_conditions weather \
    --output-path ./data/raw
```

## Data Validation

The system includes data validation tools to ensure data quality:

```python
from src.backend.data.validators.pandera_schemas import RTLMPSchema, WeatherSchema
from src.backend.data.validators.data_quality import check_completeness, check_temporal_consistency

# Validate RTLMP data structure
validation_result = RTLMPSchema.validate(rtlmp_data)
print(f"Schema validation passed: {validation_result.success}")
if not validation_result.success:
    print(f"Validation errors: {validation_result.errors}")

# Check data completeness
completeness = check_completeness(
    rtlmp_data,
    timestamp_column="timestamp",
    expected_frequency="5min",
    groupby_columns=["node_id"]
)
print(f"Data completeness: {completeness.completeness_ratio:.2%}")
print(f"Missing periods: {len(completeness.missing_periods)}")

# Check temporal consistency
consistency = check_temporal_consistency(
    rtlmp_data,
    timestamp_column="timestamp",
    value_column="price"
)
print(f"Temporal consistency issues: {len(consistency.anomalies)}")
```

These validation tools help identify data quality issues before they affect model training.

## Data Storage

The system uses Parquet files for efficient storage of time series data:

```python
from src.backend.data.storage.parquet_store import ParquetStore

# Initialize storage
store = ParquetStore(base_path="./data")

# Store raw data
store.store_dataframe(
    df=rtlmp_data,
    dataset_name="rtlmp",
    partition_cols=["node_id"],
    timestamp_col="timestamp"
)

# Retrieve data
retrieved_data = store.load_dataframe(
    dataset_name="rtlmp",
    filters=[("node_id", "=", "HB_NORTH")],
    date_range=(datetime(2023, 1, 1), datetime(2023, 1, 31))
)
```

The storage system automatically handles partitioning by date and other columns for efficient retrieval.

# Feature Engineering

This section covers how to transform raw data into model-ready features using the Feature Engineering Module.

## Feature Categories

The system supports several categories of features:

1. **Time Features**: Derived from timestamps
   - Hour of day, day of week, month, season, is_weekend, is_holiday
   - Cyclical encoding of time features (sin/cos transformations)

2. **Statistical Features**: Statistical aggregations of price data
   - Rolling means, standard deviations, min/max values
   - Price volatility measures, spike indicators
   - Lagged values and differences

3. **Weather Features**: Derived from weather data
   - Temperature, wind speed, solar irradiance, humidity
   - Forecast values and deviations from historical averages
   - Interaction terms between weather variables

4. **Market Features**: Derived from grid condition data
   - Load forecasts, generation mix, reserve margins
   - Supply-demand balance indicators
   - Day-ahead market prices and spreads

These features are combined to create a comprehensive set of predictors for the RTLMP spike prediction models.

## Using the Feature Pipeline

The `FeaturePipeline` class provides a unified interface for feature engineering:

```python
from src.backend.features.feature_pipeline import FeaturePipeline

# Initialize the feature pipeline
feature_pipeline = FeaturePipeline()

# Add data sources
feature_pipeline.add_data_source("rtlmp", rtlmp_data)
feature_pipeline.add_data_source("weather", weather_data)
feature_pipeline.add_data_source("grid", grid_data)

# Configure feature generation
feature_config = {
    "time_features": {
        "enabled": True,
        "timestamp_column": "timestamp"
    },
    "statistical_features": {
        "enabled": True,
        "price_column": "price",
        "timestamp_column": "timestamp",
        "windows": [1, 6, 24, 168]  # hours
    },
    "weather_features": {
        "enabled": True,
        "include_interactions": True
    },
    "market_features": {
        "enabled": True
    },
    "feature_selection": {
        "enabled": False
    }
}
feature_pipeline.update_feature_config(feature_config)

# Generate features
features_df = feature_pipeline.create_features()

# Get feature names
all_features = feature_pipeline.get_feature_names()
print(f"Generated {len(all_features)} features")

# Validate feature set
is_valid, inconsistent_features = feature_pipeline.validate_feature_set()
if not is_valid:
    print(f"Feature inconsistencies found: {inconsistent_features}")
```

The feature pipeline handles all the complexity of feature generation and ensures consistency between training and inference.

## Custom Feature Creation

You can create custom features by extending the existing feature modules:

```python
from src.backend.features.time_features import create_all_time_features
from src.backend.features.statistical_features import create_all_statistical_features
import pandas as pd
import numpy as np

def create_custom_features(df, price_column="price", timestamp_column="timestamp"):
    """Create custom features for RTLMP prediction"""
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Create standard time features
    result_df = create_all_time_features(result_df, timestamp_column)
    
    # Create standard statistical features
    result_df = create_all_statistical_features(
        result_df, 
        price_column, 
        timestamp_column
    )
    
    # Add custom price ratio feature
    result_df["price_ratio_24h"] = result_df[price_column] / \
        result_df.groupby("node_id")[price_column].shift(24).fillna(1)
    
    # Add custom volatility feature
    result_df["custom_volatility"] = result_df.groupby("node_id")[price_column] \
        .rolling(12).std().reset_index(0, drop=True) / \
        result_df.groupby("node_id")[price_column] \
        .rolling(12).mean().reset_index(0, drop=True)
    
    return result_df

# Use custom feature function
custom_features_df = create_custom_features(rtlmp_data)
```

You can integrate custom features into the feature pipeline by registering them in the feature registry.

## Feature Selection

The system includes feature selection tools to identify the most important features:

```python
from src.backend.features.feature_selection import select_features_pipeline
from src.backend.models.xgboost_model import XGBoostModel

# Train a model for feature importance
model = XGBoostModel(model_id="feature_importance_model")
model.train(features_df, target)

# Get feature importances
importances = model.get_feature_importance()

# Select features based on importance
selected_features = select_features_pipeline(
    features_df,
    target,
    methods=["importance"],
    importance_threshold=0.01,
    importance_values=importances
)

# Create a DataFrame with only selected features
selected_df = features_df[selected_features]
print(f"Selected {len(selected_features)} out of {len(features_df.columns)} features")
```

Feature selection helps reduce model complexity and improve performance by focusing on the most predictive features.

# Model Training

This section covers how to train, validate, and optimize prediction models using the Model Training Module.

## Model Types

The system supports several types of models:

1. **XGBoost**: Default model type with excellent performance for RTLMP spike prediction
   ```python
   from src.backend.models.xgboost_model import XGBoostModel
   
   model = XGBoostModel(
       model_id="xgb_model_20230701",
       hyperparameters={
           "learning_rate": 0.05,
           "max_depth": 6,
           "min_child_weight": 1,
           "subsample": 0.8,
           "colsample_bytree": 0.8,
           "n_estimators": 200
       }
   )
   ```

2. **LightGBM**: Alternative gradient boosting implementation
   ```python
   from src.backend.models.lightgbm_model import LightGBMModel
   
   model = LightGBMModel(
       model_id="lgbm_model_20230701",
       hyperparameters={
           "learning_rate": 0.05,
           "num_leaves": 31,
           "feature_fraction": 0.8,
           "bagging_fraction": 0.8,
           "n_estimators": 200
       }
   )
   ```

3. **Ensemble**: Combines multiple models for improved performance
   ```python
   from src.backend.models.ensemble import EnsembleModel
   from src.backend.models.xgboost_model import XGBoostModel
   from src.backend.models.lightgbm_model import LightGBMModel
   
   # Create base models
   xgb_model = XGBoostModel(model_id="xgb_base")
   lgbm_model = LightGBMModel(model_id="lgbm_base")
   
   # Create ensemble
   ensemble = EnsembleModel(
       model_id="ensemble_20230701",
       base_models=[xgb_model, lgbm_model],
       weights=[0.6, 0.4]
   )
   ```

Each model type has its strengths and can be selected based on your specific requirements.

## Basic Training Workflow

To train a model using the Python API:

```python
from src.backend.models.training import train_model, time_based_train_test_split
from src.backend.models.evaluation import evaluate_model_performance

# Split data into training and testing sets
X_train, X_test, y_train, y_test = time_based_train_test_split(
    features_df, target, test_size=0.2
)

# Train the model
model, metrics = train_model(
    model_type="xgboost",
    features=X_train,
    targets=y_train,
    hyperparameters={
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200
    }
)

# Evaluate on test set
test_metrics = evaluate_model_performance(
    model=model,
    features=X_test,
    targets=y_test,
    threshold=0.5
)

print(f"Training metrics: {metrics}")
print(f"Test metrics: {test_metrics}")
```

Alternatively, use the ModelTrainer class for a more object-oriented approach:

```python
from src.backend.models.training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    model_type="xgboost",
    hyperparameters={
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200
    },
    model_registry_path="./models"
)

# Train model
model, metrics = trainer.train(X_train, y_train)

# Evaluate model
test_metrics = trainer.evaluate(X_test, y_test)

# Save model
trainer.save_model()
```

## Cross-Validation

Time series cross-validation is essential for reliable model evaluation:

```python
from src.backend.models.training import cross_validate_time_series, train_and_evaluate

# Perform time series cross-validation
cv_metrics = cross_validate_time_series(
    model_type="xgboost",
    features=features_df,
    targets=target,
    hyperparameters={
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200
    },
    n_splits=5,
    cv_strategy="time_series"
)

# Print cross-validation metrics
for metric, values in cv_metrics.items():
    mean_value = sum(values) / len(values)
    std_value = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
    print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")

# Train with cross-validation in one step
model, cv_metrics = train_and_evaluate(
    model_type="xgboost",
    features=features_df,
    targets=target,
    hyperparameters={
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200
    },
    cv_folds=5,
    cv_strategy="time_series"
)
```

Time series cross-validation ensures that your model evaluation respects the temporal nature of the data, preventing data leakage.

## Hyperparameter Optimization

To find optimal hyperparameters for your model:

```python
from src.backend.models.training import optimize_and_train
from src.backend.models.hyperparameter_tuning import optimize_hyperparameters

# Define parameter grid
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 6, 9],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "n_estimators": [100, 200, 300]
}

# Optimize hyperparameters
best_params = optimize_hyperparameters(
    model_type="xgboost",
    features=X_train,
    targets=y_train,
    param_grid=param_grid,
    optimization_method="random_search",
    n_iterations=50,
    cv_folds=3,
    cv_strategy="time_series"
)

print(f"Best hyperparameters: {best_params}")

# Train with optimized hyperparameters in one step
model, metrics = optimize_and_train(
    model_type="xgboost",
    features=features_df,
    targets=target,
    param_grid=param_grid,
    optimization_method="random_search",
    n_iterations=50
)
```

Hyperparameter optimization helps find the best configuration for your model, improving prediction accuracy.

## Model Persistence

To save and load trained models:

```python
from src.backend.models.persistence import save_model, load_model
from src.backend.models.versioning import increment_version

# Save model to file
model_path = save_model(model, "./models/xgboost_model.joblib")

# Save model to registry with metadata
from src.backend.data.storage.model_registry import ModelRegistry

registry = ModelRegistry("./models")
registry.register_model(
    model=model,
    metadata={
        "training_date": datetime.now().isoformat(),
        "features": list(X_train.columns),
        "metrics": metrics,
        "description": "XGBoost model for RTLMP spike prediction"
    }
)

# Load model from file
loaded_model = load_model("./models/xgboost_model.joblib", model_type="xgboost")

# Load model from registry
from src.backend.models.training import get_latest_model

latest_model = get_latest_model(model_type="xgboost", registry_path="./models")

# Increment model version
new_version = increment_version(model.version, increment_type="minor")
print(f"New version: {new_version}")
```

Proper model versioning and persistence are essential for tracking model evolution and ensuring reproducibility.

# Inference and Forecasting

This section covers how to generate probability forecasts using trained models.

## Inference Engine

The `InferenceEngine` class handles the end-to-end process of generating forecasts:

```python
from src.backend.inference.engine import InferenceEngine
from src.backend.config.schema import InferenceConfig

# Create inference configuration
config = InferenceConfig(
    forecast_horizon=72,  # hours
    thresholds=[50.0, 100.0, 200.0],  # price thresholds in $/MWh
    nodes=["HB_NORTH", "HB_SOUTH"]
)

# Initialize inference engine
inference_engine = InferenceEngine(
    config=config,
    model_path="./models",
    forecast_path="./forecasts"
)

# Load model
inference_engine.load_model(model_id="xgb_model_20230701")

# Generate forecast
forecast = inference_engine.generate_forecast(
    data_sources={
        "rtlmp": rtlmp_data,
        "weather": weather_forecast,
        "grid": grid_data
    },
    feature_config=feature_config
)

# Get latest forecast
latest_forecast, metadata, timestamp = inference_engine.get_latest_forecast()

# Compare with previous forecast
comparison = inference_engine.compare_with_previous_forecast(forecast)
print(f"Mean absolute difference: {comparison['mean_abs_diff']:.4f}")
```

The inference engine handles all aspects of forecast generation, including feature engineering, model loading, and prediction.

## Probability Calibration

Calibration ensures that predicted probabilities match observed frequencies:

```python
from src.backend.inference.calibration import ProbabilityCalibrator

# Initialize calibrator
calibrator = ProbabilityCalibrator(method="isotonic")

# Fit calibrator with historical predictions and actuals
calibrator.fit(
    predictions=historical_predictions,
    actuals=historical_actuals
)

# Calibrate new predictions
calibrated_probs = calibrator.calibrate(raw_predictions)

# Initialize inference engine with calibrator
inference_engine.initialize_calibrator(
    historical_predictions=historical_predictions,
    historical_actuals=historical_actuals,
    method="isotonic"
)

# Generate calibrated forecast
calibrated_forecast = inference_engine.generate_forecast(data_sources)
```

Calibration is crucial for ensuring that the predicted probabilities are reliable and can be directly interpreted as the likelihood of price spikes.

## Handling Multiple Thresholds

The system can generate forecasts for multiple price thresholds simultaneously:

```python
from src.backend.inference.thresholds import ThresholdConfig, ThresholdApplier

# Define threshold configuration
threshold_config = ThresholdConfig(
    thresholds=[50.0, 100.0, 200.0],
    default_threshold=100.0
)

# Apply thresholds to price data
threshold_applier = ThresholdApplier(threshold_config)
spike_indicators = threshold_applier.apply_to_dataframe(
    df=rtlmp_data,
    price_column="price"
)

# Generate hourly spike indicators
hourly_indicators = threshold_applier.hourly_spike_indicators(
    df=rtlmp_data,
    price_column="price",
    timestamp_column="timestamp"
)

# Configure inference engine with multiple thresholds
config = InferenceConfig(
    forecast_horizon=72,
    thresholds=[50.0, 100.0, 200.0],
    nodes=["HB_NORTH"]
)
inference_engine = InferenceEngine(config=config)

# Generate forecast for all thresholds
forecast = inference_engine.generate_forecast(data_sources)

# Access forecast for specific threshold
forecast_100 = forecast[forecast["threshold"] == 100.0]
```

Generating forecasts for multiple thresholds allows users to understand how the probability of price spikes varies with different price levels.

## Forecast Storage and Retrieval

The system provides tools for storing and retrieving forecasts:

```python
from src.backend.data.storage.forecast_repository import ForecastRepository
from datetime import datetime

# Initialize forecast repository
repository = ForecastRepository("./forecasts")

# Store forecast
repository.store_forecast(
    forecast=forecast,
    metadata={
        "model_id": "xgb_model_20230701",
        "model_version": "1.0.0",
        "thresholds": [50.0, 100.0, 200.0],
        "nodes": ["HB_NORTH", "HB_SOUTH"],
        "description": "72-hour forecast generated on 2023-07-01"
    }
)

# Get latest forecast
latest_forecast, metadata, timestamp = repository.get_latest_forecast()

# Get forecast for specific date
historical_forecast = repository.get_forecast_by_date(
    forecast_date=datetime(2023, 6, 15)
)

# Get forecasts for date range
forecasts = repository.get_forecasts_in_range(
    start_date=datetime(2023, 6, 1),
    end_date=datetime(2023, 6, 30)
)
```

Proper forecast storage and retrieval are essential for tracking forecast performance over time and analyzing forecast evolution.

# Model Evaluation and Backtesting

This section covers how to evaluate model performance and conduct backtesting experiments.

## Performance Metrics

The system provides comprehensive metrics for evaluating model performance:

```python
from src.backend.models.evaluation import evaluate_model_performance, plot_roc_curve, plot_precision_recall_curve, plot_calibration_curve

# Evaluate model performance
metrics = evaluate_model_performance(
    model=model,
    features=X_test,
    targets=y_test,
    threshold=0.5
)

# Print key metrics
print(f"AUC-ROC: {metrics['auc']:.4f}")
print(f"Brier Score: {metrics['brier_score']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")

# Generate performance plots
fig_roc = plot_roc_curve(model, X_test, y_test)
fig_pr = plot_precision_recall_curve(model, X_test, y_test)
fig_cal = plot_calibration_curve(model, X_test, y_test)

# Save plots
fig_roc.savefig("./visualizations/roc_curve.png")
fig_pr.savefig("./visualizations/precision_recall_curve.png")
fig_cal.savefig("./visualizations/calibration_curve.png")
```

These metrics help you understand different aspects of model performance, from discrimination ability (AUC-ROC) to probability calibration (Brier Score).

## Backtesting Framework

The backtesting framework allows you to simulate historical forecasts and evaluate performance: