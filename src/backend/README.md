# ERCOT RTLMP Spike Prediction System - Backend

[![Build Status](https://github.com/username/ercot-rtlmp-prediction/workflows/CI/badge.svg)](https://github.com/username/ercot-rtlmp-prediction/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction

This is the backend component of the ERCOT RTLMP Spike Prediction System, a machine learning-based forecasting tool that predicts the probability of price spikes in the ERCOT Real-Time Locational Marginal Price (RTLMP) market. The backend provides the core functionality for data fetching, feature engineering, model training, and inference.

The system is designed with a modular, functional programming approach, emphasizing stateless components and clear interfaces between modules. This architecture ensures maintainability, testability, and extensibility of the codebase.

## Features

- **Data Fetching Interface**: Standardized interface for retrieving ERCOT market data and weather forecasts
- **Feature Engineering Module**: Transforms raw data into model-ready features with consistent formatting
- **Model Training Module**: Trains prediction models with cross-validation capabilities
- **Inference Engine**: Generates 72-hour RTLMP spike probability forecasts
- **Backtesting Framework**: Simulates historical forecasts for model evaluation
- **Visualization & Metrics Tools**: Generates performance reports and visualizations
- **Orchestration**: Manages scheduling and execution of data fetching, training, and inference tasks

## System Architecture

The backend follows a modular, pipeline-oriented architecture with the following main components:

```
Data Sources → Data Fetching → Feature Engineering → Model Training → Inference → Visualization
                                                  ↑                  ↑
                                                  |                  |
                                                  ↓                  ↓
                                           Model Registry    Forecast Repository
                                                  ↑                  ↑
                                                  |                  |
                                                  ↓                  ↓
                                           Backtesting ←→ Performance Metrics
```

Each component has a well-defined interface and can be used independently or as part of the complete pipeline. The system uses a functional programming approach with stateless components where possible, making it easier to test and maintain.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda for package management

### Development Installation

```bash
# Clone the repository
git clone https://github.com/username/ercot-rtlmp-prediction.git
cd ercot-rtlmp-prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the backend package in development mode
cd src/backend
pip install -e .
```

### Production Installation

```bash
# Install from PyPI (when available)
pip install ercot-rtlmp-prediction

# Or install from the repository
pip install git+https://github.com/username/ercot-rtlmp-prediction.git#subdirectory=src/backend
```

## Usage

### Basic Usage

```python
from ercot_rtlmp_prediction import setup_backend
from ercot_rtlmp_prediction import data, features, models, inference

# Initialize the backend
config = setup_backend()

# Fetch data
data_fetcher = data.ERCOTDataFetcher()
raw_data = data_fetcher.fetch_data(start_date='2023-01-01', end_date='2023-01-31')

# Process features
feature_pipeline = features.create_feature_pipeline()
feature_data = feature_pipeline.process(raw_data)

# Train a model
model = models.XGBoostModel()
model.train(feature_data)

# Generate forecast
engine = inference.InferenceEngine(model=model)
forecast = engine.generate_forecast(horizon=72, threshold=100)

print(forecast)
```

### Advanced Usage

For more advanced usage examples, see the [examples directory](../../examples/) in the main repository.

## Module Structure

The backend package is organized into the following modules:

- **utils**: Utility functions, type definitions, and common helpers
- **config**: Configuration management functionality
- **data**: Data fetching and storage components
  - **fetchers**: Interfaces for retrieving data from external sources
  - **validators**: Data validation and quality checking
  - **storage**: Data persistence and retrieval
- **features**: Feature engineering components
  - **time_features**: Time-based feature extraction
  - **statistical_features**: Statistical feature calculation
  - **weather_features**: Weather-related feature processing
  - **market_features**: Market-specific feature creation
  - **feature_registry**: Feature definition and metadata management
  - **feature_pipeline**: End-to-end feature processing pipeline
- **models**: Model training and evaluation components
  - **base_model**: Abstract base class for all models
  - **xgboost_model**: XGBoost implementation
  - **lightgbm_model**: LightGBM implementation
  - **ensemble**: Ensemble model implementation
  - **training**: Model training utilities
  - **evaluation**: Model evaluation metrics and tools
  - **cross_validation**: Cross-validation strategies
  - **persistence**: Model saving and loading
- **inference**: Inference and forecasting components
  - **threshold_config**: Threshold configuration management
  - **thresholds**: Threshold application functionality
  - **calibration**: Probability calibration utilities
  - **engine**: Main inference engine
  - **prediction_pipeline**: End-to-end prediction pipeline
- **backtesting**: Backtesting and historical simulation components
- **visualization**: Visualization and metrics components
- **orchestration**: Task orchestration and scheduling components
- **api**: High-level API interfaces for system components

## Configuration

The system uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are located in the `config/hydra/` directory:

- `config.yaml`: Main configuration file
- `data.yaml`: Data fetching configuration
- `features.yaml`: Feature engineering configuration
- `models.yaml`: Model training configuration
- `inference.yaml`: Inference configuration
- `visualization.yaml`: Visualization configuration

You can override configuration values programmatically or via command line when using the CLI:

```python
# Programmatic override
from ercot_rtlmp_prediction import setup_backend
from omegaconf import OmegaConf

config = setup_backend()
config.models.xgboost.learning_rate = 0.05
config.inference.threshold = 200
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/username/ercot-rtlmp-prediction.git
cd ercot-rtlmp-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
cd src/backend
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

This project uses:
- [Black](https://github.com/psf/black) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](https://mypy.readthedocs.io/) for static type checking

You can run all linters with:

```bash
# Format code and check types
black .
isort .
mypy .

# Or use the pre-commit hooks
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=ercot_rtlmp_prediction

# Run specific test modules
pytest tests/unit/test_feature_engineering.py
```

## Documentation

### Building Documentation

The documentation is built using [Sphinx](https://www.sphinx-doc.org/):

```bash
# Build HTML documentation
cd docs
make html

# View the documentation
open build/html/index.html  # On macOS
# Or use your browser to open the file directly
```

### API Documentation

Detailed API documentation is available in the `docs/source/api/` directory:

- [Data API](docs/source/api/data.rst)
- [Features API](docs/source/api/features.rst)
- [Models API](docs/source/api/models.rst)
- [Inference API](docs/source/api/inference.rst)
- [Backtesting API](docs/source/api/backtesting.rst)
- [Visualization API](docs/source/api/visualization.rst)
- [Utils API](docs/source/api/utils.rst)
- [Orchestration API](docs/source/api/orchestration.rst)

## Contributing

Contributions are welcome! Please see the [Contributing Guide](../../CONTRIBUTING.md) in the main repository for details on how to contribute to this project.

### Adding New Features

1. Create a new branch for your feature
2. Implement the feature with appropriate tests
3. Update documentation as needed
4. Submit a pull request

### Reporting Issues

If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/username/ercot-rtlmp-prediction/issues) in the main repository.

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file in the main repository for details.