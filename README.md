# ERCOT RTLMP Spike Prediction System

[![Build Status](https://github.com/username/ercot-rtlmp-prediction/workflows/CI/badge.svg)](https://github.com/username/ercot-rtlmp-prediction/actions)
[![Documentation Status](https://github.com/username/ercot-rtlmp-prediction/workflows/deploy-docs/badge.svg)](https://username.github.io/ercot-rtlmp-prediction/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Introduction

The ERCOT RTLMP Spike Prediction System is a machine learning-based forecasting tool that predicts the probability of price spikes in the ERCOT Real-Time Locational Marginal Price (RTLMP) market. It provides 72-hour forecasts before day-ahead market closure, enabling battery storage operators to optimize their bidding strategies and maximize revenue.

This system addresses a critical need in the energy storage market by quantifying the probability of RTLMP spikes, which represent both risk and opportunity for battery storage operators.

## Key Features

- **72-hour Forecast Horizon**: Predicts hourly probabilities of RTLMP spikes for the next 72 hours
- **Multiple Threshold Support**: Configurable price thresholds for spike definition
- **Automated Retraining**: Model retraining on a two-day cadence to maintain accuracy
- **Comprehensive Backtesting**: Framework for evaluating model performance on historical data
- **Performance Visualization**: Tools for visualizing forecast accuracy and model performance
- **Modular Architecture**: Clearly defined interfaces between components for maintainability and extensibility

## System Architecture

The system follows a modular, pipeline-oriented architecture with the following main components:

- **Data Fetching Interface**: Retrieves ERCOT market data and weather forecasts
- **Feature Engineering Module**: Transforms raw data into model-ready features
- **Model Training Module**: Trains and validates prediction models
- **Inference Engine**: Generates probability forecasts using trained models
- **Backtesting Framework**: Simulates historical forecasts for model evaluation
- **Visualization & Metrics Tools**: Generates performance reports and visualizations

The system is designed with a functional programming approach, emphasizing stateless components and clear interfaces.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda for package management

### Using pip

```bash
# Clone the repository
git clone https://github.com/username/ercot-rtlmp-prediction.git
cd ercot-rtlmp-prediction

# Install the package and dependencies
pip install -e .
```

### Using Docker

```bash
# Clone the repository
git clone https://github.com/username/ercot-rtlmp-prediction.git
cd ercot-rtlmp-prediction

# Build and run using Docker Compose
cd infrastructure/docker
cp .env.example .env  # Edit .env with your configuration
docker-compose up -d
```

## Quick Start

### Command Line Interface

The system provides a command-line interface for common operations:

```bash
# Generate a 72-hour forecast
rtlmp_predict --config config.yaml predict --threshold 100 --node HB_NORTH

# Train a new model
rtlmp_predict --config config.yaml train --start-date 2020-01-01 --end-date 2023-06-30

# Run backtesting
rtlmp_predict --config config.yaml backtest --start-date 2022-01-01 --end-date 2022-12-31 --threshold 100
```

### Python API

```python
from ercot_rtlmp_prediction import inference, data, features

# Fetch latest data
raw_data = data.fetch_latest_data()

# Process features
feature_data = features.engineer_features(raw_data)

# Generate forecast
forecast = inference.generate_forecast(feature_data, threshold=100)

print(forecast)
```

For more detailed examples, see the [examples directory](examples/).

## Configuration

The system uses [Hydra](https://hydra.cc/) for configuration management. Example configuration files are provided in the `examples/config/` directory:

- `inference_config.yaml`: Configuration for forecast generation
- `training_config.yaml`: Configuration for model training
- `backtest_config.yaml`: Configuration for backtesting

You can override configuration values via command line:

```bash
rtlmp_predict --config examples/config/inference_config.yaml predict threshold=200 node=HB_HOUSTON
```

## Documentation

Comprehensive documentation is available in the `docs/` directory and online:

- [System Architecture](docs/architecture/system_overview.md)
- [Data Flow](docs/architecture/data_flow.md)
- [Component Interaction](docs/architecture/component_interaction.md)
- [Local Setup Guide](docs/deployment/local_setup.md)
- [Scheduled Execution](docs/deployment/scheduled_execution.md)
- [Data Scientist Guide](docs/user_guides/data_scientists.md)
- [Energy Scientist Guide](docs/user_guides/energy_scientists.md)

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
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test modules
pytest src/backend/tests/unit/test_feature_engineering.py
```

For more information on contributing to the project, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Electric Reliability Council of Texas (ERCOT) for providing market data
- Open-source community for the excellent tools and libraries that made this project possible