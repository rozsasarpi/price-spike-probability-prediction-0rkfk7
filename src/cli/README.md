# ERCOT RTLMP Spike Prediction System CLI

Command line interface for the ERCOT RTLMP spike prediction system, providing tools for data fetching, model training, forecasting, evaluation, and visualization.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Command Reference](#command-reference)
  - [fetch-data](#fetch-data)
  - [train](#train)
  - [predict](#predict)
  - [backtest](#backtest)
  - [evaluate](#evaluate)
  - [visualize](#visualize)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Introduction

The ERCOT RTLMP Spike Prediction System CLI provides a command-line interface for data scientists and energy scientists working with the Real-Time Locational Marginal Price (RTLMP) spike prediction system. It offers comprehensive tools for managing the entire prediction workflow, from data acquisition to model training, forecasting, evaluation, and visualization.

This CLI enables you to:
- Fetch historical ERCOT market data and weather forecasts
- Train and evaluate machine learning models for price spike prediction
- Generate 72-hour hourly probability forecasts before day-ahead market closure
- Backtest models on historical periods
- Create visualizations for model performance and forecast results

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package installer)

### Install via pip

```bash
pip install ercot-rtlmp-predict
```

### Install from source

```bash
git clone https://github.com/yourorganization/ercot-rtlmp-predict.git
cd ercot-rtlmp-predict
pip install -e .
```

### Verify installation

```bash
rtlmp_predict --version
```

## Configuration

The CLI can be configured through multiple methods, with the following precedence (highest to lowest):

1. Command-line options
2. Environment variables
3. Configuration file
4. Default values

### Configuration File

Create a configuration file in one of these locations:
- `./rtlmp_config.yaml` (current directory)
- `~/.rtlmp_config.yaml` (home directory)
- `/etc/rtlmp/config.yaml` (system-wide)

Example configuration file:

```yaml
# rtlmp_config.yaml
data:
  storage_path: "/path/to/data"
  cache_enabled: true
  cache_ttl_hours: 24

models:
  storage_path: "/path/to/models"
  default_type: "xgboost"

nodes:
  default: ["HB_NORTH", "HB_SOUTH"]

thresholds:
  default: [100, 200]

output:
  default_format: "csv"
  visualization_enabled: true
```

### Environment Variables

Environment variables should be prefixed with `RTLMP_`:

```bash
export RTLMP_DATA_STORAGE_PATH="/path/to/data"
export RTLMP_MODELS_STORAGE_PATH="/path/to/models"
export RTLMP_DEFAULT_NODES="HB_NORTH,HB_SOUTH"
```

### Command Line Configuration

Specify a custom configuration file:

```bash
rtlmp_predict --config /path/to/custom/config.yaml [COMMAND]
```

Enable verbose output:

```bash
rtlmp_predict --verbose [COMMAND]
```

## Command Reference

### fetch-data

Fetches data from external sources such as ERCOT API and weather services.

```bash
rtlmp_predict fetch-data [OPTIONS]
```

#### Options

| Option | Description | Required |
|--------|-------------|----------|
| `--data-type, -t` | Type of data to fetch (rtlmp, weather, grid_conditions, all) | Yes |
| `--start-date, -s` | Start date for data retrieval (YYYY-MM-DD) | Yes |
| `--end-date, -e` | End date for data retrieval (YYYY-MM-DD) | Yes |
| `--nodes, -n` | Node IDs to fetch data for (can specify multiple) | No |
| `--output-path, -o` | Path to save the fetched data | No |
| `--output-format, -f` | Output format (text, json, csv) | No |
| `--force-refresh/--use-cache` | Force refresh data from source instead of using cache | No |

#### Examples

```bash
# Fetch RTLMP data for HB_NORTH for January 2023
rtlmp_predict fetch-data --data-type rtlmp --start-date 2023-01-01 --end-date 2023-01-31 --nodes HB_NORTH

# Fetch all data types for January 2023 and save as CSV
rtlmp_predict fetch-data --data-type all --start-date 2023-01-01 --end-date 2023-01-31 --output-format csv --output-path ./data/january.csv
```

### train

Trains a new model using historical data.

```bash
rtlmp_predict train [OPTIONS]
```

#### Options

| Option | Description | Required |
|--------|-------------|----------|
| `--start-date, -s` | Start date for training data (YYYY-MM-DD) | Yes |
| `--end-date, -e` | End date for training data (YYYY-MM-DD) | Yes |
| `--nodes, -n` | Node IDs to train model for (can specify multiple) | Yes |
| `--thresholds, -t` | Price threshold values for spike prediction (can specify multiple) | Yes |
| `--model-type, -m` | Type of model to train (xgboost, lightgbm) | No |
| `--optimize-hyperparameters` | Enable hyperparameter optimization | No |
| `--cross-validation-folds` | Number of folds for cross-validation | No |
| `--model-name` | Custom name for the trained model | No |
| `--output-path, -o` | Path to save the trained model | No |

#### Examples

```bash
# Train a model for HB_NORTH with thresholds of 100 and 200 $/MWh
rtlmp_predict train --start-date 2022-01-01 --end-date 2022-12-31 --nodes HB_NORTH --thresholds 100 200

# Train an XGBoost model with hyperparameter optimization for multiple nodes
rtlmp_predict train --start-date 2022-01-01 --end-date 2022-12-31 --nodes HB_NORTH HB_SOUTH --thresholds 100 --model-type xgboost --optimize-hyperparameters
```

### predict

Generates forecasts using a trained model.

```bash
rtlmp_predict predict [OPTIONS]
```

#### Options

| Option | Description | Required |
|--------|-------------|----------|
| `--threshold, -t` | Price threshold value for spike prediction | Yes |
| `--nodes, -n` | Node IDs to generate forecasts for | Yes |
| `--model-version, -m` | Model version to use (default: latest) | No |
| `--output-path, -o` | Path to save output | No |
| `--output-format, -f` | Output format (text, json, csv) | No |
| `--visualize, -v` | Show visualization of forecast | No |

#### Examples

```bash
# Generate forecast for HB_NORTH with threshold of 100 $/MWh
rtlmp_predict predict --threshold 100 --nodes HB_NORTH

# Generate and visualize forecasts for multiple nodes using a specific model version
rtlmp_predict predict --threshold 100 --nodes HB_NORTH HB_SOUTH --model-version v1.2.3 --visualize
```

### backtest

Runs backtesting on historical data to evaluate model performance.

```bash
rtlmp_predict backtest [OPTIONS]
```

#### Options

| Option | Description | Required |
|--------|-------------|----------|
| `--start-date, -s` | Start date for backtesting (YYYY-MM-DD) | Yes |
| `--end-date, -e` | End date for backtesting (YYYY-MM-DD) | Yes |
| `--thresholds, -t` | Price threshold values for spike prediction | Yes |
| `--nodes, -n` | Node IDs to run backtesting for | Yes |
| `--model-version, -m` | Model version to use (default: latest) | No |
| `--output-path, -o` | Path to save output | No |
| `--output-format, -f` | Output format (text, json, csv) | No |
| `--visualize, -v` | Show visualization of backtesting results | No |

#### Examples

```bash
# Backtest model for 2022 on HB_NORTH with threshold of 100 $/MWh
rtlmp_predict backtest --start-date 2022-01-01 --end-date 2022-12-31 --thresholds 100 --nodes HB_NORTH

# Backtest with visualization for multiple nodes and thresholds
rtlmp_predict backtest --start-date 2022-01-01 --end-date 2022-12-31 --thresholds 100 200 --nodes HB_NORTH HB_SOUTH --visualize
```

### evaluate

Evaluates model performance on test data.

```bash
rtlmp_predict evaluate [OPTIONS]
```

#### Options

| Option | Description | Required |
|--------|-------------|----------|
| `--model-version, -m` | Model version to evaluate (default: latest) | No |
| `--threshold, -t` | Price threshold value for evaluation | Yes |
| `--nodes, -n` | Node IDs to evaluate model for | Yes |
| `--output-path, -o` | Path to save evaluation results | No |
| `--output-format, -f` | Output format (text, json, csv) | No |
| `--visualize, -v` | Show visualization of evaluation results | No |

#### Examples

```bash
# Evaluate latest model for HB_NORTH with threshold of 100 $/MWh
rtlmp_predict evaluate --threshold 100 --nodes HB_NORTH

# Evaluate specific model version with visualization
rtlmp_predict evaluate --threshold 100 --nodes HB_NORTH HB_SOUTH --model-version v1.2.3 --visualize
```

### visualize

Generates visualizations for forecasts, model performance, and feature importance.

```bash
rtlmp_predict visualize [OPTIONS]
```

#### Options

| Option | Description | Required |
|--------|-------------|----------|
| `--type, -t` | Type of visualization (forecast, performance, calibration, feature_importance) | Yes |
| `--data-path, -d` | Path to data for visualization | No |
| `--model-version, -m` | Model version for visualization (default: latest) | No |
| `--threshold, -t` | Price threshold value for visualization | No |
| `--nodes, -n` | Node IDs for visualization | No |
| `--output-path, -o` | Path to save visualization | No |
| `--output-format, -f` | Output format (png, svg, pdf) | No |

#### Examples

```bash
# Visualize current forecast for HB_NORTH with threshold of 100 $/MWh
rtlmp_predict visualize --type forecast --threshold 100 --nodes HB_NORTH

# Generate feature importance visualization for a specific model
rtlmp_predict visualize --type feature_importance --model-version v1.2.3 --output-format png --output-path ./visualizations/features.png
```

## Usage Examples

### Daily Forecast Generation

Generate a 72-hour forecast for a specific node and threshold:

```bash
rtlmp_predict predict --threshold 100 --nodes HB_NORTH --visualize
```

### Model Training with Hyperparameter Optimization

Train a model with hyperparameter optimization for multiple nodes and thresholds:

```bash
rtlmp_predict train --start-date 2022-01-01 --end-date 2022-12-31 --nodes HB_NORTH HB_SOUTH --thresholds 100 200 --model-type xgboost --optimize-hyperparameters
```

### Backtesting Model Performance

Evaluate model performance on historical data:

```bash
rtlmp_predict backtest --start-date 2022-01-01 --end-date 2022-12-31 --thresholds 100 --nodes HB_NORTH --visualize
```

### Data Collection for Analysis

Fetch historical data for analysis:

```bash
rtlmp_predict fetch-data --data-type all --start-date 2022-01-01 --end-date 2022-12-31 --nodes HB_NORTH --output-format csv --output-path ./data/historical_data.csv
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Command not found | Ensure the CLI is properly installed and the installation directory is in your PATH |
| Configuration file not found | Create a configuration file in one of the default locations or specify the path using `--config` |
| Data fetching errors | Check network connectivity, API credentials, and try with `--force-refresh` to bypass cache |
| Model training failures | Ensure sufficient data is available for the specified date range and nodes |
| Prediction errors | Verify that models are trained for the specified threshold and nodes |

### Getting Help

For detailed help on any command:

```bash
rtlmp_predict [COMMAND] --help
```

For general help:

```bash
rtlmp_predict --help
```

## Advanced Usage

### Automation with Scripts

You can create shell scripts to automate recurring tasks:

```bash
#!/bin/bash
# daily_forecast.sh - Run daily forecasting process

# Fetch latest data
rtlmp_predict fetch-data --data-type all --start-date $(date -d "yesterday" +%Y-%m-%d) --end-date $(date +%Y-%m-%d) --nodes HB_NORTH HB_SOUTH

# Generate forecasts
rtlmp_predict predict --threshold 100 --nodes HB_NORTH HB_SOUTH --output-format csv --output-path ./forecasts/$(date +%Y-%m-%d).csv
```

### Integrating with Cron Jobs

Schedule daily forecasts using cron:

```bash
# Run daily at 6:00 AM
0 6 * * * /path/to/daily_forecast.sh
```

### Customizing Output Formats

For programmatic integration, use the JSON output format:

```bash
rtlmp_predict predict --threshold 100 --nodes HB_NORTH --output-format json > forecast.json
```

### Pipeline Integration

Chain commands together for efficient workflows:

```bash
# Fetch data, train model, and generate forecast in one pipeline
rtlmp_predict fetch-data --data-type all --start-date 2022-01-01 --end-date $(date +%Y-%m-%d) --nodes HB_NORTH && \
rtlmp_predict train --start-date 2022-01-01 --end-date $(date +%Y-%m-%d) --nodes HB_NORTH --thresholds 100 && \
rtlmp_predict predict --threshold 100 --nodes HB_NORTH --visualize
```

### Batch Processing

Process multiple nodes or thresholds efficiently:

```bash
# Create a text file with all nodes
cat > nodes.txt << EOF
HB_NORTH
HB_SOUTH
HB_WEST
HB_HOUSTON
EOF

# Process each node
while read node; do
  rtlmp_predict predict --threshold 100 --nodes $node --output-path ./forecasts/$node.csv
done < nodes.txt
```