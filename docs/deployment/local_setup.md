# Local Setup Guide

## Introduction

The ERCOT RTLMP spike prediction system is designed to forecast the probability of price spikes in the Real-Time Locational Marginal Price (RTLMP) market before day-ahead market closure. This guide provides comprehensive instructions for setting up the system in a local development or deployment environment.

The target audience for this guide includes data scientists and energy scientists working on battery storage optimization who need to install and configure the system for local use. By following these instructions, you'll set up a fully functional environment for generating price spike probability forecasts that can inform bidding strategies in the day-ahead market.

## System Requirements

### Hardware Requirements

The ERCOT RTLMP spike prediction system requires certain hardware specifications to run efficiently:

- **CPU**: Minimum 4 cores, 8+ cores recommended for model training
- **Memory**: Minimum 16GB RAM, 32GB recommended for handling larger datasets
- **Storage**: Minimum 100GB free disk space for storing historical data, features, models, and forecasts
- **Network**: Internet connection for retrieving ERCOT market data and weather forecasts

### Software Prerequisites

The following software is required for running the system:

- **Python**: Version 3.10 or higher
- **Package Managers**:
  - pip (included with Python)
  - Poetry (version 1.4+) for dependency management
- **Optional**:
  - Docker (version 20.10+) and Docker Compose (version 2.0+) for containerized deployment
  - cron or other scheduling tools (for production environments)

### Operating System Compatibility

The system is compatible with:

- **Linux**: Most distributions (Ubuntu, Debian, CentOS, etc.)
- **macOS**: 10.15 (Catalina) or higher
- **Windows**: 10 or higher with WSL2 recommended for better performance

## Installation Methods

You can set up the ERCOT RTLMP spike prediction system using one of the following methods:

### Automated Installation

The easiest way to set up the system is using the provided setup script:

```bash
# Clone the repository
git clone https://github.com/your-org/ercot-rtlmp-prediction.git
cd ercot-rtlmp-prediction

# Run the setup script
bash infrastructure/scripts/setup_environment.sh

# Verify installation
source .venv/bin/activate
python -c "import src.backend; print('Installation successful!')"
```

The setup script performs the following tasks:
- Checks system requirements
- Installs required system dependencies
- Creates necessary directory structure
- Sets up a Python virtual environment
- Installs Python dependencies
- Configures environment variables
- Sets up initial configuration files

For scheduled execution, add the `--with-cron` flag:
```bash
bash infrastructure/scripts/setup_environment.sh --with-cron
```

### Manual Installation

If you prefer to set up the system manually, follow these steps:

1. **Create the directory structure**:
```bash
mkdir -p data/raw data/features models forecasts logs config
```

2. **Set up a Python virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r src/backend/requirements.txt
# Or if using Poetry:
cd src/backend
poetry install
```

4. **Configure environment variables**:
Create a `.env` file in the project root with the necessary variables (see the Configuration section below).

5. **Set up configuration files**:
Copy example configuration files from the `examples/config` directory to the `config` directory.

### Docker Installation

For a containerized deployment, use Docker Compose:

```bash
# Clone the repository
git clone https://github.com/your-org/ercot-rtlmp-prediction.git
cd ercot-rtlmp-prediction

# Copy and edit the environment file
cp infrastructure/docker/.env.example infrastructure/docker/.env
# Edit the .env file with your API keys and settings

# Start the containers
cd infrastructure/docker
docker-compose up -d
```

This will create a containerized environment with:
- The RTLMP prediction backend
- CLI tools for interaction
- Prometheus and Grafana for monitoring
- Node Exporter for system metrics

## Directory Structure

The ERCOT RTLMP spike prediction system requires a specific directory structure:

### Data Directory

```
data/
├── raw/       # Raw data from ERCOT and weather APIs
└── features/  # Processed feature data for model training and inference
```

- **Raw Data**: Stores the original data retrieved from ERCOT and weather APIs, typically in CSV or Parquet format.
- **Features**: Contains processed features ready for model training and inference, stored in Parquet format for efficient access.

### Models Directory

```
models/
├── xgboost/   # XGBoost models
├── lightgbm/  # LightGBM models
└── meta/      # Metadata about trained models
```

The models directory stores trained model artifacts with version control. Each model is saved with metadata including:
- Training date
- Performance metrics
- Feature importance
- Hyperparameters

### Forecasts Directory

```
forecasts/
├── daily/     # Daily forecast outputs
└── backtesting/  # Backtesting results
```

Generated forecasts are stored here, organized by date. Each forecast contains:
- Timestamp of forecast generation
- 72-hour probability predictions
- Confidence intervals
- Model version information

### Logs Directory

```
logs/
├── pipeline.log   # Main pipeline execution logs
├── inference.log  # Inference process logs
├── training.log   # Model training logs
└── errors.log     # Error logs
```

Logs are crucial for monitoring system operation and troubleshooting issues. They are organized by component and rotated to manage size.

## Configuration

### Environment Variables

The system uses environment variables for configuration. Create a `.env` file in the project root with the following variables:

```bash
# Example .env file content
ERCOT_API_KEY=your_api_key_here
ERCOT_API_SECRET=your_api_secret_here
WEATHER_API_KEY=your_weather_api_key_here

BACKEND_ENVIRONMENT=development
BACKEND_LOG_LEVEL=INFO
BACKEND_PARALLEL_JOBS=4

DEFAULT_NODE_IDS=HB_NORTH,HB_SOUTH
DEFAULT_PRICE_THRESHOLDS=100.0,200.0
TIMEZONE=America/Chicago
```

Critical environment variables include:
- **API Keys**: For accessing ERCOT and weather data
- **Data Paths**: Locations for data, models, forecasts, and logs
- **Default Settings**: Default nodes, thresholds, and timezone

### Configuration Files

The system uses YAML configuration files for different components. The main configuration files are:

1. **inference_config.yaml**: Settings for generating forecasts
   - Threshold values for defining price spikes
   - Forecast horizon and frequency
   - Node selections
   - Output settings

2. **training_config.yaml**: Settings for model training
   - Date ranges for training data
   - Model type and hyperparameters
   - Cross-validation settings
   - Output settings

3. **backtesting_config.yaml**: Settings for historical performance evaluation
   - Test period dates
   - Node and threshold selections
   - Results storage location

Example configuration files can be found in the `examples/config` directory.

### API Keys

The system requires API keys to access ERCOT data and weather forecasts:

1. **ERCOT API Key**:
   - Register on the ERCOT developer portal
   - Request access to RTLMP data
   - Generate an API key and secret
   - Add these to your `.env` file

2. **Weather API Key**:
   - Sign up for a weather data provider (e.g., OpenWeatherMap, WeatherAPI)
   - Generate an API key
   - Add the key to your `.env` file

## Verification

After installation, verify that the system is set up correctly:

### Installation Verification

To verify the basic installation:

```bash
# Activate the virtual environment (if using)
source .venv/bin/activate

# Check if Python packages are installed
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, pandera, hydra; print('Dependencies installed!')"

# Check if the CLI tool is available
rtlmp_predict --help
```

### Basic Tests

Run basic tests to ensure functionality:

```bash
# Run unit tests
pytest tests/unit

# Test data fetching (will require API keys)
rtlmp_predict fetch-data --start-date 2023-01-01 --end-date 2023-01-02 --node HB_NORTH

# Generate a test forecast (using cached data if available)
rtlmp_predict predict --threshold 100 --node HB_NORTH
```

### Troubleshooting

Common installation issues and solutions:

1. **Missing Dependencies**:
   ```bash
   pip install -r src/backend/requirements.txt
   ```

2. **API Connection Errors**:
   - Verify API keys in the `.env` file
   - Check network connectivity
   - Confirm API rate limits haven't been exceeded

3. **Directory Permission Issues**:
   ```bash
   chmod -R 755 data models forecasts logs
   ```

4. **Python Version Conflicts**:
   - Ensure you're using Python 3.10+
   - Check for conflicting environments

## Next Steps

After successful installation, you can:

### Setting Up Scheduled Execution

For production use, set up scheduled execution of data fetching, inference, and model retraining. See [Scheduled Execution](scheduled_execution.md) for detailed instructions.

### Running Your First Forecast

Generate your first RTLMP spike probability forecast:

```bash
# Activate the virtual environment (if using)
source .venv/bin/activate

# Generate a forecast
rtlmp_predict predict --threshold 100 --node HB_NORTH
```

The forecast will be saved to the `forecasts` directory and can be visualized using the included tools.

### Training a Model

Train a custom model using your own data and hyperparameters:

```bash
# Activate the virtual environment (if using)
source .venv/bin/activate

# Train a model
rtlmp_predict train --config config/training_config.yaml
```

Custom training configurations can significantly improve forecast accuracy for specific nodes or market conditions.