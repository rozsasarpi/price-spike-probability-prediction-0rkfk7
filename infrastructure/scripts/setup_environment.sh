#!/bin/bash
# setup_environment.sh
#
# Sets up the development and deployment environment for the ERCOT RTLMP spike prediction system
# This script creates necessary directories, installs dependencies, configures environment variables,
# and prepares the system for data processing, model training, and inference operations.

# Exit on error
set -e

# Define global variables
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
PYTHON_VERSION="3.10"
VENV_DIR="$PROJECT_ROOT/.venv"
DATA_DIR="$PROJECT_ROOT/data"
MODELS_DIR="$PROJECT_ROOT/models"
FORECASTS_DIR="$PROJECT_ROOT/forecasts"
LOGS_DIR="$PROJECT_ROOT/logs"
CONFIG_DIR="$PROJECT_ROOT/config"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE_FILE="$PROJECT_ROOT/infrastructure/docker/.env.example"
REQUIRED_PACKAGES="python3-dev python3-pip python3-venv build-essential"

# Parse command line arguments
WITH_CRON=false
HELP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-cron)
            WITH_CRON=true
            shift
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if $HELP; then
    echo "Usage: ./setup_environment.sh [--with-cron] [--python-version VERSION] [--help]"
    echo ""
    echo "Options:"
    echo "  --with-cron          Set up cron jobs for scheduled execution"
    echo "  --python-version     Specify Python version (default: 3.10)"
    echo "  --help               Display this help message"
    exit 0
fi

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/setup_$(date +%Y-%m-%d).log"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [$level] $message"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Cleanup function
cleanup() {
    log_message "INFO" "Cleaning up..."
    # Remove any temporary files created during installation
    if [ -f "/tmp/rtlmp_setup_temp" ]; then
        rm /tmp/rtlmp_setup_temp
    fi
}

# Set up trap for cleanup on exit
trap cleanup EXIT

log_message "INFO" "Starting environment setup for ERCOT RTLMP spike prediction system"

# Check if the system meets the minimum requirements
check_system_requirements() {
    log_message "INFO" "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &>/dev/null; then
        PYTHON_INSTALLED_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_REQUIRED_VERSION=$PYTHON_VERSION
        log_message "INFO" "Found Python version: $PYTHON_INSTALLED_VERSION (Required: $PYTHON_REQUIRED_VERSION)"
        
        if [ "$(printf '%s\n' "$PYTHON_REQUIRED_VERSION" "$PYTHON_INSTALLED_VERSION" | sort -V | head -n1)" != "$PYTHON_REQUIRED_VERSION" ]; then
            log_message "ERROR" "Python version $PYTHON_REQUIRED_VERSION or higher is required"
            return 1
        fi
    else
        log_message "ERROR" "Python 3 is not installed"
        return 1
    fi
    
    # Check available disk space (minimum 100GB)
    local available_space=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    log_message "INFO" "Available disk space: ${available_space}GB (Required: 100GB)"
    if (( $(echo "$available_space < 100" | bc -l) )); then
        log_message "WARNING" "Less than 100GB of disk space available"
    fi
    
    # Check available memory (minimum 16GB)
    if [ "$(uname)" == "Linux" ]; then
        local total_memory=$(free -g | awk '/^Mem:/{print $2}')
        log_message "INFO" "Available memory: ${total_memory}GB (Required: 16GB)"
        if [ "$total_memory" -lt 16 ]; then
            log_message "WARNING" "Less than 16GB of memory available"
        fi
    elif [ "$(uname)" == "Darwin" ]; then
        local total_memory=$(sysctl hw.memsize | awk '{print $2 / 1024 / 1024 / 1024}')
        log_message "INFO" "Available memory: ${total_memory}GB (Required: 16GB)"
        if (( $(echo "$total_memory < 16" | bc -l) )); then
            log_message "WARNING" "Less than 16GB of memory available"
        fi
    else
        log_message "WARNING" "Unable to determine available memory on this OS"
    fi
    
    # Check CPU cores (minimum 4)
    if [ "$(uname)" == "Linux" ]; then
        local cpu_cores=$(nproc)
    elif [ "$(uname)" == "Darwin" ]; then
        local cpu_cores=$(sysctl -n hw.ncpu)
    else
        local cpu_cores="unknown"
    fi
    
    log_message "INFO" "Available CPU cores: $cpu_cores (Required: 4)"
    if [ "$cpu_cores" != "unknown" ] && [ "$cpu_cores" -lt 4 ]; then
        log_message "WARNING" "Less than 4 CPU cores available"
    fi
    
    log_message "INFO" "System requirements check completed"
    return 0
}

# Install required system packages
install_system_dependencies() {
    log_message "INFO" "Installing system dependencies..."
    
    # Detect OS
    if [ -f /etc/os-release ]; then
        # freedesktop.org and systemd
        . /etc/os-release
        OS=$NAME
    elif type lsb_release >/dev/null 2>&1; then
        # linuxbase.org
        OS=$(lsb_release -si)
    elif [ -f /etc/lsb-release ]; then
        # For some versions of Debian/Ubuntu without lsb_release command
        . /etc/lsb-release
        OS=$DISTRIB_ID
    elif [ -f /etc/debian_version ]; then
        # Older Debian/Ubuntu/etc.
        OS=Debian
    elif [ -f /etc/SuSe-release ]; then
        # Older SuSE/etc.
        OS=SuSE
    elif [ -f /etc/redhat-release ]; then
        # Older Red Hat, CentOS, etc.
        OS=RedHat
    elif [ "$(uname)" == "Darwin" ]; then
        OS=macOS
    else
        # Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
        OS=$(uname -s)
    fi
    
    log_message "INFO" "Detected OS: $OS"
    
    # Install packages based on OS
    case "$OS" in
        Ubuntu|Debian|Debian*|Ubuntu*)
            log_message "INFO" "Using apt to install packages"
            sudo apt-get update
            sudo apt-get install -y $REQUIRED_PACKAGES
            ;;
        CentOS*|RedHat*|Fedora*|Red*)
            log_message "INFO" "Using yum to install packages"
            sudo yum update -y
            # Convert packages to CentOS equivalents
            CENTOS_PACKAGES=$(echo "$REQUIRED_PACKAGES" | sed 's/python3-dev/python3-devel/g')
            sudo yum install -y $CENTOS_PACKAGES
            ;;
        macOS)
            log_message "INFO" "Using brew to install packages"
            if ! command -v brew &>/dev/null; then
                log_message "ERROR" "Homebrew is not installed. Please install Homebrew first."
                return 1
            fi
            # Convert packages to macOS equivalents
            brew install python@$PYTHON_VERSION
            ;;
        *)
            log_message "ERROR" "Unsupported OS: $OS"
            return 1
            ;;
    esac
    
    log_message "INFO" "System dependencies installed successfully"
    return 0
}

# Create the required directory structure
create_directory_structure() {
    log_message "INFO" "Creating directory structure..."
    
    # Create data directory and subdirectories
    mkdir -p "$DATA_DIR/raw"
    mkdir -p "$DATA_DIR/features"
    
    # Create other required directories
    mkdir -p "$MODELS_DIR"
    mkdir -p "$FORECASTS_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$CONFIG_DIR"
    
    # Set appropriate permissions
    chmod -R 755 "$DATA_DIR"
    chmod -R 755 "$MODELS_DIR"
    chmod -R 755 "$FORECASTS_DIR"
    chmod -R 755 "$LOGS_DIR"
    chmod -R 755 "$CONFIG_DIR"
    
    log_message "INFO" "Directory structure created successfully"
    log_message "INFO" "Data directory: $DATA_DIR"
    log_message "INFO" "Models directory: $MODELS_DIR"
    log_message "INFO" "Forecasts directory: $FORECASTS_DIR"
    log_message "INFO" "Logs directory: $LOGS_DIR"
    log_message "INFO" "Config directory: $CONFIG_DIR"
    
    return 0
}

# Set up Python virtual environment and install dependencies
setup_python_environment() {
    log_message "INFO" "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv "$VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install dependencies
    log_message "INFO" "Installing Python dependencies..."
    
    # Check if requirements files exist
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        log_message "INFO" "Installing from requirements.txt"
        pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        log_message "INFO" "requirements.txt not found, installing core packages"
        pip install numpy==1.24.3 pandas==2.0.2 scikit-learn==1.2.2 xgboost==1.7.5 \
            lightgbm==3.3.5 pandera==0.15.1 joblib==1.2.0 matplotlib==3.7.1 \
            seaborn==0.12.2 plotly==5.14.1 pytest==7.3.1 pydantic==2.0.3 \
            hydra-core==1.3.2 black==23.3.0 isort==5.12.0 mypy==1.3.0
    fi
    
    # Deactivate virtual environment
    deactivate
    
    log_message "INFO" "Python virtual environment set up successfully at $VENV_DIR"
    return 0
}

# Set up environment variables
setup_environment_variables() {
    log_message "INFO" "Setting up environment variables..."
    
    # Check if .env file exists, create from .env.example if not
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$ENV_EXAMPLE_FILE" ]; then
            log_message "INFO" "Creating .env file from .env.example"
            cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
        else
            log_message "INFO" "Creating new .env file"
            touch "$ENV_FILE"
        fi
    fi
    
    # Prompt for required API keys if not set
    if ! grep -q "ERCOT_API_KEY" "$ENV_FILE" || grep -q "ERCOT_API_KEY=" "$ENV_FILE"; then
        echo -n "Enter ERCOT API key (press Enter to skip): "
        read -r ERCOT_API_KEY
        if [ -n "$ERCOT_API_KEY" ]; then
            sed -i.bak "/ERCOT_API_KEY/d" "$ENV_FILE" 2>/dev/null || true
            echo "ERCOT_API_KEY=$ERCOT_API_KEY" >> "$ENV_FILE"
            log_message "INFO" "ERCOT API key configured"
        else
            log_message "WARNING" "ERCOT API key not provided"
        fi
    fi
    
    if ! grep -q "WEATHER_API_KEY" "$ENV_FILE" || grep -q "WEATHER_API_KEY=" "$ENV_FILE"; then
        echo -n "Enter Weather API key (press Enter to skip): "
        read -r WEATHER_API_KEY
        if [ -n "$WEATHER_API_KEY" ]; then
            sed -i.bak "/WEATHER_API_KEY/d" "$ENV_FILE" 2>/dev/null || true
            echo "WEATHER_API_KEY=$WEATHER_API_KEY" >> "$ENV_FILE"
            log_message "INFO" "Weather API key configured"
        else
            log_message "WARNING" "Weather API key not provided"
        fi
    fi
    
    # Set other default environment variables
    DEFAULT_ENV_VARS=(
        "DATA_DIR=$DATA_DIR"
        "MODELS_DIR=$MODELS_DIR"
        "FORECASTS_DIR=$FORECASTS_DIR"
        "LOGS_DIR=$LOGS_DIR"
        "CONFIG_DIR=$CONFIG_DIR"
        "PYTHON_PATH=$PROJECT_ROOT"
    )
    
    for var in "${DEFAULT_ENV_VARS[@]}"; do
        var_name=$(echo "$var" | cut -d= -f1)
        var_value=$(echo "$var" | cut -d= -f2-)
        
        if ! grep -q "^$var_name=" "$ENV_FILE"; then
            echo "$var_name=$var_value" >> "$ENV_FILE"
            log_message "INFO" "Set $var_name=$var_value"
        fi
    done
    
    log_message "INFO" "Environment variables configured successfully"
    return 0
}

# Set up configuration files
setup_configuration_files() {
    log_message "INFO" "Setting up configuration files..."
    
    # Create default configuration files if they don't exist
    
    # Create default inference configuration
    if [ ! -f "$CONFIG_DIR/inference_config.yaml" ]; then
        log_message "INFO" "Creating default inference configuration"
        cat > "$CONFIG_DIR/inference_config.yaml" << EOF
# Default inference configuration
model:
  version: latest
  threshold_values: [50, 100, 200]

data:
  nodes: [HB_NORTH, HB_SOUTH, HB_WEST, HB_HOUSTON]
  forecast_horizon: 72

output:
  forecast_directory: ${FORECASTS_DIR}
  log_directory: ${LOGS_DIR}
EOF
    fi
    
    # Create default training configuration
    if [ ! -f "$CONFIG_DIR/training_config.yaml" ]; then
        log_message "INFO" "Creating default training configuration"
        cat > "$CONFIG_DIR/training_config.yaml" << EOF
# Default training configuration
data:
  start_date: "2020-01-01"
  end_date: "auto"  # Will use current date - 1 day
  nodes: [HB_NORTH, HB_SOUTH, HB_WEST, HB_HOUSTON]
  threshold_values: [50, 100, 200]

model:
  type: "xgboost"
  hyperparameters:
    learning_rate: 0.05
    max_depth: 6
    min_child_weight: 1
    subsample: 0.8
    colsample_bytree: 0.8
    n_estimators: 200
  
  cross_validation:
    method: "time_based"
    n_splits: 5

output:
  model_directory: ${MODELS_DIR}
  log_directory: ${LOGS_DIR}
EOF
    fi
    
    # Create default backtesting configuration
    if [ ! -f "$CONFIG_DIR/backtesting_config.yaml" ]; then
        log_message "INFO" "Creating default backtesting configuration"
        cat > "$CONFIG_DIR/backtesting_config.yaml" << EOF
# Default backtesting configuration
data:
  start_date: "2022-01-01"
  end_date: "2022-12-31"
  nodes: [HB_NORTH, HB_SOUTH]
  threshold_values: [50, 100, 200]

model:
  version: "latest"  # or specific version

output:
  results_directory: ${FORECASTS_DIR}/backtesting
  log_directory: ${LOGS_DIR}
EOF
    fi
    
    log_message "INFO" "Configuration files set up successfully"
    return 0
}

# Set up cron jobs for scheduled execution
setup_cron_jobs() {
    log_message "INFO" "Setting up cron jobs for scheduled execution..."
    
    # Check if crontab is available
    if ! command -v crontab &>/dev/null; then
        log_message "ERROR" "crontab is not available on this system"
        return 1
    fi
    
    # Create a temporary file for crontab entries
    CRONTAB_FILE="/tmp/rtlmp_crontab"
    crontab -l > "$CRONTAB_FILE" 2>/dev/null || echo "" > "$CRONTAB_FILE"
    
    # Check if entries already exist
    if grep -q "RTLMP" "$CRONTAB_FILE"; then
        log_message "WARNING" "RTLMP cron jobs already exist. Skipping cron setup."
        rm "$CRONTAB_FILE"
        return 0
    fi
    
    # Add header comment
    echo "# RTLMP prediction system scheduled jobs" >> "$CRONTAB_FILE"
    
    # Add daily inference job (runs at 06:00)
    echo "0 6 * * * cd $PROJECT_ROOT && $VENV_DIR/bin/python -m rtlmp_predict predict --config $CONFIG_DIR/inference_config.yaml >> $LOGS_DIR/inference_\$(date +\%Y-\%m-\%d).log 2>&1" >> "$CRONTAB_FILE"
    
    # Add bi-daily model retraining job (runs at 01:00 every 2 days)
    echo "0 1 */2 * * cd $PROJECT_ROOT && $VENV_DIR/bin/python -m rtlmp_predict train --config $CONFIG_DIR/training_config.yaml >> $LOGS_DIR/training_\$(date +\%Y-\%m-\%d).log 2>&1" >> "$CRONTAB_FILE"
    
    # Add weekly backtesting job (runs on Sunday at 02:00)
    echo "0 2 * * 0 cd $PROJECT_ROOT && $VENV_DIR/bin/python -m rtlmp_predict backtest --config $CONFIG_DIR/backtesting_config.yaml >> $LOGS_DIR/backtesting_\$(date +\%Y-\%m-\%d).log 2>&1" >> "$CRONTAB_FILE"
    
    # Install new crontab
    crontab "$CRONTAB_FILE"
    rm "$CRONTAB_FILE"
    
    log_message "INFO" "Cron jobs set up successfully"
    return 0
}

# Verify that the environment is set up correctly
verify_installation() {
    log_message "INFO" "Verifying installation..."
    
    # Check directories
    for dir in "$DATA_DIR" "$MODELS_DIR" "$FORECASTS_DIR" "$LOGS_DIR" "$CONFIG_DIR"; do
        if [ ! -d "$dir" ]; then
            log_message "ERROR" "Directory $dir does not exist"
            return 1
        fi
    done
    
    # Check virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        log_message "ERROR" "Virtual environment directory $VENV_DIR does not exist"
        return 1
    fi
    
    # Activate virtual environment and test Python
    source "$VENV_DIR/bin/activate"
    
    # Check if required packages are installed
    log_message "INFO" "Checking installed packages..."
    python -c "import numpy, pandas, sklearn, xgboost" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Required Python packages are not installed correctly"
        deactivate
        return 1
    fi
    
    # Check environment variables
    if [ ! -f "$ENV_FILE" ]; then
        log_message "ERROR" "Environment file $ENV_FILE does not exist"
        deactivate
        return 1
    fi
    
    # Check configuration files
    for config in "inference_config.yaml" "training_config.yaml" "backtesting_config.yaml"; do
        if [ ! -f "$CONFIG_DIR/$config" ]; then
            log_message "ERROR" "Configuration file $CONFIG_DIR/$config does not exist"
            deactivate
            return 1
        fi
    done
    
    deactivate
    
    log_message "SUCCESS" "Installation verified successfully"
    return 0
}

# Main function
main() {
    # Check system requirements
    check_system_requirements
    if [ $? -ne 0 ]; then
        log_message "ERROR" "System requirements check failed"
        return 1
    fi
    
    # Install system dependencies
    install_system_dependencies
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Failed to install system dependencies"
        return 1
    fi
    
    # Create directory structure
    create_directory_structure
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Failed to create directory structure"
        return 1
    fi
    
    # Set up Python environment
    setup_python_environment
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Failed to set up Python environment"
        return 1
    fi
    
    # Set up environment variables
    setup_environment_variables
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Failed to set up environment variables"
        return 1
    fi
    
    # Set up configuration files
    setup_configuration_files
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Failed to set up configuration files"
        return 1
    fi
    
    # Set up cron jobs if requested
    if $WITH_CRON; then
        setup_cron_jobs
        if [ $? -ne 0 ]; then
            log_message "ERROR" "Failed to set up cron jobs"
            return 1
        fi
    fi
    
    # Verify installation
    verify_installation
    if [ $? -ne 0 ]; then
        log_message "ERROR" "Installation verification failed"
        return 1
    fi
    
    log_message "SUCCESS" "Environment setup completed successfully"
    
    # Display next steps
    echo ""
    echo "===================================================="
    echo "ENVIRONMENT SETUP COMPLETED SUCCESSFULLY"
    echo "===================================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source $VENV_DIR/bin/activate"
    echo ""
    echo "2. Set up API keys in $ENV_FILE if not already configured"
    echo ""
    echo "3. Customize configuration files in $CONFIG_DIR as needed"
    echo ""
    echo "4. Run a test inference:"
    echo "   rtlmp_predict predict --config $CONFIG_DIR/inference_config.yaml"
    echo ""
    echo "Setup log available at: $LOG_FILE"
    echo "===================================================="
    
    return 0
}

# Run main function
main
exit $?