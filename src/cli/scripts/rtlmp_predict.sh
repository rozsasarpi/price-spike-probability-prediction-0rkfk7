#!/bin/bash
#
# ERCOT RTLMP Spike Prediction System CLI
#
# This script provides a command-line interface for the ERCOT RTLMP spike prediction system.
# It handles environment setup, Python interpreter detection, and forwards commands to the Python CLI application.
#

# Global variable definitions
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CLI_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$HOME/.rtlmp_predict"
DEFAULT_CONFIG="$CONFIG_DIR/config.yaml"
LOG_DIR="$CONFIG_DIR/logs"
PYTHON_MIN_VERSION="3.10"

# Function to print an error message to stderr
print_error() {
  local message="$1"
  echo -e "\033[31mERROR: $message\033[0m" >&2
}

# Function to find a suitable Python interpreter (version 3.10+)
find_python() {
  # Check if python3 command exists and meets minimum version requirement
  if command -v python3 &> /dev/null; then
    python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" | grep -q "^${PYTHON_MIN_VERSION}"
    if [ $? -eq 0 ]; then
      echo "python3"
      return
    fi
  fi

  # If not, check if python command exists and meets minimum version requirement
  if command -v python &> /dev/null; then
    python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" | grep -q "^${PYTHON_MIN_VERSION}"
    if [ $? -eq 0 ]; then
      echo "python"
      return
    fi
  fi

  # If a suitable interpreter is not found, return empty string
  echo ""
}

# Function to check if the environment is properly set up for running the CLI
check_environment() {
  # Check if Python interpreter is available
  PYTHON_EXECUTABLE=$(find_python)
  if [ -z "$PYTHON_EXECUTABLE" ]; then
    print_error "Python ${PYTHON_MIN_VERSION}+ is required. Please install it and ensure it's in your PATH."
    return 1
  fi

  # Verify that the CONFIG_DIR exists, create it if not
  if [ ! -d "$CONFIG_DIR" ]; then
    mkdir -p "$CONFIG_DIR"
    if [ $? -ne 0 ]; then
      print_error "Failed to create configuration directory: $CONFIG_DIR"
      return 1
    fi
  fi

  # Verify that the LOG_DIR exists, create it if not
  if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    if [ $? -ne 0 ]; then
      print_error "Failed to create log directory: $LOG_DIR"
      return 1
    fi
  fi

  # Return 0 if all checks pass
  return 0
}

# Function to print version information
print_version() {
  echo "ERCOT RTLMP Spike Prediction System CLI"
  echo "Version: 1.0" # TODO: Implement dynamic version retrieval
  echo "Python: $(python3 -V 2>&1)"
  echo "System: $(uname -srm)"
}

# Function to run the Python CLI application with the provided arguments
run_cli() {
  local args=("$@")

  # Get Python interpreter path
  PYTHON_EXECUTABLE=$(find_python)
  if [ -z "$PYTHON_EXECUTABLE" ]; then
    print_error "Python ${PYTHON_MIN_VERSION}+ is required. Please install it and ensure it's in your PATH."
    return 1
  fi

  # Set PYTHONPATH to include the project root directory
  export PYTHONPATH="$CLI_ROOT:$PYTHONPATH"

  # Execute the Python CLI application (main.py) with the provided arguments
  "${PYTHON_EXECUTABLE}" "$CLI_ROOT/src/cli/main.py" "${args[@]}"
  local exit_code=$?

  # Return the exit code from the Python CLI application
  return $exit_code
}

# Main execution flow
if [ $# -eq 0 ]; then
  # No arguments provided, print usage information
  run_cli --help
  exit 0
fi

# Special argument handling
if [[ "$1" == "--version" ]]; then
  print_version
  exit 0
fi

if [[ "$1" == "--help" ]]; then
  run_cli --help
  exit 0
fi

# Check environment setup
if [ $(check_environment) -ne 0 ]; then
  exit 1
fi

# Run the CLI with argument forwarding
run_cli "$@"
exit $?