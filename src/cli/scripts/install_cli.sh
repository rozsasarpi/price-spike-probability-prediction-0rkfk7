#!/bin/bash
#
# ERCOT RTLMP Spike Prediction System CLI Installation Script
#
# This script automates the installation of the ERCOT RTLMP spike prediction system's
# command-line interface (CLI) tool on a user's system. It performs prerequisite checks,
# installs Python dependencies, sets up the CLI executable, and configures bash completion.
#
# Usage:
#   1. Run this script with sudo or root privileges:
#      sudo ./install_cli.sh
#   2. Follow the on-screen instructions.
#
# The script will:
#   - Check for necessary prerequisites (Python 3.10+, pip, sudo/root permissions)
#   - Install required Python packages using pip
#   - Copy the CLI executable to /usr/local/bin
#   - Set up bash completion for the CLI
#   - Create a configuration directory in the user's home directory
#
# Exit codes:
#   - 0: Installation successful
#   - 1: Installation failed due to an error

# Script header with description and usage information

# Global variable definitions
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CLI_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="/usr/local/bin"
COMPLETION_DIR="/etc/bash_completion.d"
CONFIG_DIR="$HOME/.rtlmp_predict"
LOG_DIR="$CONFIG_DIR/logs"
PYTHON_MIN_VERSION="3.10"
REQUIRED_PACKAGES="click pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly pandera joblib hydra-core pydantic typing_extensions"

# Function definitions for installation steps
check_prerequisites() {
  # Checks if all prerequisites for installation are met
  #
  # Returns:
  #   0 if all checks pass, non-zero otherwise

  # Check if script is run with sudo/root permissions
  if [[ $EUID -ne 0 ]]; then
    echo "This script requires sudo or root privileges. Please run with sudo." >&2
    return 1
  fi

  # Verify Python 3.10+ is installed
  if ! python3 -c "import sys; assert sys.version_info >= (3, 10)" &> /dev/null; then
    echo "Python $PYTHON_MIN_VERSION+ is required. Please install it and ensure it's in your PATH." >&2
    return 1
  fi

  # Check if pip is available
  if ! command -v pip3 &> /dev/null; then
    echo "pip3 is required. Please install it and ensure it's in your PATH." >&2
    return 1
  fi

  # Ensure required directories exist or can be created
  if [ ! -d "$INSTALL_DIR" ]; then
    echo "Creating installation directory: $INSTALL_DIR"
    sudo mkdir -p "$INSTALL_DIR"
    if [ $? -ne 0 ]; then
      echo "Failed to create installation directory: $INSTALL_DIR" >&2
      return 1
    fi
  fi

  if [ ! -d "$COMPLETION_DIR" ]; then
    echo "Creating completion directory: $COMPLETION_DIR"
    sudo mkdir -p "$COMPLETION_DIR"
    if [ $? -ne 0 ]; then
      echo "Failed to create completion directory: $COMPLETION_DIR" >&2
      return 1
    fi
  fi

  return 0
}

install_python_dependencies() {
  # Installs required Python packages
  #
  # Returns:
  #   0 if successful, non-zero otherwise

  echo "Installing required Python dependencies..."
  sudo pip3 install --upgrade $REQUIRED_PACKAGES
  if [ $? -ne 0 ]; then
    echo "Failed to install Python dependencies." >&2
    return 1
  fi

  return 0
}

install_cli_executable() {
  # Installs the CLI executable script to the system path
  #
  # Returns:
  #   0 if successful, non-zero otherwise

  echo "Installing CLI executable..."
  sudo cp "$SCRIPT_DIR/rtlmp_predict.sh" "$INSTALL_DIR/rtlmp_predict"
  if [ $? -ne 0 ]; then
    echo "Failed to copy CLI executable." >&2
    return 1
  fi

  sudo chmod +x "$INSTALL_DIR/rtlmp_predict"
  if [ $? -ne 0 ]; then
    echo "Failed to make CLI executable executable." >&2
    return 1
  fi

  # Create symbolic link if needed
  if ! sudo ln -sfn "$INSTALL_DIR/rtlmp_predict" "/usr/bin/rtlmp_predict" 2>/dev/null; then
    echo "Creating symbolic link in /usr/bin..."
    if ! sudo ln -sfn "$INSTALL_DIR/rtlmp_predict" "/usr/local/bin/rtlmp_predict"; then
      echo "Failed to create symbolic link." >&2
      return 1
    fi
  fi

  return 0
}

install_bash_completion() {
  # Installs bash completion for the CLI
  #
  # Returns:
  #   0 if successful, non-zero otherwise

  echo "Installing bash completion..."
  sudo cp "$SCRIPT_DIR/completion.sh" "$COMPLETION_DIR/rtlmp_predict"
  if [ $? -ne 0 ]; then
    echo "Failed to copy completion script." >&2
    return 1
  fi

  # Source the completion script in current session
  source "$COMPLETION_DIR/rtlmp_predict"

  # Add source command to user's .bashrc if not already present
  if ! grep -q "source $COMPLETION_DIR/rtlmp_predict" "$HOME/.bashrc"; then
    echo "Adding source command to .bashrc..."
    echo "source $COMPLETION_DIR/rtlmp_predict" >> "$HOME/.bashrc"
  fi

  return 0
}

setup_config_directory() {
  # Sets up the configuration directory for the CLI
  #
  # Returns:
  #   0 if successful, non-zero otherwise

  echo "Setting up configuration directory..."

  # Create CONFIG_DIR if it doesn't exist
  if [ ! -d "$CONFIG_DIR" ]; then
    mkdir -p "$CONFIG_DIR"
    if [ $? -ne 0 ]; then
      echo "Failed to create configuration directory: $CONFIG_DIR" >&2
      return 1
    fi
  fi

  # Create LOG_DIR if it doesn't exist
  if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    if [ $? -ne 0 ]; then
      echo "Failed to create log directory: $LOG_DIR" >&2
      return 1
    fi
  fi

  # Set appropriate permissions
  sudo chown -R "$USER:$USER" "$CONFIG_DIR"
  if [ $? -ne 0 ]; then
    echo "Failed to set permissions on configuration directory." >&2
    return 1
  fi

  # Create default configuration file if it doesn't exist
  if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    echo "Creating default configuration file..."
    echo "# Default configuration file for ERCOT RTLMP Spike Prediction System CLI" > "$CONFIG_DIR/config.yaml"
  fi

  return 0
}

print_success_message() {
  # Prints a success message with usage information
  echo "Installation successful!"
  echo "The CLI executable is located at: /usr/local/bin/rtlmp_predict"
  echo "Basic usage: rtlmp_predict --help"
  echo "Configuration directory: $CONFIG_DIR"
  echo "Please open a new terminal for bash completion to take effect."
}

handle_error() {
  # Handles installation errors
  #
  # Parameters:
  #   $1: Exit code
  #   $2: Error message
  #
  # Returns:
  #   The provided exit code

  local exit_code="$1"
  local error_message="$2"

  echo "ERROR: $error_message" >&2
  echo "Troubleshooting suggestions:" >&2
  echo "  - Ensure you have the required prerequisites installed." >&2
  echo "  - Check your internet connection." >&2
  echo "  - Try running the script again with sudo." >&2

  exit "$exit_code"
}

# Main execution flow with prerequisite checking
if ! check_prerequisites; then
  handle_error 1 "Prerequisite checks failed."
fi

# Sequential installation steps with error handling
if ! install_python_dependencies; then
  handle_error 1 "Python dependency installation failed."
fi

if ! install_cli_executable; then
  handle_error 1 "CLI executable installation failed."
fi

if ! install_bash_completion; then
  handle_error 1 "Bash completion installation failed."
fi

if ! setup_config_directory; then
  handle_error 1 "Configuration directory setup failed."
fi

# Success message and exit with appropriate status code
print_success_message
exit 0