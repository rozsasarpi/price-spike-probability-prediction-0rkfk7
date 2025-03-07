"""
Initializes the commands package and exports all CLI command functions for the ERCOT RTLMP spike prediction system.
This file serves as the central point for importing and re-exporting all command functions that can be registered with the CLI application.
"""
# Import the fetch-data command function
from .fetch_data import fetch_data_command

# Import the train command function
from .train import train_command

# Import the predict command function
from .predict import predict_command

# Import the backtest command function
from .backtest import backtest_command

# Import the evaluate command function
from .evaluate import evaluate_command

# Import the visualize command function
from .visualize import visualize_command

# Export all command functions
__all__ = [
    "fetch_data_command",
    "train_command",
    "predict_command",
    "backtest_command",
    "evaluate_command",
    "visualize_command"
]