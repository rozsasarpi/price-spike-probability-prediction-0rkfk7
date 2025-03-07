"""
Defines default configuration values for the CLI application of the ERCOT RTLMP spike prediction system.

This module provides functions to generate default configuration dictionaries for both global CLI
settings and command-specific parameters, ensuring consistent behavior when custom configurations
are not provided.
"""

import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Literal, cast

from ..cli_types import (
    CLIConfigDict, FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict,
    CommandType, LogLevel, DataType, VisualizationType, OutputFormat
)
from ...backend.utils.type_definitions import NodeID, ThresholdValue, ModelType

# Set up logger
logger = logging.getLogger(__name__)

# Default node IDs for ERCOT
DEFAULT_NODE_IDS: List[NodeID] = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"]

# Default threshold values for spike detection
DEFAULT_THRESHOLDS: List[ThresholdValue] = [50.0, 100.0, 200.0]

# Default CLI configuration
DEFAULT_CLI_CONFIG: Dict[str, Any] = {
    "config_file": None,
    "log_level": "INFO",
    "log_file": None,
    "output_dir": Path("./output"),
    "verbose": False
}

# Default parameters for fetch-data command
DEFAULT_FETCH_DATA_PARAMS: Dict[str, Any] = {
    "data_type": "all",
    "start_date": date.today() - timedelta(days=30),
    "end_date": date.today(),
    "nodes": DEFAULT_NODE_IDS,
    "output_path": None,
    "output_format": "csv",
    "force_refresh": False
}

# Default parameters for train command
DEFAULT_TRAIN_PARAMS: Dict[str, Any] = {
    "start_date": date.today() - timedelta(days=365),
    "end_date": date.today(),
    "nodes": DEFAULT_NODE_IDS,
    "thresholds": DEFAULT_THRESHOLDS,
    "model_type": "xgboost",
    "hyperparameters": {
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200
    },
    "optimize_hyperparameters": True,
    "cross_validation_folds": 5,
    "model_name": None,
    "output_path": None
}

# Default parameters for predict command
DEFAULT_PREDICT_PARAMS: Dict[str, Any] = {
    "threshold": 100.0,
    "nodes": DEFAULT_NODE_IDS,
    "model_version": None,  # Use latest version
    "output_path": None,
    "output_format": "csv",
    "visualize": True
}

# Default parameters for backtest command
DEFAULT_BACKTEST_PARAMS: Dict[str, Any] = {
    "start_date": date.today() - timedelta(days=90),
    "end_date": date.today(),
    "thresholds": DEFAULT_THRESHOLDS,
    "nodes": DEFAULT_NODE_IDS,
    "model_version": None,  # Use latest version
    "output_path": None,
    "output_format": "csv",
    "visualize": True
}

# Default parameters for evaluate command
DEFAULT_EVALUATE_PARAMS: Dict[str, Any] = {
    "model_version": None,  # Use latest version
    "compare_with": [],
    "thresholds": DEFAULT_THRESHOLDS,
    "nodes": DEFAULT_NODE_IDS,
    "start_date": date.today() - timedelta(days=90),
    "end_date": date.today(),
    "output_path": None,
    "output_format": "csv",
    "visualize": True
}

# Default parameters for visualize command
DEFAULT_VISUALIZE_PARAMS: Dict[str, Any] = {
    "visualization_type": "forecast",
    "forecast_id": None,  # Use latest forecast
    "model_version": None,  # Use latest model version
    "compare_with": [],
    "threshold": 100.0,
    "nodes": DEFAULT_NODE_IDS,
    "start_date": date.today() - timedelta(days=7),
    "end_date": date.today(),
    "output_path": None,
    "output_format": "png",
    "interactive": True
}


def get_default_cli_config() -> CLIConfigDict:
    """Returns the default CLI configuration dictionary.

    Returns:
        CLIConfigDict: Default CLI configuration dictionary
    """
    return cast(CLIConfigDict, DEFAULT_CLI_CONFIG.copy())


def get_default_fetch_data_params() -> FetchDataParamsDict:
    """Returns the default parameters for the fetch-data command.

    Returns:
        FetchDataParamsDict: Default fetch-data command parameters
    """
    return cast(FetchDataParamsDict, DEFAULT_FETCH_DATA_PARAMS.copy())


def get_default_train_params() -> TrainParamsDict:
    """Returns the default parameters for the train command.

    Returns:
        TrainParamsDict: Default train command parameters
    """
    return cast(TrainParamsDict, DEFAULT_TRAIN_PARAMS.copy())


def get_default_predict_params() -> PredictParamsDict:
    """Returns the default parameters for the predict command.

    Returns:
        PredictParamsDict: Default predict command parameters
    """
    return cast(PredictParamsDict, DEFAULT_PREDICT_PARAMS.copy())


def get_default_backtest_params() -> BacktestParamsDict:
    """Returns the default parameters for the backtest command.

    Returns:
        BacktestParamsDict: Default backtest command parameters
    """
    return cast(BacktestParamsDict, DEFAULT_BACKTEST_PARAMS.copy())


def get_default_evaluate_params() -> EvaluateParamsDict:
    """Returns the default parameters for the evaluate command.

    Returns:
        EvaluateParamsDict: Default evaluate command parameters
    """
    return cast(EvaluateParamsDict, DEFAULT_EVALUATE_PARAMS.copy())


def get_default_visualize_params() -> VisualizeParamsDict:
    """Returns the default parameters for the visualize command.

    Returns:
        VisualizeParamsDict: Default visualize command parameters
    """
    return cast(VisualizeParamsDict, DEFAULT_VISUALIZE_PARAMS.copy())


def get_default_command_params(command: CommandType) -> Union[
    FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict
]:
    """Returns the default parameters for a specific command.

    Args:
        command: The command for which to get default parameters

    Returns:
        Union[FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
              BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict]:
            Default parameters for the specified command

    Raises:
        ValueError: If an invalid command is provided
    """
    if command == "fetch-data":
        return get_default_fetch_data_params()
    elif command == "train":
        return get_default_train_params()
    elif command == "predict":
        return get_default_predict_params()
    elif command == "backtest":
        return get_default_backtest_params()
    elif command == "evaluate":
        return get_default_evaluate_params()
    elif command == "visualize":
        return get_default_visualize_params()
    else:
        raise ValueError(f"Invalid command: {command}")


def merge_with_cli_defaults(user_config: Dict[str, Any]) -> CLIConfigDict:
    """Merges user-provided CLI configuration with default values.

    Args:
        user_config: User-provided CLI configuration dictionary

    Returns:
        CLIConfigDict: Merged CLI configuration dictionary
    """
    default_config = get_default_cli_config()
    merged_config = {**default_config, **user_config}
    return cast(CLIConfigDict, merged_config)


def merge_with_command_defaults(
    command: CommandType, user_params: Dict[str, Any]
) -> Union[
    FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict
]:
    """Merges user-provided command parameters with default values.

    Args:
        command: The command for which to merge parameters
        user_params: User-provided parameter dictionary

    Returns:
        Union[FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
              BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict]:
            Merged command parameters dictionary
    """
    default_params = get_default_command_params(command)
    merged_params = {**default_params, **user_params}
    return merged_params