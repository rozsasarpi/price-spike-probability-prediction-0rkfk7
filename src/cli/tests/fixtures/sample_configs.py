"""
Provides sample configuration dictionaries for testing CLI components of the ERCOT RTLMP spike prediction system.

Contains predefined configuration samples for global CLI settings and command-specific parameters
that can be used in unit tests without requiring actual configuration files.
"""

from typing import Dict, List, Any, Optional, Union, cast, TypedDict
from pathlib import Path
from datetime import datetime
from copy import deepcopy

# Import CLI type definitions
from ../../cli_types import (
    CLIConfigDict, FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict,
    CommandType, LogLevel, DataType, VisualizationType, OutputFormat
)

# Import backend type definitions
from ../../../backend/utils/type_definitions import NodeID, ThresholdValue, ModelType

# Import default configuration dictionaries as a base
from ../../config/default_config import (
    DEFAULT_CLI_CONFIG, DEFAULT_FETCH_DATA_PARAMS, DEFAULT_TRAIN_PARAMS,
    DEFAULT_PREDICT_PARAMS, DEFAULT_BACKTEST_PARAMS, DEFAULT_EVALUATE_PARAMS,
    DEFAULT_VISUALIZE_PARAMS
)

# Sample CLI configuration for testing
SAMPLE_CLI_CONFIG: Dict[str, Any] = {
    "config_file": Path("./test_config.yaml"),
    "log_level": "DEBUG",
    "log_file": Path("./test_logs.log"),
    "output_dir": Path("./test_output"),
    "verbose": True
}

# Sample fetch-data command parameters for testing
SAMPLE_FETCH_DATA_PARAMS: Dict[str, Any] = {
    "data_type": "rtlmp",
    "start_date": datetime(2023, 1, 1).date(),
    "end_date": datetime(2023, 1, 31).date(),
    "nodes": ["HB_NORTH", "HB_SOUTH"],
    "output_path": Path("./test_data"),
    "output_format": "csv",
    "force_refresh": True
}

# Sample train command parameters for testing
SAMPLE_TRAIN_PARAMS: Dict[str, Any] = {
    "start_date": datetime(2022, 1, 1).date(),
    "end_date": datetime(2022, 12, 31).date(),
    "nodes": ["HB_NORTH", "HB_HOUSTON"],
    "thresholds": [75.0, 150.0],
    "model_type": "xgboost",
    "hyperparameters": {
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_estimators": 150
    },
    "optimize_hyperparameters": True,
    "cross_validation_folds": 3,
    "model_name": "test_model",
    "output_path": Path("./test_models")
}

# Sample predict command parameters for testing
SAMPLE_PREDICT_PARAMS: Dict[str, Any] = {
    "threshold": 150.0,
    "nodes": ["HB_NORTH"],
    "model_version": "v1.0.0",
    "output_path": Path("./test_predictions"),
    "output_format": "json",
    "visualize": True
}

# Sample backtest command parameters for testing
SAMPLE_BACKTEST_PARAMS: Dict[str, Any] = {
    "start_date": datetime(2022, 6, 1).date(),
    "end_date": datetime(2022, 6, 30).date(),
    "thresholds": [75.0, 150.0],
    "nodes": ["HB_NORTH", "HB_SOUTH"],
    "model_version": "v1.0.0",
    "output_path": Path("./test_backtest"),
    "output_format": "csv",
    "visualize": True
}

# Sample evaluate command parameters for testing
SAMPLE_EVALUATE_PARAMS: Dict[str, Any] = {
    "model_version": "v1.0.0",
    "compare_with": ["v0.9.0", "v0.8.0"],
    "thresholds": [75.0, 150.0],
    "nodes": ["HB_NORTH", "HB_SOUTH"],
    "start_date": datetime(2022, 7, 1).date(),
    "end_date": datetime(2022, 7, 31).date(),
    "output_path": Path("./test_evaluation"),
    "output_format": "html",
    "visualize": True
}

# Sample visualize command parameters for testing
SAMPLE_VISUALIZE_PARAMS: Dict[str, Any] = {
    "visualization_type": "forecast",
    "forecast_id": "forecast_20220101",
    "model_version": "v1.0.0",
    "compare_with": ["v0.9.0"],
    "threshold": 150.0,
    "nodes": ["HB_NORTH"],
    "start_date": datetime(2022, 8, 1).date(),
    "end_date": datetime(2022, 8, 31).date(),
    "output_path": Path("./test_visualization"),
    "output_format": "png",
    "interactive": True
}

# Dictionary mapping command types to their sample parameters
SAMPLE_COMMAND_PARAMS: Dict[CommandType, Dict[str, Any]] = {
    "fetch-data": SAMPLE_FETCH_DATA_PARAMS,
    "train": SAMPLE_TRAIN_PARAMS,
    "predict": SAMPLE_PREDICT_PARAMS,
    "backtest": SAMPLE_BACKTEST_PARAMS,
    "evaluate": SAMPLE_EVALUATE_PARAMS,
    "visualize": SAMPLE_VISUALIZE_PARAMS
}


def get_sample_cli_config(overrides: Optional[Dict[str, Any]] = None) -> CLIConfigDict:
    """Returns a copy of the sample CLI configuration dictionary.
    
    Args:
        overrides: Optional dictionary of values to override in the sample configuration
        
    Returns:
        CLIConfigDict: Sample CLI configuration dictionary with optional overrides
    """
    config = deepcopy(SAMPLE_CLI_CONFIG)
    if overrides:
        config.update(overrides)
    return cast(CLIConfigDict, config)


def get_sample_command_params(
    command: CommandType, 
    overrides: Optional[Dict[str, Any]] = None
) -> Union[
    FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict
]:
    """Returns a copy of the sample parameters for a specific command.
    
    Args:
        command: The command for which to get sample parameters
        overrides: Optional dictionary of values to override in the sample parameters
        
    Returns:
        Union[FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
              BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict]:
            Sample command parameters dictionary with optional overrides
        
    Raises:
        ValueError: If an invalid command is provided
    """
    if command not in SAMPLE_COMMAND_PARAMS:
        raise ValueError(f"Invalid command: {command}")
    
    params = deepcopy(SAMPLE_COMMAND_PARAMS[command])
    if overrides:
        params.update(overrides)
        
    if command == "fetch-data":
        return cast(FetchDataParamsDict, params)
    elif command == "train":
        return cast(TrainParamsDict, params)
    elif command == "predict":
        return cast(PredictParamsDict, params)
    elif command == "backtest":
        return cast(BacktestParamsDict, params)
    elif command == "evaluate":
        return cast(EvaluateParamsDict, params)
    else:  # command == "visualize"
        return cast(VisualizeParamsDict, params)


def create_test_config(
    config_path: Path, 
    overrides: Optional[Dict[str, Any]] = None
) -> bool:
    """Creates a test configuration file with sample CLI configuration.
    
    Args:
        config_path: Path where the configuration file should be created
        overrides: Optional dictionary of values to override in the sample configuration
        
    Returns:
        bool: True if the configuration file was created successfully, False otherwise
    """
    import json
    import yaml
    
    config = get_sample_cli_config(overrides)
    
    try:
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file format based on extension
        if config_path.suffix in (".yaml", ".yml"):
            with open(config_path, "w") as f:
                yaml.dump(config, f)
        elif config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {config_path.suffix}")
            
        return True
    except Exception as e:
        print(f"Error creating test config file: {e}")
        return False


def create_test_command_params(
    command: CommandType,
    config_path: Path, 
    overrides: Optional[Dict[str, Any]] = None
) -> bool:
    """Creates a test configuration file with sample command parameters.
    
    Args:
        command: The command for which to create a configuration file
        config_path: Path where the configuration file should be created
        overrides: Optional dictionary of values to override in the sample parameters
        
    Returns:
        bool: True if the configuration file was created successfully, False otherwise
    """
    import json
    import yaml
    
    params = get_sample_command_params(command, overrides)
    
    try:
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file format based on extension
        if config_path.suffix in (".yaml", ".yml"):
            with open(config_path, "w") as f:
                yaml.dump(params, f)
        elif config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(params, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {config_path.suffix}")
            
        return True
    except Exception as e:
        print(f"Error creating test command params file: {e}")
        return False