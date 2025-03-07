"""
Provides validation utilities for the CLI application of the ERCOT RTLMP spike prediction system.

This module contains functions for validating user inputs, command-line arguments, configuration parameters, 
and data formats to ensure data integrity and proper CLI operation.
"""

import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar, cast

# Internal imports
from ..exceptions import ValidationError
from ..logger import get_cli_logger
from .formatters import format_date, format_price, format_probability
from ..cli_types import (
    CommandType, LogLevel, DataType, VisualizationType, OutputFormat,
    CLIConfigDict, FetchDataParamsDict, TrainParamsDict, PredictParamsDict,
    BacktestParamsDict, EvaluateParamsDict, VisualizeParamsDict
)
from ...backend.utils.type_definitions import NodeID, ThresholdValue, ModelType

# Initialize logger
logger = get_cli_logger(__name__)

# Global constants for validation
VALID_NODE_ID_PATTERN = r'^[A-Z_]+$'
MIN_THRESHOLD_VALUE = 10.0
MAX_THRESHOLD_VALUE = 10000.0
MIN_DATE = datetime.date(2010, 1, 1)
MAX_DATE = datetime.date(2050, 12, 31)


def validate_command_type(command_type: str) -> CommandType:
    """
    Validates that a command type is one of the supported CLI commands.
    
    Args:
        command_type: Command type to validate
        
    Returns:
        CommandType: Validated command type
        
    Raises:
        ValidationError: If the command type is not supported
    """
    valid_commands = ["fetch-data", "train", "predict", "backtest", "evaluate", "visualize"]
    
    if command_type in valid_commands:
        return cast(CommandType, command_type)
    
    raise ValidationError(
        f"Invalid command type: {command_type}. Valid commands are: {', '.join(valid_commands)}",
        "command_type",
        command_type
    )


def validate_log_level(log_level: str) -> LogLevel:
    """
    Validates that a log level is one of the supported logging levels.
    
    Args:
        log_level: Log level to validate
        
    Returns:
        LogLevel: Validated log level
        
    Raises:
        ValidationError: If the log level is not supported
    """
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    if log_level.upper() in valid_log_levels:
        return cast(LogLevel, log_level.upper())
    
    raise ValidationError(
        f"Invalid log level: {log_level}. Valid log levels are: {', '.join(valid_log_levels)}",
        "log_level",
        log_level
    )


def validate_data_type(data_type: str) -> DataType:
    """
    Validates that a data type is one of the supported data types.
    
    Args:
        data_type: Data type to validate
        
    Returns:
        DataType: Validated data type
        
    Raises:
        ValidationError: If the data type is not supported
    """
    valid_data_types = ["rtlmp", "weather", "grid_conditions", "all"]
    
    if data_type.lower() in valid_data_types:
        return cast(DataType, data_type.lower())
    
    raise ValidationError(
        f"Invalid data type: {data_type}. Valid data types are: {', '.join(valid_data_types)}",
        "data_type",
        data_type
    )


def validate_visualization_type(visualization_type: str) -> VisualizationType:
    """
    Validates that a visualization type is one of the supported types.
    
    Args:
        visualization_type: Visualization type to validate
        
    Returns:
        VisualizationType: Validated visualization type
        
    Raises:
        ValidationError: If the visualization type is not supported
    """
    valid_viz_types = [
        "forecast", "performance", "calibration", 
        "feature_importance", "roc_curve", "precision_recall"
    ]
    
    if visualization_type.lower() in valid_viz_types:
        return cast(VisualizationType, visualization_type.lower())
    
    raise ValidationError(
        f"Invalid visualization type: {visualization_type}. Valid types are: {', '.join(valid_viz_types)}",
        "visualization_type",
        visualization_type
    )


def validate_output_format(output_format: str) -> OutputFormat:
    """
    Validates that an output format is one of the supported formats.
    
    Args:
        output_format: Output format to validate
        
    Returns:
        OutputFormat: Validated output format
        
    Raises:
        ValidationError: If the output format is not supported
    """
    valid_formats = ["text", "json", "csv", "html", "png"]
    
    if output_format.lower() in valid_formats:
        return cast(OutputFormat, output_format.lower())
    
    raise ValidationError(
        f"Invalid output format: {output_format}. Valid formats are: {', '.join(valid_formats)}",
        "output_format",
        output_format
    )


def validate_node_id(node_id: str) -> NodeID:
    """
    Validates that a node ID follows the expected format.
    
    Args:
        node_id: Node ID to validate
        
    Returns:
        NodeID: Validated node ID
        
    Raises:
        ValidationError: If the node ID format is invalid
    """
    if not node_id:
        raise ValidationError(
            "Node ID cannot be empty",
            "node_id",
            node_id
        )
    
    if re.match(VALID_NODE_ID_PATTERN, node_id):
        return cast(NodeID, node_id)
    
    raise ValidationError(
        f"Invalid node ID format: {node_id}. Node IDs must match pattern {VALID_NODE_ID_PATTERN}",
        "node_id",
        node_id
    )


def validate_node_ids(node_ids: List[str]) -> List[NodeID]:
    """
    Validates a list of node IDs.
    
    Args:
        node_ids: List of node IDs to validate
        
    Returns:
        List[NodeID]: List of validated node IDs
        
    Raises:
        ValidationError: If any node ID is invalid or the list is empty
    """
    if not node_ids:
        raise ValidationError(
            "Node IDs list cannot be empty",
            "node_ids",
            node_ids
        )
    
    validated_ids = []
    invalid_ids = []
    
    for node_id in node_ids:
        try:
            validated_ids.append(validate_node_id(node_id))
        except ValidationError:
            invalid_ids.append(node_id)
    
    if invalid_ids:
        raise ValidationError(
            f"Invalid node IDs: {', '.join(invalid_ids)}. Node IDs must match pattern {VALID_NODE_ID_PATTERN}",
            "node_ids",
            node_ids
        )
    
    return validated_ids


def validate_threshold_value(threshold_value: Union[float, str]) -> ThresholdValue:
    """
    Validates that a threshold value is within the acceptable range.
    
    Args:
        threshold_value: Price threshold value to validate
        
    Returns:
        ThresholdValue: Validated threshold value
        
    Raises:
        ValidationError: If the threshold value is outside the acceptable range
    """
    try:
        value = float(threshold_value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Invalid threshold value: {threshold_value}. Must be a numeric value.",
            "threshold_value",
            threshold_value
        )
    
    if value < MIN_THRESHOLD_VALUE or value > MAX_THRESHOLD_VALUE:
        raise ValidationError(
            f"Threshold value out of range: {value}. Must be between {format_price(MIN_THRESHOLD_VALUE)} and {format_price(MAX_THRESHOLD_VALUE)}.",
            "threshold_value",
            threshold_value
        )
    
    return cast(ThresholdValue, value)


def validate_threshold_values(threshold_values: List[Union[float, str]]) -> List[ThresholdValue]:
    """
    Validates a list of threshold values.
    
    Args:
        threshold_values: List of threshold values to validate
        
    Returns:
        List[ThresholdValue]: List of validated threshold values
        
    Raises:
        ValidationError: If any threshold value is invalid or the list is empty
    """
    if not threshold_values:
        raise ValidationError(
            "Threshold values list cannot be empty",
            "threshold_values",
            threshold_values
        )
    
    validated_thresholds = []
    invalid_thresholds = []
    
    for threshold in threshold_values:
        try:
            validated_thresholds.append(validate_threshold_value(threshold))
        except ValidationError:
            invalid_thresholds.append(str(threshold))
    
    if invalid_thresholds:
        raise ValidationError(
            f"Invalid threshold values: {', '.join(invalid_thresholds)}. " +
            f"Values must be numeric and between {format_price(MIN_THRESHOLD_VALUE)} and {format_price(MAX_THRESHOLD_VALUE)}.",
            "threshold_values",
            threshold_values
        )
    
    return validated_thresholds


def validate_date(date: Union[datetime.date, str]) -> datetime.date:
    """
    Validates that a date is within the acceptable range.
    
    Args:
        date: Date to validate
        
    Returns:
        datetime.date: Validated date
        
    Raises:
        ValidationError: If the date is outside the acceptable range or invalid
    """
    if isinstance(date, str):
        try:
            date_obj = datetime.datetime.fromisoformat(date).date()
        except ValueError:
            raise ValidationError(
                f"Invalid date format: {date}. Use ISO format (YYYY-MM-DD).",
                "date",
                date
            )
    elif isinstance(date, datetime.datetime):
        date_obj = date.date()
    elif isinstance(date, datetime.date):
        date_obj = date
    else:
        raise ValidationError(
            f"Invalid date type: {type(date)}. Must be string or date object.",
            "date",
            date
        )
    
    if date_obj < MIN_DATE or date_obj > MAX_DATE:
        raise ValidationError(
            f"Date out of range: {format_date(date_obj)}. " +
            f"Must be between {format_date(MIN_DATE)} and {format_date(MAX_DATE)}.",
            "date",
            date
        )
    
    return date_obj


def validate_date_range(start_date: datetime.date, end_date: datetime.date) -> tuple[datetime.date, datetime.date]:
    """
    Validates that a date range is valid (start <= end).
    
    Args:
        start_date: Range start date
        end_date: Range end date
        
    Returns:
        tuple[datetime.date, datetime.date]: Validated start and end dates
        
    Raises:
        ValidationError: If the date range is invalid
    """
    # Validate individual dates
    validated_start = validate_date(start_date)
    validated_end = validate_date(end_date)
    
    # Ensure start_date <= end_date
    if validated_start > validated_end:
        raise ValidationError(
            f"Invalid date range: start date ({format_date(validated_start)}) " +
            f"is after end date ({format_date(validated_end)})",
            "date_range",
            (start_date, end_date)
        )
    
    return (validated_start, validated_end)


def validate_file_path(file_path: Path, must_exist: bool = True) -> Path:
    """
    Validates that a file path exists and is accessible.
    
    Args:
        file_path: File path to validate
        must_exist: Whether the file must already exist
        
    Returns:
        Path: Validated file path
        
    Raises:
        ValidationError: If the file path is invalid or inaccessible
    """
    if not isinstance(file_path, Path):
        raise ValidationError(
            f"Invalid file path type: {type(file_path)}. Must be a Path object.",
            "file_path",
            file_path
        )
    
    if must_exist and not file_path.exists():
        raise ValidationError(
            f"File does not exist: {file_path}",
            "file_path",
            file_path
        )
    
    return file_path


def validate_directory_path(directory_path: Path, create_if_missing: bool = False) -> Path:
    """
    Validates that a directory path exists and is accessible.
    
    Args:
        directory_path: Directory path to validate
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        Path: Validated directory path
        
    Raises:
        ValidationError: If the directory path is invalid or inaccessible
    """
    if not isinstance(directory_path, Path):
        raise ValidationError(
            f"Invalid directory path type: {type(directory_path)}. Must be a Path object.",
            "directory_path",
            directory_path
        )
    
    if not directory_path.exists():
        if create_if_missing:
            try:
                directory_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory_path}")
            except Exception as e:
                raise ValidationError(
                    f"Failed to create directory: {directory_path}. Error: {str(e)}",
                    "directory_path",
                    directory_path
                )
        else:
            raise ValidationError(
                f"Directory does not exist: {directory_path}",
                "directory_path",
                directory_path
            )
    elif not directory_path.is_dir():
        raise ValidationError(
            f"Path exists but is not a directory: {directory_path}",
            "directory_path",
            directory_path
        )
    
    return directory_path


def validate_model_type(model_type: str) -> ModelType:
    """
    Validates that a model type is one of the supported types.
    
    Args:
        model_type: Model type to validate
        
    Returns:
        ModelType: Validated model type
        
    Raises:
        ValidationError: If the model type is not supported
    """
    valid_model_types = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "ensemble"]
    
    if model_type.lower() in valid_model_types:
        return cast(ModelType, model_type.lower())
    
    raise ValidationError(
        f"Invalid model type: {model_type}. Valid model types are: {', '.join(valid_model_types)}",
        "model_type",
        model_type
    )


def validate_model_version(model_version: str) -> str:
    """
    Validates that a model version has the correct format.
    
    Args:
        model_version: Model version to validate
        
    Returns:
        str: Validated model version
        
    Raises:
        ValidationError: If the model version format is invalid
    """
    # Expected format: v<major>.<minor>.<patch> (e.g., v1.2.3)
    version_pattern = r'^v\d+\.\d+\.\d+$'
    
    if not model_version:
        raise ValidationError(
            "Model version cannot be empty",
            "model_version",
            model_version
        )
    
    if re.match(version_pattern, model_version):
        return model_version
    
    raise ValidationError(
        f"Invalid model version format: {model_version}. Expected format: v<major>.<minor>.<patch> (e.g., v1.2.3)",
        "model_version",
        model_version
    )


def validate_hyperparameters(hyperparameters: Dict[str, Any], model_type: ModelType) -> Dict[str, Any]:
    """
    Validates that hyperparameters are in the correct format for the specified model type.
    
    Args:
        hyperparameters: Dictionary of hyperparameters to validate
        model_type: Type of model for validation context
        
    Returns:
        Dict[str, Any]: Validated hyperparameters
        
    Raises:
        ValidationError: If the hyperparameters are invalid for the model type
    """
    if not isinstance(hyperparameters, dict):
        raise ValidationError(
            f"Hyperparameters must be a dictionary, got {type(hyperparameters)}",
            "hyperparameters",
            hyperparameters
        )
    
    # Model-specific validation
    model_type_str = str(model_type)
    
    # Validate common hyperparameters
    validated_params = {}
    
    # XGBoost specific validation
    if model_type_str == "xgboost":
        # Required parameters for XGBoost
        required_params = []
        
        # Optional parameters with validation
        if "learning_rate" in hyperparameters:
            lr = float(hyperparameters["learning_rate"])
            if 0.001 <= lr <= 0.5:
                validated_params["learning_rate"] = lr
            else:
                raise ValidationError(
                    f"Invalid learning_rate: {lr}. Must be between 0.001 and 0.5.",
                    "hyperparameters.learning_rate",
                    lr
                )
        
        if "max_depth" in hyperparameters:
            max_depth = int(hyperparameters["max_depth"])
            if 1 <= max_depth <= 15:
                validated_params["max_depth"] = max_depth
            else:
                raise ValidationError(
                    f"Invalid max_depth: {max_depth}. Must be between 1 and 15.",
                    "hyperparameters.max_depth",
                    max_depth
                )
        
        # Add other parameters without specific validation
        for param, value in hyperparameters.items():
            if param not in validated_params:
                validated_params[param] = value
    
    # LightGBM specific validation
    elif model_type_str == "lightgbm":
        # Similar validation for LightGBM parameters
        if "learning_rate" in hyperparameters:
            lr = float(hyperparameters["learning_rate"])
            if 0.001 <= lr <= 0.5:
                validated_params["learning_rate"] = lr
            else:
                raise ValidationError(
                    f"Invalid learning_rate: {lr}. Must be between 0.001 and 0.5.",
                    "hyperparameters.learning_rate",
                    lr
                )
        
        if "num_leaves" in hyperparameters:
            num_leaves = int(hyperparameters["num_leaves"])
            if 2 <= num_leaves <= 256:
                validated_params["num_leaves"] = num_leaves
            else:
                raise ValidationError(
                    f"Invalid num_leaves: {num_leaves}. Must be between 2 and 256.",
                    "hyperparameters.num_leaves",
                    num_leaves
                )
        
        # Add other parameters without specific validation
        for param, value in hyperparameters.items():
            if param not in validated_params:
                validated_params[param] = value
    
    # For other model types, just pass through the hyperparameters
    else:
        validated_params = hyperparameters
    
    return validated_params


def validate_positive_integer(value: Union[int, str], parameter_name: str) -> int:
    """
    Validates that a value is a positive integer.
    
    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error messages
        
    Returns:
        int: Validated positive integer
        
    Raises:
        ValidationError: If the value is not a positive integer
    """
    try:
        int_value = int(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Invalid value for {parameter_name}: {value}. Must be an integer.",
            parameter_name,
            value
        )
    
    if int_value <= 0:
        raise ValidationError(
            f"Invalid value for {parameter_name}: {int_value}. Must be a positive integer.",
            parameter_name,
            value
        )
    
    return int_value


def validate_boolean(value: Union[bool, str], parameter_name: str) -> bool:
    """
    Validates that a value is a boolean or can be converted to one.
    
    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error messages
        
    Returns:
        bool: Validated boolean value
        
    Raises:
        ValidationError: If the value cannot be converted to a boolean
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower in ("true", "yes", "y", "1", "on"):
            return True
        elif value_lower in ("false", "no", "n", "0", "off"):
            return False
    
    raise ValidationError(
        f"Invalid boolean value for {parameter_name}: {value}. " +
        "Must be True/False, Yes/No, Y/N, 1/0, or On/Off.",
        parameter_name,
        value
    )


def validate_cli_config(config: Dict[str, Any]) -> CLIConfigDict:
    """
    Validates the CLI configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        CLIConfigDict: Validated CLI configuration
        
    Raises:
        ValidationError: If the configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError(
            f"Configuration must be a dictionary, got {type(config)}",
            "config",
            config
        )
    
    validated_config: Dict[str, Any] = {}
    
    # Validate config_file if provided
    if "config_file" in config and config["config_file"]:
        validated_config["config_file"] = validate_file_path(
            config["config_file"], 
            must_exist=True
        )
    else:
        validated_config["config_file"] = None
    
    # Validate log_level if provided
    if "log_level" in config and config["log_level"]:
        validated_config["log_level"] = validate_log_level(config["log_level"])
    else:
        validated_config["log_level"] = None
    
    # Validate log_file if provided
    if "log_file" in config and config["log_file"]:
        validated_config["log_file"] = validate_file_path(
            config["log_file"], 
            must_exist=False
        )
    else:
        validated_config["log_file"] = None
    
    # Validate output_dir if provided
    if "output_dir" in config and config["output_dir"]:
        validated_config["output_dir"] = validate_directory_path(
            config["output_dir"], 
            create_if_missing=True
        )
    else:
        validated_config["output_dir"] = None
    
    # Validate verbose flag
    if "verbose" in config:
        validated_config["verbose"] = validate_boolean(config["verbose"], "verbose")
    else:
        validated_config["verbose"] = False
    
    return cast(CLIConfigDict, validated_config)


def validate_fetch_data_params(params: Dict[str, Any]) -> FetchDataParamsDict:
    """
    Validates parameters for the fetch-data command.
    
    Args:
        params: Parameters to validate
        
    Returns:
        FetchDataParamsDict: Validated fetch-data parameters
        
    Raises:
        ValidationError: If the parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError(
            f"Parameters must be a dictionary, got {type(params)}",
            "params",
            params
        )
    
    validated_params: Dict[str, Any] = {}
    
    # Required parameters
    if "data_type" not in params:
        raise ValidationError(
            "Missing required parameter: data_type",
            "data_type",
            None
        )
    validated_params["data_type"] = validate_data_type(params["data_type"])
    
    if "start_date" not in params:
        raise ValidationError(
            "Missing required parameter: start_date",
            "start_date",
            None
        )
    
    if "end_date" not in params:
        raise ValidationError(
            "Missing required parameter: end_date",
            "end_date",
            None
        )
    
    # Validate date range
    start_date = validate_date(params["start_date"])
    end_date = validate_date(params["end_date"])
    validated_params["start_date"], validated_params["end_date"] = validate_date_range(start_date, end_date)
    
    # Validate nodes
    if "nodes" not in params or not params["nodes"]:
        raise ValidationError(
            "Missing required parameter: nodes",
            "nodes",
            None
        )
    validated_params["nodes"] = validate_node_ids(params["nodes"])
    
    # Optional parameters
    if "output_path" in params and params["output_path"]:
        validated_params["output_path"] = validate_file_path(
            params["output_path"], 
            must_exist=False
        )
    else:
        validated_params["output_path"] = None
    
    if "output_format" in params and params["output_format"]:
        validated_params["output_format"] = validate_output_format(params["output_format"])
    else:
        validated_params["output_format"] = None
    
    if "force_refresh" in params:
        validated_params["force_refresh"] = validate_boolean(params["force_refresh"], "force_refresh")
    else:
        validated_params["force_refresh"] = False
    
    return cast(FetchDataParamsDict, validated_params)


def validate_train_params(params: Dict[str, Any]) -> TrainParamsDict:
    """
    Validates parameters for the train command.
    
    Args:
        params: Parameters to validate
        
    Returns:
        TrainParamsDict: Validated train parameters
        
    Raises:
        ValidationError: If the parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError(
            f"Parameters must be a dictionary, got {type(params)}",
            "params",
            params
        )
    
    validated_params: Dict[str, Any] = {}
    
    # Required parameters
    # Validate date range
    if "start_date" not in params:
        raise ValidationError(
            "Missing required parameter: start_date",
            "start_date",
            None
        )
    
    if "end_date" not in params:
        raise ValidationError(
            "Missing required parameter: end_date",
            "end_date",
            None
        )
    
    start_date = validate_date(params["start_date"])
    end_date = validate_date(params["end_date"])
    validated_params["start_date"], validated_params["end_date"] = validate_date_range(start_date, end_date)
    
    # Validate nodes
    if "nodes" not in params or not params["nodes"]:
        raise ValidationError(
            "Missing required parameter: nodes",
            "nodes",
            None
        )
    validated_params["nodes"] = validate_node_ids(params["nodes"])
    
    # Validate thresholds
    if "thresholds" not in params or not params["thresholds"]:
        raise ValidationError(
            "Missing required parameter: thresholds",
            "thresholds",
            None
        )
    validated_params["thresholds"] = validate_threshold_values(params["thresholds"])
    
    # Validate model_type
    if "model_type" not in params:
        raise ValidationError(
            "Missing required parameter: model_type",
            "model_type",
            None
        )
    validated_params["model_type"] = validate_model_type(params["model_type"])
    
    # Optional parameters
    if "hyperparameters" in params and params["hyperparameters"]:
        validated_params["hyperparameters"] = validate_hyperparameters(
            params["hyperparameters"], 
            validated_params["model_type"]
        )
    else:
        validated_params["hyperparameters"] = None
    
    if "optimize_hyperparameters" in params:
        validated_params["optimize_hyperparameters"] = validate_boolean(
            params["optimize_hyperparameters"], 
            "optimize_hyperparameters"
        )
    else:
        validated_params["optimize_hyperparameters"] = False
    
    if "cross_validation_folds" in params:
        validated_params["cross_validation_folds"] = validate_positive_integer(
            params["cross_validation_folds"], 
            "cross_validation_folds"
        )
    else:
        validated_params["cross_validation_folds"] = 5  # Default value
    
    if "model_name" in params and params["model_name"]:
        validated_params["model_name"] = str(params["model_name"])
    else:
        validated_params["model_name"] = None
    
    if "output_path" in params and params["output_path"]:
        validated_params["output_path"] = validate_file_path(
            params["output_path"], 
            must_exist=False
        )
    else:
        validated_params["output_path"] = None
    
    return cast(TrainParamsDict, validated_params)


def validate_predict_params(params: Dict[str, Any]) -> PredictParamsDict:
    """
    Validates parameters for the predict command.
    
    Args:
        params: Parameters to validate
        
    Returns:
        PredictParamsDict: Validated predict parameters
        
    Raises:
        ValidationError: If the parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError(
            f"Parameters must be a dictionary, got {type(params)}",
            "params",
            params
        )
    
    validated_params: Dict[str, Any] = {}
    
    # Required parameters
    # Validate threshold
    if "threshold" not in params:
        raise ValidationError(
            "Missing required parameter: threshold",
            "threshold",
            None
        )
    validated_params["threshold"] = validate_threshold_value(params["threshold"])
    
    # Validate nodes
    if "nodes" not in params or not params["nodes"]:
        raise ValidationError(
            "Missing required parameter: nodes",
            "nodes",
            None
        )
    validated_params["nodes"] = validate_node_ids(params["nodes"])
    
    # Optional parameters
    if "model_version" in params and params["model_version"]:
        validated_params["model_version"] = validate_model_version(params["model_version"])
    else:
        validated_params["model_version"] = None
    
    if "output_path" in params and params["output_path"]:
        validated_params["output_path"] = validate_file_path(
            params["output_path"], 
            must_exist=False
        )
    else:
        validated_params["output_path"] = None
    
    if "output_format" in params and params["output_format"]:
        validated_params["output_format"] = validate_output_format(params["output_format"])
    else:
        validated_params["output_format"] = None
    
    if "visualize" in params:
        validated_params["visualize"] = validate_boolean(params["visualize"], "visualize")
    else:
        validated_params["visualize"] = False
    
    return cast(PredictParamsDict, validated_params)


def validate_backtest_params(params: Dict[str, Any]) -> BacktestParamsDict:
    """
    Validates parameters for the backtest command.
    
    Args:
        params: Parameters to validate
        
    Returns:
        BacktestParamsDict: Validated backtest parameters
        
    Raises:
        ValidationError: If the parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError(
            f"Parameters must be a dictionary, got {type(params)}",
            "params",
            params
        )
    
    validated_params: Dict[str, Any] = {}
    
    # Required parameters
    # Validate date range
    if "start_date" not in params:
        raise ValidationError(
            "Missing required parameter: start_date",
            "start_date",
            None
        )
    
    if "end_date" not in params:
        raise ValidationError(
            "Missing required parameter: end_date",
            "end_date",
            None
        )
    
    start_date = validate_date(params["start_date"])
    end_date = validate_date(params["end_date"])
    validated_params["start_date"], validated_params["end_date"] = validate_date_range(start_date, end_date)
    
    # Validate thresholds
    if "thresholds" not in params or not params["thresholds"]:
        raise ValidationError(
            "Missing required parameter: thresholds",
            "thresholds",
            None
        )
    validated_params["thresholds"] = validate_threshold_values(params["thresholds"])
    
    # Validate nodes
    if "nodes" not in params or not params["nodes"]:
        raise ValidationError(
            "Missing required parameter: nodes",
            "nodes",
            None
        )
    validated_params["nodes"] = validate_node_ids(params["nodes"])
    
    # Optional parameters
    if "model_version" in params and params["model_version"]:
        validated_params["model_version"] = validate_model_version(params["model_version"])
    else:
        validated_params["model_version"] = None
    
    if "output_path" in params and params["output_path"]:
        validated_params["output_path"] = validate_file_path(
            params["output_path"], 
            must_exist=False
        )
    else:
        validated_params["output_path"] = None
    
    if "output_format" in params and params["output_format"]:
        validated_params["output_format"] = validate_output_format(params["output_format"])
    else:
        validated_params["output_format"] = None
    
    if "visualize" in params:
        validated_params["visualize"] = validate_boolean(params["visualize"], "visualize")
    else:
        validated_params["visualize"] = False
    
    return cast(BacktestParamsDict, validated_params)


def validate_evaluate_params(params: Dict[str, Any]) -> EvaluateParamsDict:
    """
    Validates parameters for the evaluate command.
    
    Args:
        params: Parameters to validate
        
    Returns:
        EvaluateParamsDict: Validated evaluate parameters
        
    Raises:
        ValidationError: If the parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError(
            f"Parameters must be a dictionary, got {type(params)}",
            "params",
            params
        )
    
    validated_params: Dict[str, Any] = {}
    
    # Optional model_version
    if "model_version" in params and params["model_version"]:
        validated_params["model_version"] = validate_model_version(params["model_version"])
    else:
        validated_params["model_version"] = None
    
    # Optional compare_with (list of model versions)
    if "compare_with" in params and params["compare_with"]:
        if not isinstance(params["compare_with"], list):
            raise ValidationError(
                "compare_with must be a list of model versions",
                "compare_with",
                params["compare_with"]
            )
        
        validated_versions = []
        for version in params["compare_with"]:
            validated_versions.append(validate_model_version(version))
        
        validated_params["compare_with"] = validated_versions
    else:
        validated_params["compare_with"] = None
    
    # Required thresholds
    if "thresholds" not in params or not params["thresholds"]:
        raise ValidationError(
            "Missing required parameter: thresholds",
            "thresholds",
            None
        )
    validated_params["thresholds"] = validate_threshold_values(params["thresholds"])
    
    # Required nodes
    if "nodes" not in params or not params["nodes"]:
        raise ValidationError(
            "Missing required parameter: nodes",
            "nodes",
            None
        )
    validated_params["nodes"] = validate_node_ids(params["nodes"])
    
    # Optional date range
    if "start_date" in params and params["start_date"]:
        start_date = validate_date(params["start_date"])
        
        if "end_date" in params and params["end_date"]:
            end_date = validate_date(params["end_date"])
            validated_params["start_date"], validated_params["end_date"] = validate_date_range(start_date, end_date)
        else:
            validated_params["start_date"] = start_date
            validated_params["end_date"] = None
    else:
        validated_params["start_date"] = None
        validated_params["end_date"] = None
    
    # Optional output parameters
    if "output_path" in params and params["output_path"]:
        validated_params["output_path"] = validate_file_path(
            params["output_path"], 
            must_exist=False
        )
    else:
        validated_params["output_path"] = None
    
    if "output_format" in params and params["output_format"]:
        validated_params["output_format"] = validate_output_format(params["output_format"])
    else:
        validated_params["output_format"] = None
    
    if "visualize" in params:
        validated_params["visualize"] = validate_boolean(params["visualize"], "visualize")
    else:
        validated_params["visualize"] = False
    
    return cast(EvaluateParamsDict, validated_params)


def validate_visualize_params(params: Dict[str, Any]) -> VisualizeParamsDict:
    """
    Validates parameters for the visualize command.
    
    Args:
        params: Parameters to validate
        
    Returns:
        VisualizeParamsDict: Validated visualize parameters
        
    Raises:
        ValidationError: If the parameters are invalid
    """
    if not isinstance(params, dict):
        raise ValidationError(
            f"Parameters must be a dictionary, got {type(params)}",
            "params",
            params
        )
    
    validated_params: Dict[str, Any] = {}
    
    # Required visualization_type
    if "visualization_type" not in params:
        raise ValidationError(
            "Missing required parameter: visualization_type",
            "visualization_type",
            None
        )
    validated_params["visualization_type"] = validate_visualization_type(params["visualization_type"])
    
    # Optional parameters
    if "forecast_id" in params and params["forecast_id"]:
        validated_params["forecast_id"] = str(params["forecast_id"])
    else:
        validated_params["forecast_id"] = None
    
    if "model_version" in params and params["model_version"]:
        validated_params["model_version"] = validate_model_version(params["model_version"])
    else:
        validated_params["model_version"] = None
    
    if "compare_with" in params and params["compare_with"]:
        if not isinstance(params["compare_with"], list):
            raise ValidationError(
                "compare_with must be a list of model versions",
                "compare_with",
                params["compare_with"]
            )
        
        validated_versions = []
        for version in params["compare_with"]:
            validated_versions.append(validate_model_version(version))
        
        validated_params["compare_with"] = validated_versions
    else:
        validated_params["compare_with"] = None
    
    if "threshold" in params and params["threshold"] is not None:
        validated_params["threshold"] = validate_threshold_value(params["threshold"])
    else:
        validated_params["threshold"] = None
    
    if "nodes" in params and params["nodes"]:
        validated_params["nodes"] = validate_node_ids(params["nodes"])
    else:
        validated_params["nodes"] = None
    
    # Optional date range
    if "start_date" in params and params["start_date"]:
        start_date = validate_date(params["start_date"])
        
        if "end_date" in params and params["end_date"]:
            end_date = validate_date(params["end_date"])
            validated_params["start_date"], validated_params["end_date"] = validate_date_range(start_date, end_date)
        else:
            validated_params["start_date"] = start_date
            validated_params["end_date"] = None
    else:
        validated_params["start_date"] = None
        validated_params["end_date"] = None
    
    if "output_path" in params and params["output_path"]:
        validated_params["output_path"] = validate_file_path(
            params["output_path"], 
            must_exist=False
        )
    else:
        validated_params["output_path"] = None
    
    if "output_format" in params and params["output_format"]:
        validated_params["output_format"] = validate_output_format(params["output_format"])
    else:
        validated_params["output_format"] = None
    
    if "interactive" in params:
        validated_params["interactive"] = validate_boolean(params["interactive"], "interactive")
    else:
        validated_params["interactive"] = False
    
    return cast(VisualizeParamsDict, validated_params)


def validate_command_params(command_type: CommandType, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates parameters for a specific command type.
    
    Args:
        command_type: Type of command for validation context
        params: Parameters to validate
        
    Returns:
        Dict[str, Any]: Validated command parameters
        
    Raises:
        ValidationError: If the parameters are invalid for the command type
    """
    validation_functions = {
        "fetch-data": validate_fetch_data_params,
        "train": validate_train_params,
        "predict": validate_predict_params,
        "backtest": validate_backtest_params,
        "evaluate": validate_evaluate_params,
        "visualize": validate_visualize_params
    }
    
    if command_type not in validation_functions:
        raise ValidationError(
            f"Unknown command type: {command_type}",
            "command_type",
            command_type
        )
    
    try:
        return validation_functions[command_type](params)
    except ValidationError as e:
        # Add command context to the error
        raise e.with_context(f"Error validating {command_type} parameters")


class ValidationDecorator:
    """
    Decorator class for validating function parameters.
    
    This decorator applies validation functions to function parameters
    before calling the decorated function.
    """
    
    def __init__(self, validators: Dict[str, Callable], raise_errors: bool = True):
        """
        Initialize the validation decorator with parameter validators.
        
        Args:
            validators: Dictionary mapping parameter names to validation functions
            raise_errors: Whether to raise exceptions for validation errors
        """
        self._validators = validators
        self._raise_errors = raise_errors
    
    def __call__(self, func: Callable) -> Callable:
        """
        Apply the decorator to a function.
        
        Args:
            func: Function to decorate
            
        Returns:
            Callable: Wrapped function with validation
        """
        def wrapper(*args, **kwargs):
            # Extract function argument names and values
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            
            # Validate parameters
            validated_args = bound_args.arguments.copy()
            
            for param_name, validator in self._validators.items():
                if param_name in validated_args:
                    try:
                        validated_args[param_name] = validator(validated_args[param_name])
                    except ValidationError as e:
                        if self._raise_errors:
                            raise e
                        else:
                            logger.warning(f"Validation error: {e}")
            
            # Call function with validated arguments
            return func(**validated_args)
        
        return wrapper