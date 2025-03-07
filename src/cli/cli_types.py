"""
Defines type definitions and type aliases for the CLI application of the ERCOT RTLMP spike prediction system.

This module provides TypedDict classes for configuration and command parameters, as well as Literal types 
for command types, log levels, and other enumerated values used throughout the CLI application.
"""

from typing import TypedDict, Dict, List, Optional, Union, Any, Literal, cast
from pathlib import Path
from datetime import datetime

# Import backend type definitions
from ..backend.utils.type_definitions import NodeID, ThresholdValue, ModelType

# Define Literal types for CLI
CommandType = Literal['fetch-data', 'train', 'predict', 'backtest', 'evaluate', 'visualize']
LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
DataType = Literal['rtlmp', 'weather', 'grid_conditions', 'all']
VisualizationType = Literal['forecast', 'performance', 'calibration', 'feature_importance', 'roc_curve', 'precision_recall']
OutputFormat = Literal['text', 'json', 'csv', 'html', 'png']

# TypedDict definitions for CLI configuration and parameters

class CLIConfigDict(TypedDict):
    """TypedDict for CLI global configuration"""
    config_file: Optional[Path]
    log_level: Optional[LogLevel]
    log_file: Optional[Path]
    output_dir: Optional[Path]
    verbose: bool

class FetchDataParamsDict(TypedDict):
    """TypedDict for fetch-data command parameters"""
    data_type: DataType
    start_date: datetime.date
    end_date: datetime.date
    nodes: List[NodeID]
    output_path: Optional[Path]
    output_format: Optional[OutputFormat]
    force_refresh: bool

class TrainParamsDict(TypedDict):
    """TypedDict for train command parameters"""
    start_date: datetime.date
    end_date: datetime.date
    nodes: List[NodeID]
    thresholds: List[ThresholdValue]
    model_type: ModelType
    hyperparameters: Optional[Dict[str, Any]]
    optimize_hyperparameters: bool
    cross_validation_folds: int
    model_name: Optional[str]
    output_path: Optional[Path]

class PredictParamsDict(TypedDict):
    """TypedDict for predict command parameters"""
    threshold: ThresholdValue
    nodes: List[NodeID]
    model_version: Optional[str]
    output_path: Optional[Path]
    output_format: Optional[OutputFormat]
    visualize: bool

class BacktestParamsDict(TypedDict):
    """TypedDict for backtest command parameters"""
    start_date: datetime.date
    end_date: datetime.date
    thresholds: List[ThresholdValue]
    nodes: List[NodeID]
    model_version: Optional[str]
    output_path: Optional[Path]
    output_format: Optional[OutputFormat]
    visualize: bool

class EvaluateParamsDict(TypedDict):
    """TypedDict for evaluate command parameters"""
    model_version: Optional[str]
    compare_with: Optional[List[str]]
    thresholds: List[ThresholdValue]
    nodes: List[NodeID]
    start_date: Optional[datetime.date]
    end_date: Optional[datetime.date]
    output_path: Optional[Path]
    output_format: Optional[OutputFormat]
    visualize: bool

class VisualizeParamsDict(TypedDict):
    """TypedDict for visualize command parameters"""
    visualization_type: VisualizationType
    forecast_id: Optional[str]
    model_version: Optional[str]
    compare_with: Optional[List[str]]
    threshold: Optional[ThresholdValue]
    nodes: Optional[List[NodeID]]
    start_date: Optional[datetime.date]
    end_date: Optional[datetime.date]
    output_path: Optional[Path]
    output_format: Optional[OutputFormat]
    interactive: bool

# Union type for all command parameter dictionaries
CommandParamsDict = Union[
    FetchDataParamsDict, 
    TrainParamsDict, 
    PredictParamsDict, 
    BacktestParamsDict, 
    EvaluateParamsDict, 
    VisualizeParamsDict
]