"""
Entry point for the ERCOT RTLMP spike prediction system's API module.
This file exposes the main API classes and functions from the various API submodules,
providing a unified interface for data fetching, model management, inference,
backtesting, and visualization operations.
"""

# Internal imports
from .data_api import DataAPI, get_historical_rtlmp_data, get_historical_weather_data, get_weather_forecast, get_historical_grid_conditions, get_combined_historical_data
from .model_api import ModelAPI, train_new_model, optimize_and_train_model, get_model, get_latest_model, evaluate_model, retrain_model, schedule_model_retraining
from .inference_api import InferenceAPI, generate_forecast, get_latest_forecast, get_forecast_by_date, compare_forecasts, list_available_models, DEFAULT_FORECAST_HORIZON
from .backtesting_api import BacktestingAPI, run_backtesting_scenario, run_backtesting_scenarios, run_historical_simulation, get_backtesting_results, compare_backtesting_results, visualize_backtesting_results, generate_backtesting_report
from .visualization_api import VisualizationAPI, create_forecast_visualization, create_model_performance_visualization, create_feature_importance_visualization, create_backtesting_visualization, create_metrics_dashboard, export_visualization, export_data, run_dashboard, SUPPORTED_IMAGE_FORMATS, SUPPORTED_DATA_FORMATS
from ..utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Version string for the API module
__version__ = "0.1.0"

# Expose DataAPI class and data retrieval functions
__all__ = [
    "DataAPI",
    "get_historical_rtlmp_data",
    "get_historical_weather_data",
    "get_weather_forecast",
    "get_historical_grid_conditions",
    "get_combined_historical_data",
    "ModelAPI",
    "train_new_model",
    "optimize_and_train_model",
    "get_model",
    "get_latest_model",
    "evaluate_model",
    "retrain_model",
    "schedule_model_retraining",
    "InferenceAPI",
    "generate_forecast",
    "get_latest_forecast",
    "get_forecast_by_date",
    "compare_forecasts",
    "list_available_models",
    "DEFAULT_FORECAST_HORIZON",
    "BacktestingAPI",
    "run_backtesting_scenario",
    "run_backtesting_scenarios",
    "run_historical_simulation",
    "get_backtesting_results",
    "compare_backtesting_results",
    "visualize_backtesting_results",
    "generate_backtesting_report",
    "VisualizationAPI",
    "create_forecast_visualization",
    "create_model_performance_visualization",
    "create_feature_importance_visualization",
    "create_backtesting_visualization",
    "create_metrics_dashboard",
    "export_visualization",
    "export_data",
    "run_dashboard",
    "SUPPORTED_IMAGE_FORMATS",
    "SUPPORTED_DATA_FORMATS",
    "__version__"
]