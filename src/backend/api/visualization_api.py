"""
Provides a unified API for creating, managing, and exporting visualizations in the ERCOT RTLMP spike prediction system.
This module serves as an interface layer between client code and the visualization components,
offering simplified access to forecast visualizations, model performance metrics, feature importance plots,
and interactive dashboards.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import datetime
from pathlib import Path

import pandas  # version 2.0+
import numpy  # version 1.24+
import plotly  # version 5.14+
import plotly.graph_objects as go
import matplotlib  # version 3.7+
from matplotlib.figure import Figure

from ..visualization.forecast_plots import ForecastPlotter, plot_probability_timeline, create_interactive_timeline
from ..visualization.performance_plots import ModelPerformancePlotter, plot_roc_curve, create_interactive_roc_curve
from ..visualization.feature_importance import FeatureImportancePlotter, plot_feature_importance, create_interactive_feature_importance
from ..visualization.metrics_dashboard import MetricsDashboard, create_metrics_dashboard
from ..visualization.export import ExportManager, figure_to_base64, plotly_figure_to_base64, dataframe_to_base64, SUPPORTED_IMAGE_FORMATS, SUPPORTED_DATA_FORMATS
from ..data.storage.forecast_repository import ForecastRepository
from ..data.storage.model_registry import ModelRegistry
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, PathType, ThresholdValue, NodeID
from ..utils.logging import get_logger
from ..utils.error_handling import handle_errors, VisualizationError

logger = get_logger(__name__)

DEFAULT_REPOSITORY_PATH = Path('./data')
DEFAULT_MODEL_REGISTRY_PATH = Path('./models')
DEFAULT_FORECAST_REPOSITORY_PATH = Path('./forecasts')
DEFAULT_OUTPUT_PATH = Path('./output')

@handle_errors(logger, VisualizationError)
def create_forecast_visualization(
    forecast: Union[DataFrameType, datetime.datetime, str, None],
    visualization_type: str,
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None,
    visualization_config: Optional[Dict[str, Any]] = None,
    repository_path: Optional[PathType] = None
) -> Union[Tuple[Figure, Any], go.Figure]:
    """
    Creates a visualization of forecast data
    """
    forecast_plotter = ForecastPlotter(repository_path=repository_path)
    forecast_plotter.load_forecast(forecast=forecast, thresholds=thresholds, nodes=nodes)

    if visualization_type == 'probability_timeline':
        if visualization_config and visualization_config.get('interactive'):
            return forecast_plotter.create_interactive_timeline(threshold=thresholds[0] if thresholds else None, node_id=nodes[0] if nodes else None)
        else:
            return forecast_plotter.plot_probability_timeline(threshold=thresholds[0] if thresholds else None, node_id=nodes[0] if nodes else None)
    # elif visualization_type == 'threshold_comparison':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return forecast_plotter.create_interactive_threshold_comparison(thresholds=thresholds, node_id=nodes[0] if nodes else None)
    #     else:
    #         return forecast_plotter.plot_threshold_comparison(thresholds=thresholds, node_id=nodes[0] if nodes else None)
    # elif visualization_type == 'node_comparison':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return forecast_plotter.create_interactive_node_comparison(threshold=thresholds[0] if thresholds else None, nodes=nodes)
    #     else:
    #         return forecast_plotter.plot_node_comparison(threshold=thresholds[0] if thresholds else None, nodes=nodes)
    # elif visualization_type == 'heatmap':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return forecast_plotter.create_interactive_heatmap(heatmap_type='threshold', node_id=nodes[0] if nodes else None, threshold=thresholds[0] if thresholds else None)
    #     else:
    #         return forecast_plotter.plot_heatmap(heatmap_type='threshold', node_id=nodes[0] if nodes else None, threshold=thresholds[0] if thresholds else None)
    elif visualization_type == 'dashboard':
        return forecast_plotter.create_forecast_dashboard(thresholds=thresholds, nodes=nodes)
    else:
        raise VisualizationError(f"Unsupported visualization type: {visualization_type}")

@handle_errors(logger, VisualizationError)
def create_model_performance_visualization(
    model_id: str,
    visualization_type: str,
    version: Optional[str] = None,
    y_true: Optional[SeriesType] = None,
    y_prob: Optional[ArrayType] = None,
    y_pred: Optional[ArrayType] = None,
    visualization_config: Optional[Dict[str, Any]] = None,
    registry_path: Optional[PathType] = None
) -> Union[Tuple[Figure, Any], go.Figure]:
    """
    Creates a visualization of model performance metrics
    """
    performance_plotter = ModelPerformancePlotter(registry_path=registry_path)
    performance_plotter.load_model_data(model_id=model_id, version=version, y_true=y_true, y_prob=y_prob, y_pred=y_pred)

    if visualization_type == 'roc_curve':
        if visualization_config and visualization_config.get('interactive'):
            return performance_plotter.create_interactive_roc_curve()
        else:
            return performance_plotter.plot_roc_curve()
    # elif visualization_type == 'precision_recall_curve':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return performance_plotter.create_interactive_precision_recall_curve()
    #     else:
    #         return performance_plotter.plot_precision_recall_curve()
    # elif visualization_type == 'calibration_curve':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return performance_plotter.create_interactive_calibration_curve()
    #     else:
    #         return performance_plotter.plot_calibration_curve()
    # elif visualization_type == 'confusion_matrix':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return performance_plotter.create_interactive_confusion_matrix()
    #     else:
    #         return performance_plotter.plot_confusion_matrix()
    # elif visualization_type == 'threshold_sensitivity':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return performance_plotter.create_interactive_threshold_sensitivity()
    #     else:
    #         return performance_plotter.plot_threshold_sensitivity()
    elif visualization_type == 'dashboard':
        return performance_plotter.create_performance_dashboard()
    else:
        raise VisualizationError(f"Unsupported visualization type: {visualization_type}")

@handle_errors(logger, VisualizationError)
def create_feature_importance_visualization(
    model_id: str,
    visualization_type: str,
    version: Optional[str] = None,
    n_features: Optional[int] = None,
    visualization_config: Optional[Dict[str, Any]] = None,
    registry_path: Optional[PathType] = None
) -> Union[Tuple[Figure, Any], go.Figure]:
    """
    Creates a visualization of feature importance for a model
    """
    feature_plotter = FeatureImportancePlotter(registry_path=registry_path)
    feature_plotter.load_model_data(model_id=model_id, version=version)

    if visualization_type == 'feature_importance':
        if visualization_config and visualization_config.get('interactive'):
            return feature_plotter.create_interactive_feature_importance(n_features=n_features)
        else:
            return feature_plotter.plot_feature_importance(n_features=n_features)
    # elif visualization_type == 'feature_importance_history':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return feature_plotter.create_interactive_feature_importance_history()
    #     else:
    #         return feature_plotter.plot_feature_importance_history()
    # elif visualization_type == 'feature_group_importance':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return feature_plotter.create_interactive_feature_group_importance()
    #     else:
    #         return feature_plotter.plot_feature_group_importance()
    # elif visualization_type == 'feature_correlation':
    #     if visualization_config and visualization_config.get('interactive'):
    #         return feature_plotter.create_interactive_feature_correlation_heatmap()
    #     else:
    #         return feature_plotter.plot_feature_correlation_heatmap()
    elif visualization_type == 'dashboard':
        return feature_plotter.create_feature_importance_dashboard()
    else:
        raise VisualizationError(f"Unsupported visualization type: {visualization_type}")

@handle_errors(logger, VisualizationError)
def create_backtesting_visualization(
    backtest_results: Union[DataFrameType, Dict[str, Any]],
    visualization_type: str,
    visualization_config: Optional[Dict[str, Any]] = None
) -> Union[Tuple[Figure, Any], go.Figure]:
    """
    Creates a visualization of backtesting results
    """
    # Placeholder for backtesting visualization logic
    # Replace with actual implementation based on backtest_results and visualization_type
    return go.Figure()

@handle_errors(logger, VisualizationError)
def create_metrics_dashboard(
    forecast: Optional[Union[DataFrameType, datetime.datetime, str, None]] = None,
    model_id: Optional[str] = None,
    version: Optional[str] = None,
    dashboard_config: Optional[Dict[str, Any]] = None,
    forecast_repository_path: Optional[PathType] = None,
    model_registry_path: Optional[PathType] = None
) -> go.Figure:
    """
    Creates a comprehensive metrics dashboard combining multiple visualizations
    """
    dashboard = MetricsDashboard(forecast_repository_path=forecast_repository_path, model_registry_path=model_registry_path)
    return dashboard.create_dashboard(forecast=forecast, model_id=model_id, version=version)

@handle_errors(logger, VisualizationError)
def export_visualization(
    fig: Union[Figure, go.Figure],
    output_path: PathType,
    format: Optional[str] = None,
    export_config: Optional[Dict[str, Any]] = None
) -> PathType:
    """
    Exports a visualization to a file
    """
    export_manager = ExportManager(export_config)
    return export_manager.export_figure(fig, output_path, format)

@handle_errors(logger, VisualizationError)
def export_data(
    data: Union[DataFrameType, Dict[str, Any], List[Dict[str, Any]]],
    output_path: PathType,
    data_type: str,
    format: Optional[str] = None,
    export_config: Optional[Dict[str, Any]] = None
) -> PathType:
    """
    Exports data to a file
    """
    export_manager = ExportManager(export_config)
    return export_manager.export_dataframe(data, output_path, format)

@handle_errors(logger, VisualizationError)
def run_dashboard(
    dashboard_config: Optional[Dict[str, Any]] = None,
    forecast_repository_path: Optional[PathType] = None,
    model_registry_path: Optional[PathType] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    debug: Optional[bool] = None
) -> None:
    """
    Runs an interactive web dashboard
    """
    dashboard = MetricsDashboard(forecast_repository_path=forecast_repository_path, model_registry_path=model_registry_path, dashboard_config=dashboard_config)
    dashboard.run_dashboard(host=host, port=port, debug=debug)

class VisualizationAPI:
    """
    Main class providing a unified interface for visualization operations
    """
    def __init__(
        self,
        forecast_repository_path: Optional[PathType] = None,
        model_registry_path: Optional[PathType] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the VisualizationAPI with repository paths and configuration
        """
        self._forecast_repository = ForecastRepository(forecast_repository_path or DEFAULT_FORECAST_REPOSITORY_PATH)
        self._model_registry = ModelRegistry(model_registry_path or DEFAULT_MODEL_REGISTRY_PATH)
        self._forecast_plotter = ForecastPlotter(forecast_repository_path or DEFAULT_FORECAST_REPOSITORY_PATH)
        self._performance_plotter = ModelPerformancePlotter(model_registry_path or DEFAULT_MODEL_REGISTRY_PATH)
        self._feature_plotter = FeatureImportancePlotter(model_registry_path or DEFAULT_MODEL_REGISTRY_PATH)
        self._dashboard = MetricsDashboard(forecast_repository_path or DEFAULT_FORECAST_REPOSITORY_PATH, model_registry_path or DEFAULT_MODEL_REGISTRY_PATH)
        self._export_manager = ExportManager()
        self._config = {}
        if config:
            self._config.update(config)

    def create_forecast_visualization(
        self,
        forecast: Union[DataFrameType, datetime.datetime, str, None],
        visualization_type: str,
        thresholds: Optional[List[ThresholdValue]] = None,
        nodes: Optional[List[NodeID]] = None,
        visualization_config: Optional[Dict[str, Any]] = None
    ) -> Union[Tuple[Figure, Any], go.Figure]:
        """
        Creates a visualization of forecast data
        """
        return create_forecast_visualization(forecast, visualization_type, thresholds, nodes, visualization_config, self._forecast_repository._forecast_root)

    def create_model_performance_visualization(
        self,
        model_id: str,
        visualization_type: str,
        version: Optional[str] = None,
        y_true: Optional[SeriesType] = None,
        y_prob: Optional[ArrayType] = None,
        y_pred: Optional[ArrayType] = None,
        visualization_config: Optional[Dict[str, Any]] = None
    ) -> Union[Tuple[Figure, Any], go.Figure]:
        """
        Creates a visualization of model performance metrics
        """
        return create_model_performance_visualization(model_id, visualization_type, version, y_true, y_prob, y_pred, visualization_config, self._model_registry.registry_path)

    def create_feature_importance_visualization(
        self,
        model_id: str,
        visualization_type: str,
        version: Optional[str] = None,
        n_features: Optional[int] = None,
        visualization_config: Optional[Dict[str, Any]] = None
    ) -> Union[Tuple[Figure, Any], go.Figure]:
        """
        Creates a visualization of feature importance for a model
        """
        return create_feature_importance_visualization(model_id, visualization_type, version, n_features, visualization_config, self._model_registry.registry_path)

    def create_backtesting_visualization(
        self,
        backtest_results: Union[DataFrameType, Dict[str, Any]],
        visualization_type: str,
        visualization_config: Optional[Dict[str, Any]] = None
    ) -> Union[Tuple[Figure, Any], go.Figure]:
        """
        Creates a visualization of backtesting results
        """
        return create_backtesting_visualization(backtest_results, visualization_type, visualization_config)

    def create_metrics_dashboard(
        self,
        forecast: Optional[Union[DataFrameType, datetime.datetime, str, None]] = None,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        dashboard_config: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """
        Creates a comprehensive metrics dashboard
        """
        return create_metrics_dashboard(forecast, model_id, version, dashboard_config, self._forecast_repository._forecast_root, self._model_registry.registry_path)

    def export_visualization(
        self,
        fig: Union[Figure, go.Figure],
        output_path: Optional[PathType] = None,
        format: Optional[str] = None,
        export_config: Optional[Dict[str, Any]] = None
    ) -> PathType:
        """
        Exports a visualization to a file
        """
        if output_path is None:
            output_path = DEFAULT_OUTPUT_PATH / "visualization" / f"visualization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        return self._export_manager.export_figure(fig, output_path, format)

    def export_data(
        self,
        data: Union[DataFrameType, Dict[str, Any], List[Dict[str, Any]]],
        output_path: Optional[PathType] = None,
        data_type: str = "dataframe",
        format: Optional[str] = None,
        export_config: Optional[Dict[str, Any]] = None
    ) -> PathType:
        """
        Exports data to a file
        """
        if output_path is None:
            output_path = DEFAULT_OUTPUT_PATH / "data" / f"data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return self._export_manager.export_dataframe(data, output_path, format)

    def run_dashboard(
        self,
        dashboard_config: Optional[Dict[str, Any]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        debug: Optional[bool] = None
    ) -> None:
        """
        Runs an interactive web dashboard
        """
        run_dashboard(dashboard_config, self._forecast_repository._forecast_root, self._model_registry.registry_path, host, port, debug)

    def get_supported_formats(self, format_type: str) -> List[str]:
        """
        Gets the supported export formats
        """
        if format_type == 'image':
            return SUPPORTED_IMAGE_FORMATS
        elif format_type == 'data':
            return SUPPORTED_DATA_FORMATS
        else:
            raise ValueError(f"Invalid format_type: {format_type}. Must be 'image' or 'data'")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Updates the configuration
        """
        self._config.update(new_config)
        if 'export' in new_config:
            self._export_manager.update_config(new_config['export'])