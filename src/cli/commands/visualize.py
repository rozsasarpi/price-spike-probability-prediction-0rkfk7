"""
Implements the 'visualize' command for the CLI application of the ERCOT RTLMP spike prediction system. This module enables users to generate various visualizations including forecast probabilities, model performance metrics, calibration curves, and feature importance charts with options for different visualization types, output formats, and interactive displays.
"""

import typing
from typing import Dict, List, Optional, Any, Union, cast

import click  # version 8.0+
import pandas  # version 2.0+
from pathlib import Path
from datetime import datetime

import matplotlib  # version 3.7+
import matplotlib.pyplot as plt

import plotly  # version 5.14+
import plotly.graph_objects as go

from ..cli_types import VisualizeParamsDict
from ..exceptions import CommandError, VisualizationError
from ..logger import get_cli_logger, CLILoggingContext
from ..utils.validators import validate_visualization_params, validate_threshold_value, validate_node_ids, validate_model_version, validate_date_range
from ..utils.progress_bars import create_indeterminate_spinner, progress_bar_context
from ..utils.output_handlers import OutputHandler, display_success_message, display_error_message
from ..ui.charts import create_probability_timeline, create_threshold_comparison, create_feature_importance_chart, create_metrics_chart, create_roc_curve, create_calibration_curve
from ...backend.api.visualization_api import VisualizationAPI


logger = get_cli_logger(__name__)


@click.command('visualize', help='Generate visualizations of forecasts and model performance')
@click.option('--type', '-t', 'visualization_type', type=click.Choice(['forecast', 'performance', 'calibration', 'feature_importance', 'roc_curve', 'precision_recall']), required=True, help='Type of visualization to generate')
@click.option('--forecast-id', '-f', type=str, help='ID of the forecast to visualize (for forecast visualizations)')
@click.option('--model-version', '-m', type=str, help='Model version to use (default: latest)')
@click.option('--compare-with', '-c', multiple=True, help='Model versions to compare with')
@click.option('--threshold', '-th', type=float, help='Price threshold value for spike prediction')
@click.option('--nodes', '-n', multiple=True, help='Node IDs to include in visualization')
@click.option('--start-date', '-s', type=click.DateTime(formats=['%Y-%m-%d']), help='Start date for historical data')
@click.option('--end-date', '-e', type=click.DateTime(formats=['%Y-%m-%d']), help='End date for historical data')
@click.option('--output-path', '-o', type=click.Path(file_okay=True, dir_okay=True, writable=True), help='Path to save output')
@click.option('--output-format', '-fmt', type=click.Choice(['text', 'json', 'csv', 'html', 'png']), help='Output format')
@click.option('--interactive', '-i', is_flag=True, help='Generate interactive visualization')
def visualize_command(visualization_type: str, forecast_id: Optional[str], model_version: Optional[str], compare_with: Optional[List[str]], threshold: Optional[float], nodes: Optional[List[str]], start_date: Optional[datetime.date], end_date: Optional[datetime.date], output_path: Optional[Path], output_format: Optional[str], interactive: bool) -> int:
    """
    Main function for the 'visualize' command that generates various visualizations
    """
    try:
        logger.info(f"Starting visualize command execution")

        # Validate command parameters
        params = validate_parameters(visualization_type, forecast_id, model_version, compare_with, threshold, nodes, start_date, end_date)

        # Create an OutputHandler for handling command results
        output_handler = OutputHandler(output_path=output_path, output_format=output_format)

        # Initialize the VisualizationAPI
        viz_api = VisualizationAPI()

        # Based on visualization_type, call the appropriate visualization function
        if visualization_type == 'forecast':
            fig = generate_forecast_visualization(viz_api, params, interactive)
        elif visualization_type == 'performance':
            fig = generate_performance_visualization(viz_api, params, interactive)
        elif visualization_type == 'calibration':
            fig = generate_calibration_visualization(viz_api, params, interactive)
        elif visualization_type == 'feature_importance':
            fig = generate_feature_importance_visualization(viz_api, params, interactive)
        elif visualization_type == 'roc_curve':
            fig = generate_roc_curve_visualization(viz_api, params, interactive)
        elif visualization_type == 'precision_recall':
            fig = generate_precision_recall_visualization(viz_api, params, interactive)
        else:
            raise CommandError(f"Unsupported visualization type: {visualization_type}", command='visualize')

        # If interactive is True and output_format is None or 'html', run interactive dashboard
        if interactive and (output_format is None or output_format == 'html'):
            run_interactive_dashboard(viz_api, params)
        else:
            # Otherwise, export the visualization to the specified format
            if isinstance(fig, str):
                # Terminal visualization
                output_handler.handle_result({'visualization': fig})
            else:
                # Export the visualization to the specified format
                export_visualization(viz_api, fig, output_path, output_format)

        # Display a success message
        display_success_message(f"Successfully generated visualization of type '{visualization_type}'")

        return 0  # Exit code 0 for success

    except Exception as e:
        log_message = f"Error generating visualization of type '{visualization_type}': {e}"
        logger.error(log_message)
        display_error_message(log_message)
        return 1  # Non-zero exit code for failure


def validate_parameters(visualization_type: str, forecast_id: Optional[str], model_version: Optional[str], compare_with: Optional[List[str]], threshold: Optional[float], nodes: Optional[List[str]], start_date: Optional[datetime.date], end_date: Optional[datetime.date]) -> Dict[str, Any]:
    """
    Validates the parameters for the visualize command
    """
    params: Dict[str, Any] = {}
    params['visualization_type'] = visualization_type

    if visualization_type == 'forecast':
        if forecast_id:
            params['forecast_id'] = forecast_id
        if threshold:
            params['threshold'] = validate_threshold_value(threshold)
        if nodes:
            params['nodes'] = validate_node_ids(nodes)

    if visualization_type in ['performance', 'calibration', 'feature_importance', 'roc_curve', 'precision_recall']:
        if model_version:
            params['model_version'] = validate_model_version(model_version)
        if compare_with:
            params['compare_with'] = [validate_model_version(v) for v in compare_with]

    if start_date and end_date:
        params['start_date'], params['end_date'] = validate_date_range(start_date, end_date)

    return params


def generate_forecast_visualization(viz_api: VisualizationAPI, params: Dict[str, Any], interactive: bool) -> Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]:
    """
    Generates a forecast visualization
    """
    forecast_id = params.get('forecast_id')
    threshold = params.get('threshold')
    nodes = params.get('nodes')

    visualization_config: Dict[str, Any] = {}
    if interactive:
        visualization_config['interactive'] = True

    return viz_api.create_forecast_visualization(forecast_id, 'probability_timeline', thresholds=[threshold] if threshold else None, nodes=nodes, visualization_config=visualization_config)


def generate_performance_visualization(viz_api: VisualizationAPI, params: Dict[str, Any], interactive: bool) -> Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]:
    """
    Generates a model performance visualization
    """
    model_version = params.get('model_version')
    compare_with = params.get('compare_with')

    visualization_config: Dict[str, Any] = {'compare_with': compare_with}
    if interactive:
        visualization_config['interactive'] = True

    return viz_api.create_model_performance_visualization(model_version, 'dashboard', visualization_config=visualization_config)


def generate_calibration_visualization(viz_api: VisualizationAPI, params: Dict[str, Any], interactive: bool) -> Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]:
    """
    Generates a calibration curve visualization
    """
    model_version = params.get('model_version')
    compare_with = params.get('compare_with')

    visualization_config: Dict[str, Any] = {'compare_with': compare_with}
    if interactive:
        visualization_config['interactive'] = True

    return viz_api.create_model_performance_visualization(model_version, 'calibration_curve', visualization_config=visualization_config)


def generate_feature_importance_visualization(viz_api: VisualizationAPI, params: Dict[str, Any], interactive: bool) -> Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]:
    """
    Generates a feature importance visualization
    """
    model_version = params.get('model_version')

    visualization_config: Dict[str, Any] = {}
    if interactive:
        visualization_config['interactive'] = True

    return viz_api.create_feature_importance_visualization(model_version, 'feature_importance', visualization_config=visualization_config)


def generate_roc_curve_visualization(viz_api: VisualizationAPI, params: Dict[str, Any], interactive: bool) -> Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]:
    """
    Generates a ROC curve visualization
    """
    model_version = params.get('model_version')
    compare_with = params.get('compare_with')

    visualization_config: Dict[str, Any] = {'compare_with': compare_with}
    if interactive:
        visualization_config['interactive'] = True

    return viz_api.create_model_performance_visualization(model_version, 'roc_curve', visualization_config=visualization_config)


def generate_precision_recall_visualization(viz_api: VisualizationAPI, params: Dict[str, Any], interactive: bool) -> Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]:
    """
    Generates a precision-recall curve visualization
    """
    model_version = params.get('model_version')
    compare_with = params.get('compare_with')

    visualization_config: Dict[str, Any] = {'compare_with': compare_with}
    if interactive:
        visualization_config['interactive'] = True

    return viz_api.create_model_performance_visualization(model_version, 'precision_recall_curve', visualization_config=visualization_config)


def export_visualization(viz_api: VisualizationAPI, fig: Union[matplotlib.figure.Figure, plotly.graph_objects.Figure], output_path: Optional[Path], output_format: Optional[str]) -> Path:
    """
    Exports a visualization to a file
    """
    try:
        return viz_api.export_visualization(fig, output_path, output_format)
    except Exception as e:
        raise CommandError(f"Error exporting visualization: {e}", command='visualize')


def run_interactive_dashboard(viz_api: VisualizationAPI, params: Dict[str, Any]) -> None:
    """
    Runs an interactive dashboard for the visualization
    """
    visualization_type = params.get('visualization_type')
    model_version = params.get('model_version')
    forecast_id = params.get('forecast_id')

    dashboard_config: Dict[str, Any] = {'visualization_type': visualization_type, 'model_version': model_version, 'forecast_id': forecast_id}

    logger.info("Starting interactive dashboard...")
    viz_api.run_dashboard(dashboard_config=dashboard_config)


def create_terminal_visualization(visualization_type: str, data: Dict[str, Any], params: Dict[str, Any]) -> str:
    """
    Creates a terminal-based visualization for non-interactive mode
    """
    if visualization_type == 'forecast':
        probabilities = data.get('probabilities')
        timestamps = data.get('timestamps')
        return create_probability_timeline(probabilities, timestamps)
    elif visualization_type == 'performance':
        metrics = data.get('metrics')
        return create_metrics_chart(metrics)
    elif visualization_type == 'calibration':
        predicted_probabilities = data.get('predicted_probabilities')
        true_probabilities = data.get('true_probabilities')
        return create_calibration_curve(predicted_probabilities, true_probabilities)
    elif visualization_type == 'feature_importance':
        feature_importance_values = data.get('feature_importance')
        return create_feature_importance_chart(feature_importance_values)
    elif visualization_type == 'roc_curve':
        false_positive_rates = data.get('false_positive_rates')
        true_positive_rates = data.get('true_positive_rates')
        return create_roc_curve(false_positive_rates, true_positive_rates)
    elif visualization_type == 'precision_recall':
        precision_values = data.get('precision_values')
        recall_values = data.get('recall_values')
        # Create a custom line chart for precision-recall curve
        return "Precision-Recall Curve (Terminal visualization not fully supported)"
    else:
        return "Unsupported visualization type for terminal output"