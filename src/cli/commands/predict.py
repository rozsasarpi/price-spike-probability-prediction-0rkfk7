"""
Implements the 'predict' command for the CLI application of the ERCOT RTLMP spike prediction system. This module enables users to generate RTLMP spike probability forecasts using trained models, with options for specifying thresholds, nodes, model versions, and output formats.
"""

import typing
from typing import Dict, List, Optional, Any, cast
import click  # version 8.0+
import pandas  # version 2.0+
from pathlib import Path

from ..cli_types import PredictParamsDict  # src/cli/cli_types.py
from ..exceptions import CommandError, ModelOperationError  # src/cli/exceptions.py
from ..logger import get_cli_logger, CLILoggingContext  # src/cli/logger.py
from ..utils.formatters import format_forecast_data  # src/cli/utils/formatters.py
from ..utils.validators import validate_predict_params, validate_threshold_value, validate_node_ids, validate_model_version  # src/cli/utils/validators.py
from ..utils.progress_bars import create_indeterminate_spinner, progress_bar_context  # src/cli/utils/progress_bars.py
from ..utils.output_handlers import OutputHandler, display_success_message, display_error_message  # src/cli/utils/output_handlers.py
from ..ui.tables import create_forecast_table  # src/cli/ui/tables.py
from ..ui.charts import create_probability_timeline  # src/cli/ui/charts.py
from ...backend.api.inference_api import InferenceAPI, generate_forecast, get_latest_forecast  # src/backend/api/inference_api.py

logger = get_cli_logger(__name__)


@click.command('predict', help='Generate forecasts using trained model')
@click.option('--threshold', '-t', type=float, required=True, help='Price threshold value for spike prediction')
@click.option('--nodes', '-n', multiple=True, required=True, help='Node IDs to generate forecasts for')
@click.option('--model-version', '-m', type=str, help='Model version to use (default: latest)')
@click.option('--output-path', '-o', type=click.Path(file_okay=True, dir_okay=True, writable=True), help='Path to save output')
@click.option('--output-format', '-f', type=click.Choice(['text', 'json', 'csv']), help='Output format')
@click.option('--visualize', '-v', is_flag=True, help='Show visualization of forecast')
def predict_command(
    threshold: float,
    nodes: List[str],
    model_version: Optional[str] = None,
    output_path: Optional[Path] = None,
    output_format: Optional[str] = None,
    visualize: bool = False
) -> int:
    """
    Main function for the 'predict' command that generates RTLMP spike probability forecasts
    """
    with CLILoggingContext(logger, command='predict', threshold=threshold, nodes=nodes, model_version=model_version, output_path=output_path, output_format=output_format, visualize=visualize):
        logger.info("Starting predict command execution")

        try:
            # Validate command parameters using validate_predict_params
            validated_params: PredictParamsDict = validate_parameters(threshold, nodes, model_version)

            # Create an OutputHandler for handling command results
            output_handler = OutputHandler(output_path=output_path, output_format=output_format)

            # Initialize the InferenceAPI
            inference_api = InferenceAPI()

            # Display a spinner while loading the model
            with create_indeterminate_spinner("Loading model...") as load_spinner:
                # If model_version is provided, load the specified model
                if validated_params.get('model_version'):
                    inference_api.load_model(model_version=validated_params['model_version'])
                # Otherwise, use the latest model
                else:
                    inference_api.load_model()
                load_spinner.update("Model loaded successfully.")

            # Get model information for logging
            model_info = inference_api.get_model_info()

            # Display a spinner while generating the forecast
            with progress_bar_context(desc="Generating forecast...") as forecast_spinner:
                # Generate the forecast with the specified threshold and nodes
                forecast_df = generate_forecast(thresholds=[validated_params['threshold']], nodes=validated_params['nodes'])

            # Format the forecast results for display
            result = {
                'title': 'RTLMP Spike Probability Forecast',
                'forecast': forecast_df.to_dict('records'),
                'model_version': model_info.get('version', 'latest'),
                'threshold': validated_params['threshold'],
                'nodes': validated_params['nodes']
            }

            # If visualize is True, create and display a probability timeline chart
            if visualize:
                with create_indeterminate_spinner("Creating visualization...") as viz_spinner:
                    visualization = create_visualization(forecast_df, validated_params['threshold'])
                    result['visualization'] = visualization
                    viz_spinner.update("Visualization created.")

            # Handle the forecast output using the OutputHandler
            output_handler.handle_result(result)

            # Display a success message
            display_success_message("Forecast generated successfully.")

            # Return exit code 0 for success
            return 0

        except Exception as e:
            # Catch and handle any exceptions, returning appropriate error codes
            logger.error(f"An error occurred: {e}")
            display_error_message(f"Failed to generate forecast: {e}")
            return 1


def validate_parameters(
    threshold: float,
    nodes: List[str],
    model_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates the parameters for the predict command
    """
    # Validate threshold using validate_threshold_value
    validated_threshold = validate_threshold_value(threshold)

    # Validate nodes using validate_node_ids
    validated_nodes = validate_node_ids(nodes)

    # If model_version is provided, validate it using validate_model_version
    validated_model_version = validate_model_version(model_version) if model_version else None

    # Create and return a validated parameters dictionary
    validated_params = {
        'threshold': validated_threshold,
        'nodes': validated_nodes,
        'model_version': validated_model_version
    }
    return validated_params


def format_forecast_result(
    forecast_df: pandas.DataFrame,
    model_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Formats the forecast result for display and export
    """
    # Convert the forecast DataFrame to a list of dictionaries
    forecast_data = forecast_df.to_dict('records')

    # Format each forecast entry using format_forecast_data
    formatted_forecast = [format_forecast_data(entry) for entry in forecast_data]

    # Create a result dictionary with forecast data and model information
    result = {
        'forecast': formatted_forecast,
        'model_info': model_info
    }

    # Return the formatted result
    return result


def create_visualization(
    forecast_df: pandas.DataFrame,
    threshold: float
) -> str:
    """
    Creates a visualization of the forecast probabilities
    """
    # Extract timestamps and probability values from the forecast DataFrame
    timestamps = forecast_df['target_timestamp'].tolist()
    probabilities = forecast_df['spike_probability'].tolist()

    # Create a probability timeline chart using create_probability_timeline
    visualization = create_probability_timeline(probabilities, timestamps, threshold_label=str(threshold))

    # Return the visualization string
    return visualization