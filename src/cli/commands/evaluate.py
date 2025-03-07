# src/cli/commands/evaluate.py
"""
Implements the 'evaluate' command for the CLI application of the ERCOT RTLMP spike prediction system.
This module enables users to evaluate model performance with various metrics, compare multiple models, and visualize evaluation results.
"""
import typing
from typing import Dict, List, Optional, Any, cast
import click  # version 8.0+
import pandas  # version 2.0+
from pathlib import Path

from ..cli_types import EvaluateParamsDict  # type: ignore
from ..exceptions import CommandError, ModelOperationError  # type: ignore
from ..logger import get_cli_logger, CLILoggingContext  # type: ignore
from ..utils.validators import validate_evaluate_params, validate_model_version, validate_threshold_values, validate_node_ids  # type: ignore
from ..utils.formatters import format_metrics, format_comparison_result  # type: ignore
from ..utils.progress_bars import create_indeterminate_spinner, progress_bar_context  # type: ignore
from ..utils.output_handlers import OutputHandler, display_success_message, display_error_message  # type: ignore
from ..ui.tables import create_metrics_table, create_comparison_table  # type: ignore
from ..ui.charts import create_metrics_chart, create_roc_curve, create_calibration_curve, create_feature_importance_chart  # type: ignore
from ...backend.api.model_api import ModelAPI, evaluate_model, compare_models, get_model, get_latest_model  # type: ignore
from ...backend.api.data_api import DataFetcherAPI, fetch_historical_data  # type: ignore

logger = get_cli_logger(__name__)


@click.command('evaluate', help='Evaluate model performance')
@click.option('--model-version', '-m', type=str, help='Model version to evaluate (default: latest)')
@click.option('--compare-with', '-c', multiple=True, help='Model versions to compare with')
@click.option('--thresholds', '-t', type=float, multiple=True, required=True, help='Price threshold values for evaluation')
@click.option('--nodes', '-n', multiple=True, required=True, help='Node IDs to evaluate for')
@click.option('--start-date', '-s', type=str, help='Start date for test data (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=str, help='End date for test data (YYYY-MM-DD)')
@click.option('--output-path', '-o', type=click.Path(file_okay=True, dir_okay=True, writable=True), help='Path to save output')
@click.option('--output-format', '-f', type=click.Choice(['text', 'json', 'csv']), help='Output format')
@click.option('--visualize', '-v', is_flag=True, help='Show visualization of evaluation results')
def evaluate_command(
    model_version: Optional[str],
    compare_with: Optional[List[str]],
    thresholds: List[float],
    nodes: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    output_path: Optional[Path],
    output_format: Optional[str],
    visualize: bool
) -> int:
    """
    Main function for the 'evaluate' command that evaluates model performance
    """
    logger.info("Starting evaluate command execution")

    try:
        # Validate command parameters using validate_evaluate_params
        params = validate_parameters(model_version, compare_with, thresholds, nodes, start_date, end_date)

        # Create an OutputHandler for handling command results
        output_handler = OutputHandler(output_path, output_format)

        # Initialize the ModelAPI
        model_api = ModelAPI()

        # Display a spinner while loading the model
        with create_indeterminate_spinner("Loading model...") as spinner:
            # If model_version is provided, load the specified model
            if params.get("model_version"):
                model = model_api.get_model(params["model_version"])
                spinner.update(f"Loaded model version {params['model_version']}")
            # Otherwise, use the latest model
            else:
                model = model_api.get_latest_model()
                spinner.update(f"Loaded latest model version")

        # If start_date and end_date are provided, fetch test data for that period
        if params.get("start_date") and params.get("end_date"):
            # Fetch test data for the specified period
            features = DataFetcherAPI.fetch_historical_data(params["start_date"], params["end_date"], params["nodes"])
        # Otherwise, use default test data
        else:
            # Use default test data
            features = DataFetcherAPI.fetch_historical_data()

        # Display a spinner while evaluating the model
        with create_indeterminate_spinner("Evaluating model...") as spinner:
            # Evaluate the model with the specified thresholds and nodes
            evaluation_results = model_api.evaluate_model(model, features, params["thresholds"], params["nodes"])
            spinner.update("Model evaluation complete")

        # If compare_with is provided, compare with the specified model versions
        if params.get("compare_with"):
            # Compare with the specified model versions
            comparison_results = model_api.compare_models(params["compare_with"], features, params["thresholds"], params["nodes"])
        else:
            comparison_results = None

        # Format the evaluation results for display
        formatted_result = format_evaluation_result(evaluation_results, model.get_model_config(), comparison_results)

        # If visualize is True, create and display evaluation visualizations
        if params["visualize"]:
            # Create and display evaluation visualizations
            visualizations = create_visualizations(evaluation_results, comparison_results)
            formatted_result["visualizations"] = visualizations

        # Handle the evaluation output using the OutputHandler
        success = output_handler.handle_result(formatted_result)

        if success:
            # Display a success message
            display_success_message("Model evaluation complete")
            # Return exit code 0 for success
            return 0
        else:
            # Display an error message
            display_error_message("Model evaluation failed")
            # Return exit code 1 for failure
            return 1

    except Exception as e:
        # Log the error
        logger.exception(f"An error occurred during model evaluation: {e}")
        # Display an error message
        display_error_message(f"Model evaluation failed: {e}")
        # Return exit code 1 for failure
        return 1


def validate_parameters(
    model_version: Optional[str],
    compare_with: Optional[List[str]],
    thresholds: List[float],
    nodes: List[str],
    start_date: Optional[str],
    end_date: Optional[str]
) -> Dict[str, Any]:
    """
    Validates the parameters for the evaluate command
    """
    # Validate model_version using validate_model_version if provided
    if model_version:
        validate_model_version(model_version)

    # Validate compare_with model versions if provided
    if compare_with:
        for version in compare_with:
            validate_model_version(version)

    # Validate thresholds using validate_threshold_values
    validate_threshold_values(thresholds)

    # Validate nodes using validate_node_ids
    validate_node_ids(nodes)

    # Validate start_date and end_date if provided
    if start_date and end_date:
        # Validate start_date and end_date
        pass

    # Create and return a validated parameters dictionary
    validated_params = {
        "model_version": model_version,
        "compare_with": compare_with,
        "thresholds": thresholds,
        "nodes": nodes,
        "start_date": start_date,
        "end_date": end_date,
    }
    return validated_params


def evaluate_model_performance(
    model: Any,
    features: pandas.DataFrame,
    targets: pandas.Series,
    thresholds: List[float],
    metrics: List[str]
) -> Dict[str, Any]:
    """
    Evaluates model performance with specified parameters
    """
    # Initialize empty dictionary for results
    results = {}

    # For each threshold:
    for threshold in thresholds:
        # Extract target values for the threshold
        # Call model_api.evaluate_model with model, features, targets, and metrics
        evaluation_result = evaluate_model(model, features, targets, metrics)
        # Store results in the dictionary with threshold as key
        results[threshold] = evaluation_result

    # Return the evaluation results dictionary
    return results


def compare_model_versions(
    model_versions: List[str],
    features: pandas.DataFrame,
    targets: pandas.Series,
    thresholds: List[float],
    metrics: List[str]
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """
    Compares multiple model versions on the same dataset
    """
    # Initialize empty dictionary for results
    results = {}

    # For each model version:
    for model_version in model_versions:
        # Load the model using model_api.get_model
        model = get_model(model_version)
        # Evaluate the model using evaluate_model_performance
        evaluation_result = evaluate_model_performance(model, features, targets, thresholds, metrics)
        # Store results in the dictionary with model version as key
        results[model_version] = evaluation_result

    # Return the comparison results dictionary
    return results


def format_evaluation_result(
    evaluation_results: Dict[str, Any],
    model_info: Dict[str, Any],
    comparison_results: Optional[Dict[str, Dict[float, Dict[str, float]]]]
) -> Dict[str, Any]:
    """
    Formats the evaluation result for display and export
    """
    # Format the evaluation metrics using format_metrics
    formatted_metrics = format_metrics(evaluation_results)

    # If comparison_results is provided, format using format_comparison_result
    if comparison_results:
        formatted_comparison = format_comparison_result(comparison_results)
    else:
        formatted_comparison = None

    # Create a result dictionary with evaluation data and model information
    result = {
        "model_info": model_info,
        "metrics": formatted_metrics,
        "comparison": formatted_comparison,
    }

    # Return the formatted result
    return result


def create_visualizations(
    evaluation_results: Dict[str, Any],
    comparison_results: Optional[Dict[str, Dict[float, Dict[str, float]]]]
) -> Dict[str, str]:
    """
    Creates visualizations of the evaluation results
    """
    # Initialize empty dictionary for visualizations
    visualizations = {}

    # For each threshold in evaluation_results:
    for threshold in evaluation_results:
        # Create metrics chart using create_metrics_chart
        metrics_chart = create_metrics_chart(evaluation_results[threshold])
        visualizations[f"metrics_chart_{threshold}"] = metrics_chart

        # Create ROC curve using create_roc_curve if data available
        roc_curve = create_roc_curve(evaluation_results[threshold])
        visualizations[f"roc_curve_{threshold}"] = roc_curve

        # Create calibration curve using create_calibration_curve if data available
        calibration_curve = create_calibration_curve(evaluation_results[threshold])
        visualizations[f"calibration_curve_{threshold}"] = calibration_curve

        # Create feature importance chart using create_feature_importance_chart if data available
        feature_importance_chart = create_feature_importance_chart(evaluation_results[threshold])
        visualizations[f"feature_importance_chart_{threshold}"] = feature_importance_chart

    # If comparison_results is provided:
    if comparison_results:
        # Create comparison visualizations for each metric and threshold
        pass

    # Return the dictionary of visualization strings
    return visualizations