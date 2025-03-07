# src/cli/commands/backtest.py
"""
Implements the 'backtest' command for the CLI application of the ERCOT RTLMP spike prediction system.
This module enables users to run backtesting scenarios on historical data to evaluate model performance across different time periods, thresholds, and nodes.
"""

import typing
from typing import Dict, List, Optional, Any, cast

import click  # version 8.0+
import pandas  # version 2.0+
from pathlib import Path
from datetime import datetime

# Internal imports
from ..cli_types import BacktestParamsDict
from ..exceptions import CommandError, BacktestingError
from ..logger import get_cli_logger, CLILoggingContext
from ..utils.validators import validate_backtest_params, validate_date, validate_threshold_values, validate_node_ids, validate_model_version
from ..utils.progress_bars import create_indeterminate_spinner, progress_bar_context
from ..utils.output_handlers import OutputHandler, display_success_message, display_error_message
from ..ui.tables import create_backtesting_results_table
from ..ui.charts import create_performance_chart
from ...backend.api.backtesting_api import BacktestingAPI, run_backtesting_scenario, run_backtesting_scenarios, visualize_backtesting_results

logger = get_cli_logger(__name__)


@click.command('backtest', help='Run backtesting on historical data')
@click.option('--start-date', '-s', required=True, help='Start date for backtesting (YYYY-MM-DD)')
@click.option('--end-date', '-e', required=True, help='End date for backtesting (YYYY-MM-DD)')
@click.option('--thresholds', '-t', multiple=True, type=float, required=True, help='Price threshold values for spike prediction')
@click.option('--nodes', '-n', multiple=True, required=True, help='Node IDs to run backtesting for')
@click.option('--model-version', '-m', type=str, help='Model version to use (default: latest)')
@click.option('--output-path', '-o', type=click.Path(file_okay=True, dir_okay=True, writable=True), help='Path to save output')
@click.option('--output-format', '-f', type=click.Choice(['text', 'json', 'csv']), help='Output format')
@click.option('--visualize', '-v', is_flag=True, help='Show visualization of backtesting results')
def backtest_command(start_date: str, end_date: str, thresholds: List[float], nodes: List[str], model_version: Optional[str], output_path: Optional[Path], output_format: Optional[str], visualize: bool) -> int:
    """
    Main function for the 'backtest' command that runs backtesting scenarios on historical data
    """
    logger.info("Starting backtest command execution")

    try:
        # Parse and validate start_date and end_date strings to datetime.date objects
        start_date_dt = validate_date(start_date)
        end_date_dt = validate_date(end_date)

        # Validate command parameters using validate_backtest_params
        params: BacktestParamsDict = validate_backtest_params({
            'start_date': start_date_dt,
            'end_date': end_date_dt,
            'thresholds': thresholds,
            'nodes': nodes,
            'model_version': model_version
        })

        # Create an OutputHandler for handling command results
        output_handler = OutputHandler(output_path=output_path, output_format=output_format)

        # Initialize the BacktestingAPI
        backtesting_api = BacktestingAPI()

        # Create scenario configuration dictionary with validated parameters
        scenario_config = {}  # Placeholder for scenario configuration

        # Display a spinner while running the backtesting scenario
        with create_indeterminate_spinner("Running backtesting scenario") as spinner:
            # Run the backtesting scenario with the specified parameters
            backtesting_result = backtesting_api.run_scenario(scenario_config)

        # Format the backtesting results for display
        formatted_results = {}  # Placeholder for formatted results

        # If visualize is True, create and display performance charts
        if visualize:
            # Create and display performance charts
            visualization = {}  # Placeholder for visualization
            # Handle the backtesting output using the OutputHandler
            output_handler.handle_result(visualization)

        # Handle the backtesting output using the OutputHandler
        output_handler.handle_result(formatted_results)

        # Display a success message
        display_success_message("Backtesting completed successfully")

        # Return exit code 0 for success
        return 0

    except Exception as e:
        # Catch and handle any exceptions, returning appropriate error codes
        logger.exception(f"An error occurred during backtesting: {e}")
        display_error_message(f"Backtesting failed: {e}")
        return 1