"""
Provides a unified API for backtesting operations in the ERCOT RTLMP spike prediction system.
This module acts as a facade over the backtesting framework, offering a clean interface for configuring, executing, and analyzing backtesting scenarios.
"""

import os  # Standard
import pathlib  # Standard
from datetime import datetime  # Standard
from typing import Dict, List, Optional, Union, Tuple, Any  # Standard

import pandas  # version 2.0+

# Internal imports
from ..backtesting.framework import BacktestingFramework  # src/backend/backtesting/framework.py
from ..backtesting.historical_simulation import HistoricalSimulator  # src/backend/backtesting/historical_simulation.py
from ..backtesting.scenario_definitions import ScenarioConfig, ModelConfig, MetricsConfig  # src/backend/backtesting/scenario_definitions.py
from ..backtesting.performance_metrics import BacktestingMetricsCalculator  # src/backend/backtesting/performance_metrics.py
from .data_api import DataAPI  # src/backend/api/data_api.py
from .model_api import ModelAPI  # src/backend/api/model_api.py
from ..utils.type_definitions import DataFrameType, ModelType, ThresholdValue, NodeID, PathType  # src/backend/utils/type_definitions.py
from ..utils.logging import get_logger, log_execution_time  # src/backend/utils/logging.py
from ..utils.error_handling import handle_errors, BacktestingError  # src/backend/utils/error_handling.py

# Initialize logger
logger = get_logger(__name__)

# Define default paths from environment variables
DEFAULT_BACKTESTING_PATH = os.environ.get('BACKTESTING_PATH', 'results/backtesting')
DEFAULT_MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(BacktestingError,), error_message='Failed to run backtesting scenario')
def run_backtesting_scenario(
    scenario_config: Dict[str, Any],
    model_path: Optional[PathType] = None,
    output_path: Optional[PathType] = None,
    save_results: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Executes a single backtesting scenario with the provided configuration

    Args:
        scenario_config (Dict[str, Any]): Dictionary containing scenario configuration
        model_path (Optional[PathType]): Path to the model directory
        output_path (Optional[PathType]): Path to save the results
        save_results (Optional[bool]): Whether to save the results

    Returns:
        Dict[str, Any]: Dictionary containing scenario results and metrics
    """
    # Create ScenarioConfig from scenario_config dictionary using from_dict
    scenario = ScenarioConfig.from_dict(scenario_config)

    # Initialize DataAPI for data retrieval
    data_api = DataAPI()

    # Initialize BacktestingFramework with DataAPI, model_path, and output_path
    backtesting_framework = BacktestingFramework(data_fetcher=data_api, model_path=model_path, output_path=output_path)

    # Execute the scenario using BacktestingFramework.execute_scenario
    scenario_results = backtesting_framework.execute_scenario(scenario=scenario)

    # If save_results is True, save results using BacktestingFramework.save_results
    if save_results:
        backtesting_framework.save_results(output_path=output_path)

    # Return the scenario results
    return scenario_results


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(BacktestingError,), error_message='Failed to run backtesting scenarios')
def run_backtesting_scenarios(
    scenario_configs: List[Dict[str, Any]],
    model_path: Optional[PathType] = None,
    output_path: Optional[PathType] = None,
    parallel: Optional[bool] = False,
    n_jobs: Optional[int] = None,
    save_results: Optional[bool] = False
) -> Dict[str, Dict[str, Any]]:
    """
    Executes multiple backtesting scenarios with the provided configurations

    Args:
        scenario_configs (List[Dict[str, Any]]): List of dictionaries containing scenario configurations
        model_path (Optional[PathType]): Path to the model directory
        output_path (Optional[PathType]): Path to save the results
        parallel (Optional[bool]): Whether to run scenarios in parallel
        n_jobs (Optional[int]): Number of jobs to use for parallel execution
        save_results (Optional[bool]): Whether to save the results

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping scenario names to their results
    """
    # Convert each scenario_config dictionary to ScenarioConfig using from_dict
    scenarios = [ScenarioConfig.from_dict(config_dict) for config_dict in scenario_configs]

    # Initialize DataAPI for data retrieval
    data_api = DataAPI()

    # Initialize BacktestingFramework with DataAPI, model_path, and output_path
    backtesting_framework = BacktestingFramework(data_fetcher=data_api, model_path=model_path, output_path=output_path)

    # Execute the scenarios using BacktestingFramework.execute_scenarios with parallel and n_jobs
    scenario_results = backtesting_framework.execute_scenarios(scenarios=scenarios, parallel=parallel, n_jobs=n_jobs)

    # If save_results is True, save results using BacktestingFramework.save_results
    if save_results:
        backtesting_framework.save_results(output_path=output_path)

    # Return the dictionary of scenario results
    return scenario_results


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(BacktestingError,), error_message='Failed to run historical simulation')
def run_historical_simulation(
    scenario_config: Dict[str, Any],
    model_path: Optional[PathType] = None,
    output_path: Optional[PathType] = None,
    retrain_per_window: Optional[bool] = False,
    save_results: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Executes a historical simulation with the provided configuration

    Args:
        scenario_config (Dict[str, Any]): Dictionary containing scenario configuration
        model_path (Optional[PathType]): Path to the model directory
        output_path (Optional[PathType]): Path to save the results
        retrain_per_window (Optional[bool]): Whether to retrain the model for each time window
        save_results (Optional[bool]): Whether to save the results

    Returns:
        Dict[str, Any]: Dictionary containing simulation results
    """
    # Create ScenarioConfig from scenario_config dictionary using from_dict
    scenario = ScenarioConfig.from_dict(scenario_config)

    # Initialize DataAPI for data retrieval
    data_api = DataAPI()

    # Initialize HistoricalSimulator with DataAPI, model_path, and output_path
    historical_simulator = HistoricalSimulator(data_fetcher=data_api, model_path=model_path, output_path=output_path)

    # Execute the simulation using HistoricalSimulator.run_simulation with retrain_per_window
    simulation_results = historical_simulator.run_simulation(scenario=scenario, retrain_per_window=retrain_per_window)

    # If save_results is True, save results using HistoricalSimulator.save_results
    if save_results:
        historical_simulator.save_results(output_path=output_path)

    # Return the simulation results
    return simulation_results


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to retrieve backtesting results', default_return={})
def get_backtesting_results(
    scenario_name: Optional[str] = None,
    results_path: Optional[PathType] = None
) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieves results from previously executed backtesting scenarios

    Args:
        scenario_name (Optional[str]): Name of the scenario to retrieve results for
        results_path (Optional[PathType]): Path to the directory containing backtesting results

    Returns:
        Union[Dict[str, Dict[str, Any]], Dict[str, Any]]: Backtesting results
    """
    # Set results_path to provided value or DEFAULT_BACKTESTING_PATH
    results_path = results_path or DEFAULT_BACKTESTING_PATH

    # Initialize BacktestingFramework with None for data_fetcher and model_path
    backtesting_framework = BacktestingFramework(data_fetcher=None, model_path=None, output_path=results_path)  # type: ignore

    # Load results from results_path using BacktestingFramework.load_results
    backtesting_framework.load_results(results_paths=results_path)

    # If scenario_name is provided, return results for that scenario only
    if scenario_name:
        results = backtesting_framework.get_results(scenario_name=scenario_name)
        return results

    # Otherwise, return all results
    results = backtesting_framework.get_results()
    return results


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to compare backtesting results', default_return={})
def compare_backtesting_results(
    scenario_names: List[str],
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[ThresholdValue]] = None,
    results_path: Optional[PathType] = None
) -> Dict[str, DataFrameType]:
    """
    Compares results from multiple backtesting scenarios

    Args:
        scenario_names (List[str]): List of scenario names to compare
        metrics (Optional[List[str]]): List of metrics to include in the comparison
        thresholds (Optional[List[ThresholdValue]]): List of threshold values to include in the comparison
        results_path (Optional[PathType]): Path to the directory containing backtesting results

    Returns:
        Dict[str, DataFrameType]: Dictionary of comparison DataFrames
    """
    # Set results_path to provided value or DEFAULT_BACKTESTING_PATH
    results_path = results_path or DEFAULT_BACKTESTING_PATH

    # Initialize BacktestingFramework with None for data_fetcher and model_path
    backtesting_framework = BacktestingFramework(data_fetcher=None, model_path=None, output_path=results_path)  # type: ignore

    # Load results from results_path using BacktestingFramework.load_results
    backtesting_framework.load_results(results_paths=results_path)

    # Compare results using BacktestingFramework.compare_results with scenario_names, metrics, and thresholds
    comparison_dataframes = backtesting_framework.compare_results(scenario_names=scenario_names, metrics=metrics, thresholds=thresholds)

    # Return the comparison DataFrames
    return comparison_dataframes


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to visualize backtesting results', default_return={})
def visualize_backtesting_results(
    scenario_name: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[ThresholdValue]] = None,
    show_plot: Optional[bool] = False,
    results_path: Optional[PathType] = None
) -> Dict[str, Any]:
    """
    Generates visualizations for backtesting results

    Args:
        scenario_name (Optional[str]): Name of the scenario to visualize
        metrics (Optional[List[str]]): List of metrics to visualize
        thresholds (Optional[List[ThresholdValue]]): List of threshold values to visualize
        show_plot (Optional[bool]): Whether to display the plots
        results_path (Optional[PathType]): Path to the directory containing backtesting results

    Returns:
        Dict[str, Any]: Dictionary of plot objects
    """
    # Set results_path to provided value or DEFAULT_BACKTESTING_PATH
    results_path = results_path or DEFAULT_BACKTESTING_PATH

    # Initialize BacktestingFramework with None for data_fetcher and model_path
    backtesting_framework = BacktestingFramework(data_fetcher=None, model_path=None, output_path=results_path)  # type: ignore

    # Load results from results_path using BacktestingFramework.load_results
    backtesting_framework.load_results(results_paths=results_path)

    # Generate visualizations using BacktestingFramework.visualize_results with scenario_name, metrics, thresholds, and show_plot
    plots = backtesting_framework.visualize_results(scenario_name=scenario_name, metrics=metrics, thresholds=thresholds, show_plot=show_plot)

    # Return the dictionary of plot objects
    return plots


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=(Exception,), error_message='Failed to generate backtesting report', default_return={})
def generate_backtesting_report(
    scenario_name: str,
    additional_data: Optional[Dict[str, Any]] = None,
    results_path: Optional[PathType] = None,
    output_path: Optional[PathType] = None
) -> Dict[str, Any]:
    """
    Generates a comprehensive report for backtesting results

    Args:
        scenario_name (str): Name of the scenario to generate a report for
        additional_data (Optional[Dict[str, Any]]): Additional data to include in the report
        results_path (Optional[PathType]): Path to the directory containing backtesting results
        output_path (Optional[PathType]): Path to save the report

    Returns:
        Dict[str, Any]: Report dictionary
    """
    # Set results_path to provided value or DEFAULT_BACKTESTING_PATH
    results_path = results_path or DEFAULT_BACKTESTING_PATH

    # Set output_path to provided value or DEFAULT_BACKTESTING_PATH
    output_path = output_path or DEFAULT_BACKTESTING_PATH

    # Initialize BacktestingFramework with None for data_fetcher and model_path
    backtesting_framework = BacktestingFramework(data_fetcher=None, model_path=None, output_path=results_path)  # type: ignore

    # Load results from results_path using BacktestingFramework.load_results
    backtesting_framework.load_results(results_paths=results_path)

    # Get results for the specified scenario_name
    results = backtesting_framework.get_results(scenario_name=scenario_name)

    # Initialize BacktestingMetricsCalculator
    metrics_calculator = BacktestingMetricsCalculator()

    # Generate report using BacktestingMetricsCalculator.generate_report with additional_data and output_path
    report = metrics_calculator.generate_report(model_id=scenario_name, additional_data=additional_data, output_path=output_path)

    # Return the report dictionary
    return report


class BacktestingAPI:
    """
    Class that provides a unified interface for backtesting operations
    """

    def __init__(
        self,
        model_path: Optional[PathType] = None,
        results_path: Optional[PathType] = None
    ):
        """
        Initialize the BacktestingAPI with configuration

        Args:
            model_path (Optional[PathType]): Path to the model directory
            results_path (Optional[PathType]): Path to the directory containing backtesting results
        """
        # Set _model_path to model_path or DEFAULT_MODEL_PATH
        self._model_path = model_path or DEFAULT_MODEL_PATH

        # Set _results_path to results_path or DEFAULT_BACKTESTING_PATH
        self._results_path = results_path or DEFAULT_BACKTESTING_PATH

        # Initialize _data_api with default configuration
        self._data_api = DataAPI()

        # Initialize _backtesting_framework with _data_api, _model_path, and _results_path
        self._backtesting_framework = BacktestingFramework(data_fetcher=self._data_api, model_path=self._model_path, output_path=self._results_path)

        # Initialize _historical_simulator with _data_api, _model_path, and _results_path
        self._historical_simulator = HistoricalSimulator(data_fetcher=self._data_api, model_path=self._model_path, output_path=self._results_path)

        # Log initialization of BacktestingAPI
        logger.info("Initialized BacktestingAPI")

    @log_execution_time(logger, 'INFO')
    def run_scenario(
        self,
        scenario_config: Dict[str, Any],
        save_results: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Executes a single backtesting scenario

        Args:
            scenario_config (Dict[str, Any]): Dictionary containing scenario configuration
            save_results (Optional[bool]): Whether to save the results

        Returns:
            Dict[str, Any]: Dictionary containing scenario results and metrics
        """
        # Create ScenarioConfig from scenario_config dictionary using from_dict
        scenario = ScenarioConfig.from_dict(scenario_config)

        # Execute the scenario using _backtesting_framework.execute_scenario
        scenario_results = self._backtesting_framework.execute_scenario(scenario=scenario, save_results=save_results)

        # Return the scenario results
        return scenario_results

    @log_execution_time(logger, 'INFO')
    def run_scenarios(
        self,
        scenario_configs: List[Dict[str, Any]],
        parallel: Optional[bool] = False,
        n_jobs: Optional[int] = None,
        save_results: Optional[bool] = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes multiple backtesting scenarios

        Args:
            scenario_configs (List[Dict[str, Any]]): List of dictionaries containing scenario configurations
            parallel (Optional[bool]): Whether to run scenarios in parallel
            n_jobs (Optional[int]): Number of jobs to use for parallel execution
            save_results (Optional[bool]): Whether to save the results

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping scenario names to their results
        """
        # Convert each scenario_config dictionary to ScenarioConfig using from_dict
        scenarios = [ScenarioConfig.from_dict(config_dict) for config_dict in scenario_configs]

        # Execute the scenarios using _backtesting_framework.execute_scenarios with parallel and n_jobs
        scenario_results = self._backtesting_framework.execute_scenarios(scenarios=scenarios, parallel=parallel, n_jobs=n_jobs, save_results=save_results)

        # Return the dictionary of scenario results
        return scenario_results

    @log_execution_time(logger, 'INFO')
    def run_historical_simulation(
        self,
        scenario_config: Dict[str, Any],
        retrain_per_window: Optional[bool] = False,
        save_results: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Executes a historical simulation

        Args:
            scenario_config (Dict[str, Any]): Dictionary containing scenario configuration
            retrain_per_window (Optional[bool]): Whether to retrain the model for each time window
            save_results (Optional[bool]): Whether to save the results

        Returns:
            Dict[str, Any]: Dictionary containing simulation results
        """
        # Create ScenarioConfig from scenario_config dictionary using from_dict
        scenario = ScenarioConfig.from_dict(scenario_config)

        # Execute the simulation using _historical_simulator.run_simulation with retrain_per_window
        simulation_results = self._historical_simulator.run_simulation(scenario=scenario, retrain_per_window=retrain_per_window, save_results=save_results)

        # Return the simulation results
        return simulation_results

    @log_execution_time(logger, 'INFO')
    def get_results(
        self,
        scenario_name: Optional[str] = None
    ) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieves results from previously executed backtesting scenarios

        Args:
            scenario_name (Optional[str]): Name of the scenario to retrieve results for

        Returns:
            Union[Dict[str, Dict[str, Any]], Dict[str, Any]]: Backtesting results
        """
        # Get results using _backtesting_framework.get_results with scenario_name
        results = self._backtesting_framework.get_results(scenario_name=scenario_name)

        # Return the results
        return results

    @log_execution_time(logger, 'INFO')
    def load_results(
        self,
        results_paths: Union[PathType, List[PathType]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Loads backtesting results from disk

        Args:
            results_paths (Union[PathType, List[PathType]]): Path to the results file or a list of paths

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of loaded results
        """
        # Load results using _backtesting_framework.load_results with results_paths
        results = self._backtesting_framework.load_results(results_paths=results_paths)

        # Return the loaded results
        return results

    @log_execution_time(logger, 'INFO')
    def compare_results(
        self,
        scenario_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[List[ThresholdValue]] = None
    ) -> Dict[str, DataFrameType]:
        """
        Compares results from multiple backtesting scenarios

        Args:
            scenario_names (Optional[List[str]]): List of scenario names to compare
            metrics (Optional[List[str]]): List of metrics to include in the comparison
            thresholds (Optional[List[ThresholdValue]]): List of threshold values to include in the comparison

        Returns:
            Dict[str, DataFrameType]: Dictionary of comparison DataFrames
        """
        # Compare results using _backtesting_framework.compare_results with scenario_names, metrics, and thresholds
        comparison_dataframes = self._backtesting_framework.compare_results(scenario_names=scenario_names, metrics=metrics, thresholds=thresholds)

        # Return the comparison DataFrames
        return comparison_dataframes

    @log_execution_time(logger, 'INFO')
    def visualize_results(
        self,
        scenario_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[List[ThresholdValue]] = None,
        show_plot: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Generates visualizations for backtesting results

        Args:
            scenario_name (Optional[str]): Name of the scenario to visualize
            metrics (Optional[List[str]]): List of metrics to visualize
            thresholds (Optional[List[ThresholdValue]]): List of threshold values to visualize
            show_plot (Optional[bool]): Whether to display the plots

        Returns:
            Dict[str, Any]: Dictionary of plot objects
        """
        # Generate visualizations using _backtesting_framework.visualize_results with scenario_name, metrics, thresholds, and show_plot
        plots = self._backtesting_framework.visualize_results(scenario_name=scenario_name, metrics=metrics, thresholds=thresholds, show_plot=show_plot)

        # Return the dictionary of plot objects
        return plots

    @log_execution_time(logger, 'INFO')
    def generate_report(
        self,
        scenario_name: str,
        additional_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[PathType] = None
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive report for backtesting results

        Args:
            scenario_name (str): Name of the scenario to generate a report for
            additional_data (Optional[Dict[str, Any]]): Additional data to include in the report
            output_path (Optional[PathType]): Path to save the report

        Returns:
            Dict[str, Any]: Report dictionary
        """
        # Get results for the specified scenario_name using _backtesting_framework.get_results
        results = self._backtesting_framework.get_results(scenario_name=scenario_name)

        # Initialize BacktestingMetricsCalculator
        metrics_calculator = BacktestingMetricsCalculator()

        # Generate report using BacktestingMetricsCalculator.generate_report with additional_data and output_path
        report = metrics_calculator.generate_report(model_id=scenario_name, additional_data=additional_data, output_path=output_path)

        # Return the report dictionary
        return report

    @log_execution_time(logger, 'INFO')
    def save_results(
        self,
        output_path: Optional[PathType] = None
    ) -> Dict[str, PathType]:
        """
        Saves backtesting results to disk

        Args:
            output_path (Optional[PathType]): Path to save the results

        Returns:
            Dict[str, PathType]: Dictionary mapping scenario names to saved file paths
        """
        # Set output_path to provided value or _results_path
        output_path = output_path or self._results_path

        # Save results using _backtesting_framework.save_results with output_path
        filepaths = self._backtesting_framework.save_results(output_path=output_path)

        # Return the dictionary of file paths
        return filepaths

    @log_execution_time(logger, 'INFO')
    def clear_results(
        self,
        scenario_name: Optional[str] = None
    ) -> bool:
        """
        Clears stored backtesting results

        Args:
            scenario_name (Optional[str]): Name of the scenario to clear results for

        Returns:
            bool: True if cleared successfully, False otherwise
        """
        # Clear results using _backtesting_framework.clear_results with scenario_name
        success = self._backtesting_framework.clear_results(scenario_name=scenario_name)

        # Return True if operation was successful, False otherwise
        return success