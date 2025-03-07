"""
Module for calculating and analyzing performance metrics in the backtesting framework for ERCOT RTLMP spike prediction.

This module provides comprehensive functionality to evaluate model performance across different thresholds,
time periods, and market conditions.
"""

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss  # version 1.2+
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import datetime
import matplotlib.pyplot as plt  # version 3.7+

# Internal imports
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ThresholdValue, ModelType
from ..utils.statistics import calculate_binary_classification_metrics, calculate_probability_metrics
from ..inference.calibration import evaluate_calibration
from .scenario_definitions import MetricsConfig
from ..utils.logging import get_logger, log_execution_time
from ..utils.validation import validate_probability_values

# Set up logger
logger = get_logger(__name__)

# Default metrics to calculate
DEFAULT_CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
DEFAULT_PROBABILITY_METRICS = ["brier_score", "log_loss"]
DEFAULT_CALIBRATION_METRICS = ["expected_calibration_error", "maximum_calibration_error"]


@log_execution_time(logger, 'INFO')
def calculate_backtesting_metrics(
    predictions: DataFrameType,
    actuals: DataFrameType,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[ThresholdValue]] = None
) -> Dict[ThresholdValue, Dict[str, float]]:
    """
    Calculates performance metrics for backtesting results.
    
    Args:
        predictions: DataFrame containing predicted probabilities for each threshold
        actuals: DataFrame containing actual outcomes
        metrics: Optional list of metrics to calculate (default: uses DEFAULT_CLASSIFICATION_METRICS + DEFAULT_PROBABILITY_METRICS)
        thresholds: Optional list of threshold values to evaluate (default: extracts from predictions columns)
        
    Returns:
        Dictionary of metrics by threshold
    """
    # Validate inputs
    if not isinstance(predictions, pd.DataFrame) or not isinstance(actuals, pd.DataFrame):
        logger.error("Predictions and actuals must be pandas DataFrames")
        raise TypeError("Predictions and actuals must be pandas DataFrames")
    
    # If metrics is None, use default metrics
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS + DEFAULT_PROBABILITY_METRICS
    
    # If thresholds is None, try to extract from predictions columns
    if thresholds is None:
        # Assuming threshold values are in the column names of predictions
        threshold_cols = [col for col in predictions.columns if isinstance(col, (int, float))]
        if not threshold_cols:
            logger.error("No thresholds found in predictions columns and none provided")
            raise ValueError("No thresholds found in predictions columns and none provided")
        thresholds = threshold_cols
    
    # Initialize results dictionary
    results = {}
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Check if threshold exists in predictions
        if threshold not in predictions.columns:
            logger.warning(f"Threshold {threshold} not found in predictions, skipping")
            continue
        
        # Extract actual values (assuming they match the threshold values or have a consistent naming)
        if threshold in actuals.columns:
            y_true = actuals[threshold].values
        elif 'actual' in actuals.columns:
            y_true = actuals['actual'].values
        else:
            logger.warning(f"No actual values found for threshold {threshold}, skipping")
            continue
        
        # Extract predicted values
        y_pred = predictions[threshold].values
        
        # Validate that inputs are valid probabilities
        validate_probability_values({'prob': y_pred}, ['prob'])
        
        # Calculate classification metrics (using a threshold of 0.5 to convert probabilities to binary predictions)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        classification_metrics = calculate_binary_classification_metrics(y_true, y_pred_binary)
        
        # Calculate probability metrics
        probability_metrics = calculate_probability_metrics(y_true, y_pred)
        
        # Calculate calibration metrics
        calibration_metrics = evaluate_calibration(y_true, y_pred)
        
        # Combine all metrics
        threshold_metrics = {}
        threshold_metrics.update(classification_metrics)
        threshold_metrics.update(probability_metrics)
        for metric in DEFAULT_CALIBRATION_METRICS:
            if metric in calibration_metrics:
                threshold_metrics[metric] = calibration_metrics[metric]
        
        # Filter to only include requested metrics
        threshold_metrics = {k: v for k, v in threshold_metrics.items() if k in metrics}
        
        # Store in results dictionary
        results[threshold] = threshold_metrics
    
    return results


@log_execution_time(logger, 'INFO')
def calculate_metrics_over_time(
    predictions: DataFrameType,
    actuals: DataFrameType,
    time_column: str,
    time_grouping: str,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[ThresholdValue]] = None
) -> Dict[str, DataFrameType]:
    """
    Calculates performance metrics over different time periods.
    
    Args:
        predictions: DataFrame containing predicted probabilities for each threshold
        actuals: DataFrame containing actual outcomes
        time_column: Name of the column containing timestamps
        time_grouping: Time grouping to use (e.g., 'hour', 'day', 'month')
        metrics: Optional list of metrics to calculate
        thresholds: Optional list of threshold values to evaluate
        
    Returns:
        Dictionary of DataFrames with metrics by time period for each threshold
    """
    # Validate inputs
    if not isinstance(predictions, pd.DataFrame) or not isinstance(actuals, pd.DataFrame):
        logger.error("Predictions and actuals must be pandas DataFrames")
        raise TypeError("Predictions and actuals must be pandas DataFrames")
    
    if time_column not in predictions.columns or time_column not in actuals.columns:
        logger.error(f"Time column '{time_column}' not found in both predictions and actuals")
        raise ValueError(f"Time column '{time_column}' not found in both predictions and actuals")
    
    # If metrics is None, use default metrics
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS + DEFAULT_PROBABILITY_METRICS
    
    # If thresholds is None, try to extract from predictions columns
    if thresholds is None:
        # Assuming threshold values are in the column names of predictions
        threshold_cols = [col for col in predictions.columns if isinstance(col, (int, float))]
        if not threshold_cols:
            logger.error("No thresholds found in predictions columns and none provided")
            raise ValueError("No thresholds found in predictions columns and none provided")
        thresholds = threshold_cols
    
    # Initialize results dictionary
    results = {}
    
    # Extract time component based on time_grouping
    if time_grouping == 'hour':
        time_component = pd.to_datetime(predictions[time_column]).dt.hour
    elif time_grouping == 'day':
        time_component = pd.to_datetime(predictions[time_column]).dt.day
    elif time_grouping == 'month':
        time_component = pd.to_datetime(predictions[time_column]).dt.month
    elif time_grouping == 'year':
        time_component = pd.to_datetime(predictions[time_column]).dt.year
    elif time_grouping == 'dayofweek':
        time_component = pd.to_datetime(predictions[time_column]).dt.dayofweek
    elif time_grouping == 'hour_of_day':
        time_component = pd.to_datetime(predictions[time_column]).dt.hour
    else:
        logger.error(f"Unsupported time_grouping: {time_grouping}")
        raise ValueError(f"Unsupported time_grouping: {time_grouping}")
    
    # Calculate metrics for each threshold over time
    for threshold in thresholds:
        # Check if threshold exists in predictions
        if threshold not in predictions.columns:
            logger.warning(f"Threshold {threshold} not found in predictions, skipping")
            continue
        
        # Extract actual values (assuming they match the threshold values or have a consistent naming)
        if threshold in actuals.columns:
            y_true = actuals[threshold].values
        elif 'actual' in actuals.columns:
            y_true = actuals['actual'].values
        else:
            logger.warning(f"No actual values found for threshold {threshold}, skipping")
            continue
        
        # Extract predicted values
        y_pred = predictions[threshold].values
        
        # Combine into a DataFrame for grouping
        combined = pd.DataFrame({
            'time_component': time_component,
            'y_true': y_true,
            'y_pred': y_pred
        })
        
        # Group by time component
        grouped = combined.groupby('time_component')
        
        # Initialize a DataFrame to store metrics by time period
        metrics_by_time = []
        
        # Calculate metrics for each time group
        for time_value, group in grouped:
            # Extract true and predicted values
            group_y_true = group['y_true'].values
            group_y_pred = group['y_pred'].values
            
            # Skip if not enough data points
            if len(group_y_true) < 2:
                continue
            
            # Calculate binary predictions
            group_y_pred_binary = (group_y_pred >= 0.5).astype(int)
            
            # Calculate classification metrics
            try:
                classification_metrics = calculate_binary_classification_metrics(group_y_true, group_y_pred_binary)
            except Exception as e:
                logger.warning(f"Error calculating classification metrics for time {time_value}: {str(e)}")
                classification_metrics = {metric: np.nan for metric in DEFAULT_CLASSIFICATION_METRICS}
            
            # Calculate probability metrics
            try:
                probability_metrics = calculate_probability_metrics(group_y_true, group_y_pred)
            except Exception as e:
                logger.warning(f"Error calculating probability metrics for time {time_value}: {str(e)}")
                probability_metrics = {metric: np.nan for metric in DEFAULT_PROBABILITY_METRICS}
            
            # Combine metrics with time value
            time_metrics = {'time_value': time_value}
            time_metrics.update(classification_metrics)
            time_metrics.update(probability_metrics)
            
            # Append to list
            metrics_by_time.append(time_metrics)
        
        # Convert to DataFrame
        if metrics_by_time:
            time_metrics_df = pd.DataFrame(metrics_by_time).set_index('time_value')
            
            # Filter to only include requested metrics
            time_metrics_df = time_metrics_df.loc[:, [col for col in time_metrics_df.columns if col in metrics]]
            
            # Store in results dictionary
            results[threshold] = time_metrics_df
        else:
            logger.warning(f"No valid time periods found for threshold {threshold}")
            results[threshold] = pd.DataFrame()
    
    return results


def aggregate_metrics_across_thresholds(
    metrics_by_threshold: Dict[ThresholdValue, Dict[str, float]],
    metrics_to_aggregate: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregates metrics across multiple thresholds.
    
    Args:
        metrics_by_threshold: Dictionary mapping thresholds to metric dictionaries
        metrics_to_aggregate: Optional list of metrics to aggregate
        
    Returns:
        Dictionary of aggregated metrics with statistics
    """
    # If no metrics provided, use all metrics from the first threshold
    if not metrics_by_threshold:
        logger.warning("Empty metrics_by_threshold provided")
        return {}
    
    if metrics_to_aggregate is None:
        # Get all metric names from the first threshold
        first_threshold = next(iter(metrics_by_threshold.values()))
        metrics_to_aggregate = list(first_threshold.keys())
    
    # Initialize results dictionary
    results = {}
    
    # Process each metric
    for metric in metrics_to_aggregate:
        # Extract values for this metric across all thresholds
        metric_values = []
        for threshold, metrics_dict in metrics_by_threshold.items():
            if metric in metrics_dict:
                value = metrics_dict[metric]
                # Ensure value is a number
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_values.append(value)
        
        # Skip if no valid values
        if not metric_values:
            logger.warning(f"No valid values found for metric: {metric}")
            continue
        
        # Calculate statistics
        stats = {
            'mean': np.mean(metric_values),
            'median': np.median(metric_values),
            'min': np.min(metric_values),
            'max': np.max(metric_values),
            'std': np.std(metric_values) if len(metric_values) > 1 else 0.0
        }
        
        # Store in results
        results[metric] = stats
    
    return results


def aggregate_metrics_across_nodes(
    metrics_by_node: Dict[str, Dict[ThresholdValue, Dict[str, float]]],
    metrics_to_aggregate: Optional[List[str]] = None
) -> Dict[ThresholdValue, Dict[str, Dict[str, float]]]:
    """
    Aggregates metrics across multiple nodes.
    
    Args:
        metrics_by_node: Dictionary mapping node IDs to metric dictionaries by threshold
        metrics_to_aggregate: Optional list of metrics to aggregate
        
    Returns:
        Dictionary of aggregated metrics by threshold
    """
    # If no metrics provided, return empty
    if not metrics_by_node:
        logger.warning("Empty metrics_by_node provided")
        return {}
    
    # Get all thresholds across all nodes
    all_thresholds = set()
    for node_metrics in metrics_by_node.values():
        all_thresholds.update(node_metrics.keys())
    
    # If no metrics to aggregate specified, get all metrics from first node and threshold
    if metrics_to_aggregate is None:
        first_node = next(iter(metrics_by_node.values()))
        if first_node:
            first_threshold = next(iter(first_node.values()))
            metrics_to_aggregate = list(first_threshold.keys())
        else:
            metrics_to_aggregate = []
    
    # Initialize results dictionary
    results = {}
    
    # Process each threshold
    for threshold in all_thresholds:
        threshold_results = {}
        
        # Process each metric
        for metric in metrics_to_aggregate:
            # Extract values for this metric across all nodes
            metric_values = []
            for node, node_metrics in metrics_by_node.items():
                if threshold in node_metrics and metric in node_metrics[threshold]:
                    value = node_metrics[threshold][metric]
                    # Ensure value is a number
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metric_values.append(value)
            
            # Skip if no valid values
            if not metric_values:
                logger.warning(f"No valid values found for threshold {threshold}, metric: {metric}")
                continue
            
            # Calculate statistics
            stats = {
                'mean': np.mean(metric_values),
                'median': np.median(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values),
                'std': np.std(metric_values) if len(metric_values) > 1 else 0.0
            }
            
            # Store in results
            threshold_results[metric] = stats
        
        # Store threshold results
        results[threshold] = threshold_results
    
    return results


def aggregate_metrics_across_windows(
    metrics_by_window: List[Dict[ThresholdValue, Dict[str, float]]],
    metrics_to_aggregate: Optional[List[str]] = None
) -> Dict[ThresholdValue, Dict[str, Dict[str, float]]]:
    """
    Aggregates metrics across multiple time windows.
    
    Args:
        metrics_by_window: List of dictionaries mapping thresholds to metric dictionaries
        metrics_to_aggregate: Optional list of metrics to aggregate
        
    Returns:
        Dictionary of aggregated metrics by threshold
    """
    # If no metrics provided, return empty
    if not metrics_by_window:
        logger.warning("Empty metrics_by_window provided")
        return {}
    
    # Get all thresholds across all windows
    all_thresholds = set()
    for window_metrics in metrics_by_window:
        all_thresholds.update(window_metrics.keys())
    
    # If no metrics to aggregate specified, get all metrics from first window and threshold
    if metrics_to_aggregate is None:
        first_window = metrics_by_window[0] if metrics_by_window else {}
        if first_window:
            first_threshold = next(iter(first_window.values()))
            metrics_to_aggregate = list(first_threshold.keys())
        else:
            metrics_to_aggregate = []
    
    # Initialize results dictionary
    results = {}
    
    # Process each threshold
    for threshold in all_thresholds:
        threshold_results = {}
        
        # Process each metric
        for metric in metrics_to_aggregate:
            # Extract values for this metric across all windows
            metric_values = []
            for window_metrics in metrics_by_window:
                if threshold in window_metrics and metric in window_metrics[threshold]:
                    value = window_metrics[threshold][metric]
                    # Ensure value is a number
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metric_values.append(value)
            
            # Skip if no valid values
            if not metric_values:
                logger.warning(f"No valid values found for threshold {threshold}, metric: {metric}")
                continue
            
            # Calculate statistics
            stats = {
                'mean': np.mean(metric_values),
                'median': np.median(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values),
                'std': np.std(metric_values) if len(metric_values) > 1 else 0.0
            }
            
            # Store in results
            threshold_results[metric] = stats
        
        # Store threshold results
        results[threshold] = threshold_results
    
    return results


def compare_metrics_between_models(
    metrics_by_model: Dict[str, Dict[ThresholdValue, Dict[str, float]]],
    metrics_to_compare: Optional[List[str]] = None,
    thresholds_to_compare: Optional[List[ThresholdValue]] = None
) -> Dict[str, DataFrameType]:
    """
    Compares performance metrics between multiple models.
    
    Args:
        metrics_by_model: Dictionary mapping model IDs to metric dictionaries by threshold
        metrics_to_compare: Optional list of metrics to compare
        thresholds_to_compare: Optional list of thresholds to compare
        
    Returns:
        Dictionary of DataFrames comparing models by threshold
    """
    # If no metrics provided, return empty
    if not metrics_by_model:
        logger.warning("Empty metrics_by_model provided")
        return {}
    
    # Get all thresholds across all models
    all_thresholds = set()
    for model_metrics in metrics_by_model.values():
        all_thresholds.update(model_metrics.keys())
    
    # If thresholds_to_compare not provided, use all thresholds
    if thresholds_to_compare is None:
        thresholds_to_compare = list(all_thresholds)
    
    # Get common metrics across all models and thresholds
    common_metrics = set()
    first_iteration = True
    
    for model_id, model_metrics in metrics_by_model.items():
        for threshold in thresholds_to_compare:
            if threshold in model_metrics:
                threshold_metrics = set(model_metrics[threshold].keys())
                if first_iteration:
                    common_metrics = threshold_metrics
                    first_iteration = False
                else:
                    common_metrics = common_metrics.intersection(threshold_metrics)
    
    # If metrics_to_compare not provided, use common metrics
    if metrics_to_compare is None:
        metrics_to_compare = list(common_metrics)
    else:
        # Filter to only include metrics that are common across all models
        metrics_to_compare = [m for m in metrics_to_compare if m in common_metrics]
    
    # Initialize results dictionary
    results = {}
    
    # Process each threshold
    for threshold in thresholds_to_compare:
        # Create a DataFrame with models as rows and metrics as columns
        comparison_data = []
        
        for model_id, model_metrics in metrics_by_model.items():
            if threshold in model_metrics:
                row_data = {'model_id': model_id}
                for metric in metrics_to_compare:
                    if metric in model_metrics[threshold]:
                        row_data[metric] = model_metrics[threshold][metric]
                    else:
                        row_data[metric] = np.nan
                comparison_data.append(row_data)
        
        # Convert to DataFrame
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data).set_index('model_id')
            results[threshold] = comparison_df
        else:
            logger.warning(f"No comparison data available for threshold {threshold}")
            results[threshold] = pd.DataFrame()
    
    return results


def plot_metrics_by_threshold(
    metrics_by_threshold: Dict[ThresholdValue, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates plots of metrics across different thresholds.
    
    Args:
        metrics_by_threshold: Dictionary mapping thresholds to metric dictionaries
        metrics_to_plot: Optional list of metrics to plot
        fig: Optional figure object to plot on
        ax: Optional axes object to plot on
        
    Returns:
        Figure and axes objects
    """
    # If no metrics provided, return empty plot
    if not metrics_by_threshold:
        logger.warning("Empty metrics_by_threshold provided")
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        return fig, ax
    
    # If metrics_to_plot not provided, use all metrics from the first threshold
    if metrics_to_plot is None:
        first_threshold = next(iter(metrics_by_threshold.values()))
        metrics_to_plot = list(first_threshold.keys())
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract thresholds and sort them
    thresholds = sorted(metrics_by_threshold.keys())
    
    # Plot each metric
    for metric in metrics_to_plot:
        # Extract values for this metric across all thresholds
        metric_values = []
        valid_thresholds = []
        
        for threshold in thresholds:
            if metric in metrics_by_threshold[threshold]:
                value = metrics_by_threshold[threshold][metric]
                # Ensure value is a number
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_values.append(value)
                    valid_thresholds.append(threshold)
        
        # Skip if no valid values
        if not metric_values:
            logger.warning(f"No valid values found for metric: {metric}")
            continue
        
        # Plot the metric
        ax.plot(valid_thresholds, metric_values, 'o-', label=metric)
    
    # Set axis labels and title
    ax.set_xlabel('Threshold Value')
    ax.set_ylabel('Metric Value')
    ax.set_title('Metrics by Threshold')
    ax.grid(True)
    ax.legend()
    
    return fig, ax


def plot_metrics_over_time(
    metrics_over_time: Dict[str, DataFrameType],
    metrics_to_plot: Optional[List[str]] = None,
    thresholds_to_plot: Optional[List[ThresholdValue]] = None,
    plot_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """
    Generates plots of metrics over time.
    
    Args:
        metrics_over_time: Dictionary mapping thresholds to DataFrames with time-based metrics
        metrics_to_plot: Optional list of metrics to plot
        thresholds_to_plot: Optional list of thresholds to plot
        plot_config: Optional configuration for plots
        
    Returns:
        Dictionary of figure and axes objects by metric
    """
    # If no metrics provided, return empty
    if not metrics_over_time:
        logger.warning("Empty metrics_over_time provided")
        return {}
    
    # Get default plot configuration
    default_config = {
        'figsize': (12, 6),
        'grid': True,
        'title_template': '{metric} over Time',
        'xlabel': 'Time Period',
        'ylabel': '{metric} Value',
        'marker': 'o-',
        'legend_loc': 'best'
    }
    
    # Update with user-provided config
    if plot_config:
        default_config.update(plot_config)
    
    # If thresholds_to_plot not provided, use all thresholds
    if thresholds_to_plot is None:
        thresholds_to_plot = list(metrics_over_time.keys())
    
    # Get common metrics across all thresholds
    common_metrics = set()
    first_iteration = True
    
    for threshold, df in metrics_over_time.items():
        if threshold in thresholds_to_plot and not df.empty:
            if first_iteration:
                common_metrics = set(df.columns)
                first_iteration = False
            else:
                common_metrics = common_metrics.intersection(df.columns)
    
    # If metrics_to_plot not provided, use common metrics
    if metrics_to_plot is None:
        metrics_to_plot = list(common_metrics)
    else:
        # Filter to only include metrics that are common
        metrics_to_plot = [m for m in metrics_to_plot if m in common_metrics]
    
    # Initialize results dictionary
    results = {}
    
    # Create a plot for each metric
    for metric in metrics_to_plot:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=default_config['figsize'])
        
        # Plot each threshold
        for threshold in thresholds_to_plot:
            if threshold in metrics_over_time and not metrics_over_time[threshold].empty:
                df = metrics_over_time[threshold]
                if metric in df.columns:
                    # Plot the metric over time
                    ax.plot(
                        df.index, 
                        df[metric], 
                        default_config['marker'], 
                        label=f'Threshold {threshold}'
                    )
        
        # Set axis labels and title
        ax.set_xlabel(default_config['xlabel'])
        ax.set_ylabel(default_config['ylabel'].format(metric=metric))
        ax.set_title(default_config['title_template'].format(metric=metric))
        
        if default_config['grid']:
            ax.grid(True)
            
        ax.legend(loc=default_config['legend_loc'])
        
        # Store in results
        results[metric] = (fig, ax)
    
    return results


@log_execution_time(logger, 'INFO')
def generate_metrics_report(
    metrics_by_threshold: Dict[ThresholdValue, Dict[str, float]],
    metrics_over_time: Optional[Dict[str, DataFrameType]] = None,
    additional_data: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a comprehensive report of performance metrics.
    
    Args:
        metrics_by_threshold: Dictionary mapping thresholds to metric dictionaries
        metrics_over_time: Optional dictionary of time-based metrics
        additional_data: Optional additional data to include in the report
        output_path: Optional path to save the report JSON
        
    Returns:
        Dictionary with report data
    """
    import json
    
    # Initialize report dictionary
    report = {
        'metrics_by_threshold': metrics_by_threshold,
        'timestamp': datetime.datetime.now().isoformat(),
        'aggregate_metrics': aggregate_metrics_across_thresholds(metrics_by_threshold)
    }
    
    # Add metrics_over_time if provided
    if metrics_over_time:
        # Convert DataFrames to serializable format
        serialized_metrics_over_time = {}
        for threshold, df in metrics_over_time.items():
            if not df.empty:
                # Convert DataFrame to dictionary with index as strings
                df_dict = df.to_dict(orient='index')
                serialized_dict = {str(idx): values for idx, values in df_dict.items()}
                serialized_metrics_over_time[str(threshold)] = serialized_dict
        
        report['metrics_over_time'] = serialized_metrics_over_time
    
    # Add additional data if provided
    if additional_data:
        report.update(additional_data)
    
    # Generate summary tables
    summary_tables = {}
    
    # Summary by threshold
    threshold_summary = []
    for threshold, metrics in sorted(metrics_by_threshold.items()):
        row = {'threshold': threshold}
        row.update(metrics)
        threshold_summary.append(row)
    
    summary_tables['threshold_summary'] = threshold_summary
    
    # Add summary tables to report
    report['summary_tables'] = summary_tables
    
    # Save to JSON if output_path provided
    if output_path:
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving report to {output_path}: {str(e)}")
    
    return report


class BacktestingMetricsCalculator:
    """
    Class for calculating and analyzing performance metrics in backtesting.
    """
    
    def __init__(self, metrics_config: Optional[MetricsConfig] = None):
        """
        Initializes the BacktestingMetricsCalculator with configuration.
        
        Args:
            metrics_config: Optional metrics configuration
        """
        # Initialize with default configuration if none provided
        self._metrics_config = metrics_config or MetricsConfig()
        
        # Initialize storage for metrics
        self._metrics_by_model = {}
        self._metrics_over_time = {}
    
    @log_execution_time(logger, 'INFO')
    def calculate_all_metrics(
        self,
        predictions: DataFrameType,
        actuals: DataFrameType,
        model_id: Optional[str] = None,
        thresholds: Optional[List[ThresholdValue]] = None
    ) -> Dict[ThresholdValue, Dict[str, float]]:
        """
        Calculates all metrics for predictions and actuals.
        
        Args:
            predictions: DataFrame containing predicted probabilities
            actuals: DataFrame containing actual outcomes
            model_id: Optional model identifier to store results
            thresholds: Optional list of thresholds to evaluate
            
        Returns:
            Dictionary of metrics by threshold
        """
        # Get metrics list from configuration
        metrics = self._metrics_config.metrics
        
        # Calculate metrics
        results = calculate_backtesting_metrics(
            predictions=predictions,
            actuals=actuals,
            metrics=metrics,
            thresholds=thresholds
        )
        
        # Store results if model_id provided
        if model_id:
            self._metrics_by_model[model_id] = results
        
        return results
    
    @log_execution_time(logger, 'INFO')
    def calculate_metrics_over_time(
        self,
        predictions: DataFrameType,
        actuals: DataFrameType,
        time_column: str,
        time_grouping: str,
        model_id: Optional[str] = None,
        thresholds: Optional[List[ThresholdValue]] = None
    ) -> Dict[str, DataFrameType]:
        """
        Calculates metrics over different time periods.
        
        Args:
            predictions: DataFrame containing predicted probabilities
            actuals: DataFrame containing actual outcomes
            time_column: Name of the column containing timestamps
            time_grouping: Time grouping to use (e.g., 'hour', 'day', 'month')
            model_id: Optional model identifier to store results
            thresholds: Optional list of thresholds to evaluate
            
        Returns:
            Dictionary of DataFrames with metrics by time period
        """
        # Get metrics list from configuration
        metrics = self._metrics_config.metrics
        
        # Calculate metrics over time
        results = calculate_metrics_over_time(
            predictions=predictions,
            actuals=actuals,
            time_column=time_column,
            time_grouping=time_grouping,
            metrics=metrics,
            thresholds=thresholds
        )
        
        # Store results if model_id provided
        if model_id:
            self._metrics_over_time[model_id] = results
        
        return results
    
    def compare_models(
        self,
        model_ids: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[List[ThresholdValue]] = None
    ) -> Dict[str, DataFrameType]:
        """
        Compares metrics between multiple models.
        
        Args:
            model_ids: Optional list of model IDs to compare (default: all models)
            metrics: Optional list of metrics to compare
            thresholds: Optional list of thresholds to compare
            
        Returns:
            Dictionary of comparison DataFrames by threshold
        """
        # If model_ids not provided, use all models
        if model_ids is None:
            model_ids = list(self._metrics_by_model.keys())
        
        # Filter metrics_by_model to only include specified models
        filtered_metrics = {
            model_id: self._metrics_by_model[model_id]
            for model_id in model_ids
            if model_id in self._metrics_by_model
        }
        
        # Compare metrics
        return compare_metrics_between_models(
            metrics_by_model=filtered_metrics,
            metrics_to_compare=metrics,
            thresholds_to_compare=thresholds
        )
    
    def plot_metrics_by_threshold(
        self,
        model_id: str,
        metrics: Optional[List[str]] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Generates plots of metrics across thresholds.
        
        Args:
            model_id: Model ID to plot metrics for
            metrics: Optional list of metrics to plot
            fig: Optional figure object to plot on
            ax: Optional axes object to plot on
            
        Returns:
            Figure and axes objects
        """
        # Check if model_id exists
        if model_id not in self._metrics_by_model:
            logger.error(f"Model ID {model_id} not found in metrics")
            raise ValueError(f"Model ID {model_id} not found in metrics")
        
        # Get metrics by threshold for the model
        metrics_by_threshold = self._metrics_by_model[model_id]
        
        # Plot metrics
        return plot_metrics_by_threshold(
            metrics_by_threshold=metrics_by_threshold,
            metrics_to_plot=metrics,
            fig=fig,
            ax=ax
        )
    
    def plot_metrics_over_time(
        self,
        model_id: str,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[List[ThresholdValue]] = None,
        plot_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
        """
        Generates plots of metrics over time.
        
        Args:
            model_id: Model ID to plot metrics for
            metrics: Optional list of metrics to plot
            thresholds: Optional list of thresholds to plot
            plot_config: Optional configuration for plots
            
        Returns:
            Dictionary of figure and axes objects by metric
        """
        # Check if model_id exists
        if model_id not in self._metrics_over_time:
            logger.error(f"Model ID {model_id} not found in time-based metrics")
            raise ValueError(f"Model ID {model_id} not found in time-based metrics")
        
        # Get metrics over time for the model
        metrics_over_time = self._metrics_over_time[model_id]
        
        # Plot metrics
        return plot_metrics_over_time(
            metrics_over_time=metrics_over_time,
            metrics_to_plot=metrics,
            thresholds_to_plot=thresholds,
            plot_config=plot_config
        )
    
    @log_execution_time(logger, 'INFO')
    def generate_report(
        self,
        model_id: str,
        additional_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive metrics report.
        
        Args:
            model_id: Model ID to generate report for
            additional_data: Optional additional data to include
            output_path: Optional path to save the report
            
        Returns:
            Report dictionary
        """
        # Check if model_id exists
        if model_id not in self._metrics_by_model:
            logger.error(f"Model ID {model_id} not found in metrics")
            raise ValueError(f"Model ID {model_id} not found in metrics")
        
        # Get metrics by threshold for the model
        metrics_by_threshold = self._metrics_by_model[model_id]
        
        # Get metrics over time if available
        metrics_over_time = self._metrics_over_time.get(model_id)
        
        # Generate report
        return generate_metrics_report(
            metrics_by_threshold=metrics_by_threshold,
            metrics_over_time=metrics_over_time,
            additional_data=additional_data,
            output_path=output_path
        )
    
    def get_metric_summary(
        self,
        model_id: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[List[ThresholdValue]] = None
    ) -> DataFrameType:
        """
        Returns a summary of key metrics.
        
        Args:
            model_id: Optional model ID (default: aggregate across all models)
            metrics: Optional list of metrics to include
            thresholds: Optional list of thresholds to include
            
        Returns:
            DataFrame with metric summary
        """
        # If metrics not provided, use key metrics from config
        if metrics is None:
            metrics = self._metrics_config.metrics
        
        # If model_id provided, get metrics for that model
        if model_id:
            if model_id not in self._metrics_by_model:
                logger.error(f"Model ID {model_id} not found in metrics")
                raise ValueError(f"Model ID {model_id} not found in metrics")
            
            metrics_by_threshold = self._metrics_by_model[model_id]
        else:
            # Aggregate across all models
            all_metrics = []
            for model_metrics in self._metrics_by_model.values():
                all_metrics.append(model_metrics)
            
            # Aggregate metrics
            if all_metrics:
                metrics_by_threshold = {}
                all_thresholds = set()
                for model_metrics in all_metrics:
                    all_thresholds.update(model_metrics.keys())
                
                for threshold in all_thresholds:
                    threshold_metrics = []
                    for model_metrics in all_metrics:
                        if threshold in model_metrics:
                            threshold_metrics.append(model_metrics[threshold])
                    
                    # Average metrics across models
                    if threshold_metrics:
                        metrics_by_threshold[threshold] = {}
                        for metric in metrics:
                            values = [tm[metric] for tm in threshold_metrics if metric in tm]
                            if values:
                                metrics_by_threshold[threshold][metric] = np.mean(values)
            else:
                logger.warning("No metrics available for summary")
                return pd.DataFrame()
        
        # If thresholds not provided, use all thresholds
        if thresholds is None:
            thresholds = list(metrics_by_threshold.keys())
        
        # Create summary DataFrame
        summary_data = []
        for threshold in thresholds:
            if threshold in metrics_by_threshold:
                row = {'threshold': threshold}
                for metric in metrics:
                    if metric in metrics_by_threshold[threshold]:
                        row[metric] = metrics_by_threshold[threshold][metric]
                summary_data.append(row)
        
        # Convert to DataFrame
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            if 'threshold' in summary_df.columns:
                summary_df = summary_df.set_index('threshold')
            return summary_df
        else:
            logger.warning("No summary data available")
            return pd.DataFrame()
    
    def set_metrics_config(self, metrics_config: MetricsConfig) -> None:
        """
        Sets the metrics configuration.
        
        Args:
            metrics_config: Metrics configuration
        """
        # Validate metrics_config
        metrics_config.validate()
        
        # Set new configuration
        self._metrics_config = metrics_config
    
    def get_metrics_config(self) -> MetricsConfig:
        """
        Returns the current metrics configuration.
        
        Returns:
            Current metrics configuration
        """
        return self._metrics_config
    
    def get_all_metrics(self, model_id: Optional[str] = None) -> Dict[str, Dict[ThresholdValue, Dict[str, float]]]:
        """
        Returns all stored metrics.
        
        Args:
            model_id: Optional model ID to return metrics for
            
        Returns:
            Dictionary of metrics by model and threshold
        """
        if model_id:
            if model_id in self._metrics_by_model:
                return {model_id: self._metrics_by_model[model_id]}
            else:
                logger.warning(f"Model ID {model_id} not found in metrics")
                return {}
        else:
            return self._metrics_by_model
    
    def clear_metrics(self, model_id: Optional[str] = None) -> None:
        """
        Clears stored metrics.
        
        Args:
            model_id: Optional model ID to clear metrics for (default: all models)
        """
        if model_id:
            if model_id in self._metrics_by_model:
                del self._metrics_by_model[model_id]
            
            if model_id in self._metrics_over_time:
                del self._metrics_over_time[model_id]
        else:
            self._metrics_by_model = {}
            self._metrics_over_time = {}