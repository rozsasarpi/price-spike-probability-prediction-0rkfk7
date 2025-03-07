"""
Implements visualization functions for model performance metrics in the ERCOT RTLMP spike prediction system.

This module provides specialized plotting capabilities for visualizing model evaluation metrics
such as ROC curves, precision-recall curves, calibration curves, confusion matrices, 
and threshold sensitivity analysis to help assess model quality and performance.
"""

# Standard library imports
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import io
import base64

# External libraries
import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
import matplotlib  # version 3.7+
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns  # version 0.12+
import plotly  # version 5.14+
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics  # version 1.2+

# Internal imports
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, PathType, ModelType
from ..utils.logging import get_logger
from ..utils.error_handling import handle_errors, VisualizationError
from ..data.storage.model_registry import ModelRegistry
from ..models.evaluation import (
    calculate_roc_curve, calculate_precision_recall_curve, evaluate_calibration,
    calculate_confusion_matrix, calculate_threshold_metrics
)

# Set up logger
logger = get_logger(__name__)

# Default values for visualizations
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 100
DEFAULT_CMAP = 'viridis'

# Color mapping for consistent visualization
METRIC_COLORS = {
    'accuracy': '#1f77b4',   # blue
    'precision': '#ff7f0e',  # orange
    'recall': '#2ca02c',     # green
    'f1': '#d62728',         # red
    'auc': '#9467bd',        # purple
    'brier_score': '#8c564b' # brown
}

# Color mapping for threshold values
THRESHOLD_COLORS = {
    '50': '#1f77b4',   # blue
    '100': '#ff7f0e',  # orange
    '200': '#d62728',  # red
    '300': '#9467bd',  # purple
    '400': '#8c564b'   # brown
}


@handle_errors(logger, VisualizationError)
def plot_roc_curve(
    y_true: SeriesType,
    y_prob: ArrayType,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a ROC curve plot for model evaluation.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        figsize: Optional figure size tuple (width, height)
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        
    Returns:
        Matplotlib figure and axes objects with the ROC curve plot
    """
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    # Calculate ROC curve data
    fpr, tpr, auc_value = calculate_roc_curve(y_true, y_prob)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_value:.3f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier')
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic (AUC = {auc_value:.3f})')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_precision_recall_curve(
    y_true: SeriesType,
    y_prob: ArrayType,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a precision-recall curve plot for model evaluation.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        figsize: Optional figure size tuple (width, height)
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        
    Returns:
        Matplotlib figure and axes objects with the precision-recall curve plot
    """
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    # Calculate precision-recall curve data
    precision, recall, avg_precision = calculate_precision_recall_curve(y_true, y_prob)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (Average Precision = {avg_precision:.3f})')
    ax.legend(loc='lower left')
    ax.grid(True)
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_calibration_curve(
    y_true: SeriesType,
    y_prob: ArrayType,
    n_bins: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a calibration curve (reliability diagram) for model evaluation.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for the calibration curve
        figsize: Optional figure size tuple (width, height)
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        
    Returns:
        Matplotlib figure and axes objects with the calibration curve plot
    """
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    if n_bins is None:
        n_bins = 10
    
    # Get calibration data
    calibration_data = evaluate_calibration(y_true, y_prob, n_bins)
    prob_true = np.array(calibration_data['calibration_curve']['prob_true'])
    prob_pred = np.array(calibration_data['calibration_curve']['prob_pred'])
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 's-', label='Calibration curve')
    
    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    
    # Set labels and title
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    
    # Add metrics to the title
    title = (f"Calibration Curve\nBrier Score: {calibration_data['brier_score']:.4f}, "
             f"ECE: {calibration_data['expected_calibration_error']:.4f}")
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid for readability
    ax.grid(True)
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_confusion_matrix(
    y_true: SeriesType,
    y_pred: ArrayType,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: Optional[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a confusion matrix heatmap for model evaluation.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        figsize: Optional figure size tuple (width, height)
        cmap: Colormap for the heatmap
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        
    Returns:
        Matplotlib figure and axes objects with the confusion matrix plot
    """
    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Calculate confusion matrix
    cm_dict = calculate_confusion_matrix(y_true, y_pred)
    cm = np.array([
        [cm_dict['true_negatives'], cm_dict['false_positives']],
        [cm_dict['false_negatives'], cm_dict['true_positives']]
    ])
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap or DEFAULT_CMAP,
        square=True,
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    
    # Set tick labels
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    # Calculate accuracy, precision, and recall
    accuracy = (cm_dict['true_positives'] + cm_dict['true_negatives']) / (
        cm_dict['true_positives'] + cm_dict['true_negatives'] + 
        cm_dict['false_positives'] + cm_dict['false_negatives']
    )
    precision = cm_dict['true_positives'] / (cm_dict['true_positives'] + cm_dict['false_positives']) if (cm_dict['true_positives'] + cm_dict['false_positives']) > 0 else 0
    recall = cm_dict['true_positives'] / (cm_dict['true_positives'] + cm_dict['false_negatives']) if (cm_dict['true_positives'] + cm_dict['false_negatives']) > 0 else 0
    
    # Update title with metrics
    ax.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_threshold_sensitivity(
    y_true: SeriesType,
    y_prob: ArrayType,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a plot showing how metrics change with different probability thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metrics: List of metrics to plot
        thresholds: List of threshold values to evaluate
        figsize: Optional figure size tuple (width, height)
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        
    Returns:
        Matplotlib figure and axes objects with the threshold sensitivity plot
    """
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    # Set default metrics and thresholds if not provided
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    
    # Calculate metrics at different thresholds
    metrics_df = calculate_threshold_metrics(y_true, y_prob, thresholds)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Plot each metric
    for metric in metrics:
        if metric in metrics_df.columns:
            ax.plot(
                metrics_df['threshold'], 
                metrics_df[metric], 
                label=metric,
                color=METRIC_COLORS.get(metric, None)
            )
    
    # Add vertical line at threshold 0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', label='Threshold = 0.5')
    
    # Set labels and title
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Threshold Sensitivity Analysis')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_metric_comparison(
    model_metrics: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a bar chart comparing multiple metrics across different models.
    
    Args:
        model_metrics: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to compare
        figsize: Optional figure size tuple (width, height)
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        
    Returns:
        Matplotlib figure and axes objects with the metric comparison plot
    """
    # Validate inputs
    if not model_metrics:
        raise VisualizationError("model_metrics dictionary cannot be empty")
    
    # Get common metrics across all models if not specified
    if metrics is None:
        metrics = []
        for model_metric in model_metrics.values():
            metrics.extend(model_metric.keys())
        metrics = list(set(metrics))
    
    # Create DataFrame for plotting
    data = []
    for model_name, model_metric in model_metrics.items():
        row = {'model': model_name}
        row.update({metric: model_metric.get(metric, float('nan')) for metric in metrics})
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Plot grouped bar chart
    df_plot = df.set_index('model')
    df_plot[metrics].plot(kind='bar', ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Performance Comparison')
    ax.legend(title='Metric')
    ax.grid(True, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_temporal_performance(
    temporal_metrics: DataFrameType,
    time_column: str,
    metrics: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a plot showing how model performance varies over time.
    
    Args:
        temporal_metrics: DataFrame with time-based metrics
        time_column: Name of the column containing time values
        metrics: List of metrics to plot
        figsize: Optional figure size tuple (width, height)
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        
    Returns:
        Matplotlib figure and axes objects with the temporal performance plot
    """
    # Validate inputs
    if time_column not in temporal_metrics.columns:
        raise VisualizationError(f"Time column '{time_column}' not found in DataFrame")
    
    # If metrics not specified, use all columns except time_column
    if metrics is None:
        metrics = [col for col in temporal_metrics.columns if col != time_column]
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Plot each metric over time
    for metric in metrics:
        if metric in temporal_metrics.columns:
            ax.plot(
                temporal_metrics[time_column], 
                temporal_metrics[metric], 
                label=metric,
                color=METRIC_COLORS.get(metric, None)
            )
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Metric Value')
    ax.set_title('Temporal Performance Analysis')
    ax.legend(loc='best')
    ax.grid(True)
    
    # Format x-axis based on data type
    if pd.api.types.is_datetime64_any_dtype(temporal_metrics[time_column]):
        fig.autofmt_xdate()
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def create_interactive_roc_curve(
    y_true: SeriesType,
    y_prob: ArrayType
) -> go.Figure:
    """
    Creates an interactive ROC curve plot using Plotly.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Plotly figure object with the interactive ROC curve plot
    """
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    # Calculate ROC curve data
    fpr, tpr, auc_value = calculate_roc_curve(y_true, y_prob)
    
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {auc_value:.3f})',
            line=dict(color='blue', width=2),
            hovertemplate='False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<extra></extra>'
        )
    )
    
    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random classifier',
            line=dict(color='gray', width=2, dash='dash'),
            hoverinfo='skip'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'Receiver Operating Characteristic (AUC = {auc_value:.3f})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        hovermode='closest',
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1.05], scaleanchor='x', scaleratio=1),
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_interactive_precision_recall_curve(
    y_true: SeriesType,
    y_prob: ArrayType
) -> go.Figure:
    """
    Creates an interactive precision-recall curve plot using Plotly.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Plotly figure object with the interactive precision-recall curve plot
    """
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    # Calculate precision-recall curve data
    precision, recall, avg_precision = calculate_precision_recall_curve(y_true, y_prob)
    
    # Create figure
    fig = go.Figure()
    
    # Add precision-recall curve
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR curve (AP = {avg_precision:.3f})',
            line=dict(color='orange', width=2),
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'Precision-Recall Curve (Average Precision = {avg_precision:.3f})',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.5)'),
        hovermode='closest',
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1.05]),
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_interactive_calibration_curve(
    y_true: SeriesType,
    y_prob: ArrayType,
    n_bins: Optional[int] = None
) -> go.Figure:
    """
    Creates an interactive calibration curve plot using Plotly.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for the calibration curve
        
    Returns:
        Plotly figure object with the interactive calibration curve plot
    """
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    if n_bins is None:
        n_bins = 10
    
    # Get calibration data
    calibration_data = evaluate_calibration(y_true, y_prob, n_bins)
    prob_true = np.array(calibration_data['calibration_curve']['prob_true'])
    prob_pred = np.array(calibration_data['calibration_curve']['prob_pred'])
    
    # Create figure
    fig = go.Figure()
    
    # Add calibration curve
    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name='Calibration curve',
            line=dict(color='green', width=2),
            marker=dict(size=8),
            hovertemplate='Predicted Prob: %{x:.3f}<br>Observed Freq: %{y:.3f}<extra></extra>'
        )
    )
    
    # Add perfectly calibrated line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfectly calibrated',
            line=dict(color='gray', width=2, dash='dash'),
            hoverinfo='skip'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'Calibration Curve<br>Brier Score: {calibration_data["brier_score"]:.4f}, ECE: {calibration_data["expected_calibration_error"]:.4f}',
        xaxis_title='Mean predicted probability',
        yaxis_title='Fraction of positives',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        hovermode='closest',
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_interactive_confusion_matrix(
    y_true: SeriesType,
    y_pred: ArrayType
) -> go.Figure:
    """
    Creates an interactive confusion matrix heatmap using Plotly.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        
    Returns:
        Plotly figure object with the interactive confusion matrix plot
    """
    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Calculate confusion matrix
    cm_dict = calculate_confusion_matrix(y_true, y_pred)
    cm = np.array([
        [cm_dict['true_negatives'], cm_dict['false_positives']],
        [cm_dict['false_negatives'], cm_dict['true_positives']]
    ])
    
    # Calculate metrics
    accuracy = (cm_dict['true_positives'] + cm_dict['true_negatives']) / (
        cm_dict['true_positives'] + cm_dict['true_negatives'] + 
        cm_dict['false_positives'] + cm_dict['false_negatives']
    )
    precision = cm_dict['true_positives'] / (cm_dict['true_positives'] + cm_dict['false_positives']) if (cm_dict['true_positives'] + cm_dict['false_positives']) > 0 else 0
    recall = cm_dict['true_positives'] / (cm_dict['true_positives'] + cm_dict['false_negatives']) if (cm_dict['true_positives'] + cm_dict['false_negatives']) > 0 else 0
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            colorscale='Viridis',
            showscale=False,
            text=[[str(cm[i][j]) for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'Confusion Matrix<br>Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}',
        xaxis_title='Predicted label',
        yaxis_title='True label',
        xaxis=dict(type='category'),
        yaxis=dict(type='category'),
        width=600,
        height=600,
        template='plotly_white'
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_interactive_threshold_sensitivity(
    y_true: SeriesType,
    y_prob: ArrayType,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None
) -> go.Figure:
    """
    Creates an interactive threshold sensitivity plot using Plotly.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metrics: List of metrics to plot
        thresholds: List of threshold values to evaluate
        
    Returns:
        Plotly figure object with the interactive threshold sensitivity plot
    """
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise VisualizationError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    
    # Set default metrics and thresholds if not provided
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    
    # Calculate metrics at different thresholds
    metrics_df = calculate_threshold_metrics(y_true, y_prob, thresholds)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each metric
    for metric in metrics:
        if metric in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['threshold'],
                    y=metrics_df[metric],
                    mode='lines',
                    name=metric,
                    line=dict(color=METRIC_COLORS.get(metric, None), width=2),
                    hovertemplate='Threshold: %{x:.2f}<br>' + metric + ': %{y:.3f}<extra></extra>'
                )
            )
    
    # Add vertical line at threshold 0.5
    fig.add_shape(
        type='line',
        x0=0.5, y0=0, x1=0.5, y1=1,
        line=dict(color='gray', width=2, dash='dash')
    )
    
    # Add annotation for the reference line
    fig.add_annotation(
        x=0.5,
        y=1.03,
        text="Threshold = 0.5",
        showarrow=False,
        font=dict(color='gray')
    )
    
    # Add slider for interactive threshold selection
    steps = []
    for threshold in thresholds:
        step = dict(
            method='update',
            args=[
                {'visible': [True] * len(metrics) + [True]},  # Keep all traces visible
                {'shapes': [{'type': 'line', 'x0': threshold, 'y0': 0, 'x1': threshold, 'y1': 1, 
                            'line': {'color': 'gray', 'width': 2, 'dash': 'dash'}}],
                 'annotations': [{'x': threshold, 'y': 1.03, 'text': f"Threshold = {threshold:.2f}", 
                                 'showarrow': False, 'font': {'color': 'gray'}}]}
            ],
            label=f"{threshold:.2f}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=10,  # Default to 0.5 if available, otherwise middle of range
        steps=steps
    )]
    
    # Update layout
    fig.update_layout(
        title='Threshold Sensitivity Analysis',
        xaxis_title='Threshold',
        yaxis_title='Metric Value',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        hovermode='closest',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        width=800,
        height=600,
        template='plotly_white',
        sliders=sliders
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_performance_dashboard(
    y_true: SeriesType,
    y_prob: ArrayType,
    y_pred: ArrayType,
    additional_metrics: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Creates a comprehensive dashboard with multiple performance visualizations using Plotly.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        y_pred: Predicted binary labels
        additional_metrics: Optional dictionary with additional metrics to display
        
    Returns:
        Plotly figure object with the performance dashboard
    """
    # Validate inputs
    if y_true.shape != y_prob.shape or y_true.shape != y_pred.shape:
        raise VisualizationError("Shape mismatch between input arrays")
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'ROC Curve', 'Precision-Recall Curve', 
            'Calibration Curve', 'Confusion Matrix'
        ),
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Add ROC curve
    fpr, tpr, auc_value = calculate_roc_curve(y_true, y_prob)
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode='lines', name=f'ROC (AUC={auc_value:.3f})',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Random',
            line=dict(color='gray', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add Precision-Recall curve
    precision, recall, avg_precision = calculate_precision_recall_curve(y_true, y_prob)
    fig.add_trace(
        go.Scatter(
            x=recall, y=precision,
            mode='lines', name=f'PR (AP={avg_precision:.3f})',
            line=dict(color='orange', width=2)
        ),
        row=1, col=2
    )
    
    # Add Calibration curve
    cal_data = evaluate_calibration(y_true, y_prob)
    prob_true = np.array(cal_data['calibration_curve']['prob_true'])
    prob_pred = np.array(cal_data['calibration_curve']['prob_pred'])
    fig.add_trace(
        go.Scatter(
            x=prob_pred, y=prob_true,
            mode='lines+markers', name='Calibration',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Perfect calibration',
            line=dict(color='gray', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    # Add Confusion Matrix
    cm_dict = calculate_confusion_matrix(y_true, y_pred)
    cm = np.array([
        [cm_dict['true_negatives'], cm_dict['false_positives']],
        [cm_dict['false_negatives'], cm_dict['true_positives']]
    ])
    fig.add_trace(
        go.Heatmap(
            z=cm, x=['Neg', 'Pos'], y=['Neg', 'Pos'],
            colorscale='Viridis', showscale=False,
            text=[[str(cm[i][j]) for j in range(2)] for i in range(2)],
            texttemplate="%{text}"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Model Performance Dashboard',
        height=800,
        width=1000,
        template='plotly_white',
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text='False Positive Rate', range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text='True Positive Rate', range=[0, 1], row=1, col=1)
    
    fig.update_xaxes(title_text='Recall', range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text='Precision', range=[0, 1], row=1, col=2)
    
    fig.update_xaxes(title_text='Predicted Probability', range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text='Observed Frequency', range=[0, 1], row=2, col=1)
    
    fig.update_xaxes(title_text='Predicted', type='category', row=2, col=2)
    fig.update_yaxes(title_text='Actual', type='category', row=2, col=2)
    
    # Add annotations for additional metrics
    if additional_metrics:
        annotation_text = "<br>".join([f"{key}: {value:.4f}" for key, value in additional_metrics.items()])
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0,
            text=f"<b>Additional Metrics</b><br>{annotation_text}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
    
    return fig


@handle_errors(logger, VisualizationError)
def save_performance_plot(
    fig: Union[Figure, go.Figure],
    output_path: PathType,
    format: Optional[str] = None,
    dpi: Optional[int] = None,
    include_plotlyjs: Optional[bool] = None,
    full_html: Optional[bool] = None
) -> PathType:
    """
    Saves a performance plot to a file.
    
    Args:
        fig: Matplotlib or Plotly figure object
        output_path: Path where the figure should be saved
        format: Output format (if None, inferred from file extension)
        dpi: DPI for raster formats (matplotlib only)
        include_plotlyjs: Whether to include plotly.js in the output (plotly only)
        full_html: Whether to include full HTML wrapper (plotly only)
        
    Returns:
        Path to the saved file
    """
    # Convert string path to Path object
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get format from output_path if not specified
    if format is None:
        format = output_path.suffix.lstrip('.')
        if not format:
            format = 'png'  # Default format
            output_path = output_path.with_suffix(f'.{format}')
    
    # Handle matplotlib Figure
    if isinstance(fig, Figure):
        # Use default DPI if not specified
        if dpi is None:
            dpi = DEFAULT_DPI
        
        fig.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
        
    # Handle plotly Figure
    elif isinstance(fig, go.Figure):
        if format.lower() in ['html', 'htm']:
            # Save as HTML
            fig.write_html(
                output_path,
                include_plotlyjs=include_plotlyjs if include_plotlyjs is not None else 'cdn',
                full_html=full_html if full_html is not None else True
            )
        else:
            # Save as image
            fig.write_image(output_path, format=format)
    
    else:
        raise VisualizationError(f"Unsupported figure type: {type(fig)}")
    
    logger.info(f"Saved plot to {output_path}")
    return output_path


class ModelPerformancePlotter:
    """
    Class for creating and managing model performance visualizations.
    
    This class provides a convenient interface for generating various performance plots
    for machine learning models, with support for loading model data from a registry,
    generating static and interactive visualizations, and saving plots to files.
    """
    
    def __init__(self, registry_path: Optional[PathType] = None, plot_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ModelPerformancePlotter with a model registry path.
        
        Args:
            registry_path: Path to the model registry directory
            plot_config: Configuration dictionary for plot settings
        """
        # Initialize model registry
        self._registry = ModelRegistry(registry_path)
        
        # Initialize model data storage
        self._model_data = {}
        
        # Initialize plot configuration with defaults
        self._plot_config = {
            'figsize': DEFAULT_FIGURE_SIZE,
            'dpi': DEFAULT_DPI,
            'cmap': DEFAULT_CMAP,
            'interactive': True
        }
        
        # Update plot configuration if provided
        if plot_config:
            self._plot_config.update(plot_config)
        
        # Initialize data placeholders
        self._y_true = None
        self._y_prob = None
        self._y_pred = None
    
    def load_model_data(
        self, 
        model_id: str, 
        version: Optional[str] = None,
        y_true: Optional[SeriesType] = None,
        y_prob: Optional[ArrayType] = None,
        y_pred: Optional[ArrayType] = None
    ) -> bool:
        """
        Loads model data for visualization.
        
        Args:
            model_id: Identifier for the model
            version: Optional model version (if None, latest version is used)
            y_true: Optional true labels for custom evaluation
            y_prob: Optional predicted probabilities for custom evaluation
            y_pred: Optional predicted labels for custom evaluation
            
        Returns:
            True if model data loaded successfully, False otherwise
        """
        try:
            # Get model and metadata from registry
            model_obj, metadata = self._registry.get_model(model_id, version)
            
            # Get model performance metrics
            performance_metrics = self._registry.get_model_performance(model_id, version)
            
            # Store model data
            self._model_data = {
                'model_id': model_id,
                'version': version or metadata.get('version', 'latest'),
                'model_obj': model_obj,
                'metadata': metadata,
                'performance_metrics': performance_metrics
            }
            
            # Store evaluation data if provided
            if y_true is not None:
                self._y_true = y_true
            
            if y_prob is not None:
                self._y_prob = y_prob
                
                # If y_pred not provided but y_prob is, create y_pred using threshold 0.5
                if y_pred is None and y_true is not None and y_true.shape == y_prob.shape:
                    self._y_pred = (y_prob >= 0.5).astype(int)
            
            if y_pred is not None:
                self._y_pred = y_pred
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model data: {str(e)}")
            return False
    
    def plot_roc_curve(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a ROC curve plot for the loaded model.
        
        Args:
            figsize: Optional figure size tuple (width, height)
            fig: Optional existing figure to plot on
            ax: Optional existing axes to plot on
            
        Returns:
            Matplotlib figure and axes objects with the ROC curve plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the plot_roc_curve function
        return plot_roc_curve(
            self._y_true,
            self._y_prob,
            figsize or self._plot_config.get('figsize'),
            fig,
            ax
        )
    
    def plot_precision_recall_curve(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a precision-recall curve plot for the loaded model.
        
        Args:
            figsize: Optional figure size tuple (width, height)
            fig: Optional existing figure to plot on
            ax: Optional existing axes to plot on
            
        Returns:
            Matplotlib figure and axes objects with the precision-recall curve plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the plot_precision_recall_curve function
        return plot_precision_recall_curve(
            self._y_true,
            self._y_prob,
            figsize or self._plot_config.get('figsize'),
            fig,
            ax
        )
    
    def plot_calibration_curve(
        self,
        n_bins: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a calibration curve plot for the loaded model.
        
        Args:
            n_bins: Number of bins for the calibration curve
            figsize: Optional figure size tuple (width, height)
            fig: Optional existing figure to plot on
            ax: Optional existing axes to plot on
            
        Returns:
            Matplotlib figure and axes objects with the calibration curve plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the plot_calibration_curve function
        return plot_calibration_curve(
            self._y_true,
            self._y_prob,
            n_bins,
            figsize or self._plot_config.get('figsize'),
            fig,
            ax
        )
    
    def plot_confusion_matrix(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: Optional[str] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a confusion matrix plot for the loaded model.
        
        Args:
            figsize: Optional figure size tuple (width, height)
            cmap: Colormap for the heatmap
            fig: Optional existing figure to plot on
            ax: Optional existing axes to plot on
            
        Returns:
            Matplotlib figure and axes objects with the confusion matrix plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_pred is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the plot_confusion_matrix function
        return plot_confusion_matrix(
            self._y_true,
            self._y_pred,
            figsize or self._plot_config.get('figsize'),
            cmap or self._plot_config.get('cmap'),
            fig,
            ax
        )
    
    def plot_threshold_sensitivity(
        self,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a threshold sensitivity plot for the loaded model.
        
        Args:
            metrics: List of metrics to plot
            thresholds: List of threshold values to evaluate
            figsize: Optional figure size tuple (width, height)
            fig: Optional existing figure to plot on
            ax: Optional existing axes to plot on
            
        Returns:
            Matplotlib figure and axes objects with the threshold sensitivity plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the plot_threshold_sensitivity function
        return plot_threshold_sensitivity(
            self._y_true,
            self._y_prob,
            metrics,
            thresholds,
            figsize or self._plot_config.get('figsize'),
            fig,
            ax
        )
    
    def create_interactive_roc_curve(self) -> go.Figure:
        """
        Creates an interactive ROC curve plot for the loaded model.
        
        Returns:
            Plotly figure object with the interactive ROC curve plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the create_interactive_roc_curve function
        return create_interactive_roc_curve(self._y_true, self._y_prob)
    
    def create_interactive_precision_recall_curve(self) -> go.Figure:
        """
        Creates an interactive precision-recall curve plot for the loaded model.
        
        Returns:
            Plotly figure object with the interactive precision-recall curve plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the create_interactive_precision_recall_curve function
        return create_interactive_precision_recall_curve(self._y_true, self._y_prob)
    
    def create_interactive_calibration_curve(self, n_bins: Optional[int] = None) -> go.Figure:
        """
        Creates an interactive calibration curve plot for the loaded model.
        
        Args:
            n_bins: Number of bins for the calibration curve
            
        Returns:
            Plotly figure object with the interactive calibration curve plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the create_interactive_calibration_curve function
        return create_interactive_calibration_curve(self._y_true, self._y_prob, n_bins)
    
    def create_interactive_confusion_matrix(self) -> go.Figure:
        """
        Creates an interactive confusion matrix plot for the loaded model.
        
        Returns:
            Plotly figure object with the interactive confusion matrix plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_pred is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the create_interactive_confusion_matrix function
        return create_interactive_confusion_matrix(self._y_true, self._y_pred)
    
    def create_interactive_threshold_sensitivity(
        self,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None
    ) -> go.Figure:
        """
        Creates an interactive threshold sensitivity plot for the loaded model.
        
        Args:
            metrics: List of metrics to plot
            thresholds: List of threshold values to evaluate
            
        Returns:
            Plotly figure object with the interactive threshold sensitivity plot
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # Call the create_interactive_threshold_sensitivity function
        return create_interactive_threshold_sensitivity(self._y_true, self._y_prob, metrics, thresholds)
    
    def create_performance_dashboard(self, additional_metrics: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Creates a comprehensive performance dashboard for the loaded model.
        
        Args:
            additional_metrics: Optional dictionary with additional metrics to display
            
        Returns:
            Plotly figure object with the performance dashboard
        """
        # Check if model data and evaluation data are available
        if not self._model_data or self._y_true is None or self._y_prob is None or self._y_pred is None:
            raise VisualizationError("Model data not loaded or evaluation data not available")
        
        # If additional_metrics not provided, use metrics from model data
        if additional_metrics is None and 'performance_metrics' in self._model_data:
            additional_metrics = self._model_data['performance_metrics']
        
        # Call the create_performance_dashboard function
        return create_performance_dashboard(self._y_true, self._y_prob, self._y_pred, additional_metrics)
    
    def save_plot(
        self,
        fig: Union[Figure, go.Figure],
        output_path: PathType,
        format: Optional[str] = None,
        dpi: Optional[int] = None,
        include_plotlyjs: Optional[bool] = None,
        full_html: Optional[bool] = None
    ) -> PathType:
        """
        Saves a plot to a file.
        
        Args:
            fig: Matplotlib or Plotly figure object
            output_path: Path where the figure should be saved
            format: Output format (if None, inferred from file extension)
            dpi: DPI for raster formats (matplotlib only)
            include_plotlyjs: Whether to include plotly.js in the output (plotly only)
            full_html: Whether to include full HTML wrapper (plotly only)
            
        Returns:
            Path to the saved file
        """
        # Call the save_performance_plot function
        return save_performance_plot(
            fig, 
            output_path, 
            format, 
            dpi or self._plot_config.get('dpi'), 
            include_plotlyjs, 
            full_html
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gets information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        # Check if model data is available
        if not self._model_data:
            raise VisualizationError("No model data loaded")
        
        # Return selected model information
        return {
            'model_id': self._model_data['model_id'],
            'version': self._model_data['version'],
            'performance_metrics': self._model_data.get('performance_metrics', {})
        }
    
    def compare_models(
        self,
        model_ids: List[str],
        versions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Compares performance metrics between multiple models.
        
        Args:
            model_ids: List of model identifiers to compare
            versions: Optional list of model versions (must match length of model_ids)
            metrics: List of metrics to compare
            figsize: Optional figure size tuple (width, height)
            
        Returns:
            Matplotlib figure and axes objects with the model comparison plot
        """
        # Initialize metrics dictionary
        model_metrics = {}
        
        # Get metrics for each model
        for i, model_id in enumerate(model_ids):
            version = None if versions is None else versions[i]
            
            try:
                # Get model performance from registry
                performance = self._registry.get_model_performance(model_id, version)
                
                # Create model name with version if available
                model_name = f"{model_id}"
                if version:
                    model_name += f" (v{version})"
                
                # Store performance metrics
                model_metrics[model_name] = performance
                
            except Exception as e:
                logger.warning(f"Error getting metrics for model {model_id} version {version}: {str(e)}")
        
        # Call the plot_metric_comparison function
        return plot_metric_comparison(
            model_metrics,
            metrics,
            figsize or self._plot_config.get('figsize')
        )