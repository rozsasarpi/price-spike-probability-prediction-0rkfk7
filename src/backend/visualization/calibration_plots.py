"""
Implements visualization functionality for calibration curves in the ERCOT RTLMP spike prediction system.

This module provides tools to create, customize, and export calibration plots (reliability diagrams)
that assess how well the model's predicted probabilities align with observed frequencies of price spikes.
"""

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
import matplotlib.pyplot as plt  # version 3.7+
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns  # version 0.12+
import plotly  # version 5.14+
import plotly.graph_objects as go  # version 5.14+
from plotly.subplots import make_subplots  # version 5.14+
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path

# Internal imports
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, PathType, ModelType
from ..utils.logging import get_logger
from ..utils.error_handling import handle_errors, VisualizationError
from ..inference.calibration import evaluate_calibration
from ..data.storage.model_registry import ModelRegistry
from .export import export_figure_to_file, figure_to_base64

# Initialize logger
logger = get_logger(__name__)

# Global constants
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 100
DEFAULT_N_BINS = 10
MODEL_COLORS = {'current': '#1f77b4', 'previous': '#ff7f0e', 'baseline': '#2ca02c'}


@handle_errors(logger, VisualizationError)
def create_calibration_curve(y_true: SeriesType, y_prob: ArrayType, n_bins: Optional[int] = None) -> Dict[str, Any]:
    """
    Creates calibration curve data from true labels and predicted probabilities.
    
    Args:
        y_true: Series of true binary labels
        y_prob: Array of predicted probabilities
        n_bins: Number of bins for calibration curve calculation
        
    Returns:
        Dictionary with calibration curve data and metrics
    """
    # Validate inputs
    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise ValueError(f"y_true must be a Series or array, got {type(y_true)}")
    
    if not isinstance(y_prob, (pd.Series, np.ndarray)):
        raise ValueError(f"y_prob must be a Series or array, got {type(y_prob)}")
    
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true and y_prob must have the same length, got {len(y_true)} and {len(y_prob)}")
    
    # Convert Series to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
        
    if isinstance(y_prob, pd.Series):
        y_prob = y_prob.values
    
    # Use default number of bins if not specified
    if n_bins is None:
        n_bins = DEFAULT_N_BINS
    
    # Calculate calibration metrics using the evaluate_calibration function
    calibration_data = evaluate_calibration(y_true, y_prob, n_bins)
    
    logger.debug(f"Created calibration curve with {n_bins} bins")
    return calibration_data


@handle_errors(logger, VisualizationError)
def plot_calibration_curve(
    y_true: SeriesType, 
    y_prob: ArrayType, 
    n_bins: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None, 
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    label: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a static calibration curve plot using matplotlib.
    
    Args:
        y_true: Series of true binary labels
        y_prob: Array of predicted probabilities
        n_bins: Number of bins for calibration curve calculation
        figsize: Figure size as (width, height) in inches
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        title: Optional plot title
        label: Optional label for the calibration curve
        
    Returns:
        Matplotlib figure and axes objects with the calibration curve plot
    """
    # Get calibration curve data
    cal_data = create_calibration_curve(y_true, y_prob, n_bins)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Extract curve data
    prob_true = np.array(cal_data['calibration_curve']['prob_true'])
    prob_pred = np.array(cal_data['calibration_curve']['prob_pred'])
    
    # Plot the calibration curve
    ax.plot(
        prob_pred, 
        prob_true, 
        marker='o', 
        linestyle='-', 
        label=label or 'Calibration curve'
    )
    
    # Plot the perfectly calibrated line (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Set labels
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    
    # Set title (include metrics if no title provided)
    if title is None:
        brier_score = cal_data.get('brier_score', 0)
        ece = cal_data.get('expected_calibration_error', 0)
        title = f"Calibration Curve (Brier: {brier_score:.4f}, ECE: {ece:.4f})"
    
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def create_interactive_calibration_curve(
    y_true: SeriesType, 
    y_prob: ArrayType, 
    n_bins: Optional[int] = None,
    title: Optional[str] = None,
    label: Optional[str] = None
) -> go.Figure:
    """
    Creates an interactive calibration curve plot using Plotly.
    
    Args:
        y_true: Series of true binary labels
        y_prob: Array of predicted probabilities
        n_bins: Number of bins for calibration curve calculation
        title: Optional plot title
        label: Optional label for the calibration curve
        
    Returns:
        Plotly figure object with the interactive calibration curve plot
    """
    # Get calibration curve data
    cal_data = create_calibration_curve(y_true, y_prob, n_bins)
    
    # Extract curve data
    prob_true = np.array(cal_data['calibration_curve']['prob_true'])
    prob_pred = np.array(cal_data['calibration_curve']['prob_pred'])
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add the calibration curve
    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name=label or 'Calibration curve',
            marker=dict(size=8),
            line=dict(width=2),
            hovertemplate='Predicted: %{x:.4f}<br>Observed: %{y:.4f}'
        )
    )
    
    # Add the perfectly calibrated line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfectly calibrated',
            line=dict(color='black', width=2, dash='dash'),
            hoverinfo='skip'
        )
    )
    
    # Set title (include metrics if no title provided)
    if title is None:
        brier_score = cal_data.get('brier_score', 0)
        ece = cal_data.get('expected_calibration_error', 0)
        title = f"Calibration Curve (Brier: {brier_score:.4f}, ECE: {ece:.4f})"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Mean predicted probability',
        yaxis_title='Fraction of positives',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        hovermode='closest'
    )
    
    # Add a grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


@handle_errors(logger, VisualizationError)
def plot_multiple_calibration_curves(
    model_data: Dict[str, Tuple[SeriesType, ArrayType]],
    n_bins: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None, 
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a plot with multiple calibration curves for comparison.
    
    Args:
        model_data: Dictionary mapping model IDs to tuples of (y_true, y_prob)
        n_bins: Number of bins for calibration curve calculation
        figsize: Figure size as (width, height) in inches
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        title: Optional plot title
        labels: Optional dictionary mapping model IDs to display labels
        colors: Optional dictionary mapping model IDs to colors
        
    Returns:
        Matplotlib figure and axes objects with multiple calibration curves
    """
    if not model_data:
        raise ValueError("model_data dictionary must not be empty")
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Plot the perfectly calibrated line (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Use default colors if not provided
    if colors is None:
        colors = {}
    
    # Use default labels if not provided
    if labels is None:
        labels = {}
    
    # Store metrics for legend
    metrics = {}
    
    # Plot each model's calibration curve
    for i, (model_id, (y_true, y_prob)) in enumerate(model_data.items()):
        # Get calibration curve data
        cal_data = create_calibration_curve(y_true, y_prob, n_bins)
        
        # Extract curve data
        prob_true = np.array(cal_data['calibration_curve']['prob_true'])
        prob_pred = np.array(cal_data['calibration_curve']['prob_pred'])
        
        # Get label for the model
        label = labels.get(model_id, model_id)
        
        # Get color for the model
        color = colors.get(model_id)
        if color is None:
            # Use preset colors for special model types, otherwise use matplotlib's default cycle
            if model_id in MODEL_COLORS:
                color = MODEL_COLORS[model_id]
            else:
                # Let matplotlib pick the color automatically
                color = None
        
        # Plot the calibration curve
        ax.plot(
            prob_pred, 
            prob_true, 
            marker='o', 
            linestyle='-', 
            label=label,
            color=color
        )
        
        # Store metrics for this model
        metrics[model_id] = {
            'brier_score': cal_data.get('brier_score', 0),
            'ece': cal_data.get('expected_calibration_error', 0)
        }
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Set labels
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    
    # Set title
    if title is None:
        title = "Calibration Curve Comparison"
    ax.set_title(title)
    
    # Add legend with metrics
    handles, legend_labels = ax.get_legend_handles_labels()
    enhanced_labels = []
    
    # Add metrics to legend labels for each model except the reference line
    for label in legend_labels:
        if label == 'Perfectly calibrated':
            enhanced_labels.append(label)
        else:
            # Find the model ID that matches this label
            model_id = None
            for mid, lbl in labels.items() if labels else model_data.keys():
                if labels.get(mid, mid) == label:
                    model_id = mid
                    break
            
            if model_id and model_id in metrics:
                model_metrics = metrics[model_id]
                enhanced_label = f"{label} (Brier: {model_metrics['brier_score']:.4f}, ECE: {model_metrics['ece']:.4f})"
                enhanced_labels.append(enhanced_label)
            else:
                enhanced_labels.append(label)
    
    ax.legend(handles, enhanced_labels, loc='best')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def create_interactive_multiple_calibration_curves(
    model_data: Dict[str, Tuple[SeriesType, ArrayType]],
    n_bins: Optional[int] = None,
    title: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Creates an interactive plot with multiple calibration curves using Plotly.
    
    Args:
        model_data: Dictionary mapping model IDs to tuples of (y_true, y_prob)
        n_bins: Number of bins for calibration curve calculation
        title: Optional plot title
        labels: Optional dictionary mapping model IDs to display labels
        colors: Optional dictionary mapping model IDs to colors
        
    Returns:
        Plotly figure object with multiple interactive calibration curves
    """
    if not model_data:
        raise ValueError("model_data dictionary must not be empty")
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add the perfectly calibrated line (diagonal)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfectly calibrated',
            line=dict(color='black', width=2, dash='dash'),
            hoverinfo='skip'
        )
    )
    
    # Use default colors if not provided
    if colors is None:
        colors = {}
    
    # Use default labels if not provided
    if labels is None:
        labels = {}
    
    # Plot each model's calibration curve
    for i, (model_id, (y_true, y_prob)) in enumerate(model_data.items()):
        # Get calibration curve data
        cal_data = create_calibration_curve(y_true, y_prob, n_bins)
        
        # Extract curve data
        prob_true = np.array(cal_data['calibration_curve']['prob_true'])
        prob_pred = np.array(cal_data['calibration_curve']['prob_pred'])
        
        # Get label for the model
        label = labels.get(model_id, model_id)
        
        # Get color for the model
        color = colors.get(model_id)
        if color is None:
            # Use preset colors for special model types
            if model_id in MODEL_COLORS:
                color = MODEL_COLORS[model_id]
            else:
                # Let plotly pick the color automatically
                color = None
        
        # Get metrics for this model for hover information
        brier_score = cal_data.get('brier_score', 0)
        ece = cal_data.get('expected_calibration_error', 0)
        
        # Create hover template with metrics
        hovertemplate = (
            f"Model: {label}<br>" +
            "Predicted: %{x:.4f}<br>" +
            "Observed: %{y:.4f}<br>" +
            f"Brier Score: {brier_score:.4f}<br>" +
            f"ECE: {ece:.4f}"
        )
        
        # Add the calibration curve
        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode='lines+markers',
                name=f"{label} (Brier: {brier_score:.4f})",
                marker=dict(size=8, color=color),
                line=dict(width=2, color=color),
                hovertemplate=hovertemplate
            )
        )
    
    # Set title
    if title is None:
        title = "Calibration Curve Comparison"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Mean predicted probability',
        yaxis_title='Fraction of positives',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        hovermode='closest'
    )
    
    # Add a grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


@handle_errors(logger, VisualizationError)
def calculate_calibration_metrics(
    model_data: Dict[str, Tuple[SeriesType, ArrayType]],
    n_bins: Optional[int] = None
) -> DataFrameType:
    """
    Calculates calibration metrics for multiple models.
    
    Args:
        model_data: Dictionary mapping model IDs to tuples of (y_true, y_prob)
        n_bins: Number of bins for calibration curve calculation
        
    Returns:
        DataFrame with calibration metrics for each model
    """
    if not model_data:
        raise ValueError("model_data dictionary must not be empty")
    
    # Initialize lists to store results
    model_ids = []
    brier_scores = []
    expected_calibration_errors = []
    maximum_calibration_errors = []
    log_losses = []
    auc_rocs = []
    
    # Calculate metrics for each model
    for model_id, (y_true, y_prob) in model_data.items():
        # Get calibration metrics
        cal_data = create_calibration_curve(y_true, y_prob, n_bins)
        
        # Extract metrics
        model_ids.append(model_id)
        brier_scores.append(cal_data.get('brier_score', None))
        expected_calibration_errors.append(cal_data.get('expected_calibration_error', None))
        maximum_calibration_errors.append(cal_data.get('maximum_calibration_error', None))
        log_losses.append(cal_data.get('log_loss', None))
        auc_rocs.append(cal_data.get('auc_roc', None))
    
    # Create DataFrame with metrics
    metrics_df = pd.DataFrame({
        'model_id': model_ids,
        'brier_score': brier_scores,
        'expected_calibration_error': expected_calibration_errors,
        'maximum_calibration_error': maximum_calibration_errors,
        'log_loss': log_losses,
        'auc_roc': auc_rocs
    })
    
    return metrics_df


@handle_errors(logger, VisualizationError)
def save_calibration_plot(
    fig: Union[Figure, go.Figure],
    output_path: PathType,
    format: Optional[str] = None,
    dpi: Optional[int] = None,
    include_plotlyjs: Optional[bool] = None,
    full_html: Optional[bool] = None
) -> PathType:
    """
    Saves a calibration plot to a file.
    
    Args:
        fig: Matplotlib or Plotly figure to save
        output_path: Path where the figure should be saved
        format: Output format (png, pdf, svg, etc.)
        dpi: Resolution for raster formats (matplotlib only)
        include_plotlyjs: Whether to include plotly.js in HTML output (plotly only)
        full_html: Whether to include full HTML boilerplate (plotly only)
        
    Returns:
        Path to the saved file
    """
    # Convert path to Path object if it's a string
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If format is not provided, try to infer from file extension
    if format is None:
        format = output_path.suffix.lstrip('.')
        if not format:
            format = 'png'  # Default format
    
    # Handle based on figure type
    if isinstance(fig, Figure):
        # Use matplotlib's savefig for matplotlib figures
        if dpi is None:
            dpi = DEFAULT_DPI
        
        fig.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved matplotlib figure to {output_path}")
    
    elif isinstance(fig, go.Figure):
        # Use plotly's write_image or write_html for plotly figures
        if format == 'html':
            if include_plotlyjs is None:
                include_plotlyjs = True
            if full_html is None:
                full_html = True
            
            fig.write_html(
                str(output_path),
                include_plotlyjs=include_plotlyjs,
                full_html=full_html
            )
            logger.info(f"Saved plotly figure as HTML to {output_path}")
        else:
            fig.write_image(str(output_path), format=format)
            logger.info(f"Saved plotly figure as {format} to {output_path}")
    
    else:
        raise ValueError(f"Unsupported figure type: {type(fig)}")
    
    return output_path


class CalibrationPlotter:
    """Class for creating and managing calibration visualizations."""
    
    def __init__(self, plot_config: Optional[Dict[str, Any]] = None, 
                 registry_path: Optional[PathType] = None):
        """
        Initializes the CalibrationPlotter with optional configuration.
        
        Args:
            plot_config: Optional dictionary with plot configuration options
            registry_path: Optional path to the model registry
        """
        # Initialize data storage
        self._model_data: Dict[str, Tuple[SeriesType, ArrayType]] = {}
        self._calibration_results: Dict[str, Dict[str, Any]] = {}
        
        # Initialize plot configuration with defaults
        self._plot_config: Dict[str, Any] = {
            'n_bins': DEFAULT_N_BINS,
            'figsize': DEFAULT_FIGURE_SIZE,
            'dpi': DEFAULT_DPI,
            'colors': MODEL_COLORS.copy(),
            'labels': {},
            'title': "Calibration Curve",
            'interactive': True
        }
        
        # Update with provided configuration if any
        if plot_config:
            self._plot_config.update(plot_config)
        
        # Initialize model registry if path provided
        self._registry = ModelRegistry(registry_path) if registry_path else None
    
    def add_model_data(self, model_id: str, y_true: SeriesType, y_prob: ArrayType, 
                      label: Optional[str] = None) -> bool:
        """
        Adds model data for calibration visualization.
        
        Args:
            model_id: Identifier for the model
            y_true: Series of true binary labels
            y_prob: Array of predicted probabilities
            label: Optional display label for the model
            
        Returns:
            True if data was added successfully
        """
        # Validate inputs
        if not isinstance(y_true, (pd.Series, np.ndarray)):
            raise ValueError(f"y_true must be a Series or array, got {type(y_true)}")
        
        if not isinstance(y_prob, (pd.Series, np.ndarray)):
            raise ValueError(f"y_prob must be a Series or array, got {type(y_prob)}")
        
        if len(y_true) != len(y_prob):
            raise ValueError(f"y_true and y_prob must have the same length")
        
        # Store the data
        self._model_data[model_id] = (y_true, y_prob)
        
        # Store label if provided
        if label is not None:
            if 'labels' not in self._plot_config:
                self._plot_config['labels'] = {}
            self._plot_config['labels'][model_id] = label
        
        logger.debug(f"Added data for model {model_id}")
        return True
    
    def add_model_from_registry(self, model_id: str, version: Optional[str] = None, 
                               label: Optional[str] = None) -> bool:
        """
        Adds model data from the model registry.
        
        Args:
            model_id: Identifier for the model
            version: Optional specific version of the model
            label: Optional display label for the model
            
        Returns:
            True if data was added successfully
        """
        if self._registry is None:
            raise ValueError("Model registry not initialized")
        
        try:
            # Get model performance data from registry
            performance = self._registry.get_model_performance(model_id, version)
            
            if not performance:
                logger.warning(f"No performance data found for model {model_id}")
                return False
            
            # Extract y_true and y_prob from performance data
            if 'validation_data' in performance:
                val_data = performance['validation_data']
                y_true = val_data.get('y_true')
                y_prob = val_data.get('y_prob')
                
                if y_true is None or y_prob is None:
                    logger.warning(f"Validation data missing y_true or y_prob for model {model_id}")
                    return False
                
                # Convert to numpy arrays if needed
                if isinstance(y_true, list):
                    y_true = np.array(y_true)
                if isinstance(y_prob, list):
                    y_prob = np.array(y_prob)
                
                # Add the model data
                return self.add_model_data(model_id, y_true, y_prob, label)
            else:
                logger.warning(f"No validation data found for model {model_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding model from registry: {str(e)}")
            return False
    
    def plot_calibration_curve(self, model_id: str, n_bins: Optional[int] = None,
                              figsize: Optional[Tuple[int, int]] = None,
                              fig: Optional[Figure] = None, ax: Optional[Axes] = None,
                              title: Optional[str] = None) -> Tuple[Figure, Axes]:
        """
        Creates a calibration curve plot for a specific model.
        
        Args:
            model_id: Identifier for the model
            n_bins: Number of bins for calibration curve calculation
            figsize: Figure size as (width, height) in inches
            fig: Optional existing figure to plot on
            ax: Optional existing axes to plot on
            title: Optional plot title
            
        Returns:
            Matplotlib figure and axes objects with the calibration curve plot
        """
        if model_id not in self._model_data:
            raise ValueError(f"Model {model_id} not found in stored data")
        
        # Get model data
        y_true, y_prob = self._model_data[model_id]
        
        # Get label for the model if available
        label = self._plot_config.get('labels', {}).get(model_id, model_id)
        
        # Use provided parameters or defaults from config
        if n_bins is None:
            n_bins = self._plot_config.get('n_bins', DEFAULT_N_BINS)
        
        if figsize is None:
            figsize = self._plot_config.get('figsize', DEFAULT_FIGURE_SIZE)
        
        if title is None:
            title = self._plot_config.get('title', f"Calibration Curve - {label}")
        
        # Create the plot
        fig, ax = plot_calibration_curve(
            y_true, 
            y_prob, 
            n_bins=n_bins,
            figsize=figsize,
            fig=fig,
            ax=ax,
            title=title,
            label=label
        )
        
        # Store the calibration results
        cal_data = create_calibration_curve(y_true, y_prob, n_bins)
        self._calibration_results[model_id] = cal_data
        
        return fig, ax
    
    def create_interactive_calibration_curve(self, model_id: str, n_bins: Optional[int] = None,
                                           title: Optional[str] = None) -> go.Figure:
        """
        Creates an interactive calibration curve plot for a specific model.
        
        Args:
            model_id: Identifier for the model
            n_bins: Number of bins for calibration curve calculation
            title: Optional plot title
            
        Returns:
            Plotly figure object with the interactive calibration curve plot
        """
        if model_id not in self._model_data:
            raise ValueError(f"Model {model_id} not found in stored data")
        
        # Get model data
        y_true, y_prob = self._model_data[model_id]
        
        # Get label for the model if available
        label = self._plot_config.get('labels', {}).get(model_id, model_id)
        
        # Use provided parameters or defaults from config
        if n_bins is None:
            n_bins = self._plot_config.get('n_bins', DEFAULT_N_BINS)
        
        if title is None:
            title = self._plot_config.get('title', f"Calibration Curve - {label}")
        
        # Create the interactive plot
        fig = create_interactive_calibration_curve(
            y_true, 
            y_prob, 
            n_bins=n_bins,
            title=title,
            label=label
        )
        
        # Store the calibration results
        cal_data = create_calibration_curve(y_true, y_prob, n_bins)
        self._calibration_results[model_id] = cal_data
        
        return fig
    
    def plot_multiple_calibration_curves(self, model_ids: Optional[List[str]] = None,
                                       n_bins: Optional[int] = None,
                                       figsize: Optional[Tuple[int, int]] = None,
                                       fig: Optional[Figure] = None, ax: Optional[Axes] = None,
                                       title: Optional[str] = None) -> Tuple[Figure, Axes]:
        """
        Creates a plot with calibration curves for all added models.
        
        Args:
            model_ids: Optional list of model IDs to include (all if None)
            n_bins: Number of bins for calibration curve calculation
            figsize: Figure size as (width, height) in inches
            fig: Optional existing figure to plot on
            ax: Optional existing axes to plot on
            title: Optional plot title
            
        Returns:
            Matplotlib figure and axes objects with multiple calibration curves
        """
        # If no model_ids provided, use all models
        if model_ids is None:
            model_ids = list(self._model_data.keys())
        
        # Check that all requested models exist
        missing_models = [mid for mid in model_ids if mid not in self._model_data]
        if missing_models:
            raise ValueError(f"Models not found: {missing_models}")
        
        # Filter model data to include only requested models
        filtered_data = {
            mid: self._model_data[mid] for mid in model_ids
        }
        
        # Use provided parameters or defaults from config
        if n_bins is None:
            n_bins = self._plot_config.get('n_bins', DEFAULT_N_BINS)
        
        if figsize is None:
            figsize = self._plot_config.get('figsize', DEFAULT_FIGURE_SIZE)
        
        if title is None:
            title = self._plot_config.get('title', "Calibration Curve Comparison")
        
        # Get labels and colors from config
        labels = self._plot_config.get('labels', {})
        colors = self._plot_config.get('colors', {})
        
        # Create the plot
        fig, ax = plot_multiple_calibration_curves(
            filtered_data,
            n_bins=n_bins,
            figsize=figsize,
            fig=fig,
            ax=ax,
            title=title,
            labels=labels,
            colors=colors
        )
        
        # Store the calibration results for each model
        for model_id in model_ids:
            y_true, y_prob = self._model_data[model_id]
            cal_data = create_calibration_curve(y_true, y_prob, n_bins)
            self._calibration_results[model_id] = cal_data
        
        return fig, ax
    
    def create_interactive_multiple_calibration_curves(self, model_ids: Optional[List[str]] = None,
                                                     n_bins: Optional[int] = None,
                                                     title: Optional[str] = None) -> go.Figure:
        """
        Creates an interactive plot with calibration curves for all added models.
        
        Args:
            model_ids: Optional list of model IDs to include (all if None)
            n_bins: Number of bins for calibration curve calculation
            title: Optional plot title
            
        Returns:
            Plotly figure object with multiple interactive calibration curves
        """
        # If no model_ids provided, use all models
        if model_ids is None:
            model_ids = list(self._model_data.keys())
        
        # Check that all requested models exist
        missing_models = [mid for mid in model_ids if mid not in self._model_data]
        if missing_models:
            raise ValueError(f"Models not found: {missing_models}")
        
        # Filter model data to include only requested models
        filtered_data = {
            mid: self._model_data[mid] for mid in model_ids
        }
        
        # Use provided parameters or defaults from config
        if n_bins is None:
            n_bins = self._plot_config.get('n_bins', DEFAULT_N_BINS)
        
        if title is None:
            title = self._plot_config.get('title', "Calibration Curve Comparison")
        
        # Get labels and colors from config
        labels = self._plot_config.get('labels', {})
        colors = self._plot_config.get('colors', {})
        
        # Create the interactive plot
        fig = create_interactive_multiple_calibration_curves(
            filtered_data,
            n_bins=n_bins,
            title=title,
            labels=labels,
            colors=colors
        )
        
        # Store the calibration results for each model
        for model_id in model_ids:
            y_true, y_prob = self._model_data[model_id]
            cal_data = create_calibration_curve(y_true, y_prob, n_bins)
            self._calibration_results[model_id] = cal_data
        
        return fig
    
    def get_calibration_metrics(self, model_ids: Optional[List[str]] = None,
                               n_bins: Optional[int] = None) -> DataFrameType:
        """
        Gets calibration metrics for all added models.
        
        Args:
            model_ids: Optional list of model IDs to include (all if None)
            n_bins: Number of bins for calibration curve calculation
            
        Returns:
            DataFrame with calibration metrics for each model
        """
        # If no model_ids provided, use all models
        if model_ids is None:
            model_ids = list(self._model_data.keys())
        
        # Check that all requested models exist
        missing_models = [mid for mid in model_ids if mid not in self._model_data]
        if missing_models:
            raise ValueError(f"Models not found: {missing_models}")
        
        # Filter model data to include only requested models
        filtered_data = {
            mid: self._model_data[mid] for mid in model_ids
        }
        
        # Use provided parameters or defaults from config
        if n_bins is None:
            n_bins = self._plot_config.get('n_bins', DEFAULT_N_BINS)
        
        # Calculate metrics
        return calculate_calibration_metrics(filtered_data, n_bins)
    
    def save_plot(self, fig: Union[Figure, go.Figure], output_path: PathType,
                 format: Optional[str] = None, dpi: Optional[int] = None) -> PathType:
        """
        Saves a calibration plot to a file.
        
        Args:
            fig: Matplotlib or Plotly figure to save
            output_path: Path where the figure should be saved
            format: Output format (png, pdf, svg, etc.)
            dpi: Resolution for raster formats
            
        Returns:
            Path to the saved file
        """
        if dpi is None:
            dpi = self._plot_config.get('dpi', DEFAULT_DPI)
        
        return save_calibration_plot(fig, output_path, format, dpi)
    
    def get_results(self, model_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Gets stored calibration results.
        
        Args:
            model_id: Optional model ID to get results for
            
        Returns:
            Dictionary with calibration results
        """
        if model_id is not None:
            # Return results for a specific model
            return {model_id: self._calibration_results.get(model_id, {})}
        else:
            # Return all results
            return self._calibration_results
    
    def clear_data(self) -> None:
        """
        Clears all stored model data and results.
        """
        self._model_data = {}
        self._calibration_results = {}
        logger.debug("Cleared all model data and results")
    
    def set_plot_config(self, config: Dict[str, Any]) -> None:
        """
        Sets plot configuration options.
        
        Args:
            config: Dictionary with configuration options
        """
        self._plot_config.update(config)
        logger.debug("Updated plot configuration")