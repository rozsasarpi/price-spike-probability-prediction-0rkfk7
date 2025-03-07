"""
Implements visualization functions for RTLMP spike probability forecasts in the ERCOT RTLMP spike prediction system.
This module provides specialized plotting capabilities for visualizing forecast probabilities over time,
comparing different thresholds, and creating interactive dashboards for forecast analysis.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import datetime
from pathlib import Path
import io
from io import BytesIO
import base64

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns  # version 0.12+
import plotly  # version 5.14+
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.type_definitions import DataFrameType, SeriesType, PathType, ThresholdValue, NodeID, ForecastResultDict
from ..utils.logging import get_logger
from ..utils.error_handling import handle_errors, VisualizationError
from ..data.storage.forecast_repository import ForecastRepository

# Set up logger
logger = get_logger(__name__)

# Default visualization settings
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 100
DEFAULT_CMAP = 'viridis'
PROBABILITY_COLORS = {
    'low': '#1f77b4',  # blue
    'medium': '#ff7f0e',  # orange
    'high': '#d62728'  # red
}
THRESHOLD_COLORS = {
    '50': '#1f77b4',  # blue
    '100': '#ff7f0e',  # orange
    '200': '#d62728',  # red
    '300': '#9467bd',  # purple
    '400': '#8c564b'   # brown
}

@handle_errors(logger, VisualizationError)
def _figure_to_base64(fig: Figure, format: Optional[str] = 'png', dpi: Optional[int] = None) -> str:
    """
    Converts a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure to convert
        format: Output format (png, jpg, svg, pdf)
        dpi: Resolution for raster formats
        
    Returns:
        Base64 encoded string of the figure
    """
    # Use the default DPI if not specified
    if dpi is None:
        dpi = DEFAULT_DPI
    
    # Create a BytesIO object to store the image
    buf = BytesIO()
    
    # Save the figure to the BytesIO object
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    
    # Get the image data from the BytesIO object
    buf.seek(0)
    img_data = buf.getvalue()
    
    # Encode the image data as base64
    base64_encoded = base64.b64encode(img_data).decode('utf-8')
    
    return base64_encoded

@handle_errors(logger, VisualizationError)
def _plotly_figure_to_base64(
    fig: go.Figure, 
    format: Optional[str] = 'png', 
    include_plotlyjs: Optional[bool] = True,
    full_html: Optional[bool] = False
) -> str:
    """
    Converts a plotly figure to a base64 encoded string.
    
    Args:
        fig: Plotly figure to convert
        format: Output format (png, jpg, svg, html)
        include_plotlyjs: Whether to include plotly.js in HTML output
        full_html: Whether to include full HTML wrapper
        
    Returns:
        Base64 encoded string of the figure
    """
    if format == 'html':
        # Convert to HTML
        html_str = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html)
        # Encode as base64
        base64_encoded = base64.b64encode(html_str.encode('utf-8')).decode('utf-8')
    else:
        # Convert to image
        img_bytes = fig.to_image(format=format, engine="kaleido")
        # Encode as base64
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
    
    return base64_encoded

@handle_errors(logger, VisualizationError)
def _export_figure_to_file(
    fig: Figure, 
    output_path: PathType, 
    format: Optional[str] = None,
    dpi: Optional[int] = None
) -> PathType:
    """
    Saves a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Path to save the figure to
        format: Output format (png, jpg, svg, pdf)
        dpi: Resolution for raster formats
        
    Returns:
        Path to the saved file
    """
    # Convert to Path object
    output_path = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If format is not specified, infer it from the file extension
    if format is None:
        format = output_path.suffix[1:]  # Remove the leading dot
    
    # Use the default DPI if not specified
    if dpi is None:
        dpi = DEFAULT_DPI
    
    # Save the figure
    fig.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
    
    logger.info(f"Figure saved to {output_path}")
    return output_path

@handle_errors(logger, VisualizationError)
def _export_plotly_figure_to_file(
    fig: go.Figure, 
    output_path: PathType,
    include_plotlyjs: Optional[bool] = True,
    full_html: Optional[bool] = False
) -> PathType:
    """
    Saves a plotly figure to a file.
    
    Args:
        fig: Plotly figure to save
        output_path: Path to save the figure to
        include_plotlyjs: Whether to include plotly.js in HTML output
        full_html: Whether to include full HTML wrapper
        
    Returns:
        Path to the saved file
    """
    # Convert to Path object
    output_path = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the file extension
    ext = output_path.suffix[1:].lower()  # Remove the leading dot
    
    if ext == 'html':
        # Save as HTML
        fig.write_html(
            output_path, 
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )
    elif ext == 'json':
        # Save as JSON
        fig.write_json(output_path)
    else:
        # Save as image (png, jpg, svg, etc.)
        fig.write_image(output_path, engine="kaleido")
    
    logger.info(f"Plotly figure saved to {output_path}")
    return output_path

@handle_errors(logger, VisualizationError)
def plot_probability_timeline(
    forecast_df: DataFrameType,
    threshold: Optional[ThresholdValue] = None,
    node_id: Optional[NodeID] = None,
    figsize: Optional[Tuple[int, int]] = None,
    show_confidence_intervals: Optional[bool] = True
) -> Tuple[Figure, Axes]:
    """
    Creates a timeline plot of spike probabilities for a given forecast.
    
    Args:
        forecast_df: DataFrame containing forecast data
        threshold: Optional threshold value to filter by
        node_id: Optional node ID to filter by
        figsize: Optional figure size (width, height) in inches
        show_confidence_intervals: Whether to show confidence intervals
        
    Returns:
        Matplotlib figure and axes objects with the probability timeline plot
    """
    # Validate DataFrame has required columns
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    # Filter by threshold if provided
    if threshold is not None and 'threshold_value' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['threshold_value'] == threshold]
    
    # Filter by node_id if provided
    if node_id is not None and 'node_id' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['node_id'] == node_id]
    
    # Sort by target_timestamp
    forecast_df = forecast_df.sort_values('target_timestamp')
    
    # Create the figure
    if figsize is None:
        figsize = DEFAULT_FIGURE_SIZE
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the probabilities
    ax.plot(
        forecast_df['target_timestamp'], 
        forecast_df['spike_probability'],
        marker='o',
        linestyle='-',
        color=PROBABILITY_COLORS['medium'],
        label='Spike Probability'
    )
    
    # Add confidence intervals if available and requested
    if show_confidence_intervals and 'confidence_interval_lower' in forecast_df.columns and 'confidence_interval_upper' in forecast_df.columns:
        # Check if there are valid confidence intervals
        has_valid_intervals = (~forecast_df['confidence_interval_lower'].isna() & 
                               ~forecast_df['confidence_interval_upper'].isna()).any()
        
        if has_valid_intervals:
            ax.fill_between(
                forecast_df['target_timestamp'],
                forecast_df['confidence_interval_lower'],
                forecast_df['confidence_interval_upper'],
                alpha=0.2,
                color=PROBABILITY_COLORS['medium'],
                label='Confidence Interval'
            )
    
    # Add labels and title
    threshold_str = f" (Threshold: ${threshold} /MWh)" if threshold is not None else ""
    node_str = f" for {node_id}" if node_id is not None else ""
    ax.set_title(f"RTLMP Spike Probability{threshold_str}{node_str}")
    ax.set_xlabel("Target Time")
    ax.set_ylabel("Probability")
    
    # Format x-axis as datetime
    fig.autofmt_xdate()
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if we have confidence intervals
    if show_confidence_intervals and 'confidence_interval_lower' in forecast_df.columns and has_valid_intervals:
        ax.legend()
    
    return fig, ax

@handle_errors(logger, VisualizationError)
def plot_threshold_comparison(
    forecast_df: DataFrameType,
    thresholds: Optional[List[ThresholdValue]] = None,
    node_id: Optional[NodeID] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a plot comparing probabilities across different thresholds.
    
    Args:
        forecast_df: DataFrame containing forecast data
        thresholds: Optional list of threshold values to include
        node_id: Optional node ID to filter by
        figsize: Optional figure size (width, height) in inches
        
    Returns:
        Matplotlib figure and axes objects with the threshold comparison plot
    """
    # Validate DataFrame has required columns
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    # Filter by node_id if provided
    if node_id is not None and 'node_id' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['node_id'] == node_id]
    
    # If 'threshold_value' is not in the DataFrame, we can't compare thresholds
    if 'threshold_value' not in forecast_df.columns:
        raise VisualizationError("Forecast DataFrame does not contain 'threshold_value' column")
    
    # Use all unique thresholds if not specified
    if thresholds is None:
        thresholds = sorted(forecast_df['threshold_value'].unique())
    
    # Create the figure
    if figsize is None:
        figsize = DEFAULT_FIGURE_SIZE
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each threshold
    for threshold in thresholds:
        # Filter for this threshold
        threshold_df = forecast_df[forecast_df['threshold_value'] == threshold]
        
        if threshold_df.empty:
            logger.warning(f"No data found for threshold {threshold}")
            continue
        
        # Sort by target_timestamp
        threshold_df = threshold_df.sort_values('target_timestamp')
        
        # Get color from THRESHOLD_COLORS or use a default if not found
        color = THRESHOLD_COLORS.get(str(int(threshold)), None)
        
        # Plot the probabilities
        ax.plot(
            threshold_df['target_timestamp'], 
            threshold_df['spike_probability'],
            marker='o',
            linestyle='-',
            color=color,
            label=f"${threshold} /MWh"
        )
    
    # Add labels and title
    node_str = f" for {node_id}" if node_id is not None else ""
    ax.set_title(f"RTLMP Spike Probability Comparison{node_str}")
    ax.set_xlabel("Target Time")
    ax.set_ylabel("Probability")
    
    # Format x-axis as datetime
    fig.autofmt_xdate()
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(title="Threshold")
    
    return fig, ax

@handle_errors(logger, VisualizationError)
def plot_node_comparison(
    forecast_df: DataFrameType,
    threshold: Optional[ThresholdValue] = None,
    nodes: Optional[List[NodeID]] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a plot comparing probabilities across different nodes.
    
    Args:
        forecast_df: DataFrame containing forecast data
        threshold: Optional threshold value to filter by
        nodes: Optional list of node IDs to include
        figsize: Optional figure size (width, height) in inches
        
    Returns:
        Matplotlib figure and axes objects with the node comparison plot
    """
    # Validate DataFrame has required columns
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    # Filter by threshold if provided
    if threshold is not None and 'threshold_value' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['threshold_value'] == threshold]
    
    # If 'node_id' is not in the DataFrame, we can't compare nodes
    if 'node_id' not in forecast_df.columns:
        raise VisualizationError("Forecast DataFrame does not contain 'node_id' column")
    
    # Use all unique nodes if not specified
    if nodes is None:
        nodes = sorted(forecast_df['node_id'].unique())
    
    # Create the figure
    if figsize is None:
        figsize = DEFAULT_FIGURE_SIZE
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each node
    for i, node in enumerate(nodes):
        # Filter for this node
        node_df = forecast_df[forecast_df['node_id'] == node]
        
        if node_df.empty:
            logger.warning(f"No data found for node {node}")
            continue
        
        # Sort by target_timestamp
        node_df = node_df.sort_values('target_timestamp')
        
        # Plot the probabilities
        ax.plot(
            node_df['target_timestamp'], 
            node_df['spike_probability'],
            marker='o',
            linestyle='-',
            label=node
        )
    
    # Add labels and title
    threshold_str = f" (Threshold: ${threshold} /MWh)" if threshold is not None else ""
    ax.set_title(f"RTLMP Spike Probability by Node{threshold_str}")
    ax.set_xlabel("Target Time")
    ax.set_ylabel("Probability")
    
    # Format x-axis as datetime
    fig.autofmt_xdate()
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(title="Node")
    
    return fig, ax

@handle_errors(logger, VisualizationError)
def plot_heatmap(
    forecast_df: DataFrameType,
    heatmap_type: str,
    node_id: Optional[NodeID] = None,
    threshold: Optional[ThresholdValue] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a heatmap of spike probabilities across time and thresholds/nodes.
    
    Args:
        forecast_df: DataFrame containing forecast data
        heatmap_type: Type of heatmap ('threshold' or 'node')
        node_id: Optional node ID to filter by (for 'threshold' type)
        threshold: Optional threshold value to filter by (for 'node' type)
        figsize: Optional figure size (width, height) in inches
        cmap: Optional colormap name
        
    Returns:
        Matplotlib figure and axes objects with the heatmap plot
    """
    # Validate DataFrame has required columns
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    # Validate heatmap_type
    if heatmap_type not in ['threshold', 'node']:
        raise VisualizationError(f"Invalid heatmap_type: {heatmap_type}. Must be 'threshold' or 'node'")
    
    # Filter data based on heatmap_type
    if heatmap_type == 'threshold':
        if 'threshold_value' not in forecast_df.columns:
            raise VisualizationError("Forecast DataFrame does not contain 'threshold_value' column")
        
        # Filter by node_id if provided
        if node_id is not None and 'node_id' in forecast_df.columns:
            forecast_df = forecast_df[forecast_df['node_id'] == node_id]
        
        # Pivot the data: rows are timestamps, columns are thresholds
        pivot_df = forecast_df.pivot_table(
            index='target_timestamp',
            columns='threshold_value',
            values='spike_probability',
            aggfunc='mean'  # Use mean in case of duplicates
        )
        
        # Sort columns (thresholds) in ascending order
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Format column labels
        pivot_df.columns = [f"${x} /MWh" for x in pivot_df.columns]
        
        # Set title
        title = "RTLMP Spike Probability by Threshold"
        if node_id is not None:
            title += f" for {node_id}"
    
    else:  # heatmap_type == 'node'
        if 'node_id' not in forecast_df.columns:
            raise VisualizationError("Forecast DataFrame does not contain 'node_id' column")
        
        # Filter by threshold if provided
        if threshold is not None and 'threshold_value' in forecast_df.columns:
            forecast_df = forecast_df[forecast_df['threshold_value'] == threshold]
        
        # Pivot the data: rows are timestamps, columns are nodes
        pivot_df = forecast_df.pivot_table(
            index='target_timestamp',
            columns='node_id',
            values='spike_probability',
            aggfunc='mean'  # Use mean in case of duplicates
        )
        
        # Sort columns (nodes) alphabetically
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Set title
        title = "RTLMP Spike Probability by Node"
        if threshold is not None:
            title += f" (Threshold: ${threshold} /MWh)"
    
    # Create the figure
    if figsize is None:
        figsize = DEFAULT_FIGURE_SIZE
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default colormap if not provided
    if cmap is None:
        cmap = DEFAULT_CMAP
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        cmap=cmap,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Probability'}
    )
    
    # Set title and labels
    ax.set_title(title)
    ax.set_ylabel("Target Time")
    
    if heatmap_type == 'threshold':
        ax.set_xlabel("Threshold")
    else:
        ax.set_xlabel("Node")
    
    # Format x-axis ticks
    plt.xticks(rotation=45, ha='right')
    
    return fig, ax

@handle_errors(logger, VisualizationError)
def create_interactive_timeline(
    forecast_df: DataFrameType,
    threshold: Optional[ThresholdValue] = None,
    node_id: Optional[NodeID] = None,
    show_confidence_intervals: Optional[bool] = True
) -> go.Figure:
    """
    Creates an interactive timeline plot of spike probabilities using Plotly.
    
    Args:
        forecast_df: DataFrame containing forecast data
        threshold: Optional threshold value to filter by
        node_id: Optional node ID to filter by
        show_confidence_intervals: Whether to show confidence intervals
        
    Returns:
        Plotly figure object with the interactive probability timeline
    """
    # Validate DataFrame has required columns
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    # Filter by threshold if provided
    if threshold is not None and 'threshold_value' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['threshold_value'] == threshold]
    
    # Filter by node_id if provided
    if node_id is not None and 'node_id' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['node_id'] == node_id]
    
    # Sort by target_timestamp
    forecast_df = forecast_df.sort_values('target_timestamp')
    
    # Create the figure
    fig = go.Figure()
    
    # Add the probability line
    fig.add_trace(
        go.Scatter(
            x=forecast_df['target_timestamp'],
            y=forecast_df['spike_probability'],
            mode='lines+markers',
            name='Spike Probability',
            line=dict(color=PROBABILITY_COLORS['medium'], width=2),
            marker=dict(size=8),
            hovertemplate='<b>Time:</b> %{x}<br><b>Probability:</b> %{y:.2f}<extra></extra>'
        )
    )
    
    # Add confidence intervals if available and requested
    if show_confidence_intervals and 'confidence_interval_lower' in forecast_df.columns and 'confidence_interval_upper' in forecast_df.columns:
        # Check if there are valid confidence intervals
        has_valid_intervals = (~forecast_df['confidence_interval_lower'].isna() & 
                               ~forecast_df['confidence_interval_upper'].isna()).any()
        
        if has_valid_intervals:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['target_timestamp'],
                    y=forecast_df['confidence_interval_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['target_timestamp'],
                    y=forecast_df['confidence_interval_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=f'rgba{tuple(list(matplotlib.colors.to_rgb(PROBABILITY_COLORS["medium"])) + [0.2])}',
                    fill='tonexty',
                    name='Confidence Interval',
                    hoverinfo='skip'
                )
            )
    
    # Set title and labels
    threshold_str = f" (Threshold: ${threshold} /MWh)" if threshold is not None else ""
    node_str = f" for {node_id}" if node_id is not None else ""
    
    fig.update_layout(
        title=f"RTLMP Spike Probability{threshold_str}{node_str}",
        xaxis_title="Target Time",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Add range slider for interactive time navigation
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        )
    )
    
    return fig

@handle_errors(logger, VisualizationError)
def create_interactive_comparison(
    forecast_df: DataFrameType,
    comparison_type: str,
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None
) -> go.Figure:
    """
    Creates an interactive comparison plot of spike probabilities using Plotly.
    
    Args:
        forecast_df: DataFrame containing forecast data
        comparison_type: Type of comparison ('threshold' or 'node')
        thresholds: Optional list of threshold values to include (for 'node' type)
        nodes: Optional list of node IDs to include (for 'threshold' type)
        
    Returns:
        Plotly figure object with the interactive comparison plot
    """
    # Validate DataFrame has required columns
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    # Validate comparison_type
    if comparison_type not in ['threshold', 'node']:
        raise VisualizationError(f"Invalid comparison_type: {comparison_type}. Must be 'threshold' or 'node'")
    
    # Create the figure
    fig = go.Figure()
    
    if comparison_type == 'threshold':
        # Comparison across thresholds
        if 'threshold_value' not in forecast_df.columns:
            raise VisualizationError("Forecast DataFrame does not contain 'threshold_value' column")
        
        # Filter by node_id if provided
        if nodes is not None and 'node_id' in forecast_df.columns:
            if len(nodes) == 1:
                forecast_df = forecast_df[forecast_df['node_id'] == nodes[0]]
            else:
                raise VisualizationError("For threshold comparison, only one node_id should be provided")
        
        # Use all unique thresholds if not specified
        if thresholds is None:
            thresholds = sorted(forecast_df['threshold_value'].unique())
        
        # Plot each threshold
        for threshold in thresholds:
            # Filter for this threshold
            threshold_df = forecast_df[forecast_df['threshold_value'] == threshold]
            
            if threshold_df.empty:
                logger.warning(f"No data found for threshold {threshold}")
                continue
            
            # Sort by target_timestamp
            threshold_df = threshold_df.sort_values('target_timestamp')
            
            # Get color from THRESHOLD_COLORS or use a default if not found
            color = THRESHOLD_COLORS.get(str(int(threshold)), None)
            
            # Add trace for this threshold
            fig.add_trace(
                go.Scatter(
                    x=threshold_df['target_timestamp'],
                    y=threshold_df['spike_probability'],
                    mode='lines+markers',
                    name=f"${threshold} /MWh",
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    hovertemplate='<b>Time:</b> %{x}<br><b>Probability:</b> %{y:.2f}<extra></extra>'
                )
            )
        
        # Set title and labels
        node_str = f" for {nodes[0]}" if nodes is not None and len(nodes) == 1 else ""
        title = f"RTLMP Spike Probability Comparison by Threshold{node_str}"
        
    else:  # comparison_type == 'node'
        # Comparison across nodes
        if 'node_id' not in forecast_df.columns:
            raise VisualizationError("Forecast DataFrame does not contain 'node_id' column")
        
        # Filter by threshold if provided
        if thresholds is not None and 'threshold_value' in forecast_df.columns:
            if len(thresholds) == 1:
                forecast_df = forecast_df[forecast_df['threshold_value'] == thresholds[0]]
            else:
                raise VisualizationError("For node comparison, only one threshold should be provided")
        
        # Use all unique nodes if not specified
        if nodes is None:
            nodes = sorted(forecast_df['node_id'].unique())
        
        # Plot each node
        for node in nodes:
            # Filter for this node
            node_df = forecast_df[forecast_df['node_id'] == node]
            
            if node_df.empty:
                logger.warning(f"No data found for node {node}")
                continue
            
            # Sort by target_timestamp
            node_df = node_df.sort_values('target_timestamp')
            
            # Add trace for this node
            fig.add_trace(
                go.Scatter(
                    x=node_df['target_timestamp'],
                    y=node_df['spike_probability'],
                    mode='lines+markers',
                    name=node,
                    marker=dict(size=8),
                    hovertemplate='<b>Time:</b> %{x}<br><b>Probability:</b> %{y:.2f}<extra></extra>'
                )
            )
        
        # Set title and labels
        threshold_str = f" (Threshold: ${thresholds[0]} /MWh)" if thresholds is not None and len(thresholds) == 1 else ""
        title = f"RTLMP Spike Probability Comparison by Node{threshold_str}"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Target Time",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Add range slider for interactive time navigation
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        )
    )
    
    # Add buttons for selecting specific series
    if comparison_type == 'threshold' and len(thresholds) > 1:
        # Create buttons for each threshold
        buttons = []
        for i, threshold in enumerate(thresholds):
            visibility = [i == j for j in range(len(thresholds))]
            buttons.append(
                dict(
                    args=[{"visible": visibility}],
                    label=f"${threshold} /MWh",
                    method="update"
                )
            )
        
        # Add "All Thresholds" button
        buttons.append(
            dict(
                args=[{"visible": [True] * len(thresholds)}],
                label="All Thresholds",
                method="update"
            )
        )
        
        # Add dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=buttons,
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )
    
    elif comparison_type == 'node' and len(nodes) > 1:
        # Create buttons for each node
        buttons = []
        for i, node in enumerate(nodes):
            visibility = [i == j for j in range(len(nodes))]
            buttons.append(
                dict(
                    args=[{"visible": visibility}],
                    label=node,
                    method="update"
                )
            )
        
        # Add "All Nodes" button
        buttons.append(
            dict(
                args=[{"visible": [True] * len(nodes)}],
                label="All Nodes",
                method="update"
            )
        )
        
        # Add dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=buttons,
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )
    
    return fig

@handle_errors(logger, VisualizationError)
def create_interactive_heatmap(
    forecast_df: DataFrameType,
    heatmap_type: str,
    node_id: Optional[NodeID] = None,
    threshold: Optional[ThresholdValue] = None
) -> go.Figure:
    """
    Creates an interactive heatmap of spike probabilities using Plotly.
    
    Args:
        forecast_df: DataFrame containing forecast data
        heatmap_type: Type of heatmap ('threshold' or 'node')
        node_id: Optional node ID to filter by (for 'threshold' type)
        threshold: Optional threshold value to filter by (for 'node' type)
        
    Returns:
        Plotly figure object with the interactive heatmap
    """
    # Validate DataFrame has required columns
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    # Validate heatmap_type
    if heatmap_type not in ['threshold', 'node']:
        raise VisualizationError(f"Invalid heatmap_type: {heatmap_type}. Must be 'threshold' or 'node'")
    
    # Filter data based on heatmap_type
    if heatmap_type == 'threshold':
        if 'threshold_value' not in forecast_df.columns:
            raise VisualizationError("Forecast DataFrame does not contain 'threshold_value' column")
        
        # Filter by node_id if provided
        if node_id is not None and 'node_id' in forecast_df.columns:
            forecast_df = forecast_df[forecast_df['node_id'] == node_id]
        
        # Pivot the data: rows are timestamps, columns are thresholds
        pivot_df = forecast_df.pivot_table(
            index='target_timestamp',
            columns='threshold_value',
            values='spike_probability',
            aggfunc='mean'  # Use mean in case of duplicates
        )
        
        # Sort columns (thresholds) in ascending order
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Format column labels
        column_labels = [f"${x} /MWh" for x in pivot_df.columns]
        
        # Set title
        title = "RTLMP Spike Probability by Threshold"
        if node_id is not None:
            title += f" for {node_id}"
        
        # Axis titles
        x_title = "Threshold ($/MWh)"
        y_title = "Target Time"
    
    else:  # heatmap_type == 'node'
        if 'node_id' not in forecast_df.columns:
            raise VisualizationError("Forecast DataFrame does not contain 'node_id' column")
        
        # Filter by threshold if provided
        if threshold is not None and 'threshold_value' in forecast_df.columns:
            forecast_df = forecast_df[forecast_df['threshold_value'] == threshold]
        
        # Pivot the data: rows are timestamps, columns are nodes
        pivot_df = forecast_df.pivot_table(
            index='target_timestamp',
            columns='node_id',
            values='spike_probability',
            aggfunc='mean'  # Use mean in case of duplicates
        )
        
        # Sort columns (nodes) alphabetically
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Format column labels
        column_labels = pivot_df.columns.tolist()
        
        # Set title
        title = "RTLMP Spike Probability by Node"
        if threshold is not None:
            title += f" (Threshold: ${threshold} /MWh)"
        
        # Axis titles
        x_title = "Node"
        y_title = "Target Time"
    
    # Get the data for the heatmap
    z = pivot_df.values
    x = column_labels
    y = pivot_df.index
    
    # Create the figure
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        colorbar=dict(title="Probability"),
        hovertemplate='<b>Time:</b> %{y}<br><b>Value:</b> %{x}<br><b>Probability:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(
            tickangle=45,
            tickmode='array',
            tickvals=x
        ),
        yaxis=dict(
            autorange="reversed"  # To have earliest dates at the top
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

@handle_errors(logger, VisualizationError)
def create_forecast_dashboard(
    forecast_df: DataFrameType,
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None
) -> go.Figure:
    """
    Creates a comprehensive dashboard with multiple forecast visualizations.
    
    Args:
        forecast_df: DataFrame containing forecast data
        thresholds: Optional list of threshold values to include
        nodes: Optional list of node IDs to include
        
    Returns:
        Plotly figure object with the forecast dashboard
    """
    # Validate DataFrame has required columns
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    # Check if we have threshold and node columns
    has_thresholds = 'threshold_value' in forecast_df.columns
    has_nodes = 'node_id' in forecast_df.columns
    
    # Use all unique thresholds/nodes if not specified
    if has_thresholds and thresholds is None:
        thresholds = sorted(forecast_df['threshold_value'].unique())
    
    if has_nodes and nodes is None:
        nodes = sorted(forecast_df['node_id'].unique())
    
    # Create subplot figure
    if has_nodes and len(nodes) > 1:
        # Four subplots: timeline, threshold comparison, node comparison, heatmap
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                "Probability Timeline", 
                "Threshold Comparison", 
                "Node Comparison", 
                "Probability Heatmap"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
    else:
        # Three subplots: timeline, threshold comparison, heatmap
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                "Probability Timeline", 
                "Threshold Comparison", 
                "", 
                "Probability Heatmap"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter", "colspan": 1}, {"type": "heatmap"}]
            ]
        )
    
    # Add probability timeline
    # Filter for first threshold and node for timeline if needed
    timeline_df = forecast_df
    
    if has_thresholds and thresholds:
        timeline_df = timeline_df[timeline_df['threshold_value'] == thresholds[0]]
    
    if has_nodes and nodes:
        timeline_df = timeline_df[timeline_df['node_id'] == nodes[0]]
    
    timeline_df = timeline_df.sort_values('target_timestamp')
    
    fig.add_trace(
        go.Scatter(
            x=timeline_df['target_timestamp'],
            y=timeline_df['spike_probability'],
            mode='lines+markers',
            name='Spike Probability',
            line=dict(color=PROBABILITY_COLORS['medium'], width=2),
            marker=dict(size=6),
            hovertemplate='<b>Time:</b> %{x}<br><b>Probability:</b> %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add confidence intervals if available
    if ('confidence_interval_lower' in timeline_df.columns and 
        'confidence_interval_upper' in timeline_df.columns and
        (~timeline_df['confidence_interval_lower'].isna() & 
         ~timeline_df['confidence_interval_upper'].isna()).any()):
        
        fig.add_trace(
            go.Scatter(
                x=timeline_df['target_timestamp'],
                y=timeline_df['confidence_interval_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timeline_df['target_timestamp'],
                y=timeline_df['confidence_interval_lower'],
                mode='lines',
                line=dict(width=0),
                fillcolor=f'rgba{tuple(list(matplotlib.colors.to_rgb(PROBABILITY_COLORS["medium"])) + [0.2])}',
                fill='tonexty',
                name='Confidence Interval',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    # Add threshold comparison if we have multiple thresholds
    if has_thresholds and len(thresholds) > 1:
        # Filter for first node if we have nodes
        comparison_df = forecast_df
        if has_nodes and nodes:
            comparison_df = comparison_df[comparison_df['node_id'] == nodes[0]]
        
        for threshold in thresholds:
            threshold_df = comparison_df[comparison_df['threshold_value'] == threshold]
            if threshold_df.empty:
                continue
            
            threshold_df = threshold_df.sort_values('target_timestamp')
            color = THRESHOLD_COLORS.get(str(int(threshold)), None)
            
            fig.add_trace(
                go.Scatter(
                    x=threshold_df['target_timestamp'],
                    y=threshold_df['spike_probability'],
                    mode='lines+markers',
                    name=f"${threshold} /MWh",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    legendgroup=f"threshold_{threshold}",
                    hovertemplate='<b>Time:</b> %{x}<br><b>Probability:</b> %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
    
    # Add node comparison if we have multiple nodes
    if has_nodes and len(nodes) > 1:
        # Filter for first threshold if we have thresholds
        node_comparison_df = forecast_df
        if has_thresholds and thresholds:
            node_comparison_df = node_comparison_df[node_comparison_df['threshold_value'] == thresholds[0]]
        
        for node in nodes:
            node_df = node_comparison_df[node_comparison_df['node_id'] == node]
            if node_df.empty:
                continue
            
            node_df = node_df.sort_values('target_timestamp')
            
            fig.add_trace(
                go.Scatter(
                    x=node_df['target_timestamp'],
                    y=node_df['spike_probability'],
                    mode='lines+markers',
                    name=node,
                    marker=dict(size=6),
                    legendgroup=f"node_{node}",
                    hovertemplate='<b>Time:</b> %{x}<br><b>Probability:</b> %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Add heatmap
    # Determine heatmap type based on available data
    if has_thresholds and has_nodes:
        # Create a threshold heatmap
        if has_nodes and nodes:
            # Filter for first node
            heatmap_df = forecast_df[forecast_df['node_id'] == nodes[0]]
        else:
            heatmap_df = forecast_df
        
        # Pivot the data
        pivot_df = heatmap_df.pivot_table(
            index='target_timestamp',
            columns='threshold_value',
            values='spike_probability',
            aggfunc='mean'
        )
        
        # Sort columns
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Format column labels
        column_labels = [f"${x} /MWh" for x in pivot_df.columns]
        
        # Prepare data for heatmap
        z = pivot_df.values
        x = column_labels
        y = pivot_df.index
        
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale='Viridis',
                zmin=0,
                zmax=1,
                colorbar=dict(title="Probability", len=0.5, y=0.25),
                hovertemplate='<b>Time:</b> %{y}<br><b>Threshold:</b> %{x}<br><b>Probability:</b> %{z:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update heatmap axis titles
        fig.update_xaxes(title_text="Threshold", row=2, col=2)
        fig.update_yaxes(title_text="Target Time", row=2, col=2)
    
    # Update subplot layouts
    fig.update_xaxes(title_text="Target Time", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=1, range=[0, 1])
    
    fig.update_xaxes(title_text="Target Time", row=1, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=2, range=[0, 1])
    
    if has_nodes and len(nodes) > 1:
        fig.update_xaxes(title_text="Target Time", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=1, range=[0, 1])
    
    # Update overall layout
    threshold_str = f" (Threshold: ${thresholds[0]} /MWh)" if has_thresholds and thresholds and len(thresholds) == 1 else ""
    node_str = f" for {nodes[0]}" if has_nodes and nodes and len(nodes) == 1 else ""
    
    fig.update_layout(
        title=f"RTLMP Spike Probability Forecast{threshold_str}{node_str}",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Add dropdown menu for selecting different views if we have multiple thresholds/nodes
    if (has_thresholds and len(thresholds) > 1) or (has_nodes and len(nodes) > 1):
        dropdown_buttons = []
        
        if has_thresholds and len(thresholds) > 1:
            # Add buttons for each threshold
            for threshold in thresholds:
                dropdown_buttons.append(
                    dict(
                        args=[
                            {
                                "visible": [
                                    True if "threshold_" + str(threshold) in trace.legendgroup 
                                    else True if "threshold_" not in getattr(trace, "legendgroup", "") 
                                    else False 
                                    for trace in fig.data
                                ]
                            }
                        ],
                        label=f"Threshold: ${threshold} /MWh",
                        method="update"
                    )
                )
        
        if has_nodes and len(nodes) > 1:
            # Add buttons for each node
            for node in nodes:
                dropdown_buttons.append(
                    dict(
                        args=[
                            {
                                "visible": [
                                    True if "node_" + node in trace.legendgroup 
                                    else True if "node_" not in getattr(trace, "legendgroup", "") 
                                    else False 
                                    for trace in fig.data
                                ]
                            }
                        ],
                        label=f"Node: {node}",
                        method="update"
                    )
                )
        
        # Add "Show All" button
        dropdown_buttons.append(
            dict(
                args=[{"visible": [True] * len(fig.data)}],
                label="Show All",
                method="update"
            )
        )
        
        # Add dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=dropdown_buttons,
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )
    
    return fig

@handle_errors(logger, VisualizationError)
def get_forecast_for_visualization(
    forecast: Union[DataFrameType, datetime.datetime, str, None] = None,
    thresholds: Optional[List[ThresholdValue]] = None,
    nodes: Optional[List[NodeID]] = None,
    repository_path: Optional[PathType] = None
) -> DataFrameType:
    """
    Retrieves forecast data for visualization purposes.
    
    Args:
        forecast: DataFrame with forecast data, datetime for specific forecast, or None for latest
        thresholds: Optional list of threshold values to filter by
        nodes: Optional list of node IDs to filter by
        repository_path: Optional path to the forecast repository
        
    Returns:
        Forecast DataFrame ready for visualization
    """
    # If forecast is None, get the latest forecast
    if forecast is None:
        repo = ForecastRepository(repository_path)
        forecast_df, metadata, _ = repo.get_latest_forecast(thresholds=thresholds, nodes=nodes)
        
        if forecast_df is None:
            raise VisualizationError("No forecast data found")
        
        logger.info("Retrieved latest forecast for visualization")
        
    # If forecast is a datetime or string, get the forecast for that date
    elif isinstance(forecast, (datetime.datetime, str)):
        if isinstance(forecast, str):
            try:
                forecast_date = datetime.datetime.fromisoformat(forecast)
            except ValueError:
                raise VisualizationError(f"Invalid date format: {forecast}")
        else:
            forecast_date = forecast
        
        # Get the next day as the end date
        end_date = forecast_date + datetime.timedelta(days=1)
        
        repo = ForecastRepository(repository_path)
        forecast_df = repo.retrieve_forecasts_by_date_range(
            start_date=forecast_date,
            end_date=end_date,
            thresholds=thresholds,
            nodes=nodes
        )
        
        if forecast_df.empty:
            raise VisualizationError(f"No forecast data found for date {forecast_date}")
        
        logger.info(f"Retrieved forecast for {forecast_date} for visualization")
        
    # If forecast is already a DataFrame, use it directly
    elif isinstance(forecast, pd.DataFrame):
        forecast_df = forecast
        
        # Filter by thresholds and nodes if provided
        if thresholds is not None and 'threshold_value' in forecast_df.columns:
            forecast_df = forecast_df[forecast_df['threshold_value'].isin(thresholds)]
        
        if nodes is not None and 'node_id' in forecast_df.columns:
            forecast_df = forecast_df[forecast_df['node_id'].isin(nodes)]
    
    else:
        raise VisualizationError(f"Invalid forecast type: {type(forecast)}")
    
    # Validate that we have the required columns for visualization
    required_columns = ['target_timestamp', 'spike_probability']
    if not all(col in forecast_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in forecast_df.columns]
        raise VisualizationError(f"Forecast DataFrame missing required columns: {missing}")
    
    return forecast_df

@handle_errors(logger, VisualizationError)
def save_plot(
    fig: Figure, 
    output_path: PathType, 
    format: Optional[str] = None,
    dpi: Optional[int] = None
) -> PathType:
    """
    Saves a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Path to save the figure to
        format: Output format (png, jpg, svg, pdf)
        dpi: Resolution for raster formats
        
    Returns:
        Path to the saved file
    """
    return _export_figure_to_file(fig, output_path, format, dpi)

@handle_errors(logger, VisualizationError)
def save_interactive_plot(
    fig: go.Figure, 
    output_path: PathType,
    include_plotlyjs: Optional[bool] = True,
    full_html: Optional[bool] = False
) -> PathType:
    """
    Saves a Plotly figure to an HTML file.
    
    Args:
        fig: Plotly figure to save
        output_path: Path to save the figure to
        include_plotlyjs: Whether to include plotly.js in HTML output
        full_html: Whether to include full HTML wrapper
        
    Returns:
        Path to the saved file
    """
    return _export_plotly_figure_to_file(fig, output_path, include_plotlyjs, full_html)

class ForecastPlotter:
    """
    Class for creating and managing forecast visualizations.
    """
    
    def __init__(self, repository_path: Optional[PathType] = None, plot_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ForecastPlotter with a forecast repository path.
        
        Args:
            repository_path: Path to the forecast repository
            plot_config: Configuration dictionary for plotting
        """
        self._repository = ForecastRepository(repository_path)
        self._forecast_df = None
        self._metadata = {}
        
        # Set default plot configuration
        self._plot_config = {
            'figsize': DEFAULT_FIGURE_SIZE,
            'dpi': DEFAULT_DPI,
            'cmap': DEFAULT_CMAP,
            'show_confidence_intervals': True
        }
        
        # Update with user-provided configuration
        if plot_config:
            self._plot_config.update(plot_config)
    
    def load_forecast(
        self,
        forecast: Union[DataFrameType, datetime.datetime, str, None] = None,
        thresholds: Optional[List[ThresholdValue]] = None,
        nodes: Optional[List[NodeID]] = None
    ) -> bool:
        """
        Loads forecast data for visualization.
        
        Args:
            forecast: DataFrame with forecast data, datetime for specific forecast, or None for latest
            thresholds: Optional list of threshold values to filter by
            nodes: Optional list of node IDs to filter by
            
        Returns:
            True if forecast loaded successfully, False otherwise
        """
        try:
            self._forecast_df = get_forecast_for_visualization(
                forecast=forecast,
                thresholds=thresholds,
                nodes=nodes,
                repository_path=self._repository._forecast_root
            )
            
            # Load metadata if available (for None or datetime forecast)
            if forecast is None or isinstance(forecast, (datetime.datetime, str)):
                # Get a sample row to extract metadata
                if not self._forecast_df.empty:
                    self._metadata = {
                        'forecast_timestamp': self._forecast_df['forecast_timestamp'].iloc[0],
                        'threshold_values': sorted(self._forecast_df['threshold_value'].unique().tolist()) 
                                        if 'threshold_value' in self._forecast_df.columns else [],
                        'nodes': sorted(self._forecast_df['node_id'].unique().tolist())
                                if 'node_id' in self._forecast_df.columns else []
                    }
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading forecast: {str(e)}")
            return False
    
    def plot_probability_timeline(
        self, 
        threshold: Optional[ThresholdValue] = None,
        node_id: Optional[NodeID] = None,
        figsize: Optional[Tuple[int, int]] = None,
        show_confidence_intervals: Optional[bool] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a timeline plot of spike probabilities.
        
        Args:
            threshold: Optional threshold value to filter by
            node_id: Optional node ID to filter by
            figsize: Optional figure size (width, height) in inches
            show_confidence_intervals: Whether to show confidence intervals
            
        Returns:
            Matplotlib figure and axes objects with the probability timeline plot
        """
        if self._forecast_df is None:
            raise VisualizationError("No forecast data loaded. Call load_forecast() first.")
        
        if figsize is None:
            figsize = self._plot_config.get('figsize')
        
        if show_confidence_intervals is None:
            show_confidence_intervals = self._plot_config.get('show_confidence_intervals')
        
        return plot_probability_timeline(
            self._forecast_df,
            threshold=threshold,
            node_id=node_id,
            figsize=figsize,
            show_confidence_intervals=show_confidence_intervals
        )
    
    def plot_threshold_comparison(
        self, 
        thresholds: Optional[List[ThresholdValue]] = None,
        node_id: Optional[NodeID] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a plot comparing probabilities across different thresholds.
        
        Args:
            thresholds: Optional list of threshold values to include
            node_id: Optional node ID to filter by
            figsize: Optional figure size (width, height) in inches
            
        Returns:
            Matplotlib figure and axes objects with the threshold comparison plot
        """
        if self._forecast_df is None:
            raise VisualizationError("No forecast data loaded. Call load_forecast() first.")
        
        if figsize is None:
            figsize = self._plot_config.get('figsize')
        
        return plot_threshold_comparison(
            self._forecast_df,
            thresholds=thresholds,
            node_id=node_id,
            figsize=figsize
        )
    
    def plot_node_comparison(
        self, 
        threshold: Optional[ThresholdValue] = None,
        nodes: Optional[List[NodeID]] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a plot comparing probabilities across different nodes.
        
        Args:
            threshold: Optional threshold value to filter by
            nodes: Optional list of node IDs to include
            figsize: Optional figure size (width, height) in inches
            
        Returns:
            Matplotlib figure and axes objects with the node comparison plot
        """
        if self._forecast_df is None:
            raise VisualizationError("No forecast data loaded. Call load_forecast() first.")
        
        if figsize is None:
            figsize = self._plot_config.get('figsize')
        
        return plot_node_comparison(
            self._forecast_df,
            threshold=threshold,
            nodes=nodes,
            figsize=figsize
        )
    
    def plot_heatmap(
        self, 
        heatmap_type: str,
        node_id: Optional[NodeID] = None,
        threshold: Optional[ThresholdValue] = None,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a heatmap of spike probabilities.
        
        Args:
            heatmap_type: Type of heatmap ('threshold' or 'node')
            node_id: Optional node ID to filter by (for 'threshold' type)
            threshold: Optional threshold value to filter by (for 'node' type)
            figsize: Optional figure size (width, height) in inches
            cmap: Optional colormap name
            
        Returns:
            Matplotlib figure and axes objects with the heatmap plot
        """
        if self._forecast_df is None:
            raise VisualizationError("No forecast data loaded. Call load_forecast() first.")
        
        if figsize is None:
            figsize = self._plot_config.get('figsize')
        
        if cmap is None:
            cmap = self._plot_config.get('cmap')
        
        return plot_heatmap(
            self._forecast_df,
            heatmap_type=heatmap_type,
            node_id=node_id,
            threshold=threshold,
            figsize=figsize,
            cmap=cmap
        )
    
    def create_interactive_timeline(
        self, 
        threshold: Optional[ThresholdValue] = None,
        node_id: Optional[NodeID] = None,
        show_confidence_intervals: Optional[bool] = None
    ) -> go.Figure:
        """
        Creates an interactive timeline plot of spike probabilities.
        
        Args:
            threshold: Optional threshold value to filter by
            node_id: Optional node ID to filter by
            show_confidence_intervals: Whether to show confidence intervals
            
        Returns:
            Plotly figure object with the interactive probability timeline
        """
        if self._forecast_df is None:
            raise VisualizationError("No forecast data loaded. Call load_forecast() first.")
        
        if show_confidence_intervals is None:
            show_confidence_intervals = self._plot_config.get('show_confidence_intervals')
        
        return create_interactive_timeline(
            self._forecast_df,
            threshold=threshold,
            node_id=node_id,
            show_confidence_intervals=show_confidence_intervals
        )
    
    def create_interactive_comparison(
        self, 
        comparison_type: str,
        thresholds: Optional[List[ThresholdValue]] = None,
        nodes: Optional[List[NodeID]] = None
    ) -> go.Figure:
        """
        Creates an interactive comparison plot of spike probabilities.
        
        Args:
            comparison_type: Type of comparison ('threshold' or 'node')
            thresholds: Optional list of threshold values to include (for 'node' type)
            nodes: Optional list of node IDs to include (for 'threshold' type)
            
        Returns:
            Plotly figure object with the interactive comparison plot
        """
        if self._forecast_df is None:
            raise VisualizationError("No forecast data loaded. Call load_forecast() first.")
        
        return create_interactive_comparison(
            self._forecast_df,
            comparison_type=comparison_type,
            thresholds=thresholds,
            nodes=nodes
        )
    
    def create_forecast_dashboard(
        self, 
        thresholds: Optional[List[ThresholdValue]] = None,
        nodes: Optional[List[NodeID]] = None
    ) -> go.Figure:
        """
        Creates a comprehensive dashboard with multiple forecast visualizations.
        
        Args:
            thresholds: Optional list of threshold values to include
            nodes: Optional list of node IDs to include
            
        Returns:
            Plotly figure object with the forecast dashboard
        """
        if self._forecast_df is None:
            raise VisualizationError("No forecast data loaded. Call load_forecast() first.")
        
        return create_forecast_dashboard(
            self._forecast_df,
            thresholds=thresholds,
            nodes=nodes
        )
    
    def save_plot(
        self, 
        fig: Figure, 
        output_path: PathType, 
        format: Optional[str] = None,
        dpi: Optional[int] = None
    ) -> PathType:
        """
        Saves a matplotlib figure to a file.
        
        Args:
            fig: Matplotlib figure to save
            output_path: Path to save the figure to
            format: Output format (png, jpg, svg, pdf)
            dpi: Resolution for raster formats
            
        Returns:
            Path to the saved file
        """
        if dpi is None:
            dpi = self._plot_config.get('dpi')
        
        return save_plot(fig, output_path, format, dpi)
    
    def save_interactive_plot(
        self, 
        fig: go.Figure, 
        output_path: PathType,
        include_plotlyjs: Optional[bool] = True,
        full_html: Optional[bool] = False
    ) -> PathType:
        """
        Saves a Plotly figure to an HTML file.
        
        Args:
            fig: Plotly figure to save
            output_path: Path to save the figure to
            include_plotlyjs: Whether to include plotly.js in HTML output
            full_html: Whether to include full HTML wrapper
            
        Returns:
            Path to the saved file
        """
        return save_interactive_plot(fig, output_path, include_plotlyjs, full_html)
    
    def get_forecast_info(self) -> Dict[str, Any]:
        """
        Gets information about the currently loaded forecast.
        
        Returns:
            Dictionary with forecast information
        """
        if self._forecast_df is None:
            raise VisualizationError("No forecast data loaded. Call load_forecast() first.")
        
        # Get basic forecast information
        forecast_info = {
            'row_count': len(self._forecast_df),
            'forecast_timestamp': self._forecast_df['forecast_timestamp'].iloc[0] if 'forecast_timestamp' in self._forecast_df.columns else None,
            'target_timestamp_range': [
                self._forecast_df['target_timestamp'].min(),
                self._forecast_df['target_timestamp'].max()
            ] if 'target_timestamp' in self._forecast_df.columns else None
        }
        
        # Add threshold information if available
        if 'threshold_value' in self._forecast_df.columns:
            forecast_info['thresholds'] = sorted(self._forecast_df['threshold_value'].unique().tolist())
        
        # Add node information if available
        if 'node_id' in self._forecast_df.columns:
            forecast_info['nodes'] = sorted(self._forecast_df['node_id'].unique().tolist())
        
        # Add metadata if available
        if self._metadata:
            forecast_info['metadata'] = self._metadata
        
        return forecast_info