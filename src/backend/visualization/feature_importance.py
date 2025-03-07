"""
Implements visualization tools for feature importance in the ERCOT RTLMP spike prediction system.

This module provides functions and classes for creating static and interactive visualizations
of feature importance metrics, helping users understand which features have the most
significant impact on model predictions.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path

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
import networkx as nx  # version 2.8+

from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, PathType, ModelType
from ..utils.logging import get_logger
from ..utils.error_handling import handle_errors, VisualizationError
from ..data.storage.model_registry import ModelRegistry
from ..features.feature_registry import FeatureRegistry
from .export import (
    export_figure_to_file, 
    export_plotly_figure_to_file, 
    figure_to_base64, 
    plotly_figure_to_base64
)

# Initialize logger
logger = get_logger(__name__)

# Global constants
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 100
DEFAULT_CMAP = 'viridis'
GROUP_COLORS = {
    'time': '#1f77b4',
    'statistical': '#ff7f0e',
    'weather': '#2ca02c',
    'market': '#d62728'
}
MAX_FEATURES_TO_DISPLAY = 20


@handle_errors(logger, VisualizationError)
def plot_feature_importance(
    feature_importance: Dict[str, float], 
    n_features: Optional[int] = None, 
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a bar chart of feature importance scores.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        n_features: Maximum number of features to display (default: MAX_FEATURES_TO_DISPLAY)
        figsize: Figure size as (width, height) in inches
        fig: Existing figure to plot on (optional)
        ax: Existing axes to plot on (optional)
        
    Returns:
        Tuple containing the figure and axes objects
    """
    # Validate input
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = MAX_FEATURES_TO_DISPLAY
    
    # Sort features by importance (descending) and take top n
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:n_features]
    
    # Extract feature names and importance values
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_feature_importance_history(
    importance_history: DataFrameType,
    features: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a line plot showing how feature importance has changed over time.
    
    Args:
        importance_history: DataFrame with timestamp index and feature importance columns
        features: List of features to include (default: top MAX_FEATURES_TO_DISPLAY by latest values)
        figsize: Figure size as (width, height) in inches
        fig: Existing figure to plot on (optional)
        ax: Existing axes to plot on (optional)
        
    Returns:
        Tuple containing the figure and axes objects
    """
    # Validate input
    if importance_history.empty:
        raise VisualizationError("Feature importance history DataFrame is empty")
    
    # If features not specified, select top N features from the latest timestamp
    if features is None:
        latest_importance = importance_history.iloc[-1].sort_values(ascending=False)
        features = latest_importance.index[:MAX_FEATURES_TO_DISPLAY].tolist()
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Plot importance values over time for each feature
    for feature in features:
        if feature in importance_history.columns:
            ax.plot(importance_history.index, importance_history[feature], label=feature)
    
    # Add labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance Over Time')
    ax.legend(loc='best')
    
    # Format x-axis as dates
    fig.autofmt_xdate()
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_feature_group_importance(
    feature_importance: Dict[str, float],
    feature_registry: Optional[FeatureRegistry] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a pie chart showing the relative importance of feature groups.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        feature_registry: Feature registry to get feature group information (optional)
        figsize: Figure size as (width, height) in inches
        fig: Existing figure to plot on (optional)
        ax: Existing axes to plot on (optional)
        
    Returns:
        Tuple containing the figure and axes objects
    """
    # Validate input
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Create feature registry if not provided
    if feature_registry is None:
        feature_registry = FeatureRegistry()
    
    # Group features by category
    group_importance = {'time': 0.0, 'statistical': 0.0, 'weather': 0.0, 'market': 0.0}
    
    for feature, importance in feature_importance.items():
        # Get feature metadata
        metadata = feature_registry.get_feature(feature)
        
        if metadata and 'group' in metadata:
            group = metadata['group']
            if group in group_importance:
                group_importance[group] += importance
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Create pie chart
    labels = [f"{group} ({value:.2f})" for group, value in group_importance.items() if value > 0]
    values = [value for value in group_importance.values() if value > 0]
    colors = [GROUP_COLORS[group] for group, value in group_importance.items() if value > 0]
    
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%', 
        startangle=90,
        wedgeprops={'edgecolor': 'w'}
    )
    
    # Enhance the appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Feature Importance by Group')
    ax.axis('equal')  # Equal aspect ratio ensures pie is circular
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_feature_correlation_heatmap(
    features_df: DataFrameType,
    feature_importance: Dict[str, float],
    n_features: Optional[int] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a heatmap showing correlations between top features.
    
    Args:
        features_df: DataFrame containing feature values
        feature_importance: Dictionary mapping feature names to importance scores
        n_features: Maximum number of features to display (default: MAX_FEATURES_TO_DISPLAY)
        cmap: Colormap for the heatmap (default: DEFAULT_CMAP)
        figsize: Figure size as (width, height) in inches
        fig: Existing figure to plot on (optional)
        ax: Existing axes to plot on (optional)
        
    Returns:
        Tuple containing the figure and axes objects
    """
    # Validate input
    if features_df.empty:
        raise VisualizationError("Features DataFrame is empty")
    
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = MAX_FEATURES_TO_DISPLAY
    
    # Set default colormap if not specified
    if cmap is None:
        cmap = DEFAULT_CMAP
    
    # Sort features by importance (descending) and take top n
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:n_features]
    
    # Extract feature names
    feature_names = [item[0] for item in sorted_features]
    
    # Filter features that exist in the DataFrame
    existing_features = [f for f in feature_names if f in features_df.columns]
    
    if not existing_features:
        raise VisualizationError(
            "None of the top features found in the features DataFrame", 
            {"missing_features": feature_names}
        )
    
    # Calculate correlation matrix
    corr_matrix = features_df[existing_features].corr()
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap=cmap, 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        fmt='.2f', 
        ax=ax
    )
    
    ax.set_title('Feature Correlation Heatmap')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def plot_feature_dependency_graph(
    feature_importance: Dict[str, float],
    feature_registry: Optional[FeatureRegistry] = None,
    n_features: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Creates a network graph showing dependencies between features.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        feature_registry: Feature registry to get dependency information (optional)
        n_features: Maximum number of features to display (default: MAX_FEATURES_TO_DISPLAY)
        figsize: Figure size as (width, height) in inches
        fig: Existing figure to plot on (optional)
        ax: Existing axes to plot on (optional)
        
    Returns:
        Tuple containing the figure and axes objects
    """
    # Validate input
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Create feature registry if not provided
    if feature_registry is None:
        feature_registry = FeatureRegistry()
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = MAX_FEATURES_TO_DISPLAY
    
    # Sort features by importance (descending) and take top n
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:n_features]
    
    # Extract feature names and importance values
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes with importance as size
    for i, feature in enumerate(feature_names):
        # Get feature metadata
        metadata = feature_registry.get_feature(feature)
        
        # Determine node color based on feature group
        group = metadata.get('group', 'unknown') if metadata else 'unknown'
        color = GROUP_COLORS.get(group, '#888888')
        
        # Add node with attributes
        G.add_node(
            feature, 
            importance=importance_values[i],
            group=group,
            color=color
        )
        
        # Add edges for dependencies
        if metadata and 'dependencies' in metadata:
            for dependency in metadata['dependencies']:
                if dependency in feature_names:
                    G.add_edge(dependency, feature)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    
    # Create layout for the graph
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_sizes = [G.nodes[n]['importance'] * 1000 for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    nx.draw_networkx_nodes(
        G, pos, 
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        ax=ax,
        arrows=True,
        arrowsize=15,
        width=1.5,
        alpha=0.7
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, 
        ax=ax,
        font_size=10,
        font_family='sans-serif'
    )
    
    ax.set_title('Feature Dependency Graph')
    ax.axis('off')  # Hide axes
    
    return fig, ax


@handle_errors(logger, VisualizationError)
def create_interactive_feature_importance(
    feature_importance: Dict[str, float],
    n_features: Optional[int] = None
) -> go.Figure:
    """
    Creates an interactive bar chart of feature importance scores using Plotly.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        n_features: Maximum number of features to display (default: MAX_FEATURES_TO_DISPLAY)
        
    Returns:
        Plotly figure object with the interactive feature importance plot
    """
    # Validate input
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = MAX_FEATURES_TO_DISPLAY
    
    # Sort features by importance (descending) and take top n
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:n_features]
    
    # Extract feature names and importance values
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    # Create figure
    fig = go.Figure()
    
    # Add horizontal bar chart
    fig.add_trace(
        go.Bar(
            y=feature_names,
            x=importance_values,
            orientation='h',
            marker=dict(
                color=importance_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Importance',
                    thickness=20
                )
            ),
            hovertemplate=
            '<b>%{y}</b><br>' +
            'Importance: %{x:.4f}<br>' +
            '<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        yaxis=dict(
            autorange='reversed'  # To match matplotlib's behavior
        )
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_interactive_feature_group_importance(
    feature_importance: Dict[str, float],
    feature_registry: Optional[FeatureRegistry] = None
) -> go.Figure:
    """
    Creates an interactive pie chart showing the relative importance of feature groups using Plotly.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        feature_registry: Feature registry to get feature group information (optional)
        
    Returns:
        Plotly figure object with the interactive feature group importance plot
    """
    # Validate input
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Create feature registry if not provided
    if feature_registry is None:
        feature_registry = FeatureRegistry()
    
    # Group features by category
    group_importance = {'time': 0.0, 'statistical': 0.0, 'weather': 0.0, 'market': 0.0}
    group_features = {'time': [], 'statistical': [], 'weather': [], 'market': []}
    
    for feature, importance in feature_importance.items():
        # Get feature metadata
        metadata = feature_registry.get_feature(feature)
        
        if metadata and 'group' in metadata:
            group = metadata['group']
            if group in group_importance:
                group_importance[group] += importance
                group_features[group].append(feature)
    
    # Prepare data for pie chart
    labels = [group for group, value in group_importance.items() if value > 0]
    values = [value for value in group_importance.values() if value > 0]
    colors = [GROUP_COLORS[group] for group, value in group_importance.items() if value > 0]
    
    # Create hover text with feature details
    hover_text = []
    for group in labels:
        features_str = '<br>'.join([
            f"- {feature}: {feature_importance[feature]:.4f}" 
            for feature in group_features[group][:5]  # Show top 5 features
        ])
        if len(group_features[group]) > 5:
            features_str += f"<br>- and {len(group_features[group]) - 5} more..."
        
        hover_text.append(
            f"<b>{group.capitalize()}</b><br>" +
            f"Total Importance: {group_importance[group]:.4f}<br>" +
            f"Features:<br>{features_str}"
        )
    
    # Create figure
    fig = go.Figure()
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hovertext=hover_text,
            hoverinfo='text',
            textinfo='percent',
            textfont=dict(size=14, color='white'),
            pull=[0.05] * len(labels)  # Slight pull for all slices
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Feature Importance by Group',
        height=600,
        showlegend=True
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_interactive_feature_correlation_heatmap(
    features_df: DataFrameType,
    feature_importance: Dict[str, float],
    n_features: Optional[int] = None
) -> go.Figure:
    """
    Creates an interactive heatmap showing correlations between top features using Plotly.
    
    Args:
        features_df: DataFrame containing feature values
        feature_importance: Dictionary mapping feature names to importance scores
        n_features: Maximum number of features to display (default: MAX_FEATURES_TO_DISPLAY)
        
    Returns:
        Plotly figure object with the interactive correlation heatmap
    """
    # Validate input
    if features_df.empty:
        raise VisualizationError("Features DataFrame is empty")
    
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = MAX_FEATURES_TO_DISPLAY
    
    # Sort features by importance (descending) and take top n
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:n_features]
    
    # Extract feature names
    feature_names = [item[0] for item in sorted_features]
    
    # Filter features that exist in the DataFrame
    existing_features = [f for f in feature_names if f in features_df.columns]
    
    if not existing_features:
        raise VisualizationError(
            "None of the top features found in the features DataFrame", 
            {"missing_features": feature_names}
        )
    
    # Calculate correlation matrix
    corr_matrix = features_df[existing_features].corr().round(2)
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',  # Red-Blue reversed (red for negative, blue for positive)
            zmid=0,  # Center colorscale at 0
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            hovertemplate=
            '<b>%{y}</b> vs <b>%{x}</b><br>' +
            'Correlation: %{z:.4f}<br>' +
            '<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=700,
        width=800,
        xaxis=dict(
            title='Feature',
            tickangle=-45
        ),
        yaxis=dict(
            title='Feature',
            autorange='reversed'  # To match traditional heatmap layout
        )
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_interactive_feature_dependency_graph(
    feature_importance: Dict[str, float],
    feature_registry: Optional[FeatureRegistry] = None,
    n_features: Optional[int] = None
) -> go.Figure:
    """
    Creates an interactive network graph showing dependencies between features using Plotly.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        feature_registry: Feature registry to get dependency information (optional)
        n_features: Maximum number of features to display (default: MAX_FEATURES_TO_DISPLAY)
        
    Returns:
        Plotly figure object with the interactive dependency graph
    """
    # Validate input
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Create feature registry if not provided
    if feature_registry is None:
        feature_registry = FeatureRegistry()
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = MAX_FEATURES_TO_DISPLAY
    
    # Sort features by importance (descending) and take top n
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:n_features]
    
    # Extract feature names and importance values
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes with importance as size
    for i, feature in enumerate(feature_names):
        # Get feature metadata
        metadata = feature_registry.get_feature(feature)
        
        # Determine node color based on feature group
        group = metadata.get('group', 'unknown') if metadata else 'unknown'
        color = GROUP_COLORS.get(group, '#888888')
        
        # Add node with attributes
        G.add_node(
            feature, 
            importance=importance_values[i],
            group=group,
            color=color
        )
        
        # Add edges for dependencies
        if metadata and 'dependencies' in metadata:
            for dependency in metadata['dependencies']:
                if dependency in feature_names:
                    G.add_edge(dependency, feature)
    
    # Create layout for the graph
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare node and edge data for plotting
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_group = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{node}</b><br>Importance: {G.nodes[node]['importance']:.4f}")
        node_size.append(G.nodes[node]['importance'] * 50)
        node_color.append(G.nodes[node]['color'])
        node_group.append(G.nodes[node]['group'])
    
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[0]} → {edge[1]}")
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
    )
    
    # Add nodes (grouped by feature group for legend)
    for group in set(node_group):
        indices = [i for i, g in enumerate(node_group) if g == group]
        fig.add_trace(
            go.Scatter(
                x=[node_x[i] for i in indices],
                y=[node_y[i] for i in indices],
                mode='markers+text',
                marker=dict(
                    color=[node_color[i] for i in indices],
                    size=[node_size[i] for i in indices],
                    sizemode='area',
                    sizeref=2. * max(node_size) / (40**2),
                    sizemin=4
                ),
                text=[node.split('_')[0] for i, node in enumerate(G.nodes()) if i in indices],
                textposition='top center',
                hovertext=[node_text[i] for i in indices],
                hoverinfo='text',
                name=group.capitalize()
            )
        )
    
    # Add invisible scatter trace for hover on edges
    if edge_x:
        # Find the midpoint of each edge for hover text
        edge_hover_x = []
        edge_hover_y = []
        hover_text = []
        
        for i in range(0, len(edge_x)-2, 3):
            if edge_x[i] is not None and edge_x[i+1] is not None:
                mid_x = (edge_x[i] + edge_x[i+1]) / 2
                mid_y = (edge_y[i] + edge_y[i+1]) / 2
                edge_hover_x.append(mid_x)
                edge_hover_y.append(mid_y)
                
                # Find the nodes for this edge
                idx = i // 3
                if idx < len(edge_text):
                    hover_text.append(edge_text[idx])
        
        fig.add_trace(
            go.Scatter(
                x=edge_hover_x,
                y=edge_hover_y,
                mode='markers',
                marker=dict(
                    size=1,
                    color='rgba(0,0,0,0)'  # Invisible markers
                ),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Feature Dependency Graph',
        showlegend=True,
        hovermode='closest',
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def create_feature_importance_dashboard(
    feature_importance: Dict[str, float],
    features_df: Optional[DataFrameType] = None,
    feature_registry: Optional[FeatureRegistry] = None,
    n_features: Optional[int] = None
) -> go.Figure:
    """
    Creates a comprehensive dashboard with multiple feature importance visualizations using Plotly.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        features_df: Optional DataFrame containing feature values for correlation analysis
        feature_registry: Feature registry to get feature metadata (optional)
        n_features: Maximum number of features to display (default: MAX_FEATURES_TO_DISPLAY)
        
    Returns:
        Plotly figure object with the feature importance dashboard
    """
    # Validate input
    if not feature_importance:
        raise VisualizationError("Feature importance dictionary is empty")
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = MAX_FEATURES_TO_DISPLAY
    
    # Determine number of subplots based on available data
    num_plots = 2  # Feature importance and group importance are always included
    if features_df is not None and not features_df.empty:
        num_plots += 1  # Add correlation heatmap
    if feature_registry is not None:
        num_plots += 1  # Add dependency graph
    
    # Create subplot rows and cols based on number of plots
    if num_plots <= 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2
    
    # Create subplot titles
    subplot_titles = ['Feature Importance', 'Feature Importance by Group']
    if features_df is not None and not features_df.empty:
        subplot_titles.append('Feature Correlation Heatmap')
    if feature_registry is not None:
        subplot_titles.append('Feature Dependency Graph')
    
    # Create subplots
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'bar'}, {'type': 'domain'}]] + 
              ([[{'type': 'heatmap'}, {'type': 'scatter'}]] if rows > 1 else [])
    )
    
    # --- Feature Importance Bar Chart (always included) ---
    # Sort features by importance (descending) and take top n
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:n_features]
    
    # Extract feature names and importance values
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    # Add horizontal bar chart
    fig.add_trace(
        go.Bar(
            y=feature_names,
            x=importance_values,
            orientation='h',
            marker=dict(
                color=importance_values,
                colorscale='Viridis',
                showscale=False
            ),
            hovertemplate=
            '<b>%{y}</b><br>' +
            'Importance: %{x:.4f}<br>' +
            '<extra></extra>',
            name='Feature Importance'
        ),
        row=1, col=1
    )
    
    # Update axis labels
    fig.update_xaxes(title_text='Importance', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    
    # --- Feature Group Importance Pie Chart (always included) ---
    # Create feature registry if not provided
    if feature_registry is None:
        feature_registry = FeatureRegistry()
    
    # Group features by category
    group_importance = {'time': 0.0, 'statistical': 0.0, 'weather': 0.0, 'market': 0.0}
    group_features = {'time': [], 'statistical': [], 'weather': [], 'market': []}
    
    for feature, importance in feature_importance.items():
        # Get feature metadata
        metadata = feature_registry.get_feature(feature)
        
        if metadata and 'group' in metadata:
            group = metadata['group']
            if group in group_importance:
                group_importance[group] += importance
                group_features[group].append(feature)
    
    # Prepare data for pie chart
    labels = [group for group, value in group_importance.items() if value > 0]
    values = [value for value in group_importance.values() if value > 0]
    colors = [GROUP_COLORS[group] for group, value in group_importance.items() if value > 0]
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='percent',
            textfont=dict(size=12),
            name='Feature Groups'
        ),
        row=1, col=2
    )
    
    # --- Feature Correlation Heatmap (if features_df provided) ---
    if features_df is not None and not features_df.empty and rows > 1:
        # Filter features that exist in the DataFrame
        existing_features = [f for f in feature_names if f in features_df.columns]
        
        if existing_features:
            # Calculate correlation matrix
            corr_matrix = features_df[existing_features].corr().round(2)
            
            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    hovertemplate=
                    '<b>%{y}</b> vs <b>%{x}</b><br>' +
                    'Correlation: %{z:.4f}<br>' +
                    '<extra></extra>',
                    name='Feature Correlation'
                ),
                row=2, col=1
            )
            
            # Update axis settings
            fig.update_xaxes(tickangle=-45, row=2, col=1)
            fig.update_yaxes(autorange='reversed', row=2, col=1)
    
    # --- Feature Dependency Graph (if feature_registry provided) ---
    if feature_registry is not None and rows > 1:
        # Create a graph
        G = nx.DiGraph()
        
        # Add nodes with importance as size
        for i, feature in enumerate(feature_names):
            # Get feature metadata
            metadata = feature_registry.get_feature(feature)
            
            # Determine node color based on feature group
            group = metadata.get('group', 'unknown') if metadata else 'unknown'
            color = GROUP_COLORS.get(group, '#888888')
            
            # Add node with attributes
            G.add_node(
                feature, 
                importance=importance_values[i],
                group=group,
                color=color
            )
            
            # Add edges for dependencies
            if metadata and 'dependencies' in metadata:
                for dependency in metadata['dependencies']:
                    if dependency in feature_names:
                        G.add_edge(dependency, feature)
        
        # Create layout for the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Prepare node and edge data for plotting
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_group = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"<b>{node}</b><br>Importance: {G.nodes[node]['importance']:.4f}")
            node_size.append(G.nodes[node]['importance'] * 50)
            node_color.append(G.nodes[node]['color'])
            node_group.append(G.nodes[node]['group'])
        
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Add edges
        if edge_x:
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False,
                    name='Dependencies'
                ),
                row=2, col=2
            )
        
        # Add nodes
        for i, node in enumerate(G.nodes()):
            fig.add_trace(
                go.Scatter(
                    x=[node_x[i]],
                    y=[node_y[i]],
                    mode='markers+text',
                    marker=dict(
                        color=node_color[i],
                        size=node_size[i],
                        sizemode='area',
                        sizeref=2. * max(node_size) / (40**2),
                        sizemin=4
                    ),
                    text=node.split('_')[0],
                    textposition='top center',
                    hovertext=node_text[i],
                    hoverinfo='text',
                    showlegend=False,
                    name=node
                ),
                row=2, col=2
            )
        
        # Update axis settings
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=2)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title='Feature Importance Dashboard',
        height=1000 if rows > 1 else 600,
        width=1200,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


@handle_errors(logger, VisualizationError)
def save_feature_importance_plot(
    fig: Union[Figure, go.Figure],
    output_path: PathType,
    format: Optional[str] = None,
    dpi: Optional[int] = None,
    include_plotlyjs: Optional[bool] = None,
    full_html: Optional[bool] = None
) -> PathType:
    """
    Saves a feature importance plot to a file.
    
    Args:
        fig: Matplotlib or Plotly figure object
        output_path: Path to save the figure to
        format: Output format (png, jpg, svg, pdf, html for Plotly)
        dpi: Resolution in dots per inch (for Matplotlib only)
        include_plotlyjs: Whether to include plotly.js in HTML output (for Plotly only)
        full_html: Whether to include full HTML boilerplate (for Plotly only)
        
    Returns:
        Path to the saved file
    """
    # Validate input
    if not isinstance(fig, (Figure, go.Figure)):
        raise VisualizationError(
            "Input must be a matplotlib Figure or plotly Figure object",
            {"received_type": str(type(fig))}
        )
    
    # Convert path to Path object if it's a string
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If format is not specified, try to infer from the file extension
    if format is None:
        format = output_path.suffix.lstrip('.')
    
    # Save based on figure type
    if isinstance(fig, Figure):
        return export_figure_to_file(fig, output_path, format, dpi)
    else:  # plotly.graph_objects.Figure
        return export_plotly_figure_to_file(fig, output_path, include_plotlyjs, full_html)


class FeatureImportancePlotter:
    """
    Class for creating and managing feature importance visualizations.
    """
    
    def __init__(
        self,
        registry_path: Optional[PathType] = None,
        feature_registry_path: Optional[PathType] = None,
        plot_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the FeatureImportancePlotter with model and feature registries.
        
        Args:
            registry_path: Path to model registry
            feature_registry_path: Path to feature registry
            plot_config: Configuration for plot appearance
        """
        # Initialize registries
        self._registry = ModelRegistry(registry_path)
        self._feature_registry = FeatureRegistry(file_path=feature_registry_path)
        
        # Initialize data storage
        self._model_data = {}
        self._feature_importance = {}
        self._features_df = None
        
        # Set default plot configuration
        self._plot_config = {
            'figsize': DEFAULT_FIGURE_SIZE,
            'dpi': DEFAULT_DPI,
            'cmap': DEFAULT_CMAP,
            'n_features': MAX_FEATURES_TO_DISPLAY,
            'include_plotlyjs': True,
            'full_html': True
        }
        
        # Update with user configuration if provided
        if plot_config:
            self._plot_config.update(plot_config)
    
    def load_model_data(
        self,
        model_id: str,
        version: Optional[str] = None,
        features_df: Optional[DataFrameType] = None
    ) -> bool:
        """
        Loads model data and feature importance for visualization.
        
        Args:
            model_id: Model identifier
            version: Optional model version (uses latest if not specified)
            features_df: Optional DataFrame containing feature values for correlation analysis
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Get model type from ID (assuming ID format like "xgboost_spike_100")
            model_parts = model_id.split('_')
            model_type = model_parts[0] if len(model_parts) > 0 else 'unknown'
            
            # Get model and metadata from registry
            model_obj, metadata = self._registry.get_model(model_type, model_id, version)
            
            # Get model performance metrics
            performance = self._registry.get_model_performance(model_type, model_id, version)
            
            # Get feature importance from model
            feature_importance = model_obj.get_feature_importance()
            
            # Store the data
            self._model_data = {
                'model_id': model_id,
                'model_type': model_type,
                'version': version or 'latest',
                'metadata': metadata,
                'performance': performance,
                'model': model_obj
            }
            
            self._feature_importance = feature_importance
            
            # Store features DataFrame if provided
            if features_df is not None:
                self._features_df = features_df
            
            logger.info(f"Loaded model data for {model_id} (version: {version or 'latest'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model data: {str(e)}")
            return False
    
    def plot_feature_importance(
        self,
        n_features: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a bar chart of feature importance scores.
        
        Args:
            n_features: Maximum number of features to display
            figsize: Figure size as (width, height) in inches
            fig: Existing figure to plot on (optional)
            ax: Existing axes to plot on (optional)
            
        Returns:
            Tuple containing the figure and axes objects
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        return plot_feature_importance(
            self._feature_importance,
            n_features or self._plot_config['n_features'],
            figsize or self._plot_config['figsize'],
            fig,
            ax
        )
    
    def plot_feature_group_importance(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a pie chart showing the relative importance of feature groups.
        
        Args:
            figsize: Figure size as (width, height) in inches
            fig: Existing figure to plot on (optional)
            ax: Existing axes to plot on (optional)
            
        Returns:
            Tuple containing the figure and axes objects
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        return plot_feature_group_importance(
            self._feature_importance,
            self._feature_registry,
            figsize or self._plot_config['figsize'],
            fig,
            ax
        )
    
    def plot_feature_correlation_heatmap(
        self,
        n_features: Optional[int] = None,
        cmap: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a heatmap showing correlations between top features.
        
        Args:
            n_features: Maximum number of features to display
            cmap: Colormap for the heatmap
            figsize: Figure size as (width, height) in inches
            fig: Existing figure to plot on (optional)
            ax: Existing axes to plot on (optional)
            
        Returns:
            Tuple containing the figure and axes objects
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        if self._features_df is None:
            raise VisualizationError("Features DataFrame not loaded. Provide features_df to load_model_data().")
        
        return plot_feature_correlation_heatmap(
            self._features_df,
            self._feature_importance,
            n_features or self._plot_config['n_features'],
            cmap or self._plot_config['cmap'],
            figsize or self._plot_config['figsize'],
            fig,
            ax
        )
    
    def plot_feature_dependency_graph(
        self,
        n_features: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Creates a network graph showing dependencies between features.
        
        Args:
            n_features: Maximum number of features to display
            figsize: Figure size as (width, height) in inches
            fig: Existing figure to plot on (optional)
            ax: Existing axes to plot on (optional)
            
        Returns:
            Tuple containing the figure and axes objects
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        return plot_feature_dependency_graph(
            self._feature_importance,
            self._feature_registry,
            n_features or self._plot_config['n_features'],
            figsize or self._plot_config['figsize'],
            fig,
            ax
        )
    
    def create_interactive_feature_importance(
        self,
        n_features: Optional[int] = None
    ) -> go.Figure:
        """
        Creates an interactive bar chart of feature importance scores using Plotly.
        
        Args:
            n_features: Maximum number of features to display
            
        Returns:
            Plotly figure object with the interactive feature importance plot
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        return create_interactive_feature_importance(
            self._feature_importance,
            n_features or self._plot_config['n_features']
        )
    
    def create_interactive_feature_group_importance(self) -> go.Figure:
        """
        Creates an interactive pie chart showing the relative importance of feature groups using Plotly.
        
        Returns:
            Plotly figure object with the interactive feature group importance plot
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        return create_interactive_feature_group_importance(
            self._feature_importance,
            self._feature_registry
        )
    
    def create_interactive_feature_correlation_heatmap(
        self,
        n_features: Optional[int] = None
    ) -> go.Figure:
        """
        Creates an interactive heatmap showing correlations between top features using Plotly.
        
        Args:
            n_features: Maximum number of features to display
            
        Returns:
            Plotly figure object with the interactive correlation heatmap
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        if self._features_df is None:
            raise VisualizationError("Features DataFrame not loaded. Provide features_df to load_model_data().")
        
        return create_interactive_feature_correlation_heatmap(
            self._features_df,
            self._feature_importance,
            n_features or self._plot_config['n_features']
        )
    
    def create_interactive_feature_dependency_graph(
        self,
        n_features: Optional[int] = None
    ) -> go.Figure:
        """
        Creates an interactive network graph showing dependencies between features using Plotly.
        
        Args:
            n_features: Maximum number of features to display
            
        Returns:
            Plotly figure object with the interactive dependency graph
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        return create_interactive_feature_dependency_graph(
            self._feature_importance,
            self._feature_registry,
            n_features or self._plot_config['n_features']
        )
    
    def create_feature_importance_dashboard(
        self,
        n_features: Optional[int] = None
    ) -> go.Figure:
        """
        Creates a comprehensive dashboard with multiple feature importance visualizations using Plotly.
        
        Args:
            n_features: Maximum number of features to display
            
        Returns:
            Plotly figure object with the feature importance dashboard
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        return create_feature_importance_dashboard(
            self._feature_importance,
            self._features_df,
            self._feature_registry,
            n_features or self._plot_config['n_features']
        )
    
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
            output_path: Path to save the figure to
            format: Output format (png, jpg, svg, pdf, html for Plotly)
            dpi: Resolution in dots per inch (for Matplotlib only)
            include_plotlyjs: Whether to include plotly.js in HTML output (for Plotly only)
            full_html: Whether to include full HTML boilerplate (for Plotly only)
            
        Returns:
            Path to the saved file
        """
        return save_feature_importance_plot(
            fig,
            output_path,
            format,
            dpi or self._plot_config['dpi'],
            include_plotlyjs or self._plot_config['include_plotlyjs'],
            full_html or self._plot_config['full_html']
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gets information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self._model_data:
            raise VisualizationError("Model data not loaded. Call load_model_data() first.")
        
        return {
            'model_id': self._model_data['model_id'],
            'model_type': self._model_data['model_type'],
            'version': self._model_data['version'],
            'performance': self._model_data['performance']
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Gets the feature importance dictionary for the loaded model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._feature_importance:
            raise VisualizationError("Feature importance data not loaded. Call load_model_data() first.")
        
        return self._feature_importance.copy()
    
    def compare_feature_importance(
        self,
        model_ids: List[str],
        versions: Optional[List[str]] = None,
        n_features: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Compares feature importance between multiple models.
        
        Args:
            model_ids: List of model identifiers to compare
            versions: Optional list of model versions (uses latest if not specified)
            n_features: Maximum number of features to display
            figsize: Figure size as (width, height) in inches
            
        Returns:
            Tuple containing the figure and axes objects
        """
        # Validate input
        if not model_ids:
            raise VisualizationError("No model IDs provided for comparison")
        
        # Set default values
        if n_features is None:
            n_features = self._plot_config['n_features']
        
        if figsize is None:
            figsize = self._plot_config['figsize']
        
        # Initialize dictionary to store feature importance for each model
        model_importance = {}
        
        # Get feature importance for each model
        for i, model_id in enumerate(model_ids):
            try:
                # Get model type from ID
                model_parts = model_id.split('_')
                model_type = model_parts[0] if len(model_parts) > 0 else 'unknown'
                
                # Get version for this model
                version = versions[i] if versions and i < len(versions) else None
                
                # Get model and extract feature importance
                model_obj, _ = self._registry.get_model(model_type, model_id, version)
                importance = model_obj.get_feature_importance()
                
                # Store with a readable label
                label = f"{model_id} (v{version})" if version else f"{model_id} (latest)"
                model_importance[label] = importance
                
            except Exception as e:
                logger.warning(f"Failed to get feature importance for model {model_id}: {str(e)}")
        
        if not model_importance:
            raise VisualizationError("Could not retrieve feature importance for any of the specified models")
        
        # Identify the top N features across all models
        all_features = {}
        for model, importance in model_importance.items():
            for feature, value in importance.items():
                if feature not in all_features:
                    all_features[feature] = 0
                all_features[feature] += value
        
        top_features = [f for f, v in sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:n_features]]
        
        # Create a DataFrame for comparison
        comparison_data = []
        for model, importance in model_importance.items():
            row = {'Model': model}
            for feature in top_features:
                row[feature] = importance.get(feature, 0)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create grouped bar chart
        x = np.arange(len(top_features))
        width = 0.8 / len(model_importance)
        
        for i, model in enumerate(comparison_df['Model']):
            values = [comparison_df.loc[comparison_df['Model'] == model, feature].values[0] for feature in top_features]
            ax.bar(x + i * width - 0.4 + width/2, values, width, label=model)
        
        # Add labels and legend
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        
        # Adjust layout
        fig.tight_layout()
        
        return fig, ax