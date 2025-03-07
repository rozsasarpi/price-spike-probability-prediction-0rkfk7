"""
Provides functionality for exporting visualizations and data to various file formats 
in the ERCOT RTLMP spike prediction system.

This module handles the conversion and saving of matplotlib figures, plotly figures,
and pandas DataFrames to different formats including images (PNG, SVG, JPEG),
data files (CSV, Excel, Parquet), and interactive HTML.
"""

import os
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
import matplotlib  # version 3.7+
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly  # version 5.14+
import plotly.graph_objects as go

from ..utils.type_definitions import DataFrameType, SeriesType, PathType, ForecastResultDict
from ..utils.logging import get_logger
from ..utils.error_handling import handle_errors, VisualizationError

# Initialize logger
logger = get_logger(__name__)

# Constants
SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'svg', 'pdf']
SUPPORTED_DATA_FORMATS = ['csv', 'excel', 'xlsx', 'parquet', 'json']
DEFAULT_DPI = 100

@handle_errors(logger, VisualizationError)
def figure_to_base64(fig: Figure, format: Optional[str] = None, dpi: Optional[int] = None) -> str:
    """
    Converts a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure object
        format: Output format (png, jpg, svg, pdf)
        dpi: Resolution in dots per inch
        
    Returns:
        Base64 encoded string of the figure
    
    Raises:
        VisualizationError: If conversion fails
    """
    if not isinstance(fig, Figure):
        raise VisualizationError("Input must be a matplotlib Figure object", 
                                {"received_type": str(type(fig))})
    
    if format is None:
        format = 'png'
    
    if dpi is None:
        dpi = DEFAULT_DPI
    
    if format not in SUPPORTED_IMAGE_FORMATS:
        raise VisualizationError(f"Unsupported image format: {format}",
                                {"supported_formats": SUPPORTED_IMAGE_FORMATS})
    
    buffer = BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    
    img_bytes = buffer.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    
    return encoded

@handle_errors(logger, VisualizationError)
def plotly_figure_to_base64(fig: go.Figure, format: Optional[str] = None, 
                           include_plotlyjs: Optional[bool] = None,
                           full_html: Optional[bool] = None) -> str:
    """
    Converts a plotly figure to a base64 encoded string.
    
    Args:
        fig: Plotly figure object
        format: Output format (png, jpg, svg, pdf, html)
        include_plotlyjs: Whether to include plotly.js in the HTML output
        full_html: Whether to include full HTML boilerplate
        
    Returns:
        Base64 encoded string of the figure
    
    Raises:
        VisualizationError: If conversion fails
    """
    if not isinstance(fig, go.Figure):
        raise VisualizationError("Input must be a plotly Figure object",
                                {"received_type": str(type(fig))})
    
    if format is None:
        format = 'png'
    
    buffer = BytesIO()
    
    if format == 'html':
        # Default values for HTML export options
        if include_plotlyjs is None:
            include_plotlyjs = True
        if full_html is None:
            full_html = True
            
        html_str = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html)
        buffer.write(html_str.encode('utf-8'))
    elif format in SUPPORTED_IMAGE_FORMATS:
        fig.write_image(buffer, format=format)
    else:
        raise VisualizationError(f"Unsupported format for plotly figure: {format}",
                               {"supported_formats": SUPPORTED_IMAGE_FORMATS + ['html']})
    
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    
    return encoded

@handle_errors(logger, VisualizationError)
def dataframe_to_base64(df: DataFrameType, format: str) -> str:
    """
    Converts a pandas DataFrame to a base64 encoded string.
    
    Args:
        df: Pandas DataFrame to convert
        format: Output format (csv, excel, parquet, json)
        
    Returns:
        Base64 encoded string of the DataFrame
    
    Raises:
        VisualizationError: If conversion fails
    """
    if not isinstance(df, pd.DataFrame):
        raise VisualizationError("Input must be a pandas DataFrame",
                               {"received_type": str(type(df))})
    
    if format not in SUPPORTED_DATA_FORMATS:
        raise VisualizationError(f"Unsupported data format: {format}",
                               {"supported_formats": SUPPORTED_DATA_FORMATS})
    
    buffer = BytesIO()
    
    if format == 'csv':
        df.to_csv(buffer, index=True)
    elif format in ['excel', 'xlsx']:
        df.to_excel(buffer, index=True)
    elif format == 'parquet':
        df.to_parquet(buffer, index=True)
    elif format == 'json':
        df.to_json(buffer, orient='records')
    
    buffer.seek(0)
    data_bytes = buffer.getvalue()
    encoded = base64.b64encode(data_bytes).decode('utf-8')
    
    return encoded

@handle_errors(logger, VisualizationError)
def export_figure_to_file(fig: Figure, output_path: PathType, 
                         format: Optional[str] = None, dpi: Optional[int] = None) -> PathType:
    """
    Exports a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure object
        output_path: Path to save the figure to
        format: Output format (png, jpg, svg, pdf)
        dpi: Resolution in dots per inch
        
    Returns:
        Path to the saved file
        
    Raises:
        VisualizationError: If export fails
    """
    if not isinstance(fig, Figure):
        raise VisualizationError("Input must be a matplotlib Figure object",
                               {"received_type": str(type(fig))})
    
    # Convert path to Path object if it's a string
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If format is not specified, try to infer from the file extension
    if format is None:
        format = output_path.suffix.lstrip('.')
    
    if format not in SUPPORTED_IMAGE_FORMATS:
        raise VisualizationError(f"Unsupported image format: {format}",
                               {"supported_formats": SUPPORTED_IMAGE_FORMATS})
    
    if dpi is None:
        dpi = DEFAULT_DPI
    
    fig.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
    logger.info(f"Exported figure to {output_path}")
    
    return output_path

@handle_errors(logger, VisualizationError)
def export_plotly_figure_to_file(fig: go.Figure, output_path: PathType,
                                include_plotlyjs: Optional[bool] = None,
                                full_html: Optional[bool] = None) -> PathType:
    """
    Exports a plotly figure to a file.
    
    Args:
        fig: Plotly figure object
        output_path: Path to save the figure to
        include_plotlyjs: Whether to include plotly.js in the HTML output
        full_html: Whether to include full HTML boilerplate
        
    Returns:
        Path to the saved file
        
    Raises:
        VisualizationError: If export fails
    """
    if not isinstance(fig, go.Figure):
        raise VisualizationError("Input must be a plotly Figure object",
                               {"received_type": str(type(fig))})
    
    # Convert path to Path object if it's a string
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine the format from the file extension
    format = output_path.suffix.lstrip('.')
    
    if format == 'html':
        # Default values for HTML export options
        if include_plotlyjs is None:
            include_plotlyjs = True
        if full_html is None:
            full_html = True
            
        fig.write_html(output_path, include_plotlyjs=include_plotlyjs, full_html=full_html)
    elif format == 'json':
        fig.write_json(output_path)
    elif format in SUPPORTED_IMAGE_FORMATS:
        fig.write_image(output_path)
    else:
        raise VisualizationError(f"Unsupported format for plotly figure: {format}",
                               {"supported_formats": SUPPORTED_IMAGE_FORMATS + ['html', 'json']})
    
    logger.info(f"Exported plotly figure to {output_path}")
    
    return output_path

@handle_errors(logger, VisualizationError)
def export_dataframe_to_file(df: DataFrameType, output_path: PathType,
                            format: Optional[str] = None,
                            export_options: Optional[Dict[str, Any]] = None) -> PathType:
    """
    Exports a pandas DataFrame to a file.
    
    Args:
        df: Pandas DataFrame to export
        output_path: Path to save the DataFrame to
        format: Output format (csv, excel, parquet, json)
        export_options: Additional options to pass to the export function
        
    Returns:
        Path to the saved file
        
    Raises:
        VisualizationError: If export fails
    """
    if not isinstance(df, pd.DataFrame):
        raise VisualizationError("Input must be a pandas DataFrame",
                               {"received_type": str(type(df))})
    
    # Convert path to Path object if it's a string
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If format is not specified, try to infer from the file extension
    if format is None:
        format = output_path.suffix.lstrip('.')
        
        # Handle xlsx extension specially
        if format == 'xlsx':
            format = 'excel'
    
    if format not in SUPPORTED_DATA_FORMATS:
        raise VisualizationError(f"Unsupported data format: {format}",
                               {"supported_formats": SUPPORTED_DATA_FORMATS})
    
    # Default export options
    if export_options is None:
        export_options = {}
    
    if format == 'csv':
        df.to_csv(output_path, **export_options)
    elif format in ['excel', 'xlsx']:
        df.to_excel(output_path, **export_options)
    elif format == 'parquet':
        df.to_parquet(output_path, **export_options)
    elif format == 'json':
        df.to_json(output_path, **export_options)
    
    logger.info(f"Exported DataFrame to {output_path}")
    
    return output_path

@handle_errors(logger, VisualizationError)
def export_forecast_to_file(forecast_data: Union[DataFrameType, List[ForecastResultDict]],
                           output_path: PathType, format: Optional[str] = None,
                           export_options: Optional[Dict[str, Any]] = None) -> PathType:
    """
    Exports forecast data to a file.
    
    Args:
        forecast_data: Forecast data as DataFrame or list of dictionaries
        output_path: Path to save the forecast data to
        format: Output format (csv, excel, parquet, json)
        export_options: Additional options to pass to the export function
        
    Returns:
        Path to the saved file
        
    Raises:
        VisualizationError: If export fails
    """
    # Convert to DataFrame if it's a list of dictionaries
    if isinstance(forecast_data, list):
        forecast_df = pd.DataFrame(forecast_data)
    else:
        forecast_df = forecast_data
    
    return export_dataframe_to_file(forecast_df, output_path, format, export_options)

@handle_errors(logger, VisualizationError)
def export_model_performance_to_file(performance_metrics: Dict[str, Any],
                                    output_path: PathType, format: Optional[str] = None,
                                    export_options: Optional[Dict[str, Any]] = None) -> PathType:
    """
    Exports model performance metrics to a file.
    
    Args:
        performance_metrics: Dictionary of performance metrics
        output_path: Path to save the metrics to
        format: Output format (csv, excel, parquet, json)
        export_options: Additional options to pass to the export function
        
    Returns:
        Path to the saved file
        
    Raises:
        VisualizationError: If export fails
    """
    # Convert the metrics dictionary to a DataFrame
    metrics_list = []
    
    for metric_name, metric_value in performance_metrics.items():
        # Handle nested dictionaries
        if isinstance(metric_value, dict):
            for sub_name, sub_value in metric_value.items():
                metrics_list.append({
                    'metric': f"{metric_name}_{sub_name}",
                    'value': sub_value
                })
        else:
            metrics_list.append({
                'metric': metric_name,
                'value': metric_value
            })
    
    metrics_df = pd.DataFrame(metrics_list)
    
    return export_dataframe_to_file(metrics_df, output_path, format, export_options)

@handle_errors(logger, VisualizationError)
def export_backtesting_results_to_file(backtest_results: Union[DataFrameType, Dict[str, Any]],
                                      output_path: PathType, format: Optional[str] = None,
                                      export_options: Optional[Dict[str, Any]] = None) -> PathType:
    """
    Exports backtesting results to a file.
    
    Args:
        backtest_results: Backtesting results as DataFrame or dictionary
        output_path: Path to save the results to
        format: Output format (csv, excel, parquet, json)
        export_options: Additional options to pass to the export function
        
    Returns:
        Path to the saved file
        
    Raises:
        VisualizationError: If export fails
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(backtest_results, dict):
        # We need to handle potentially complex nested dictionary structures
        # This is a simplified approach that flattens one level of nesting
        flattened_data = []
        
        for key, value in backtest_results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened_data.append({
                        'category': key,
                        'metric': sub_key,
                        'value': sub_value
                    })
            else:
                flattened_data.append({
                    'metric': key,
                    'value': value
                })
        
        backtest_df = pd.DataFrame(flattened_data)
    else:
        backtest_df = backtest_results
    
    return export_dataframe_to_file(backtest_df, output_path, format, export_options)

class ExportError(VisualizationError):
    """
    Exception raised for errors during export operations.
    """
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        """
        Initialize the export error.
        
        Args:
            message: Error message
            context: Additional context about the error
        """
        super().__init__(message, context)

class ExportManager:
    """
    Class for managing exports of visualizations and data to various file formats.
    """
    
    def __init__(self, export_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ExportManager with configuration options.
        
        Args:
            export_config: Configuration dictionary for export operations
        """
        # Default export configuration
        self._export_config = {
            'dpi': DEFAULT_DPI,
            'include_plotlyjs': True,
            'full_html': True,
            'default_image_format': 'png',
            'default_data_format': 'csv'
        }
        
        # Update with provided configuration if available
        if export_config:
            self._export_config.update(export_config)
    
    def export_figure(self, fig: Figure, output_path: PathType,
                     format: Optional[str] = None, dpi: Optional[int] = None) -> PathType:
        """
        Exports a matplotlib figure to a file.
        
        Args:
            fig: Matplotlib figure object
            output_path: Path to save the figure to
            format: Output format (png, jpg, svg, pdf)
            dpi: Resolution in dots per inch
            
        Returns:
            Path to the saved file
        """
        if dpi is None:
            dpi = self._export_config.get('dpi', DEFAULT_DPI)
        
        return export_figure_to_file(fig, output_path, format, dpi)
    
    def export_plotly_figure(self, fig: go.Figure, output_path: PathType,
                            include_plotlyjs: Optional[bool] = None,
                            full_html: Optional[bool] = None) -> PathType:
        """
        Exports a plotly figure to a file.
        
        Args:
            fig: Plotly figure object
            output_path: Path to save the figure to
            include_plotlyjs: Whether to include plotly.js in the HTML output
            full_html: Whether to include full HTML boilerplate
            
        Returns:
            Path to the saved file
        """
        if include_plotlyjs is None:
            include_plotlyjs = self._export_config.get('include_plotlyjs', True)
        
        if full_html is None:
            full_html = self._export_config.get('full_html', True)
        
        return export_plotly_figure_to_file(fig, output_path, include_plotlyjs, full_html)
    
    def export_dataframe(self, df: DataFrameType, output_path: PathType,
                        format: Optional[str] = None,
                        export_options: Optional[Dict[str, Any]] = None) -> PathType:
        """
        Exports a pandas DataFrame to a file.
        
        Args:
            df: Pandas DataFrame to export
            output_path: Path to save the DataFrame to
            format: Output format (csv, excel, parquet, json)
            export_options: Additional options to pass to the export function
            
        Returns:
            Path to the saved file
        """
        if export_options is None:
            export_options = self._export_config.get('dataframe_export_options', {})
        
        return export_dataframe_to_file(df, output_path, format, export_options)
    
    def export_forecast(self, forecast_data: Union[DataFrameType, List[ForecastResultDict]],
                       output_path: PathType, format: Optional[str] = None,
                       export_options: Optional[Dict[str, Any]] = None) -> PathType:
        """
        Exports forecast data to a file.
        
        Args:
            forecast_data: Forecast data as DataFrame or list of dictionaries
            output_path: Path to save the forecast data to
            format: Output format (csv, excel, parquet, json)
            export_options: Additional options to pass to the export function
            
        Returns:
            Path to the saved file
        """
        if export_options is None:
            export_options = self._export_config.get('forecast_export_options', {})
        
        return export_forecast_to_file(forecast_data, output_path, format, export_options)
    
    def export_model_performance(self, performance_metrics: Dict[str, Any],
                                output_path: PathType, format: Optional[str] = None,
                                export_options: Optional[Dict[str, Any]] = None) -> PathType:
        """
        Exports model performance metrics to a file.
        
        Args:
            performance_metrics: Dictionary of performance metrics
            output_path: Path to save the metrics to
            format: Output format (csv, excel, parquet, json)
            export_options: Additional options to pass to the export function
            
        Returns:
            Path to the saved file
        """
        if export_options is None:
            export_options = self._export_config.get('metrics_export_options', {})
        
        return export_model_performance_to_file(performance_metrics, output_path, format, export_options)
    
    def export_backtesting_results(self, backtest_results: Union[DataFrameType, Dict[str, Any]],
                                  output_path: PathType, format: Optional[str] = None,
                                  export_options: Optional[Dict[str, Any]] = None) -> PathType:
        """
        Exports backtesting results to a file.
        
        Args:
            backtest_results: Backtesting results as DataFrame or dictionary
            output_path: Path to save the results to
            format: Output format (csv, excel, parquet, json)
            export_options: Additional options to pass to the export function
            
        Returns:
            Path to the saved file
        """
        if export_options is None:
            export_options = self._export_config.get('backtest_export_options', {})
        
        return export_backtesting_results_to_file(backtest_results, output_path, format, export_options)
    
    def figure_to_base64(self, fig: Figure, format: Optional[str] = None,
                        dpi: Optional[int] = None) -> str:
        """
        Converts a matplotlib figure to a base64 encoded string.
        
        Args:
            fig: Matplotlib figure object
            format: Output format (png, jpg, svg, pdf)
            dpi: Resolution in dots per inch
            
        Returns:
            Base64 encoded string of the figure
        """
        if dpi is None:
            dpi = self._export_config.get('dpi', DEFAULT_DPI)
        
        return figure_to_base64(fig, format, dpi)
    
    def plotly_figure_to_base64(self, fig: go.Figure, format: Optional[str] = None,
                               include_plotlyjs: Optional[bool] = None,
                               full_html: Optional[bool] = None) -> str:
        """
        Converts a plotly figure to a base64 encoded string.
        
        Args:
            fig: Plotly figure object
            format: Output format (png, jpg, svg, pdf, html)
            include_plotlyjs: Whether to include plotly.js in the HTML output
            full_html: Whether to include full HTML boilerplate
            
        Returns:
            Base64 encoded string of the figure
        """
        if include_plotlyjs is None:
            include_plotlyjs = self._export_config.get('include_plotlyjs', True)
        
        if full_html is None:
            full_html = self._export_config.get('full_html', True)
        
        return plotly_figure_to_base64(fig, format, include_plotlyjs, full_html)
    
    def dataframe_to_base64(self, df: DataFrameType, format: str) -> str:
        """
        Converts a pandas DataFrame to a base64 encoded string.
        
        Args:
            df: Pandas DataFrame to convert
            format: Output format (csv, excel, parquet, json)
            
        Returns:
            Base64 encoded string of the DataFrame
        """
        return dataframe_to_base64(df, format)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Updates the export configuration.
        
        Args:
            new_config: New configuration values to update
        """
        self._export_config.update(new_config)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Gets the current export configuration.
        
        Returns:
            Current export configuration
        """
        return self._export_config.copy()