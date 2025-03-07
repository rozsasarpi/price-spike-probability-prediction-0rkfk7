"""
Provides output handling utilities for the CLI application of the ERCOT RTLMP spike prediction system.

This module handles formatting, displaying, and exporting command outputs in various formats
(text, JSON, CSV) and manages the routing of outputs to appropriate destinations.
"""

import json
import csv
from io import StringIO
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar, cast
from pathlib import Path

import pandas as pd  # version 2.0+
import click  # version 8.0+
from datetime import datetime

from ..cli_types import OutputFormat
from .formatters import (
    format_price, format_probability, format_date, format_datetime_str,
    format_number, format_rtlmp_data, format_forecast_data, format_dataframe,
    format_metrics
)
from ..ui.tables import (
    create_table, create_dataframe_table, create_metrics_table,
    create_forecast_table, create_comparison_table,
    create_feature_importance_table, create_backtesting_results_table
)
from ..ui.colors import colorize, supports_color
from ..logger import get_cli_logger

# Define TypeVar for generic type hints
T = TypeVar('T')

# Logger for this module
logger = get_cli_logger('output_handlers')

# Default output settings
DEFAULT_OUTPUT_FORMAT = 'text'
DEFAULT_INDENT = 2
DEFAULT_CSV_DELIMITER = ','

def format_command_result(
    result: Dict[str, Any],
    output_format: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats the result of a command execution for display.
    
    Args:
        result: The result dictionary from command execution.
        output_format: The output format (text, json, csv).
        use_colors: Whether to use colored output.
        
    Returns:
        Formatted result string ready for display.
    """
    if use_colors is None:
        use_colors = supports_color()
        
    if output_format is None:
        output_format = DEFAULT_OUTPUT_FORMAT
    
    if output_format == 'json':
        return json.dumps(result, indent=DEFAULT_INDENT, default=str)
    
    elif output_format == 'text':
        # Format basic text output
        if not result:
            return "No results to display."
        
        # If result has a 'message' key, display it prominently
        if 'message' in result:
            message = result['message']
            if use_colors:
                message = colorize(message, color='green', use_colors=use_colors)
            return message
        
        # Handle different types of results based on content
        if 'forecast' in result:
            return format_forecast_result(result, output_format, use_colors)
        elif 'metrics' in result:
            return format_metrics_result(result, output_format, use_colors)
        elif 'backtesting' in result:
            return format_backtesting_result(result, output_format, use_colors)
        elif 'feature_importance' in result:
            return format_feature_importance_result(result, output_format, use_colors)
        else:
            # Generic formatting for other types of results
            lines = []
            for key, value in result.items():
                if isinstance(value, dict):
                    lines.append(f"{key}:")
                    for sub_key, sub_value in value.items():
                        lines.append(f"  {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    lines.append(f"{key}:")
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
    
    elif output_format == 'csv':
        # Attempt to convert result to CSV if it's in a suitable format
        if isinstance(result, dict):
            return dict_to_csv(result)
        elif isinstance(result, list) and all(isinstance(item, dict) for item in result):
            return list_of_dicts_to_csv(result)
        else:
            # Fall back to JSON if the structure is not suitable for CSV
            logger.warning("Result structure not suitable for CSV. Falling back to JSON.")
            return json.dumps(result, indent=DEFAULT_INDENT, default=str)
    
    else:
        logger.warning(f"Unsupported output format: {output_format}. Falling back to text format.")
        return format_command_result(result, 'text', use_colors)

def format_forecast_result(
    forecast_result: Dict[str, Any],
    output_format: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats forecast results for display.
    
    Args:
        forecast_result: The forecast result dictionary.
        output_format: The output format (text, json, csv).
        use_colors: Whether to use colored output.
        
    Returns:
        Formatted forecast result string.
    """
    if use_colors is None:
        use_colors = supports_color()
        
    if output_format is None:
        output_format = DEFAULT_OUTPUT_FORMAT
    
    # Extract forecast data from the result
    forecasts = forecast_result.get('forecast', [])
    title = forecast_result.get('title', 'Forecast Results')
    
    if output_format == 'json':
        return json.dumps(forecast_result, indent=DEFAULT_INDENT, default=str)
    
    elif output_format == 'text':
        if not forecasts:
            return "No forecast data available."
        
        # Create a table from the forecast data
        table = create_forecast_table(forecasts, title=title, use_colors=use_colors)
        
        # Add any additional information
        additional_info = []
        for key, value in forecast_result.items():
            if key not in ('forecast', 'title'):
                additional_info.append(f"{key}: {value}")
        
        if additional_info:
            return f"{table}\n\n" + "\n".join(additional_info)
        else:
            return table
    
    elif output_format == 'csv':
        if not forecasts:
            return "No forecast data available."
        
        # Convert forecast data to CSV
        return list_of_dicts_to_csv(forecasts)
    
    else:
        logger.warning(f"Unsupported output format: {output_format}. Falling back to text format.")
        return format_forecast_result(forecast_result, 'text', use_colors)

def format_metrics_result(
    metrics_result: Dict[str, Any],
    output_format: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats model metrics results for display.
    
    Args:
        metrics_result: The metrics result dictionary.
        output_format: The output format (text, json, csv).
        use_colors: Whether to use colored output.
        
    Returns:
        Formatted metrics result string.
    """
    if use_colors is None:
        use_colors = supports_color()
        
    if output_format is None:
        output_format = DEFAULT_OUTPUT_FORMAT
    
    # Extract metrics data from the result
    metrics = metrics_result.get('metrics', {})
    title = metrics_result.get('title', 'Model Performance Metrics')
    
    if output_format == 'json':
        return json.dumps(metrics_result, indent=DEFAULT_INDENT, default=str)
    
    elif output_format == 'text':
        if not metrics:
            return "No metrics data available."
        
        # Create a formatted metrics table
        table = create_metrics_table(metrics, title=title, use_colors=use_colors)
        
        # Add any additional information
        additional_info = []
        for key, value in metrics_result.items():
            if key not in ('metrics', 'title'):
                additional_info.append(f"{key}: {value}")
        
        if additional_info:
            return f"{table}\n\n" + "\n".join(additional_info)
        else:
            return table
    
    elif output_format == 'csv':
        if not metrics:
            return "No metrics data available."
        
        # Convert metrics to CSV
        return dict_to_csv(metrics)
    
    else:
        logger.warning(f"Unsupported output format: {output_format}. Falling back to text format.")
        return format_metrics_result(metrics_result, 'text', use_colors)

def format_backtesting_result(
    backtesting_result: Dict[str, Any],
    output_format: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats backtesting results for display.
    
    Args:
        backtesting_result: The backtesting result dictionary.
        output_format: The output format (text, json, csv).
        use_colors: Whether to use colored output.
        
    Returns:
        Formatted backtesting result string.
    """
    if use_colors is None:
        use_colors = supports_color()
        
    if output_format is None:
        output_format = DEFAULT_OUTPUT_FORMAT
    
    # Extract backtesting data from the result
    results = backtesting_result.get('results', [])
    metrics = backtesting_result.get('metrics', {})
    title = backtesting_result.get('title', 'Backtesting Results')
    
    if output_format == 'json':
        return json.dumps(backtesting_result, indent=DEFAULT_INDENT, default=str)
    
    elif output_format == 'text':
        if not results and not metrics:
            return "No backtesting data available."
        
        output_parts = []
        
        # Add the backtesting results table if available
        if results:
            # Create a backtesting results table using appropriate table function
            results_table = create_backtesting_results_table(
                results, title=title, use_colors=use_colors
            )
            output_parts.append(results_table)
        
        # Add metrics table if available
        if metrics:
            metrics_title = backtesting_result.get('metrics_title', 'Backtesting Metrics')
            metrics_table = create_metrics_table(
                metrics, title=metrics_title, use_colors=use_colors
            )
            output_parts.append(metrics_table)
        
        # Add any additional information
        additional_info = []
        exclude_keys = ('results', 'metrics', 'title', 'metrics_title')
        for key, value in backtesting_result.items():
            if key not in exclude_keys:
                additional_info.append(f"{key}: {value}")
        
        if additional_info:
            output_parts.append("\n".join(additional_info))
        
        return "\n\n".join(output_parts)
    
    elif output_format == 'csv':
        if not results:
            # Fall back to metrics if no results
            if metrics:
                return dict_to_csv(metrics)
            return "No backtesting data available."
        
        # Convert backtesting results to CSV
        return list_of_dicts_to_csv(results)
    
    else:
        logger.warning(f"Unsupported output format: {output_format}. Falling back to text format.")
        return format_backtesting_result(backtesting_result, 'text', use_colors)

def format_feature_importance_result(
    feature_importance_result: Dict[str, Any],
    output_format: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats feature importance results for display.
    
    Args:
        feature_importance_result: The feature importance result dictionary.
        output_format: The output format (text, json, csv).
        use_colors: Whether to use colored output.
        
    Returns:
        Formatted feature importance result string.
    """
    if use_colors is None:
        use_colors = supports_color()
        
    if output_format is None:
        output_format = DEFAULT_OUTPUT_FORMAT
    
    # Extract feature importance data from the result
    feature_importance = feature_importance_result.get('feature_importance', {})
    title = feature_importance_result.get('title', 'Feature Importance')
    
    if output_format == 'json':
        return json.dumps(feature_importance_result, indent=DEFAULT_INDENT, default=str)
    
    elif output_format == 'text':
        if not feature_importance:
            return "No feature importance data available."
        
        # Create a feature importance table
        table = create_feature_importance_table(
            feature_importance, title=title, use_colors=use_colors
        )
        
        # Add any additional information
        additional_info = []
        for key, value in feature_importance_result.items():
            if key not in ('feature_importance', 'title'):
                additional_info.append(f"{key}: {value}")
        
        if additional_info:
            return f"{table}\n\n" + "\n".join(additional_info)
        else:
            return table
    
    elif output_format == 'csv':
        if not feature_importance:
            return "No feature importance data available."
        
        # Convert feature importance to CSV
        return dict_to_csv(feature_importance)
    
    else:
        logger.warning(f"Unsupported output format: {output_format}. Falling back to text format.")
        return format_feature_importance_result(feature_importance_result, 'text', use_colors)

def handle_command_output(
    result: Dict[str, Any],
    output_path: Optional[Path] = None,
    output_format: Optional[str] = None,
    use_colors: Optional[bool] = None,
    formatter_func: Optional[Callable[[Dict[str, Any], Optional[str], Optional[bool]], str]] = None
) -> bool:
    """
    Handles the output of a command execution, including display and file export.
    
    Args:
        result: The result dictionary from command execution.
        output_path: Optional path to save the output to a file.
        output_format: The output format (text, json, csv).
        use_colors: Whether to use colored output.
        formatter_func: Optional custom formatter function.
        
    Returns:
        True if output was handled successfully, False otherwise.
    """
    if formatter_func is None:
        formatter_func = format_command_result
    
    if use_colors is None:
        use_colors = supports_color()
    
    if output_format is None:
        output_format = DEFAULT_OUTPUT_FORMAT
    
    # Format the result using the formatter function
    formatted_result = formatter_func(result, output_format, use_colors)
    
    # Display the formatted result to the console
    print(formatted_result)
    
    # Export to file if path is provided
    if output_path:
        success = export_to_file(result, output_path, output_format)
        if success:
            logger.info(f"Output saved to {output_path}")
            return True
        else:
            logger.error(f"Failed to save output to {output_path}")
            return False
    
    return True

def export_to_file(
    result: Dict[str, Any],
    output_path: Path,
    output_format: Optional[str] = None
) -> bool:
    """
    Exports command results to a file in the specified format.
    
    Args:
        result: The result dictionary to export.
        output_path: Path to save the output.
        output_format: The output format (text, json, csv).
        
    Returns:
        True if export was successful, False otherwise.
    """
    try:
        # If output_format is not provided, try to determine from file extension
        if output_format is None:
            suffix = output_path.suffix.lower().lstrip('.')
            if suffix in ('json', 'csv', 'txt'):
                output_format = suffix
            else:
                output_format = DEFAULT_OUTPUT_FORMAT
        
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'json':
            # Export as JSON
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=DEFAULT_INDENT, default=str)
        
        elif output_format == 'csv':
            # Export as CSV
            with open(output_path, 'w', newline='') as f:
                # Determine if result is a dictionary or list of dictionaries
                if isinstance(result, dict):
                    csv_content = dict_to_csv(result)
                    f.write(csv_content)
                elif isinstance(result, list) and all(isinstance(item, dict) for item in result):
                    csv_content = list_of_dicts_to_csv(result)
                    f.write(csv_content)
                else:
                    # Handle nested structures by extracting relevant data
                    for key, value in result.items():
                        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                            csv_content = list_of_dicts_to_csv(value)
                            f.write(csv_content)
                            break
                    else:
                        # If no suitable list found, just convert the whole dict
                        csv_content = dict_to_csv(result)
                        f.write(csv_content)
        
        else:  # Default to text format
            # Export as text
            with open(output_path, 'w') as f:
                formatted_text = format_command_result(result, 'text', False)
                f.write(formatted_text)
        
        logger.info(f"Successfully exported data to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to export data to {output_path}: {str(e)}")
        return False

def export_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    output_format: Optional[str] = None
) -> bool:
    """
    Exports a pandas DataFrame to a file in the specified format.
    
    Args:
        df: The DataFrame to export.
        output_path: Path to save the output.
        output_format: The output format (text, json, csv).
        
    Returns:
        True if export was successful, False otherwise.
    """
    try:
        # If output_format is not provided, try to determine from file extension
        if output_format is None:
            suffix = output_path.suffix.lower().lstrip('.')
            if suffix in ('json', 'csv', 'txt'):
                output_format = suffix
            else:
                output_format = DEFAULT_OUTPUT_FORMAT
        
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'json':
            # Export as JSON
            df.to_json(output_path, orient='records', date_format='iso')
        
        elif output_format == 'csv':
            # Export as CSV
            df.to_csv(output_path, index=False)
        
        else:  # Default to text format
            # Export as text
            with open(output_path, 'w') as f:
                f.write(df.to_string())
        
        logger.info(f"Successfully exported DataFrame to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to export DataFrame to {output_path}: {str(e)}")
        return False

def dict_to_csv(data: Dict[str, Any], delimiter: Optional[str] = None) -> str:
    """
    Converts a dictionary to a CSV string.
    
    Args:
        data: The dictionary to convert.
        delimiter: The CSV delimiter to use.
        
    Returns:
        CSV formatted string.
    """
    if delimiter is None:
        delimiter = DEFAULT_CSV_DELIMITER
    
    if not data:
        return ""
    
    # Create a StringIO object to hold the CSV data
    output = StringIO()
    writer = csv.writer(output, delimiter=delimiter)
    
    # Write header row (keys)
    writer.writerow(data.keys())
    
    # Write data row (values)
    writer.writerow(data.values())
    
    # Get the CSV as a string and return it
    return output.getvalue()

def list_of_dicts_to_csv(data: List[Dict[str, Any]], delimiter: Optional[str] = None) -> str:
    """
    Converts a list of dictionaries to a CSV string.
    
    Args:
        data: The list of dictionaries to convert.
        delimiter: The CSV delimiter to use.
        
    Returns:
        CSV formatted string.
    """
    if delimiter is None:
        delimiter = DEFAULT_CSV_DELIMITER
    
    if not data:
        return ""
    
    # Create a StringIO object to hold the CSV data
    output = StringIO()
    
    # Get all unique keys from all dictionaries
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    
    # Convert set to sorted list for consistent output
    fieldnames_list = sorted(fieldnames)
    
    # Create a CSV writer
    writer = csv.DictWriter(output, fieldnames=fieldnames_list, delimiter=delimiter)
    
    # Write header row and all data rows
    writer.writeheader()
    writer.writerows(data)
    
    # Get the CSV as a string and return it
    return output.getvalue()

def format_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats an error message for display to the user.
    
    Args:
        error: The exception that occurred.
        context: Additional context information.
        use_colors: Whether to use colored output.
        
    Returns:
        Formatted error message.
    """
    if use_colors is None:
        use_colors = supports_color()
    
    # Extract error information
    error_type = type(error).__name__
    error_message = str(error)
    
    # Format the basic error message
    formatted_message = f"Error: {error_message}"
    
    # Add error type for clarity
    formatted_message = f"{error_type}: {formatted_message}"
    
    # Add context if provided
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        formatted_message = f"{formatted_message} [{context_str}]"
    
    # Apply color formatting if requested
    if use_colors:
        formatted_message = colorize(formatted_message, color="red", use_colors=True)
    
    return formatted_message

def display_success_message(message: str, use_colors: Optional[bool] = None) -> None:
    """
    Displays a success message to the user.
    
    Args:
        message: The success message to display.
        use_colors: Whether to use colored output.
    """
    if use_colors is None:
        use_colors = supports_color()
    
    if use_colors:
        formatted_message = colorize(message, color="green", use_colors=True)
    else:
        formatted_message = message
    
    print(formatted_message)
    logger.info(message)

def display_warning_message(message: str, use_colors: Optional[bool] = None) -> None:
    """
    Displays a warning message to the user.
    
    Args:
        message: The warning message to display.
        use_colors: Whether to use colored output.
    """
    if use_colors is None:
        use_colors = supports_color()
    
    if use_colors:
        formatted_message = colorize(message, color="yellow", use_colors=True)
    else:
        formatted_message = message
    
    print(formatted_message)
    logger.warning(message)

def display_error_message(message: str, use_colors: Optional[bool] = None) -> None:
    """
    Displays an error message to the user.
    
    Args:
        message: The error message to display.
        use_colors: Whether to use colored output.
    """
    if use_colors is None:
        use_colors = supports_color()
    
    if use_colors:
        formatted_message = colorize(message, color="red", use_colors=True)
    else:
        formatted_message = message
    
    print(formatted_message)
    logger.error(message)

class OutputHandler:
    """
    Class for handling command output with consistent formatting and export options.
    """
    
    def __init__(
        self,
        output_path: Optional[Path] = None,
        output_format: Optional[str] = None,
        use_colors: Optional[bool] = None
    ):
        """
        Initialize the OutputHandler with output options.
        
        Args:
            output_path: Optional path to save the output to a file.
            output_format: The output format (text, json, csv).
            use_colors: Whether to use colored output.
        """
        self._output_path = output_path
        
        # Determine output format based on path if not explicitly provided
        if output_format is None and output_path is not None:
            suffix = output_path.suffix.lower().lstrip('.')
            if suffix in ('json', 'csv', 'txt'):
                output_format = suffix
            else:
                output_format = DEFAULT_OUTPUT_FORMAT
        elif output_format is None:
            output_format = DEFAULT_OUTPUT_FORMAT
        
        if use_colors is None:
            use_colors = supports_color()
        
        self._output_format = output_format
        self._use_colors = use_colors
    
    def handle_result(
        self,
        result: Dict[str, Any],
        formatter_func: Optional[Callable[[Dict[str, Any], Optional[str], Optional[bool]], str]] = None
    ) -> bool:
        """
        Handles a command result with display and optional export.
        
        Args:
            result: The result dictionary from command execution.
            formatter_func: Optional custom formatter function.
            
        Returns:
            True if output was handled successfully, False otherwise.
        """
        if formatter_func is None:
            formatter_func = format_command_result
        
        return handle_command_output(
            result,
            self._output_path,
            self._output_format,
            self._use_colors,
            formatter_func
        )
    
    def handle_forecast(self, forecast_result: Dict[str, Any]) -> bool:
        """
        Handles forecast results with appropriate formatting.
        
        Args:
            forecast_result: The forecast result dictionary.
            
        Returns:
            True if output was handled successfully, False otherwise.
        """
        return handle_command_output(
            forecast_result,
            self._output_path,
            self._output_format,
            self._use_colors,
            format_forecast_result
        )
    
    def handle_metrics(self, metrics_result: Dict[str, Any]) -> bool:
        """
        Handles metrics results with appropriate formatting.
        
        Args:
            metrics_result: The metrics result dictionary.
            
        Returns:
            True if output was handled successfully, False otherwise.
        """
        return handle_command_output(
            metrics_result,
            self._output_path,
            self._output_format,
            self._use_colors,
            format_metrics_result
        )
    
    def handle_backtesting(self, backtesting_result: Dict[str, Any]) -> bool:
        """
        Handles backtesting results with appropriate formatting.
        
        Args:
            backtesting_result: The backtesting result dictionary.
            
        Returns:
            True if output was handled successfully, False otherwise.
        """
        return handle_command_output(
            backtesting_result,
            self._output_path,
            self._output_format,
            self._use_colors,
            format_backtesting_result
        )
    
    def handle_feature_importance(self, feature_importance_result: Dict[str, Any]) -> bool:
        """
        Handles feature importance results with appropriate formatting.
        
        Args:
            feature_importance_result: The feature importance result dictionary.
            
        Returns:
            True if output was handled successfully, False otherwise.
        """
        return handle_command_output(
            feature_importance_result,
            self._output_path,
            self._output_format,
            self._use_colors,
            format_feature_importance_result
        )
    
    def export_dataframe(self, df: pd.DataFrame, custom_path: Optional[Path] = None) -> bool:
        """
        Exports a DataFrame to a file with current output settings.
        
        Args:
            df: The DataFrame to export.
            custom_path: Optional custom path to override the default.
            
        Returns:
            True if export was successful, False otherwise.
        """
        path = custom_path if custom_path is not None else self._output_path
        
        if path is None:
            logger.error("No output path specified for DataFrame export")
            return False
        
        return export_dataframe(df, path, self._output_format)