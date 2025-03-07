"""
Provides table creation and formatting utilities for the CLI interface of the ERCOT RTLMP spike prediction system.

This module enables the display of tabular data such as RTLMP values, forecast probabilities,
model metrics, and other structured data in a readable format within the terminal.
"""

from typing import Any, Dict, List, Optional, Union
import datetime
import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from tabulate import tabulate  # version 0.9+

from .colors import colorize, bold, supports_color, get_color_safe_length
from .formatters import get_terminal_size, truncate_text
from ..utils.formatters import (
    format_price, format_probability, format_number, format_datetime_str
)

# Default table formatting options
DEFAULT_TABLE_FORMAT = "grid"
DEFAULT_MAX_COLUMN_WIDTH = 30
DEFAULT_MAX_ROWS = 100
DEFAULT_MAX_COLS = 10

# Mapping of style names to tabulate formats
TABLE_STYLES = {
    "grid": "fancy_grid",
    "simple": "simple",
    "plain": "plain",
    "markdown": "pipe",
    "rst": "rst"
}

# Special formatters for specific column types
SPECIAL_COLUMN_FORMATTERS = {
    "price": format_price,
    "probability": format_probability,
    "datetime": format_datetime_str,
    "number": format_number
}


def create_table(
    data: List[List[Any]],
    headers: List[str],
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None,
    max_width: Optional[int] = None
) -> str:
    """
    Creates a formatted table from data and headers.

    Args:
        data: List of rows, where each row is a list of cell values.
        headers: List of column headers.
        tablefmt: Table format (from TABLE_STYLES).
        use_colors: Whether to use colors. If None, determined automatically.
        max_width: Maximum width of the table. If None, uses terminal width.

    Returns:
        Formatted table string.
    """
    # Set default values
    if tablefmt is None:
        tablefmt = DEFAULT_TABLE_FORMAT
    
    if tablefmt in TABLE_STYLES:
        tablefmt = TABLE_STYLES[tablefmt]

    if use_colors is None:
        use_colors = supports_color()

    if max_width is None:
        max_width, _ = get_terminal_size()
        # Leave some margin
        max_width = max(40, max_width - 4)

    # Format headers with bold if colors are enabled
    if use_colors:
        headers = [bold(header, use_colors=True) for header in headers]

    # Calculate optimal column widths
    col_widths = calculate_column_widths(data, headers, max_width)

    # Truncate cell content if needed and format cells
    formatted_data = []
    for row in data:
        formatted_row = []
        for i, cell in enumerate(row):
            # Determine width for this column
            col_width = col_widths[i] if i < len(col_widths) else DEFAULT_MAX_COLUMN_WIDTH
            
            # Format the cell
            formatted_cell = format_cell(cell, None, use_colors)
            
            # Truncate if needed
            if get_color_safe_length(formatted_cell) > col_width:
                formatted_cell = truncate_text(formatted_cell, col_width)
            
            formatted_row.append(formatted_cell)
        
        formatted_data.append(formatted_row)

    # Create the table using tabulate
    table = tabulate(formatted_data, headers=headers, tablefmt=tablefmt)
    
    return table


def create_simple_table(
    data: List[List[Any]],
    headers: List[str],
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a simple table with minimal formatting.

    Args:
        data: List of rows, where each row is a list of cell values.
        headers: List of column headers.
        use_colors: Whether to use colors. If None, determined automatically.

    Returns:
        Formatted simple table string.
    """
    return create_table(data, headers, tablefmt="simple", use_colors=use_colors)


def create_markdown_table(
    data: List[List[Any]],
    headers: List[str]
) -> str:
    """
    Creates a markdown-formatted table.

    Args:
        data: List of rows, where each row is a list of cell values.
        headers: List of column headers.

    Returns:
        Markdown table string.
    """
    return create_table(data, headers, tablefmt="markdown", use_colors=False)


def create_dataframe_table(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None,
    column_formats: Optional[Dict[str, str]] = None
) -> str:
    """
    Creates a table from a pandas DataFrame.

    Args:
        df: The DataFrame to display.
        columns: Specific columns to include. If None, include all columns.
        max_rows: Maximum number of rows to display. If None, uses DEFAULT_MAX_ROWS.
        max_cols: Maximum number of columns to display. If None, uses DEFAULT_MAX_COLS.
        tablefmt: Table format (from TABLE_STYLES).
        use_colors: Whether to use colors. If None, determined automatically.
        column_formats: Dictionary mapping column names to format types.

    Returns:
        Formatted DataFrame table string.
    """
    # Handle empty DataFrame
    if df.empty:
        return "Empty DataFrame"

    # Select columns if specified
    if columns is not None:
        # Only include columns that exist in the DataFrame
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            return "No valid columns selected"
        df_display = df[valid_columns].copy()
    else:
        df_display = df.copy()

    # Apply row and column limits
    if max_rows is None:
        max_rows = DEFAULT_MAX_ROWS
    
    if max_cols is None:
        max_cols = DEFAULT_MAX_COLS

    # Limit rows
    if len(df_display) > max_rows:
        # Show first half and last half of rows
        half_rows = max_rows // 2
        df_display = pd.concat([
            df_display.head(half_rows),
            pd.DataFrame([{col: "..." for col in df_display.columns}]),
            df_display.tail(half_rows)
        ])

    # Limit columns
    if len(df_display.columns) > max_cols:
        # Show first set of columns
        df_display = df_display.iloc[:, :max_cols]
        df_display["..."] = "..."

    # Get headers and data
    headers = [str(col) for col in df_display.columns]
    data = df_display.values.tolist()

    # Apply column formatting if provided
    formatted_data = []
    for row in data:
        formatted_row = []
        for i, value in enumerate(row):
            col_name = headers[i]
            col_format = None
            
            # Check if we have a specified format for this column
            if column_formats and col_name in column_formats:
                col_format = column_formats[col_name]
            # Auto-detect format based on column name
            elif any(keyword in col_name.lower() for keyword in ["price", "cost"]):
                col_format = "price"
            elif any(keyword in col_name.lower() for keyword in ["probability", "confidence"]):
                col_format = "probability"
            elif any(keyword in col_name.lower() for keyword in ["date", "time", "timestamp"]):
                col_format = "datetime"
            
            formatted_row.append(format_cell(value, col_format, use_colors))
        formatted_data.append(formatted_row)

    # Create the table
    return create_table(formatted_data, headers, tablefmt=tablefmt, use_colors=use_colors)


def create_key_value_table(
    data: Dict[str, Any],
    key_header: Optional[str] = None,
    value_header: Optional[str] = None,
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a two-column table from key-value pairs.

    Args:
        data: Dictionary of key-value pairs.
        key_header: Header for key column. Defaults to "Key".
        value_header: Header for value column. Defaults to "Value".
        tablefmt: Table format (from TABLE_STYLES).
        use_colors: Whether to use colors. If None, determined automatically.

    Returns:
        Formatted key-value table string.
    """
    if key_header is None:
        key_header = "Key"
    
    if value_header is None:
        value_header = "Value"
    
    # Convert dictionary to list of [key, value] pairs
    table_data = [[key, value] for key, value in data.items()]
    
    return create_table(
        table_data,
        [key_header, value_header],
        tablefmt=tablefmt,
        use_colors=use_colors
    )


def create_metrics_table(
    metrics: Dict[str, float],
    title: Optional[str] = None,
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a table displaying model performance metrics.

    Args:
        metrics: Dictionary of metric names and values.
        title: Optional title for the table.
        tablefmt: Table format (from TABLE_STYLES).
        use_colors: Whether to use colors. If None, determined automatically.

    Returns:
        Formatted metrics table string.
    """
    # Convert metrics to a list of [metric_name, value] pairs
    table_data = []
    
    for metric_name, value in metrics.items():
        metric_lower = metric_name.lower()
        
        # Format based on metric type
        if any(term in metric_lower for term in ["auc", "precision", "recall", "f1", "accuracy"]):
            # Classification metrics are typically between 0 and 1
            formatted_value = format_probability(value, use_colors=use_colors)
        elif "brier" in metric_lower:
            # Brier score - lower is better
            formatted_value = format_number(value, 3)
        elif "time" in metric_lower or "duration" in metric_lower:
            # Time metrics in seconds
            formatted_value = f"{format_number(value, 1)}s"
        else:
            # Default formatting
            formatted_value = format_number(value, 3)
        
        table_data.append([metric_name, formatted_value])
    
    # Create the table
    table = create_table(
        table_data,
        ["Metric", "Value"],
        tablefmt=tablefmt,
        use_colors=use_colors
    )
    
    # Add title if provided
    if title:
        return f"{title}\n{table}"
    
    return table


def create_comparison_table(
    comparison_data: Dict[str, Dict[str, float]],
    metrics_to_include: Optional[List[str]] = None,
    title: Optional[str] = None,
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a table comparing multiple sets of metrics.

    Args:
        comparison_data: Dictionary mapping names to metric dictionaries.
        metrics_to_include: List of metrics to include in the comparison.
            If None, include all metrics.
        title: Optional title for the table.
        tablefmt: Table format (from TABLE_STYLES).
        use_colors: Whether to use colors. If None, determined automatically.

    Returns:
        Formatted comparison table string.
    """
    if not comparison_data:
        return "No comparison data provided"
    
    # Get all unique metric names across all comparison sets
    all_metrics = set()
    for metrics in comparison_data.values():
        all_metrics.update(metrics.keys())
    
    # Filter metrics if specified
    if metrics_to_include is not None:
        all_metrics = [m for m in metrics_to_include if m in all_metrics]
    else:
        all_metrics = sorted(all_metrics)
    
    if not all_metrics:
        return "No matching metrics found"
    
    # Create headers: "Metric" followed by all comparison names
    headers = ["Metric"] + list(comparison_data.keys())
    
    # Create rows for each metric
    table_data = []
    for metric in all_metrics:
        row = [metric]
        
        for comparison_name in comparison_data.keys():
            # Get the metric value for this comparison, or N/A if not available
            value = comparison_data[comparison_name].get(metric, None)
            
            # Format based on metric type
            metric_lower = metric.lower()
            if value is None:
                formatted_value = "N/A"
            elif any(term in metric_lower for term in ["auc", "precision", "recall", "f1", "accuracy"]):
                formatted_value = format_probability(value, use_colors=use_colors)
            elif "brier" in metric_lower:
                formatted_value = format_number(value, 3)
            elif "time" in metric_lower or "duration" in metric_lower:
                formatted_value = f"{format_number(value, 1)}s"
            else:
                formatted_value = format_number(value, 3)
            
            row.append(formatted_value)
        
        table_data.append(row)
    
    # Create the table
    table = create_table(
        table_data,
        headers,
        tablefmt=tablefmt,
        use_colors=use_colors
    )
    
    # Add title if provided
    if title:
        return f"{title}\n{table}"
    
    return table


def create_forecast_table(
    forecasts: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a table displaying forecast probabilities.

    Args:
        forecasts: List of forecast dictionaries.
        columns: List of columns to include. If None, uses default forecast columns.
        title: Optional title for the table.
        tablefmt: Table format (from TABLE_STYLES).
        use_colors: Whether to use colors. If None, determined automatically.

    Returns:
        Formatted forecast table string.
    """
    if not forecasts:
        return "No forecast data provided"
    
    # Default columns if not specified
    if columns is None:
        columns = [
            "target_timestamp",
            "threshold_value",
            "spike_probability",
            "confidence_interval_lower",
            "confidence_interval_upper"
        ]
    
    # Ensure all requested columns exist in at least one forecast
    all_keys = set()
    for forecast in forecasts:
        all_keys.update(forecast.keys())
    
    valid_columns = [col for col in columns if col in all_keys]
    if not valid_columns:
        return "No valid columns found in forecast data"
    
    # Create headers
    headers = valid_columns
    
    # Extract data from forecasts
    table_data = []
    for forecast in forecasts:
        row = []
        for col in valid_columns:
            value = forecast.get(col, None)
            
            # Apply special formatting based on column name
            col_lower = col.lower()
            if "timestamp" in col_lower or "date" in col_lower or "time" in col_lower:
                formatted_value = format_datetime_str(value)
            elif "probability" in col_lower or "confidence" in col_lower:
                formatted_value = format_probability(value, use_colors=use_colors)
            elif "price" in col_lower or "threshold" in col_lower or "value" in col_lower:
                formatted_value = format_price(value, use_colors=use_colors)
            else:
                formatted_value = str(value) if value is not None else "N/A"
            
            row.append(formatted_value)
        
        table_data.append(row)
    
    # Create the table
    table = create_table(
        table_data,
        headers,
        tablefmt=tablefmt,
        use_colors=use_colors
    )
    
    # Add title if provided
    if title:
        return f"{title}\n{table}"
    
    return table


def create_feature_importance_table(
    feature_importance: Dict[str, float],
    max_features: Optional[int] = None,
    title: Optional[str] = None,
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a table displaying feature importance values.

    Args:
        feature_importance: Dictionary mapping feature names to importance values.
        max_features: Maximum number of features to display.
        title: Optional title for the table.
        tablefmt: Table format (from TABLE_STYLES).
        use_colors: Whether to use colors. If None, determined automatically.

    Returns:
        Formatted feature importance table string.
    """
    if not feature_importance:
        return "No feature importance data provided"
    
    # Sort features by importance value in descending order
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Limit to max_features if specified
    if max_features is not None:
        sorted_features = sorted_features[:max_features]
    
    # Create table data with feature names and formatted importance values
    table_data = [
        [feature, format_number(importance, 3)]
        for feature, importance in sorted_features
    ]
    
    # Create the table
    table = create_table(
        table_data,
        ["Feature", "Importance"],
        tablefmt=tablefmt,
        use_colors=use_colors
    )
    
    # Add title if provided
    if title:
        return f"{title}\n{table}"
    
    return table


def create_confusion_matrix_table(
    matrix: List[List[int]],
    class_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a table displaying a confusion matrix.

    Args:
        matrix: 2D list representing the confusion matrix.
        class_labels: Labels for the classes. If None, uses numeric indices.
        title: Optional title for the table.
        tablefmt: Table format (from TABLE_STYLES).
        use_colors: Whether to use colors. If None, determined automatically.

    Returns:
        Formatted confusion matrix table string.
    """
    if not matrix:
        return "No confusion matrix data provided"
    
    # Create default class labels if not provided
    if class_labels is None:
        class_labels = [str(i) for i in range(len(matrix))]
    
    # Ensure we have the right number of labels
    if len(class_labels) != len(matrix):
        # Use the provided labels as far as they go, then add numeric indices
        if len(class_labels) < len(matrix):
            class_labels = class_labels + [str(i) for i in range(len(class_labels), len(matrix))]
        else:
            class_labels = class_labels[:len(matrix)]
    
    # Create headers: "Actual/Predicted" followed by predicted class labels
    headers = ["Actual \\ Predicted"] + class_labels
    
    # Create table data: each row starts with the actual class label
    table_data = []
    for i, row in enumerate(matrix):
        table_row = [class_labels[i]]
        
        for j, value in enumerate(row):
            if use_colors:
                # Color coding based on value: higher values get stronger colors
                # Diagonal (correct predictions) in green, off-diagonal in red
                if i == j:  # On diagonal
                    color = "green"
                else:  # Off diagonal
                    color = "red"
                
                formatted_value = colorize(str(value), color=color, use_colors=use_colors)
            else:
                formatted_value = str(value)
            
            table_row.append(formatted_value)
        
        table_data.append(table_row)
    
    # Create the table
    table = create_table(
        table_data,
        headers,
        tablefmt=tablefmt,
        use_colors=use_colors
    )
    
    # Add title if provided
    if title:
        return f"{title}\n{table}"
    
    return table


def calculate_column_widths(
    data: List[List[Any]],
    headers: List[str],
    max_table_width: int,
    max_column_width: Optional[int] = None
) -> List[int]:
    """
    Calculates optimal column widths based on content and available space.

    Args:
        data: List of rows, where each row is a list of cell values.
        headers: List of column headers.
        max_table_width: Maximum table width in characters.
        max_column_width: Maximum width for any column. If None, uses DEFAULT_MAX_COLUMN_WIDTH.

    Returns:
        List of column widths.
    """
    if max_column_width is None:
        max_column_width = DEFAULT_MAX_COLUMN_WIDTH
    
    # Determine the number of columns
    num_columns = max(
        max(len(row) for row in data) if data else 0,
        len(headers)
    )
    
    if num_columns == 0:
        return []
    
    # Calculate maximum content width for each column
    max_widths = [0] * num_columns
    
    # Check headers
    for i, header in enumerate(headers):
        if i < num_columns:
            max_widths[i] = max(max_widths[i], get_color_safe_length(str(header)))
    
    # Check data rows
    for row in data:
        for i, cell in enumerate(row):
            if i < num_columns:
                max_widths[i] = max(max_widths[i], get_color_safe_length(str(cell)))
    
    # Limit each column to max_column_width
    max_widths = [min(width, max_column_width) for width in max_widths]
    
    # Account for table structure (borders, padding)
    # Each column typically adds 3 characters for borders and spacing in most table formats
    structure_overhead = num_columns * 3
    available_width = max_table_width - structure_overhead
    
    # Check if we need to adjust widths
    total_width = sum(max_widths)
    if total_width > available_width:
        # Scale down all columns proportionally
        scale_factor = available_width / total_width
        adjusted_widths = [max(4, int(width * scale_factor)) for width in max_widths]
        
        # Make sure we're within the available width
        while sum(adjusted_widths) > available_width:
            # Find the widest column and reduce it by 1
            widest_idx = adjusted_widths.index(max(adjusted_widths))
            adjusted_widths[widest_idx] -= 1
            
            # Ensure we don't go below a reasonable minimum
            if adjusted_widths[widest_idx] < 4:
                break
        
        return adjusted_widths
    
    return max_widths


def format_cell(
    value: Any,
    column_format: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats a cell value based on its type and column format.

    Args:
        value: The cell value to format.
        column_format: Optional format type from SPECIAL_COLUMN_FORMATTERS.
        use_colors: Whether to use colors. If None, determined automatically.

    Returns:
        Formatted cell value as string.
    """
    if value is None:
        return "N/A"
    
    # Use special formatter if specified and available
    if column_format and column_format in SPECIAL_COLUMN_FORMATTERS:
        return SPECIAL_COLUMN_FORMATTERS[column_format](value, use_colors=use_colors)
    
    # Format based on value type
    if isinstance(value, float):
        if 0 <= value <= 1:
            # Likely a probability
            return format_probability(value, use_colors=use_colors)
        else:
            # Regular float
            return format_number(value)
    elif isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        # Date/time value
        return format_datetime_str(value)
    elif isinstance(value, bool):
        # Boolean value
        return "Yes" if value else "No"
    else:
        # Default to string representation
        return str(value)