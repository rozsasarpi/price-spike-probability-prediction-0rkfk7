"""
Provides data formatting utilities for the CLI application of the ERCOT RTLMP spike prediction system.

This module contains functions for formatting various data types (prices, probabilities, dates, etc.)
in a consistent manner for display in the command-line interface.
"""

from typing import Any, Dict, List, Optional, Union
import datetime
import pandas as pd  # version 2.0+
from tabulate import tabulate  # version 0.9+
import numpy as np  # version 1.24+

from ..ui.colors import colorize, color_by_value, color_by_probability, supports_color
from ...backend.utils.type_definitions import NodeID, ThresholdValue, RTLMPDataDict, ForecastResultDict

# Format strings for consistent display
PRICE_FORMAT = "${:.2f}/MWh"
PROBABILITY_FORMAT = "{:.1%}"
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M"

# Table display constraints
TABLE_FORMAT = "grid"
MAX_TABLE_WIDTH = 80
MAX_TABLE_ROWS = 100
MAX_TABLE_COLS = 10


def format_price(price: Union[float, str, None], use_colors: Optional[bool] = None) -> str:
    """
    Formats a price value with currency symbol and units.
    
    Args:
        price: The price value to format.
        use_colors: Whether to apply color formatting based on price value.
        
    Returns:
        Formatted price string with currency symbol and units.
    """
    if price is None:
        return "N/A"
    
    try:
        float_price = float(price)
    except (ValueError, TypeError):
        return str(price)
    
    formatted_price = PRICE_FORMAT.format(float_price)
    
    if use_colors:
        return color_by_value(formatted_price, low_threshold=50.0, high_threshold=100.0, use_colors=use_colors)
    
    return formatted_price


def format_probability(probability: Union[float, str, None], use_colors: Optional[bool] = None) -> str:
    """
    Formats a probability value as a percentage.
    
    Args:
        probability: The probability value to format (0-1).
        use_colors: Whether to apply color formatting based on probability value.
        
    Returns:
        Formatted probability string as percentage.
    """
    if probability is None:
        return "N/A"
    
    try:
        float_prob = float(probability)
    except (ValueError, TypeError):
        return str(probability)
    
    # Ensure probability is between 0 and 1
    float_prob = max(0.0, min(1.0, float_prob))
    
    formatted_prob = PROBABILITY_FORMAT.format(float_prob)
    
    if use_colors:
        return color_by_probability(formatted_prob, use_colors=use_colors)
    
    return formatted_prob


def format_number(value: Union[float, int, str, None], decimal_places: int = 2) -> str:
    """
    Formats a numeric value with thousands separators and decimal places.
    
    Args:
        value: The numeric value to format.
        decimal_places: Number of decimal places to include.
        
    Returns:
        Formatted numeric string.
    """
    if value is None:
        return "N/A"
    
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    format_str = f"{{:,.{decimal_places}f}}"
    return format_str.format(float_value)


def format_integer(value: Union[int, str, None]) -> str:
    """
    Formats an integer value with thousands separators.
    
    Args:
        value: The integer value to format.
        
    Returns:
        Formatted integer string.
    """
    if value is None:
        return "N/A"
    
    try:
        int_value = int(value)
    except (ValueError, TypeError):
        return str(value)
    
    return f"{int_value:,}"


def format_date(date: Union[datetime.date, datetime.datetime, str, None]) -> str:
    """
    Formats a date object as a string.
    
    Args:
        date: The date to format.
        
    Returns:
        Formatted date string.
    """
    if date is None:
        return "N/A"
    
    if isinstance(date, str):
        try:
            date = datetime.datetime.fromisoformat(date)
        except ValueError:
            return date
    
    if isinstance(date, (datetime.date, datetime.datetime)):
        return date.strftime(DATE_FORMAT)
    
    return str(date)


def format_datetime_str(dt: Union[datetime.datetime, str, None]) -> str:
    """
    Formats a datetime object as a string.
    
    Args:
        dt: The datetime to format.
        
    Returns:
        Formatted datetime string.
    """
    if dt is None:
        return "N/A"
    
    if isinstance(dt, str):
        try:
            dt = datetime.datetime.fromisoformat(dt)
        except ValueError:
            return dt
    
    if isinstance(dt, datetime.datetime):
        return dt.strftime(DATETIME_FORMAT)
    
    return str(dt)


def format_rtlmp_data(data: Union[RTLMPDataDict, Dict[str, Any]], use_colors: Optional[bool] = None) -> Dict[str, str]:
    """
    Formats RTLMP data for display.
    
    Args:
        data: The RTLMP data dictionary.
        use_colors: Whether to apply color formatting.
        
    Returns:
        Dictionary with formatted RTLMP data.
    """
    formatted_data = {}
    
    # Format timestamp
    if "timestamp" in data:
        formatted_data["timestamp"] = format_datetime_str(data["timestamp"])
    
    # Format price
    if "price" in data:
        formatted_data["price"] = format_price(data["price"], use_colors=use_colors)
    
    # Format other price components
    for price_component in ["congestion_price", "loss_price", "energy_price"]:
        if price_component in data:
            formatted_data[price_component] = format_price(data[price_component], use_colors=use_colors)
    
    # Copy any other fields directly
    for key, value in data.items():
        if key not in formatted_data:
            formatted_data[key] = str(value)
    
    return formatted_data


def format_forecast_data(data: Union[ForecastResultDict, Dict[str, Any]], use_colors: Optional[bool] = None) -> Dict[str, str]:
    """
    Formats forecast data for display.
    
    Args:
        data: The forecast data dictionary.
        use_colors: Whether to apply color formatting.
        
    Returns:
        Dictionary with formatted forecast data.
    """
    formatted_data = {}
    
    # Format timestamps
    if "forecast_timestamp" in data:
        formatted_data["forecast_timestamp"] = format_datetime_str(data["forecast_timestamp"])
    
    if "target_timestamp" in data:
        formatted_data["target_timestamp"] = format_datetime_str(data["target_timestamp"])
    
    # Format threshold value
    if "threshold_value" in data:
        formatted_data["threshold_value"] = format_price(data["threshold_value"], use_colors=use_colors)
    
    # Format probability
    if "spike_probability" in data:
        formatted_data["spike_probability"] = format_probability(data["spike_probability"], use_colors=use_colors)
    
    # Format confidence intervals
    for interval in ["confidence_interval_lower", "confidence_interval_upper"]:
        if interval in data:
            formatted_data[interval] = format_probability(data[interval], use_colors=use_colors)
    
    # Copy any other fields directly
    for key, value in data.items():
        if key not in formatted_data:
            formatted_data[key] = str(value)
    
    return formatted_data


def format_table(data: List[List[Any]], headers: List[str], tablefmt: Optional[str] = None, use_colors: Optional[bool] = None) -> str:
    """
    Formats tabular data for CLI display.
    
    Args:
        data: The data to display (list of rows, each row is a list of values).
        headers: The column headers.
        tablefmt: The table format to use (from tabulate).
        use_colors: Whether to apply color formatting.
        
    Returns:
        Formatted table string.
    """
    if tablefmt is None:
        tablefmt = TABLE_FORMAT
    
    # Format cell values based on their type
    formatted_data = []
    for row in data:
        formatted_row = []
        for cell in row:
            if isinstance(cell, (float, int)) and cell >= 0 and cell <= 1:
                # Treat as probability
                formatted_row.append(format_probability(cell, use_colors=use_colors))
            elif isinstance(cell, float):
                # Generic float
                formatted_row.append(format_number(cell, decimal_places=2))
            elif isinstance(cell, int):
                # Integer
                formatted_row.append(format_integer(cell))
            elif isinstance(cell, (datetime.date, datetime.datetime)):
                # Date/datetime
                formatted_row.append(format_datetime_str(cell))
            else:
                # Default to string
                formatted_row.append(str(cell))
        formatted_data.append(formatted_row)
    
    return tabulate(formatted_data, headers=headers, tablefmt=tablefmt)


def format_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    tablefmt: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats a pandas DataFrame for CLI display.
    
    Args:
        df: The DataFrame to format.
        columns: Specific columns to include (if None, include all).
        max_rows: Maximum number of rows to display.
        max_cols: Maximum number of columns to display.
        tablefmt: The table format to use (from tabulate).
        use_colors: Whether to apply color formatting.
        
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
        if not valid_columns:  # No valid columns
            return "No valid columns selected"
        df_display = df[valid_columns].copy()
    else:
        df_display = df.copy()
    
    # Apply row and column limits
    if max_rows is None:
        max_rows = MAX_TABLE_ROWS
    
    if max_cols is None:
        max_cols = MAX_TABLE_COLS
    
    if len(df_display) > max_rows:
        # Show first half and last half of rows
        half_rows = max_rows // 2
        df_display = pd.concat([
            df_display.head(half_rows),
            pd.DataFrame([{col: "..." for col in df_display.columns}]),
            df_display.tail(half_rows)
        ])
    
    if len(df_display.columns) > max_cols:
        # Show first set of columns
        df_display = df_display.iloc[:, :max_cols]
        df_display["..."] = "..."
    
    # Format special columns
    for col in df_display.columns:
        col_lower = str(col).lower()
        
        # Check if the column contains price data
        if "price" in col_lower:
            df_display[col] = df_display[col].apply(
                lambda x: format_price(x, use_colors=use_colors)
            )
        
        # Check if the column contains probability data
        elif "probability" in col_lower or "confidence" in col_lower:
            df_display[col] = df_display[col].apply(
                lambda x: format_probability(x, use_colors=use_colors)
            )
        
        # Check if the column contains date/time data
        elif "date" in col_lower or "time" in col_lower:
            df_display[col] = df_display[col].apply(format_datetime_str)
    
    # Convert DataFrame to list format for tabulate
    headers = df_display.columns.tolist()
    data = df_display.values.tolist()
    
    return format_table(data, headers, tablefmt, use_colors)


def format_metrics(metrics: Dict[str, float], use_colors: Optional[bool] = None) -> str:
    """
    Formats model performance metrics for CLI display.
    
    Args:
        metrics: Dictionary of metric names and values.
        use_colors: Whether to apply color formatting.
        
    Returns:
        Formatted metrics table string.
    """
    formatted_metrics = []
    
    for metric_name, value in metrics.items():
        metric_lower = metric_name.lower()
        
        # Format based on metric type
        if any(term in metric_lower for term in ["auc", "precision", "recall", "f1", "accuracy"]):
            # Classification metrics are typically between 0 and 1
            formatted_value = format_probability(value, use_colors=use_colors)
        elif "brier" in metric_lower:
            # Brier score - lower is better
            formatted_value = format_number(value, decimal_places=3)
        elif "time" in metric_lower or "duration" in metric_lower:
            # Time metrics in seconds
            formatted_value = f"{format_number(value, decimal_places=1)}s"
        else:
            # Default formatting
            formatted_value = format_number(value, decimal_places=3)
        
        formatted_metrics.append([metric_name, formatted_value])
    
    # Create a table with the metrics
    return format_table(formatted_metrics, ["Metric", "Value"], use_colors=use_colors)


def truncate_string(text: str, max_length: int = 80, suffix: str = "...") -> str:
    """
    Truncates a long string for display.
    
    Args:
        text: The text to truncate.
        max_length: Maximum length before truncation.
        suffix: String to append when truncated.
        
    Returns:
        Truncated string.
    """
    if text is None:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def align_text(text: str, width: int = 20, alignment: str = "left") -> str:
    """
    Aligns text within a specified width.
    
    Args:
        text: The text to align.
        width: The width to align within.
        alignment: Alignment type ('left', 'right', 'center').
        
    Returns:
        Aligned text string.
    """
    if text is None:
        return ""
    
    if alignment == "left":
        return text.ljust(width)
    elif alignment == "right":
        return text.rjust(width)
    elif alignment == "center":
        return text.center(width)
    else:
        return text