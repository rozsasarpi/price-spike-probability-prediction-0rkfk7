"""
Provides ASCII/Unicode-based chart visualization utilities for the CLI interface of the ERCOT RTLMP spike prediction system.

This module enables the display of various charts and graphs directly in the terminal, including bar charts, 
line charts, histograms, and sparklines for visualizing forecast probabilities, model metrics, and other numerical data.
"""

import math
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+

from .colors import (
    colorize, 
    color_by_value, 
    color_by_probability, 
    supports_color,
    get_color_safe_length
)
from .formatters import get_terminal_size, truncate_text, align_text
from ..utils.formatters import format_number, format_price, format_probability
from ..logger import get_cli_logger

# Initialize logger for chart operations
CHART_LOGGER = get_cli_logger('charts')

# Default chart dimensions
DEFAULT_CHART_WIDTH = 80
DEFAULT_CHART_HEIGHT = 20

# Chart characters
BAR_CHAR = "█"
HALF_BAR_CHAR = "▄"
EMPTY_CHAR = " "
LINE_CHARS = ["⠁", "⠂", "⠄", "⠠", "⠐", "⠈"]
AXIS_HORIZONTAL = "─"
AXIS_VERTICAL = "│"
AXIS_CORNER = "┼"
SPARKLINE_CHARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

# Color mappings
PROBABILITY_COLORS = {"low": "green", "medium": "yellow", "high": "red"}
METRIC_COLORS = {
    "accuracy": "cyan", 
    "precision": "green", 
    "recall": "yellow", 
    "f1": "magenta", 
    "auc": "blue", 
    "brier_score": "red"
}


def create_bar_chart(
    data: Dict[str, float],
    width: Optional[int] = None,
    max_label_width: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None
) -> str:
    """
    Creates a horizontal bar chart for displaying categorical data.
    
    Args:
        data: Dictionary mapping labels to values
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        max_label_width: Maximum width for labels (default: auto-calculated)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the chart
    
    Returns:
        Formatted bar chart as a string
    """
    # Set defaults
    if width is None:
        term_width, _ = get_terminal_size()
        width = min(term_width, DEFAULT_CHART_WIDTH)
    
    if use_colors is None:
        use_colors = supports_color()
    
    # Find the maximum label length if not specified
    if max_label_width is None:
        max_label_width = max([len(label) for label in data.keys()], default=10) + 2
    
    # Find the maximum value for scaling
    max_value = max(data.values(), default=1.0)
    
    # Calculate available width for bars (accounting for label, spacing, and value)
    bar_width = width - max_label_width - 10  # Allow 10 chars for value display
    
    # Build the chart
    lines = []
    
    # Add title if provided
    if title:
        padding = (width - len(title)) // 2
        lines.append(" " * padding + title)
        lines.append("")
    
    # Generate each bar
    for label, value in data.items():
        # Format the label with appropriate padding
        formatted_label = label.ljust(max_label_width)
        
        # Calculate the bar length (ensure it's at least 1 if value > 0)
        bar_length = int(value / max_value * bar_width)
        if value > 0 and bar_length == 0:
            bar_length = 1
        
        # Create the bar
        bar = BAR_CHAR * bar_length
        
        # Apply color if enabled
        if use_colors:
            # Calculate color intensity based on value relative to max
            intensity = value / max_value
            if intensity < 0.3:
                color = "green"
            elif intensity < 0.7:
                color = "yellow"
            else:
                color = "red"
            bar = colorize(bar, color=color)
        
        # Format the value
        formatted_value = format_number(value)
        
        # Add the line to our output
        lines.append(f"{formatted_label} {bar} {formatted_value}")
    
    return "\n".join(lines)


def create_horizontal_bar_chart(
    data: Dict[str, float],
    width: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None,
    color_scheme: Optional[str] = None
) -> str:
    """
    Creates a horizontal bar chart with optional color coding.
    
    Args:
        data: Dictionary mapping labels to values
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the chart
        color_scheme: Color scheme to use ('default', 'probability', 'metric')
    
    Returns:
        Formatted horizontal bar chart as a string
    """
    return create_bar_chart(data, width, None, use_colors, title)


def create_vertical_bar_chart(
    data: Dict[str, float],
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None
) -> str:
    """
    Creates a vertical bar chart for displaying categorical data.
    
    Args:
        data: Dictionary mapping labels to values
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the chart
    
    Returns:
        Formatted vertical bar chart as a string
    """
    # Set defaults
    if width is None:
        term_width, _ = get_terminal_size()
        width = min(term_width, DEFAULT_CHART_WIDTH)
    
    if height is None:
        height = DEFAULT_CHART_HEIGHT
    
    if use_colors is None:
        use_colors = supports_color()
    
    # Find the maximum value for scaling
    max_value = max(data.values(), default=1.0)
    
    # Calculate bar width and spacing
    num_bars = len(data)
    bar_width = 2  # Default bar width
    spacing = 1  # Space between bars
    
    # Adjust if too many bars for the width
    available_width = width - 5  # Reserve space for y-axis
    total_width_needed = num_bars * (bar_width + spacing) - spacing
    
    if total_width_needed > available_width:
        # Reduce bar width if needed
        bar_width = max(1, (available_width + spacing) // num_bars - spacing)
    
    # Create a 2D grid for the chart (filled with spaces)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Add y-axis
    for i in range(height - 1):
        grid[i][2] = AXIS_VERTICAL
    
    # Add x-axis
    for i in range(3, width):
        grid[height - 1][i] = AXIS_HORIZONTAL
    
    # Add axis intersection
    grid[height - 1][2] = AXIS_CORNER
    
    # Calculate y-axis scale
    y_scale_factor = (height - 2) / max_value
    
    # Plot each bar
    color_grid = {}  # To store color information
    
    x_pos = 3  # Starting x position after y-axis
    x_labels = []  # To store label positions
    
    for label, value in data.items():
        # Calculate bar height
        bar_height = int(value * y_scale_factor)
        if value > 0 and bar_height == 0:
            bar_height = 1
        
        # Add bar to grid
        for i in range(bar_width):
            for j in range(bar_height):
                y_pos = height - 2 - j
                grid[y_pos][x_pos + i] = BAR_CHAR
                
                # Store color information if colors enabled
                if use_colors:
                    # Calculate color based on value
                    intensity = value / max_value
                    if intensity < 0.3:
                        color = "green"
                    elif intensity < 0.7:
                        color = "yellow"
                    else:
                        color = "red"
                    color_grid[(y_pos, x_pos + i)] = color
        
        # Store center of bar for label
        x_labels.append((x_pos + bar_width // 2, label))
        x_pos += bar_width + spacing
    
    # Add y-axis labels (scale)
    y_values = [max_value, max_value / 2, 0]
    for i, val in enumerate(y_values):
        label = format_number(val)
        y_pos = i * (height - 2) // 2
        for j, char in enumerate(label):
            if j < 2:  # Limit length to avoid overwriting axis
                grid[y_pos][1 - j] = char
    
    # Convert grid to text
    lines = []
    
    # Add title if provided
    if title:
        padding = (width - len(title)) // 2
        lines.append(" " * padding + title)
    
    # Add chart body
    for i in range(height):
        line = ""
        for j in range(width):
            char = grid[i][j]
            if use_colors and (i, j) in color_grid:
                line += colorize(char, color=color_grid[(i, j)])
            else:
                line += char
        lines.append(line)
    
    # Add x-axis labels
    x_label_line = " " * 3  # Align with chart
    for x_pos, label in x_labels:
        # Ensure label fits in its position
        truncated_label = truncate_text(label, bar_width + spacing, "")
        padding = x_pos - 3 - len(x_label_line)
        x_label_line += " " * padding + truncated_label
    
    lines.append(x_label_line)
    
    return "\n".join(lines)


def create_line_chart(
    data: List[float],
    labels: Optional[List[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> str:
    """
    Creates a line chart for displaying time series or sequential data.
    
    Args:
        data: List of values to plot
        labels: Optional list of labels for x-axis points
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the chart
        min_value: Minimum value for y-axis (default: min of data)
        max_value: Maximum value for y-axis (default: max of data)
    
    Returns:
        Formatted line chart as a string
    """
    # Set defaults
    if width is None:
        term_width, _ = get_terminal_size()
        width = min(term_width, DEFAULT_CHART_WIDTH)
    
    if height is None:
        height = DEFAULT_CHART_HEIGHT
    
    if use_colors is None:
        use_colors = supports_color()
    
    # Handle empty data case
    if not data:
        return "No data to display"
    
    # Determine y-axis range
    if min_value is None:
        min_value = min(data)
        # Add a small buffer below the minimum value
        min_value = min_value - abs(min_value) * 0.1 if min_value != 0 else 0
    
    if max_value is None:
        max_value = max(data)
        # Add a small buffer above the maximum value
        max_value = max_value + abs(max_value) * 0.1 if max_value != 0 else 1.0
    
    # Ensure min_value < max_value
    if min_value >= max_value:
        max_value = min_value + 1.0
    
    # Create a 2D grid for the chart (filled with spaces)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Add y-axis
    for i in range(height - 1):
        grid[i][2] = AXIS_VERTICAL
    
    # Add x-axis
    for i in range(3, width):
        grid[height - 1][i] = AXIS_HORIZONTAL
    
    # Add axis intersection
    grid[height - 1][2] = AXIS_CORNER
    
    # Calculate scaling factors
    y_scale_factor = (height - 2) / (max_value - min_value) if max_value > min_value else 1
    x_scale_factor = (width - 4) / (len(data) - 1) if len(data) > 1 else 1
    
    # Plot the line
    points = []
    for i, value in enumerate(data):
        x = 3 + int(i * x_scale_factor)
        y = height - 2 - int((value - min_value) * y_scale_factor)
        
        # Ensure y is within bounds
        y = max(0, min(height - 2, y))
        
        points.append((x, y))
    
    # Draw line by connecting points
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # Draw a simple line between points
        if x1 == x2:
            # Vertical line
            start, end = sorted([y1, y2])
            for y in range(start, end + 1):
                grid[y][x1] = "│"
        elif y1 == y2:
            # Horizontal line
            start, end = sorted([x1, x2])
            for x in range(start, end + 1):
                grid[y1][x] = "─"
        else:
            # Diagonal line using Bresenham's algorithm
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            
            while x1 != x2 or y1 != y2:
                grid[y1][x1] = select_line_char(x1, y1, x2, y2)
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
            
            grid[y2][x2] = select_line_char(x2, y2, x1, y1)
    
    # Mark data points
    for x, y in points:
        grid[y][x] = "●"
    
    # Add y-axis labels
    y_values = [max_value, (max_value + min_value) / 2, min_value]
    for i, val in enumerate(y_values):
        label = format_number(val)
        y_pos = i * (height - 2) // 2
        for j, char in enumerate(label):
            if 0 <= 1 - j < 2 and 0 <= y_pos < height:  # Ensure position is valid
                grid[y_pos][1 - j] = char
    
    # Convert grid to text with optional color
    lines = []
    
    # Add title if provided
    if title:
        padding = (width - len(title)) // 2
        lines.append(" " * padding + title)
    
    # Add chart body
    for i in range(height):
        line = ""
        for j in range(width):
            char = grid[i][j]
            if use_colors and char in ["●", LINE_CHARS[0], LINE_CHARS[1], LINE_CHARS[2], "─", "│", "/", "\\"]:
                # Determine color based on position (for points and lines)
                y_val = min_value + (height - 2 - i) / y_scale_factor * (max_value - min_value)
                # Normalize to 0-1 for color selection
                intensity = (y_val - min_value) / (max_value - min_value) if max_value > min_value else 0.5
                if intensity < 0.3:
                    color = "blue"
                elif intensity < 0.7:
                    color = "cyan"
                else:
                    color = "green"
                line += colorize(char, color=color)
            else:
                line += char
        lines.append(line)
    
    # Add x-axis labels if provided
    if labels:
        # Create evenly spaced labels based on available width
        if len(labels) > width - 4:
            # If too many labels, show a selection
            step = len(labels) // (width - 4)
            selected_labels = labels[::step]
        else:
            selected_labels = labels
        
        # Calculate positions for labels
        positions = []
        for i, _ in enumerate(selected_labels):
            pos = 3 + int(i * (width - 4) / max(1, len(selected_labels) - 1))
            positions.append(pos)
        
        # Create label line
        label_line = " " * 3  # Align with chart
        for pos, label in zip(positions, selected_labels):
            # Truncate label if needed
            short_label = truncate_text(label, 10)
            # Calculate padding to position
            padding = pos - len(label_line)
            if padding > 0:
                label_line += " " * padding + short_label
        
        lines.append(label_line)
    
    return "\n".join(lines)


def select_line_char(x1: int, y1: int, x2: int, y2: int) -> str:
    """
    Selects an appropriate character for drawing a line segment.
    
    Args:
        x1, y1: Starting point coordinates
        x2, y2: Ending point coordinates
    
    Returns:
        Character to use for the line segment
    """
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0:
        return "│"
    elif dy == 0:
        return "─"
    elif dx > 0 and dy < 0:
        return "/"
    elif dx < 0 and dy < 0:
        return "\\"
    elif dx > 0 and dy > 0:
        return "\\"
    elif dx < 0 and dy > 0:
        return "/"
    else:
        return "·"  # fallback


def create_multi_line_chart(
    data_series: Dict[str, List[float]],
    labels: Optional[List[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None
) -> str:
    """
    Creates a line chart with multiple lines for comparing multiple data series.
    
    Args:
        data_series: Dictionary mapping series names to lists of values
        labels: Optional list of labels for x-axis points
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the chart
        colors: Dictionary mapping series names to colors
    
    Returns:
        Formatted multi-line chart as a string
    """
    # Set defaults
    if width is None:
        term_width, _ = get_terminal_size()
        width = min(term_width, DEFAULT_CHART_WIDTH)
    
    if height is None:
        height = DEFAULT_CHART_HEIGHT
    
    if use_colors is None:
        use_colors = supports_color()
    
    if colors is None:
        colors = {}
    
    # Handle empty data case
    if not data_series:
        return "No data to display"
    
    # Determine overall y-axis range
    min_value = min([min(values) for values in data_series.values() if values])
    max_value = max([max(values) for values in data_series.values() if values])
    
    # Add a small buffer below and above
    y_buffer = (max_value - min_value) * 0.1 if max_value > min_value else 0.1
    min_value = min_value - y_buffer
    max_value = max_value + y_buffer
    
    # Create a 2D grid for the chart (filled with spaces)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Add y-axis
    for i in range(height - 1):
        grid[i][2] = AXIS_VERTICAL
    
    # Add x-axis
    for i in range(3, width):
        grid[height - 1][i] = AXIS_HORIZONTAL
    
    # Add axis intersection
    grid[height - 1][2] = AXIS_CORNER
    
    # Calculate scaling factors
    y_scale_factor = (height - 2) / (max_value - min_value) if max_value > min_value else 1
    
    # Store line styles and colors for legend
    line_styles = {}
    default_colors = ["cyan", "magenta", "yellow", "green", "blue", "red"]
    
    # Plot each data series
    color_grid = {}  # To store color information
    char_grid = {}   # To store character information for line style
    
    for idx, (series_name, values) in enumerate(data_series.items()):
        if not values:
            continue
        
        # Select line style and color
        line_char = "*" if idx < len(LINE_CHARS) else "+"
        color = colors.get(series_name, default_colors[idx % len(default_colors)])
        
        line_styles[series_name] = (line_char, color)
        
        # Calculate x-scale for this series
        x_scale_factor = (width - 4) / (len(values) - 1) if len(values) > 1 else 1
        
        # Calculate points
        points = []
        for i, value in enumerate(values):
            x = 3 + int(i * x_scale_factor)
            y = height - 2 - int((value - min_value) * y_scale_factor)
            
            # Ensure y is within bounds
            y = max(0, min(height - 2, y))
            
            points.append((x, y))
        
        # Draw line by connecting points
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # Use Bresenham's algorithm to draw line
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            
            while x1 != x2 or y1 != y2:
                # Only overwrite empty cells or cells from the same series
                if grid[y1][x1] == " " or (y1, x1) in char_grid and char_grid[(y1, x1)] == line_char:
                    grid[y1][x1] = line_char
                    char_grid[(y1, x1)] = line_char
                    color_grid[(y1, x1)] = color
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
            
            # Mark endpoint
            if grid[y2][x2] == " " or (y2, x2) in char_grid and char_grid[(y2, x2)] == line_char:
                grid[y2][x2] = line_char
                char_grid[(y2, x2)] = line_char
                color_grid[(y2, x2)] = color
    
    # Add y-axis labels
    y_values = [max_value, (max_value + min_value) / 2, min_value]
    for i, val in enumerate(y_values):
        label = format_number(val)
        y_pos = i * (height - 2) // 2
        for j, char in enumerate(label):
            if 0 <= 1 - j < 2 and 0 <= y_pos < height:
                grid[y_pos][1 - j] = char
    
    # Convert grid to text with color
    lines = []
    
    # Add title if provided
    if title:
        padding = (width - len(title)) // 2
        lines.append(" " * padding + title)
    
    # Add chart body
    for i in range(height):
        line = ""
        for j in range(width):
            char = grid[i][j]
            if use_colors and (i, j) in color_grid:
                line += colorize(char, color=color_grid[(i, j)])
            else:
                line += char
        lines.append(line)
    
    # Add x-axis labels if provided
    if labels:
        # Create evenly spaced labels
        if len(labels) > width - 4:
            step = len(labels) // (width - 4)
            selected_labels = labels[::step]
        else:
            selected_labels = labels
        
        # Calculate positions for labels
        positions = []
        for i, _ in enumerate(selected_labels):
            pos = 3 + int(i * (width - 4) / max(1, len(selected_labels) - 1))
            positions.append(pos)
        
        # Create label line
        label_line = " " * 3  # Align with chart
        for pos, label in zip(positions, selected_labels):
            short_label = truncate_text(label, 10)
            padding = pos - len(label_line)
            if padding > 0:
                label_line += " " * padding + short_label
        
        lines.append(label_line)
    
    # Add legend
    legend_lines = ["Legend:"]
    for series_name, (char, color) in line_styles.items():
        legend_entry = f" {char} {series_name}"
        if use_colors:
            legend_entry = f" {colorize(char, color=color)} {series_name}"
        legend_lines.append(legend_entry)
    
    # Arrange legend entries in columns if enough space
    if width >= 60:
        # Multi-column legend
        legend_formatted = ["Legend:"]
        entries_per_row = max(1, width // 20)
        for i in range(1, len(legend_lines), entries_per_row):
            row = "  ".join(legend_lines[i:i+entries_per_row])
            legend_formatted.append(" " + row)
        legend_lines = legend_formatted
    
    # Add legend to chart
    lines.append("")
    lines.extend(legend_lines)
    
    return "\n".join(lines)


def create_histogram(
    data: List[float],
    bins: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None
) -> str:
    """
    Creates a histogram for displaying the distribution of values.
    
    Args:
        data: List of values to plot in the histogram
        bins: Number of bins (default: auto-calculated)
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the chart
    
    Returns:
        Formatted histogram as a string
    """
    if not data:
        return "No data to display"
    
    # Set defaults
    if width is None:
        term_width, _ = get_terminal_size()
        width = min(term_width, DEFAULT_CHART_WIDTH)
    
    if height is None:
        height = DEFAULT_CHART_HEIGHT
    
    if use_colors is None:
        use_colors = supports_color()
    
    # Calculate bins if not specified
    if bins is None:
        # Estimate number of bins using Freedman-Diaconis rule
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr * (len(data) ** (-1/3)) if iqr > 0 else 1
        bins = max(5, min(20, int(np.ceil((max(data) - min(data)) / bin_width))))
    
    # Calculate histogram using numpy
    hist, bin_edges = np.histogram(data, bins=bins)
    
    # Create labels for bin ranges
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        bin_labels.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}")
    
    # Create data dictionary for the vertical bar chart
    hist_data = {label: count for label, count in zip(bin_labels, hist)}
    
    # Create the chart
    if title is None:
        title = "Histogram"
    
    return create_vertical_bar_chart(hist_data, width, height, use_colors, title)


def create_sparkline(
    data: List[float],
    width: Optional[int] = None,
    use_colors: Optional[bool] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> str:
    """
    Creates a compact sparkline visualization for trend display.
    
    Args:
        data: List of values to plot
        width: Width of the sparkline (default: length of data)
        use_colors: Whether to use colors (default: auto-detected)
        min_value: Minimum value for scaling (default: min of data)
        max_value: Maximum value for scaling (default: max of data)
    
    Returns:
        Formatted sparkline as a string
    """
    if not data:
        return ""
    
    # Set defaults
    if width is None:
        width = len(data)
    
    if use_colors is None:
        use_colors = supports_color()
    
    if min_value is None:
        min_value = min(data)
    
    if max_value is None:
        max_value = max(data)
    
    # Ensure min_value < max_value
    if min_value >= max_value:
        max_value = min_value + 1.0
    
    # Resample data to fit width if necessary
    if len(data) != width:
        indices = np.linspace(0, len(data) - 1, width)
        resampled_data = [data[int(i)] if i.is_integer() else 
                         (data[int(i)] + data[int(i) + 1]) / 2 
                         for i in indices]
        data = resampled_data
    
    # Create sparkline
    spark_chars = ""
    for value in data:
        # Scale value to fit in range of SPARKLINE_CHARS
        if max_value > min_value:
            normalized = (value - min_value) / (max_value - min_value)
        else:
            normalized = 0.5
        
        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        
        # Map to sparkline character
        char_idx = min(len(SPARKLINE_CHARS) - 1, int(normalized * len(SPARKLINE_CHARS)))
        char = SPARKLINE_CHARS[char_idx]
        
        # Apply color if enabled
        if use_colors:
            if normalized < 0.3:
                color = "blue"
            elif normalized < 0.7:
                color = "yellow"
            else:
                color = "red"
            char = colorize(char, color=color)
        
        spark_chars += char
    
    return spark_chars


def create_probability_sparkline(
    probabilities: List[float],
    width: Optional[int] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a sparkline specifically for displaying probability values.
    
    Args:
        probabilities: List of probability values (0-1)
        width: Width of the sparkline (default: length of data)
        use_colors: Whether to use colors (default: auto-detected)
    
    Returns:
        Formatted probability sparkline as a string
    """
    # Validate probabilities
    for p in probabilities:
        if not 0 <= p <= 1:
            CHART_LOGGER.warning(f"Invalid probability value: {p}")
    
    # Create sparkline with fixed min=0, max=1 for probabilities
    return create_sparkline(probabilities, width, use_colors, 0.0, 1.0)


def create_heatmap(
    data: List[List[float]],
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    width: Optional[int] = None,
    cell_width: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None
) -> str:
    """
    Creates a heatmap visualization for 2D data.
    
    Args:
        data: 2D array of values
        row_labels: Labels for rows
        col_labels: Labels for columns
        width: Width of the heatmap (default: terminal width or DEFAULT_CHART_WIDTH)
        cell_width: Width of each cell (default: auto-calculated)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the heatmap
    
    Returns:
        Formatted heatmap as a string
    """
    if not data or not data[0]:
        return "No data to display"
    
    # Set defaults
    if width is None:
        term_width, _ = get_terminal_size()
        width = min(term_width, DEFAULT_CHART_WIDTH)
    
    if use_colors is None:
        use_colors = supports_color()
    
    # Calculate cell dimensions
    num_cols = len(data[0])
    row_label_width = max([len(str(label)) for label in (row_labels or [])] + [3]) + 1
    
    available_width = width - row_label_width
    if cell_width is None:
        cell_width = max(3, min(10, available_width // num_cols))
    
    # Find min and max values for scaling
    flat_data = [item for sublist in data for item in sublist]
    min_value = min(flat_data)
    max_value = max(flat_data)
    
    # Normalize if min != max
    if min_value == max_value:
        normalized_data = [[0.5 for _ in row] for row in data]
    else:
        normalized_data = [[(val - min_value) / (max_value - min_value) for val in row] for row in data]
    
    # Build the heatmap
    lines = []
    
    # Add title if provided
    if title:
        padding = (width - len(title)) // 2
        lines.append(" " * padding + title)
        lines.append("")
    
    # Add column headers if provided
    if col_labels:
        header = " " * row_label_width
        for i, label in enumerate(col_labels):
            # Truncate and center cell labels
            cell_label = truncate_text(str(label), cell_width - 1)
            cell_content = cell_label.center(cell_width)
            header += cell_content
        lines.append(header)
        
        # Add separator line
        separator = " " * row_label_width + "─" * (cell_width * num_cols)
        lines.append(separator)
    
    # Add data rows
    for i, row in enumerate(normalized_data):
        # Add row label if provided
        if row_labels and i < len(row_labels):
            line = str(row_labels[i]).ljust(row_label_width)
        else:
            line = f"R{i}".ljust(row_label_width)
        
        # Add cells
        for j, val in enumerate(row):
            # Select block character based on value
            cell_text = format_number(data[i][j], decimal_places=1).center(cell_width)
            
            # Apply color if enabled
            if use_colors:
                # Color gradient based on value
                if val < 0.2:
                    color = "blue"
                elif val < 0.4:
                    color = "cyan"
                elif val < 0.6:
                    color = "green"
                elif val < 0.8:
                    color = "yellow"
                else:
                    color = "red"
                
                # Apply background color
                cell_text = colorize(cell_text, color=color)
            
            line += cell_text
        
        lines.append(line)
    
    # Add scale at the bottom
    scale_line = f"Scale: {format_number(min_value)} to {format_number(max_value)}"
    lines.append("")
    lines.append(scale_line)
    
    return "\n".join(lines)


def create_confusion_matrix(
    matrix: List[List[int]],
    class_labels: Optional[List[str]] = None,
    cell_width: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None
) -> str:
    """
    Creates a confusion matrix visualization.
    
    Args:
        matrix: 2D array containing the confusion matrix values
        class_labels: Labels for the classes
        cell_width: Width of each cell (default: auto-calculated)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the confusion matrix
    
    Returns:
        Formatted confusion matrix as a string
    """
    if not matrix:
        return "No data to display"
    
    # Set default class labels if not provided
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(len(matrix))]
    
    # Set default title if not provided
    if title is None:
        title = "Confusion Matrix"
    
    # Calculate total for percentage display
    total = sum(sum(row) for row in matrix)
    
    # Create matrix with percentages
    annotated_matrix = []
    for row in matrix:
        annotated_row = []
        for val in row:
            percentage = (val / total) * 100 if total > 0 else 0
            annotated_row.append(f"{val} ({percentage:.1f}%)")
        annotated_matrix.append(annotated_row)
    
    # Create the heatmap
    return create_heatmap(
        matrix,
        row_labels=class_labels,
        col_labels=class_labels,
        cell_width=cell_width,
        use_colors=use_colors,
        title=title
    )


def create_roc_curve(
    fpr: List[float],
    tpr: List[float],
    auc: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a simple ASCII ROC curve visualization.
    
    Args:
        fpr: List of false positive rates
        tpr: List of true positive rates
        auc: Area under the curve value
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
    
    Returns:
        Formatted ROC curve as a string
    """
    if len(fpr) != len(tpr):
        raise ValueError("FPR and TPR must have the same length")
    
    # Set defaults
    if width is None:
        term_width, _ = get_terminal_size()
        width = min(term_width, DEFAULT_CHART_WIDTH)
    
    if height is None:
        height = DEFAULT_CHART_HEIGHT
    
    if use_colors is None:
        use_colors = supports_color()
    
    # Create a 2D grid for the chart (filled with spaces)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Add y-axis (TPR)
    for i in range(height - 1):
        grid[i][2] = AXIS_VERTICAL
    
    # Add x-axis (FPR)
    for i in range(3, width):
        grid[height - 1][i] = AXIS_HORIZONTAL
    
    # Add axis intersection
    grid[height - 1][2] = AXIS_CORNER
    
    # Add diagonal reference line (random classifier)
    for i in range(height - 2):
        x = 3 + i * (width - 4) // (height - 2)
        y = height - 2 - i
        if grid[y][x] == " ":
            grid[y][x] = "·"
    
    # Plot ROC curve
    points = []
    for i, (x_val, y_val) in enumerate(zip(fpr, tpr)):
        x = 3 + int(x_val * (width - 4))
        y = height - 2 - int(y_val * (height - 2))
        
        # Ensure coordinates are within grid bounds
        x = max(3, min(width - 1, x))
        y = max(0, min(height - 2, y))
        
        points.append((x, y))
    
    # Draw line connecting points
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # Use Bresenham's algorithm to draw line
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while x1 != x2 or y1 != y2:
            grid[y1][x1] = "*"
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        
        grid[y2][x2] = "*"
    
    # Add axis labels
    grid[height - 1][width // 2] = "FPR"
    for i, char in enumerate("TPR"):
        if i < height - 2:
            grid[height // 2 - i][0] = char
    
    # Create title with AUC value
    title = f"ROC Curve (AUC: {auc:.3f})"
    
    # Convert grid to text with color
    lines = []
    
    # Add title
    padding = (width - len(title)) // 2
    lines.append(" " * padding + title)
    
    # Add chart body
    for i in range(height):
        line = ""
        for j in range(width):
            char = grid[i][j]
            if use_colors and char == "*":
                line += colorize(char, color="green")
            elif use_colors and char == "·":
                line += colorize(char, color="red")
            else:
                line += char
        lines.append(line)
    
    return "\n".join(lines)


def create_calibration_curve(
    pred_probs: List[float],
    true_probs: List[float],
    brier_score: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a simple ASCII calibration curve visualization.
    
    Args:
        pred_probs: List of predicted probabilities
        true_probs: List of true probabilities
        brier_score: Brier score value (optional)
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
    
    Returns:
        Formatted calibration curve as a string
    """
    if len(pred_probs) != len(true_probs):
        raise ValueError("Predicted and true probabilities must have the same length")
    
    # Set defaults
    if width is None:
        term_width, _ = get_terminal_size()
        width = min(term_width, DEFAULT_CHART_WIDTH)
    
    if height is None:
        height = DEFAULT_CHART_HEIGHT
    
    if use_colors is None:
        use_colors = supports_color()
    
    # Create a 2D grid for the chart (filled with spaces)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Add y-axis (Observed frequency)
    for i in range(height - 1):
        grid[i][2] = AXIS_VERTICAL
    
    # Add x-axis (Predicted probability)
    for i in range(3, width):
        grid[height - 1][i] = AXIS_HORIZONTAL
    
    # Add axis intersection
    grid[height - 1][2] = AXIS_CORNER
    
    # Add diagonal reference line (perfect calibration)
    for i in range(height - 2):
        x = 3 + i * (width - 4) // (height - 2)
        y = height - 2 - i
        if grid[y][x] == " ":
            grid[y][x] = "·"
    
    # Plot calibration curve
    points = []
    for x_val, y_val in zip(pred_probs, true_probs):
        x = 3 + int(x_val * (width - 4))
        y = height - 2 - int(y_val * (height - 2))
        
        # Ensure coordinates are within grid bounds
        x = max(3, min(width - 1, x))
        y = max(0, min(height - 2, y))
        
        points.append((x, y))
    
    # Draw points
    for x, y in points:
        grid[y][x] = "o"
    
    # Add axis labels
    grid[height - 1][width // 2] = "Predicted Probability"
    vertical_label = "Observed Frequency"
    for i, char in enumerate(vertical_label):
        if i < min(height - 2, len(vertical_label)):
            grid[i + 1][0] = char
    
    # Create title with Brier score if provided
    if brier_score is not None:
        title = f"Calibration Curve (Brier Score: {brier_score:.3f})"
    else:
        title = "Calibration Curve"
    
    # Convert grid to text with color
    lines = []
    
    # Add title
    padding = (width - len(title)) // 2
    lines.append(" " * padding + title)
    
    # Add chart body
    for i in range(height):
        line = ""
        for j in range(width):
            char = grid[i][j]
            if use_colors and char == "o":
                line += colorize(char, color="blue")
            elif use_colors and char == "·":
                line += colorize(char, color="green")
            else:
                line += char
        lines.append(line)
    
    return "\n".join(lines)


def create_feature_importance_chart(
    feature_importance: Dict[str, float],
    max_features: Optional[int] = None,
    width: Optional[int] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a horizontal bar chart showing feature importance values.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance values
        max_features: Maximum number of features to display (default: all features)
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        use_colors: Whether to use colors (default: auto-detected)
    
    Returns:
        Formatted feature importance chart as a string
    """
    if not feature_importance:
        return "No feature importance data to display"
    
    # Sort features by importance (descending)
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Limit the number of features if specified
    if max_features is not None:
        sorted_features = sorted_features[:max_features]
    
    # Format importance values as percentages
    total_importance = sum(feature_importance.values())
    
    # Handle case where total is zero
    if total_importance == 0:
        total_importance = 1
    
    formatted_importance = {
        name: value / total_importance
        for name, value in sorted_features
    }
    
    # Create the bar chart
    return create_horizontal_bar_chart(
        formatted_importance,
        width=width,
        use_colors=use_colors,
        title="Feature Importance"
    )


def create_metrics_chart(
    metrics: Dict[str, float],
    width: Optional[int] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a bar chart displaying model performance metrics.
    
    Args:
        metrics: Dictionary mapping metric names to values
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        use_colors: Whether to use colors (default: auto-detected)
    
    Returns:
        Formatted metrics chart as a string
    """
    if not metrics:
        return "No metrics data to display"
    
    # Format metrics appropriately
    formatted_metrics = {}
    for name, value in metrics.items():
        name_lower = name.lower()
        
        # Format based on metric type
        if any(term in name_lower for term in ["auc", "precision", "recall", "f1", "accuracy"]):
            # Classification metrics are typically between 0 and 1
            formatted_metrics[name] = value
        elif "brier" in name_lower:
            # Brier score - lower is better, use inverse for visualization
            formatted_metrics[name] = 1.0 - value if value <= 1.0 else 0.0
        else:
            # Default handling for unknown metrics
            formatted_metrics[name] = value
    
    # Create the bar chart
    return create_horizontal_bar_chart(
        formatted_metrics,
        width=width,
        use_colors=use_colors,
        title="Model Performance Metrics",
        color_scheme="metric"
    )


def create_probability_timeline(
    probabilities: List[float],
    timestamps: Optional[List[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None,
    threshold_label: Optional[str] = None
) -> str:
    """
    Creates a line chart showing probability values over time.
    
    Args:
        probabilities: List of probability values (0-1)
        timestamps: Optional list of timestamp labels for x-axis
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
        threshold_label: Label describing the threshold for these probabilities
    
    Returns:
        Formatted probability timeline chart as a string
    """
    # Validate probabilities
    for p in probabilities:
        if not 0 <= p <= 1:
            CHART_LOGGER.warning(f"Invalid probability value: {p}")
            # Clamp to valid range
            p = max(0.0, min(1.0, p))
    
    # Create title based on threshold if provided
    if threshold_label:
        title = f"Spike Probability Forecast (Threshold: {threshold_label})"
    else:
        title = "Spike Probability Forecast"
    
    # Create line chart with fixed min=0, max=1 for probabilities
    return create_line_chart(
        probabilities,
        labels=timestamps,
        width=width,
        height=height,
        use_colors=use_colors,
        title=title,
        min_value=0.0,
        max_value=1.0
    )


def create_threshold_comparison(
    threshold_probabilities: Dict[str, List[float]],
    timestamps: Optional[List[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Creates a multi-line chart comparing probabilities across different thresholds.
    
    Args:
        threshold_probabilities: Dictionary mapping threshold labels to list of probabilities
        timestamps: Optional list of timestamp labels for x-axis
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
    
    Returns:
        Formatted threshold comparison chart as a string
    """
    # Validate probabilities
    for threshold, probs in threshold_probabilities.items():
        for i, p in enumerate(probs):
            if not 0 <= p <= 1:
                CHART_LOGGER.warning(f"Invalid probability value: {p} for threshold {threshold}")
                # Clamp to valid range
                probs[i] = max(0.0, min(1.0, p))
    
    # Create color mapping for thresholds
    colors = {}
    default_colors = ["green", "yellow", "red", "blue", "magenta", "cyan"]
    
    for i, threshold in enumerate(threshold_probabilities.keys()):
        colors[threshold] = default_colors[i % len(default_colors)]
    
    # Create multi-line chart
    return create_multi_line_chart(
        threshold_probabilities,
        labels=timestamps,
        width=width,
        height=height,
        use_colors=use_colors,
        title="Threshold Comparison",
        colors=colors
    )


def create_from_dataframe(
    df: pd.DataFrame,
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    group_column: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_colors: Optional[bool] = None,
    title: Optional[str] = None
) -> str:
    """
    Creates a chart from a pandas DataFrame.
    
    Args:
        df: DataFrame containing the data
        chart_type: Type of chart to create ('bar', 'line', 'multi_line', 'histogram', 'heatmap')
        x_column: Column to use for x-axis (required for most chart types)
        y_column: Column to use for y-axis (required for most chart types)
        group_column: Column to use for grouping (for multi_line charts)
        width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
        height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
        use_colors: Whether to use colors (default: auto-detected)
        title: Optional title for the chart
    
    Returns:
        Formatted chart as a string
    """
    if df.empty:
        return "No data to display"
    
    # Handle different chart types
    if chart_type == 'bar':
        if not x_column or not y_column:
            raise ValueError("x_column and y_column are required for bar charts")
        
        # Create data dictionary
        values = df[y_column].tolist()
        labels = df[x_column].tolist()
        data = {str(label): value for label, value in zip(labels, values)}
        
        return create_vertical_bar_chart(data, width, height, use_colors, title)
    
    elif chart_type == 'line':
        if not y_column:
            raise ValueError("y_column is required for line charts")
        
        # Get data
        values = df[y_column].tolist()
        
        # Get labels if x_column is provided
        labels = None
        if x_column:
            labels = [str(x) for x in df[x_column].tolist()]
        
        return create_line_chart(values, labels, width, height, use_colors, title)
    
    elif chart_type == 'multi_line':
        if not y_column or not group_column:
            raise ValueError("y_column and group_column are required for multi_line charts")
        
        # Group data by the group column
        data_series = {}
        for group, group_df in df.groupby(group_column):
            data_series[str(group)] = group_df[y_column].tolist()
        
        # Get labels if x_column is provided
        labels = None
        if x_column:
            # Use the first group's x values
            first_group = next(iter(df.groupby(group_column)))
            labels = [str(x) for x in first_group[1][x_column].tolist()]
        
        return create_multi_line_chart(data_series, labels, width, height, use_colors, title)
    
    elif chart_type == 'histogram':
        if not y_column:
            raise ValueError("y_column is required for histograms")
        
        # Get data
        values = df[y_column].tolist()
        
        return create_histogram(values, None, width, height, use_colors, title)
    
    elif chart_type == 'heatmap':
        if not x_column or not y_column:
            raise ValueError("x_column and y_column are required for heatmaps")
        
        # Pivot DataFrame to create a 2D array
        if group_column:
            pivot_df = df.pivot(index=y_column, columns=x_column, values=group_column)
        else:
            # Without a group column, we need to determine what to put in the cells
            raise ValueError("group_column is required for heatmaps to provide cell values")
        
        # Convert to list of lists
        data = pivot_df.values.tolist()
        row_labels = [str(label) for label in pivot_df.index.tolist()]
        col_labels = [str(label) for label in pivot_df.columns.tolist()]
        
        return create_heatmap(data, row_labels, col_labels, width, None, use_colors, title)
    
    else:
        return f"Unsupported chart type: {chart_type}"


class Chart:
    """Base class for creating ASCII/Unicode charts in the terminal."""
    
    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        use_colors: Optional[bool] = None,
        title: Optional[str] = None
    ):
        """
        Initialize a Chart instance with the specified parameters.
        
        Args:
            width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
            height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
            use_colors: Whether to use colors (default: auto-detected)
            title: Optional title for the chart
        """
        # Set width
        if width is None:
            term_width, _ = get_terminal_size()
            self._width = min(term_width, DEFAULT_CHART_WIDTH)
        else:
            self._width = width
        
        # Set height
        self._height = height if height is not None else DEFAULT_CHART_HEIGHT
        
        # Set color usage
        self._use_colors = use_colors if use_colors is not None else supports_color()
        
        # Set title
        self._title = title
        
        # Initialize buffer for chart content
        self._buffer = []
    
    def render(self) -> str:
        """
        Render the chart as a string.
        
        Returns:
            Rendered chart as a string
        """
        return "\n".join(self._buffer)
    
    def add_title(self, title: str) -> None:
        """
        Add a title to the chart.
        
        Args:
            title: The title text
        """
        padding = (self._width - len(title)) // 2
        if self._use_colors:
            title = colorize(title, style="bold")
        self._buffer.append(" " * padding + title)
    
    def add_line(self, line: str) -> None:
        """
        Add a line of text to the chart buffer.
        
        Args:
            line: The text to add
        """
        self._buffer.append(line)
    
    def clear(self) -> None:
        """Clear the chart buffer."""
        self._buffer = []
    
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions of the chart.
        
        Returns:
            Width and height of the chart
        """
        return (self._width, self._height)


class BarChart(Chart):
    """Class for creating bar charts in the terminal."""
    
    def __init__(
        self,
        data: Dict[str, float],
        width: Optional[int] = None,
        max_label_width: Optional[int] = None,
        use_colors: Optional[bool] = None,
        title: Optional[str] = None,
        color_scheme: Optional[str] = None
    ):
        """
        Initialize a BarChart instance with the specified parameters.
        
        Args:
            data: Dictionary mapping labels to values
            width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
            max_label_width: Maximum width for labels (default: auto-calculated)
            use_colors: Whether to use colors (default: auto-detected)
            title: Optional title for the chart
            color_scheme: Color scheme to use (default, probability, metric)
        """
        super().__init__(width, None, use_colors, title)
        self._data = data
        
        # Set max label width
        if max_label_width is None:
            self._max_label_width = max([len(label) for label in data.keys()], default=10) + 2
        else:
            self._max_label_width = max_label_width
        
        # Set color scheme
        self._color_scheme = color_scheme
    
    def render(self) -> str:
        """
        Render the bar chart as a string.
        
        Returns:
            Rendered bar chart as a string
        """
        self.clear()
        
        # Add title if provided
        if self._title:
            self.add_title(self._title)
            self.add_line("")
        
        # Find the maximum value for scaling
        max_value = max(self._data.values(), default=1.0)
        
        # Calculate available width for bars
        bar_width = self._width - self._max_label_width - 10  # Allow 10 chars for value display
        
        # Generate each bar
        for label, value in self._data.items():
            # Format the label with appropriate padding
            formatted_label = label.ljust(self._max_label_width)
            
            # Calculate the bar length
            bar_length = int(value / max_value * bar_width)
            if value > 0 and bar_length == 0:
                bar_length = 1
            
            # Create the bar
            bar = BAR_CHAR * bar_length
            
            # Apply color if enabled
            if self._use_colors:
                # Select color based on scheme
                if self._color_scheme == "probability":
                    # For probability data
                    if value < 0.3:
                        color = PROBABILITY_COLORS["low"]
                    elif value < 0.7:
                        color = PROBABILITY_COLORS["medium"]
                    else:
                        color = PROBABILITY_COLORS["high"]
                elif self._color_scheme == "metric":
                    # For metric data, use predefined colors
                    color = METRIC_COLORS.get(label.lower(), "cyan")
                else:
                    # Default color scheme based on intensity
                    intensity = value / max_value
                    if intensity < 0.3:
                        color = "green"
                    elif intensity < 0.7:
                        color = "yellow"
                    else:
                        color = "red"
                
                bar = colorize(bar, color=color)
            
            # Format the value
            formatted_value = format_number(value)
            
            # Add the line to our output
            self.add_line(f"{formatted_label} {bar} {formatted_value}")
        
        return super().render()
    
    @classmethod
    def horizontal(
        cls,
        data: Dict[str, float],
        width: Optional[int] = None,
        use_colors: Optional[bool] = None,
        title: Optional[str] = None,
        color_scheme: Optional[str] = None
    ) -> "BarChart":
        """
        Create a horizontal bar chart.
        
        Args:
            data: Dictionary mapping labels to values
            width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
            use_colors: Whether to use colors (default: auto-detected)
            title: Optional title for the chart
            color_scheme: Color scheme to use (default, probability, metric)
        
        Returns:
            A new BarChart instance
        """
        return cls(data, width, None, use_colors, title, color_scheme)


class LineChart(Chart):
    """Class for creating line charts in the terminal."""
    
    def __init__(
        self,
        data: List[float],
        labels: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        use_colors: Optional[bool] = None,
        title: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ):
        """
        Initialize a LineChart instance with the specified parameters.
        
        Args:
            data: List of values to plot
            labels: Optional list of labels for x-axis points
            width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
            height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
            use_colors: Whether to use colors (default: auto-detected)
            title: Optional title for the chart
            min_value: Minimum value for y-axis (default: min of data)
            max_value: Maximum value for y-axis (default: max of data)
        """
        super().__init__(width, height, use_colors, title)
        self._data = data
        self._labels = labels
        
        # Determine y-axis range
        if min_value is None and data:
            self._min_value = min(data)
            # Add a small buffer below the minimum value
            self._min_value = self._min_value - abs(self._min_value) * 0.1 if self._min_value != 0 else 0
        else:
            self._min_value = min_value if min_value is not None else 0
        
        if max_value is None and data:
            self._max_value = max(data)
            # Add a small buffer above the maximum value
            self._max_value = self._max_value + abs(self._max_value) * 0.1 if self._max_value != 0 else 1.0
        else:
            self._max_value = max_value if max_value is not None else 1.0
        
        # Ensure min_value < max_value
        if self._min_value >= self._max_value:
            self._max_value = self._min_value + 1.0
    
    def render(self) -> str:
        """
        Render the line chart as a string.
        
        Returns:
            Rendered line chart as a string
        """
        self.clear()
        
        # Add title if provided
        if self._title:
            self.add_title(self._title)
        
        # Handle empty data case
        if not self._data:
            self.add_line("No data to display")
            return super().render()
        
        # Create a 2D grid for the chart (filled with spaces)
        grid = [[" " for _ in range(self._width)] for _ in range(self._height)]
        
        # Add y-axis
        for i in range(self._height - 1):
            grid[i][2] = AXIS_VERTICAL
        
        # Add x-axis
        for i in range(3, self._width):
            grid[self._height - 1][i] = AXIS_HORIZONTAL
        
        # Add axis intersection
        grid[self._height - 1][2] = AXIS_CORNER
        
        # Calculate scaling factors
        y_scale_factor = (self._height - 2) / (self._max_value - self._min_value) if self._max_value > self._min_value else 1
        x_scale_factor = (self._width - 4) / (len(self._data) - 1) if len(self._data) > 1 else 1
        
        # Plot the line
        points = []
        for i, value in enumerate(self._data):
            x = 3 + int(i * x_scale_factor)
            y = self._height - 2 - int((value - self._min_value) * y_scale_factor)
            
            # Ensure y is within bounds
            y = max(0, min(self._height - 2, y))
            
            points.append((x, y))
        
        # Draw line by connecting points
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # Draw a simple line between points
            if x1 == x2:
                # Vertical line
                start, end = sorted([y1, y2])
                for y in range(start, end + 1):
                    grid[y][x1] = "│"
            elif y1 == y2:
                # Horizontal line
                start, end = sorted([x1, x2])
                for x in range(start, end + 1):
                    grid[y1][x] = "─"
            else:
                # Diagonal line using Bresenham's algorithm
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy
                
                while x1 != x2 or y1 != y2:
                    grid[y1][x1] = select_line_char(x1, y1, x2, y2)
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x1 += sx
                    if e2 < dx:
                        err += dx
                        y1 += sy
                
                grid[y2][x2] = select_line_char(x2, y2, x1, y1)
        
        # Mark data points
        for x, y in points:
            grid[y][x] = "●"
        
        # Add y-axis labels
        y_values = [self._max_value, (self._max_value + self._min_value) / 2, self._min_value]
        for i, val in enumerate(y_values):
            label = format_number(val)
            y_pos = i * (self._height - 2) // 2
            for j, char in enumerate(label):
                if 0 <= 1 - j < 2 and 0 <= y_pos < self._height:
                    grid[y_pos][1 - j] = char
        
        # Convert grid to text with optional color
        for i in range(self._height):
            line = ""
            for j in range(self._width):
                char = grid[i][j]
                if self._use_colors and char in ["●", "\\", "/", "│", "─", "·"]:
                    # Determine color based on position (for points and lines)
                    y_val = self._min_value + (self._height - 2 - i) / y_scale_factor * (self._max_value - self._min_value)
                    # Normalize to 0-1 for color selection
                    intensity = (y_val - self._min_value) / (self._max_value - self._min_value) if self._max_value > self._min_value else 0.5
                    if intensity < 0.3:
                        color = "blue"
                    elif intensity < 0.7:
                        color = "cyan"
                    else:
                        color = "green"
                    line += colorize(char, color=color)
                else:
                    line += char
            self.add_line(line)
        
        # Add x-axis labels if provided
        if self._labels:
            # Create evenly spaced labels based on available width
            if len(self._labels) > self._width - 4:
                # If too many labels, show a selection
                step = len(self._labels) // (self._width - 4)
                selected_labels = self._labels[::step]
            else:
                selected_labels = self._labels
            
            # Calculate positions for labels
            positions = []
            for i, _ in enumerate(selected_labels):
                pos = 3 + int(i * (self._width - 4) / max(1, len(selected_labels) - 1))
                positions.append(pos)
            
            # Create label line
            label_line = " " * 3  # Align with chart
            for pos, label in zip(positions, selected_labels):
                # Truncate label if needed
                short_label = truncate_text(label, 10)
                # Calculate padding to position
                padding = pos - len(label_line)
                if padding > 0:
                    label_line += " " * padding + short_label
            
            self.add_line(label_line)
        
        return super().render()
    
    @classmethod
    def probability_timeline(
        cls,
        probabilities: List[float],
        timestamps: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        use_colors: Optional[bool] = None,
        threshold_label: Optional[str] = None
    ) -> "LineChart":
        """
        Create a line chart for probability values over time.
        
        Args:
            probabilities: List of probability values (0-1)
            timestamps: Optional list of timestamp labels for x-axis
            width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
            height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
            use_colors: Whether to use colors (default: auto-detected)
            threshold_label: Label describing the threshold for these probabilities
        
        Returns:
            A new LineChart instance
        """
        # Validate probabilities
        for i, p in enumerate(probabilities):
            if not 0 <= p <= 1:
                probabilities[i] = max(0.0, min(1.0, p))
        
        # Create title based on threshold if provided
        if threshold_label:
            title = f"Spike Probability Forecast (Threshold: {threshold_label})"
        else:
            title = "Spike Probability Forecast"
        
        return cls(
            probabilities,
            timestamps,
            width,
            height,
            use_colors,
            title,
            0.0,
            1.0
        )


class MultiLineChart(Chart):
    """Class for creating charts with multiple lines in the terminal."""
    
    def __init__(
        self,
        data_series: Dict[str, List[float]],
        labels: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        use_colors: Optional[bool] = None,
        title: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None
    ):
        """
        Initialize a MultiLineChart instance with the specified parameters.
        
        Args:
            data_series: Dictionary mapping series names to lists of values
            labels: Optional list of labels for x-axis points
            width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
            height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
            use_colors: Whether to use colors (default: auto-detected)
            title: Optional title for the chart
            colors: Dictionary mapping series names to colors
        """
        super().__init__(width, height, use_colors, title)
        self._data_series = data_series
        self._labels = labels
        self._colors = colors or {}
    
    def render(self) -> str:
        """
        Render the multi-line chart as a string.
        
        Returns:
            Rendered multi-line chart as a string
        """
        self.clear()
        
        # Add title if provided
        if self._title:
            self.add_title(self._title)
        
        # Handle empty data case
        if not self._data_series:
            self.add_line("No data to display")
            return super().render()
        
        # Determine overall y-axis range
        all_values = [val for values in self._data_series.values() for val in values if values]
        if not all_values:
            self.add_line("No data points to display")
            return super().render()
        
        min_value = min(all_values)
        max_value = max(all_values)
        
        # Add a small buffer below and above
        y_buffer = (max_value - min_value) * 0.1 if max_value > min_value else 0.1
        min_value = min_value - y_buffer
        max_value = max_value + y_buffer
        
        # Create a 2D grid for the chart (filled with spaces)
        grid = [[" " for _ in range(self._width)] for _ in range(self._height)]
        
        # Add y-axis
        for i in range(self._height - 1):
            grid[i][2] = AXIS_VERTICAL
        
        # Add x-axis
        for i in range(3, self._width):
            grid[self._height - 1][i] = AXIS_HORIZONTAL
        
        # Add axis intersection
        grid[self._height - 1][2] = AXIS_CORNER
        
        # Calculate y scaling factor
        y_scale_factor = (self._height - 2) / (max_value - min_value) if max_value > min_value else 1
        
        # Store line styles and colors for legend
        line_styles = {}
        default_colors = ["cyan", "magenta", "yellow", "green", "blue", "red"]
        
        # Plot each data series
        color_grid = {}  # To store color information
        char_grid = {}   # To store character information for line style
        
        for idx, (series_name, values) in enumerate(self._data_series.items()):
            if not values:
                continue
            
            # Select line style and color
            line_char = chr(ord('A') + idx % 26)  # A, B, C, ... as line markers
            color = self._colors.get(series_name, default_colors[idx % len(default_colors)])
            
            line_styles[series_name] = (line_char, color)
            
            # Calculate x-scale for this series
            x_scale_factor = (self._width - 4) / (len(values) - 1) if len(values) > 1 else 1
            
            # Calculate points
            points = []
            for i, value in enumerate(values):
                x = 3 + int(i * x_scale_factor)
                y = self._height - 2 - int((value - min_value) * y_scale_factor)
                
                # Ensure y is within bounds
                y = max(0, min(self._height - 2, y))
                
                points.append((x, y))
            
            # Draw line by connecting points
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                # Use Bresenham's algorithm to draw line
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy
                
                while x1 != x2 or y1 != y2:
                    # Only overwrite empty cells or cells from the same series
                    if grid[y1][x1] == " " or (y1, x1) in char_grid and char_grid[(y1, x1)] == line_char:
                        grid[y1][x1] = line_char
                        char_grid[(y1, x1)] = line_char
                        color_grid[(y1, x1)] = color
                    
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x1 += sx
                    if e2 < dx:
                        err += dx
                        y1 += sy
                
                # Mark endpoint
                if grid[y2][x2] == " " or (y2, x2) in char_grid and char_grid[(y2, x2)] == line_char:
                    grid[y2][x2] = line_char
                    char_grid[(y2, x2)] = line_char
                    color_grid[(y2, x2)] = color
        
        # Add y-axis labels
        y_values = [max_value, (max_value + min_value) / 2, min_value]
        for i, val in enumerate(y_values):
            label = format_number(val)
            y_pos = i * (self._height - 2) // 2
            for j, char in enumerate(label):
                if 0 <= 1 - j < 2 and 0 <= y_pos < self._height:
                    grid[y_pos][1 - j] = char
        
        # Convert grid to text with color
        for i in range(self._height):
            line = ""
            for j in range(self._width):
                char = grid[i][j]
                if self._use_colors and (i, j) in color_grid:
                    line += colorize(char, color=color_grid[(i, j)])
                else:
                    line += char
            self.add_line(line)
        
        # Add x-axis labels if provided
        if self._labels:
            # Create evenly spaced labels
            if len(self._labels) > self._width - 4:
                step = len(self._labels) // (self._width - 4)
                selected_labels = self._labels[::step]
            else:
                selected_labels = self._labels
            
            # Calculate positions for labels
            positions = []
            for i, _ in enumerate(selected_labels):
                pos = 3 + int(i * (self._width - 4) / max(1, len(selected_labels) - 1))
                positions.append(pos)
            
            # Create label line
            label_line = " " * 3  # Align with chart
            for pos, label in zip(positions, selected_labels):
                short_label = truncate_text(label, 10)
                padding = pos - len(label_line)
                if padding > 0:
                    label_line += " " * padding + short_label
            
            self.add_line(label_line)
        
        # Add legend
        legend_lines = ["Legend:"]
        for series_name, (char, color) in line_styles.items():
            legend_entry = f" {char} {series_name}"
            if self._use_colors:
                legend_entry = f" {colorize(char, color=color)} {series_name}"
            legend_lines.append(legend_entry)
        
        # Arrange legend entries in columns if enough space
        if self._width >= 60:
            # Multi-column legend
            legend_formatted = ["Legend:"]
            entries_per_row = max(1, self._width // 20)
            for i in range(1, len(legend_lines), entries_per_row):
                row = "  ".join(legend_lines[i:i+entries_per_row])
                legend_formatted.append(" " + row)
            legend_lines = legend_formatted
        
        # Add legend to chart
        self.add_line("")
        for line in legend_lines:
            self.add_line(line)
        
        return super().render()
    
    @classmethod
    def threshold_comparison(
        cls,
        threshold_probabilities: Dict[str, List[float]],
        timestamps: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        use_colors: Optional[bool] = None
    ) -> "MultiLineChart":
        """
        Create a multi-line chart comparing probabilities across different thresholds.
        
        Args:
            threshold_probabilities: Dictionary mapping threshold labels to list of probabilities
            timestamps: Optional list of timestamp labels for x-axis
            width: Width of the chart (default: terminal width or DEFAULT_CHART_WIDTH)
            height: Height of the chart (default: DEFAULT_CHART_HEIGHT)
            use_colors: Whether to use colors (default: auto-detected)
        
        Returns:
            A new MultiLineChart instance
        """
        # Validate probabilities
        for threshold, probs in threshold_probabilities.items():
            for i, p in enumerate(probs):
                if not 0 <= p <= 1:
                    probs[i] = max(0.0, min(1.0, p))
        
        # Create color mapping for thresholds
        colors = {}
        default_colors = ["green", "yellow", "red", "blue", "magenta", "cyan"]
        
        for i, threshold in enumerate(threshold_probabilities.keys()):
            colors[threshold] = default_colors[i % len(default_colors)]
        
        return cls(
            threshold_probabilities,
            timestamps,
            width,
            height,
            use_colors,
            "Threshold Comparison",
            colors
        )