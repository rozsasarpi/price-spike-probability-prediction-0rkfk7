"""
Initializes and exports the UI components for the CLI interface of the ERCOT RTLMP spike prediction system.

This module provides a unified interface for accessing various UI components including
color formatting, text formatting, tables, spinners, and visualizations. These components
are designed to create a consistent and user-friendly command-line experience.
"""

# Import all UI components
from .colors import *
from .formatters import *
from .tables import *
from .spinners import *
from .charts import *

# Define what is exported from this package
__all__ = [
    # Color utilities
    "colorize", "bold", "italic", "underline", "color_by_status", "color_by_level", 
    "color_by_value", "color_by_probability", "supports_color", "strip_color", 
    "get_color_safe_length", "COLORS", "BACKGROUNDS", "STYLES", "STATUS_COLORS", 
    "LOG_LEVEL_COLORS",
    
    # Text formatting utilities
    "format_header", "format_subheader", "format_title", "format_paragraph", 
    "format_bullet_list", "format_numbered_list", "format_key_value", 
    "format_key_value_list", "format_section", "format_subsection", "format_box", 
    "get_terminal_size", "wrap_text", "truncate_text", "center_text", "align_text", 
    "DEFAULT_TERMINAL_WIDTH", "HEADER_CHAR", "SUBHEADER_CHAR",
    
    # Table utilities
    "create_table", "create_simple_table", "create_markdown_table", 
    "create_dataframe_table", "create_key_value_table", "create_metrics_table", 
    "create_comparison_table", "create_forecast_table", "create_feature_importance_table", 
    "create_confusion_matrix_table", "DEFAULT_TABLE_FORMAT", "TABLE_STYLES",
    
    # Spinner components
    "Spinner", "create_spinner", "spinner_context", "with_spinner", "get_spinner_frames", 
    "SPINNER_TYPES", "DEFAULT_SPINNER_TYPE",
    
    # Chart utilities
    "create_bar_chart", "create_horizontal_bar_chart", "create_vertical_bar_chart", 
    "create_line_chart", "create_multi_line_chart", "create_histogram", "create_sparkline", 
    "create_probability_sparkline", "create_heatmap", "create_confusion_matrix", 
    "create_roc_curve", "create_calibration_curve", "create_feature_importance_chart", 
    "create_metrics_chart", "create_probability_timeline", "create_threshold_comparison", 
    "create_from_dataframe", "Chart", "BarChart", "LineChart", "MultiLineChart", 
    "DEFAULT_CHART_WIDTH", "DEFAULT_CHART_HEIGHT", "PROBABILITY_COLORS", "METRIC_COLORS"
]