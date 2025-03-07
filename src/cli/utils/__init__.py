"""
Entry point for the CLI utilities module of the ERCOT RTLMP spike prediction system.

This module exposes key functionality from various utility modules including formatters,
validators, error handlers, configuration helpers, progress bars, and output handlers
to provide a unified interface for CLI operations.
"""

__version__ = "0.1.0"

# Imports from formatters module
from .formatters import (
    format_price, format_probability, format_number, format_integer,
    format_date, format_datetime_str, format_rtlmp_data, format_forecast_data,
    format_table, format_dataframe, format_metrics, truncate_string, align_text,
    PRICE_FORMAT, PROBABILITY_FORMAT, TABLE_FORMAT
)

# Imports from validators module
from .validators import (
    validate_command_type, validate_log_level, validate_data_type,
    validate_visualization_type, validate_output_format, validate_node_id,
    validate_node_ids, validate_threshold_value, validate_threshold_values,
    validate_date, validate_date_range, validate_file_path, validate_directory_path,
    validate_model_type, validate_model_version, validate_hyperparameters,
    validate_positive_integer, validate_boolean, validate_cli_config,
    validate_command_params, ValidationDecorator
)

# Imports from error_handlers module
from .error_handlers import (
    ErrorHandler, ErrorHandlingContext, format_error_message,
    print_error_message, handle_error, with_error_handling
)

# Imports from config_helpers module
from .config_helpers import (
    find_config_file, load_cli_config, load_command_config,
    load_config_from_file, load_config_from_env, merge_configs,
    save_config_to_file, create_default_config_file,
    ConfigHelper, ConfigManager
)

# Imports from progress_bars module
from .progress_bars import (
    ProgressBar, IndeterminateSpinner, create_progress_bar,
    progress_bar_context, update_progress_bar, create_indeterminate_spinner,
    with_progress_bar, progress_callback, format_progress_message
)

# Imports from output_handlers module
from .output_handlers import (
    format_command_result, format_forecast_result, format_metrics_result,
    format_backtesting_result, format_feature_importance_result,
    handle_command_output, export_to_file, export_dataframe,
    dict_to_csv, list_of_dicts_to_csv, format_error,
    display_success_message, display_warning_message, display_error_message,
    OutputHandler
)