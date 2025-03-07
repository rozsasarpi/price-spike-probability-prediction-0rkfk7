import pytest
import datetime
from pathlib import Path
import io
import tempfile
import pandas as pd
from typing import Dict, List, Any, Optional, Union

# Internal imports
from ../../cli/utils/formatters import (
    format_price, format_probability, format_date, format_datetime_str, 
    format_number, format_integer, format_rtlmp_data, format_forecast_data,
    format_table, format_dataframe, format_metrics, truncate_string, align_text
)
from ../../cli/utils/validators import (
    validate_command_type, validate_log_level, validate_data_type,
    validate_visualization_type, validate_output_format, validate_node_id,
    validate_node_ids, validate_threshold_value, validate_threshold_values,
    validate_date, validate_date_range, validate_file_path, validate_directory_path,
    validate_model_type, validate_model_version, validate_hyperparameters,
    validate_positive_integer, validate_boolean, validate_cli_config,
    validate_command_params, ValidationDecorator
)
from ../../cli/utils/error_handlers import (
    format_error_message, print_error_message, handle_error, with_error_handling,
    ErrorHandler, ErrorHandlingContext
)
from ../../cli/utils/config_helpers import (
    find_config_file, load_config_from_file, load_config_from_env,
    merge_configs, save_config_to_file, load_cli_config, load_command_config,
    create_default_config_file, ConfigHelper, ConfigManager
)
from ../../cli/utils/progress_bars import (
    ProgressBar, IndeterminateSpinner, create_progress_bar,
    progress_bar_context, update_progress_bar, create_indeterminate_spinner,
    with_progress_bar, progress_callback, format_progress_message
)
from ../../cli/utils/output_handlers import (
    format_command_result, format_forecast_result, format_metrics_result,
    format_backtesting_result, format_feature_importance_result,
    handle_command_output, export_to_file, export_dataframe,
    dict_to_csv, list_of_dicts_to_csv, format_error,
    display_success_message, display_warning_message, display_error_message,
    OutputHandler
)
from ../../cli/exceptions import CLIException, ValidationError, ConfigurationError, FileError


class TestFormatters:
    """Test class for formatter utility functions"""

    def setup_method(self, method):
        """Set up test fixtures for each test method"""
        # Create sample data for testing formatters
        self.rtlmp_data = {
            "timestamp": datetime.datetime(2023, 7, 15, 12, 0),
            "node_id": "HB_NORTH",
            "price": 45.75,
            "congestion_price": 10.25,
            "loss_price": 2.50,
            "energy_price": 33.00
        }
        
        self.forecast_data = {
            "forecast_timestamp": datetime.datetime(2023, 7, 15, 6, 0),
            "target_timestamp": datetime.datetime(2023, 7, 16, 14, 0),
            "threshold_value": 100.0,
            "spike_probability": 0.65,
            "confidence_interval_lower": 0.55,
            "confidence_interval_upper": 0.75,
            "model_version": "v1.2.3"
        }
        
        self.sample_df = pd.DataFrame({
            "timestamp": [datetime.datetime(2023, 7, 15, 12, 0), datetime.datetime(2023, 7, 15, 12, 5)],
            "node_id": ["HB_NORTH", "HB_NORTH"],
            "price": [45.75, 48.50],
            "probability": [0.65, 0.70]
        })

    def test_format_price(self):
        """Test the format_price function"""
        # Test with positive price value
        assert format_price(100.0) == "$100.00/MWh"
        
        # Test with zero price value
        assert format_price(0.0) == "$0.00/MWh"
        
        # Test with negative price value
        assert format_price(-10.5) == "$-10.50/MWh"
        
        # Test with None value
        assert format_price(None) == "N/A"
        
        # Test with string value that can be converted to float
        assert format_price("75.25") == "$75.25/MWh"
        
        # Test with color formatting
        colored_price = format_price(120.0, use_colors=True)
        assert "$120.00/MWh" in colored_price

    def test_format_probability(self):
        """Test the format_probability function"""
        # Test with probability value between 0 and 1
        assert format_probability(0.75) == "75.0%"
        
        # Test with 0 probability value
        assert format_probability(0) == "0.0%"
        
        # Test with 1 probability value
        assert format_probability(1) == "100.0%"
        
        # Test with None value
        assert format_probability(None) == "N/A"
        
        # Test with string value that can be converted to float
        assert format_probability("0.5") == "50.0%"
        
        # Test with out-of-range probability (should be clamped)
        assert format_probability(1.5) == "100.0%"
        assert format_probability(-0.5) == "0.0%"
        
        # Test with color formatting
        colored_prob = format_probability(0.85, use_colors=True)
        assert "85.0%" in colored_prob


class TestValidators:
    """Test class for validator utility functions"""

    def setup_method(self, method):
        """Set up test fixtures for each test method"""
        # Sample values for validation testing
        self.valid_command_type = "fetch-data"
        self.invalid_command_type = "invalid-command"
        
        self.valid_node_id = "HB_NORTH"
        self.invalid_node_id = "hb_north"  # lowercase not allowed
        
        self.valid_threshold = 150.0
        self.invalid_threshold_low = 5.0  # below MIN_THRESHOLD_VALUE
        self.invalid_threshold_high = 12000.0  # above MAX_THRESHOLD_VALUE
        
        self.valid_date = datetime.date(2023, 7, 15)
        self.invalid_date_early = datetime.date(2000, 1, 1)  # before MIN_DATE
        self.invalid_date_late = datetime.date(2060, 1, 1)  # after MAX_DATE
        
        # Create a temporary test file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        self.test_file_path = Path(self.temp_file.name)
        self.nonexistent_file_path = Path("/path/does/not/exist.txt")

    def teardown_method(self, method):
        """Clean up test fixtures after each test method"""
        if hasattr(self, 'test_file_path') and self.test_file_path.exists():
            self.test_file_path.unlink()

    def test_validate_command_type(self):
        """Test the validate_command_type function"""
        # Test with valid command types
        assert validate_command_type("fetch-data") == "fetch-data"
        assert validate_command_type("train") == "train"
        assert validate_command_type("predict") == "predict"
        assert validate_command_type("backtest") == "backtest"
        assert validate_command_type("evaluate") == "evaluate"
        assert validate_command_type("visualize") == "visualize"
        
        # Test with invalid command type
        with pytest.raises(ValidationError):
            validate_command_type("invalid-command")

    def test_validate_threshold_value(self):
        """Test the validate_threshold_value function"""
        # Test with valid threshold value
        assert validate_threshold_value(100.0) == 100.0
        assert validate_threshold_value(500.0) == 500.0
        
        # Test with threshold value as string
        assert validate_threshold_value("100.0") == 100.0
        
        # Test with threshold below minimum
        with pytest.raises(ValidationError):
            validate_threshold_value(5.0)
        
        # Test with threshold above maximum
        with pytest.raises(ValidationError):
            validate_threshold_value(12000.0)
        
        # Test with non-numeric value
        with pytest.raises(ValidationError):
            validate_threshold_value("not_a_number")


class TestErrorHandlers:
    """Test class for error handling utility functions"""

    def setup_method(self, method):
        """Set up test fixtures for each test method"""
        # Create sample exceptions for testing error handlers
        self.simple_error = ValueError("Test error message")
        self.cli_error = CLIException("CLI test error", {"param": "value"})
        self.nested_error = CLIException(
            "Outer error", 
            {"outer_param": "outer_value"}, 
            cause=ValueError("Inner error")
        )

    def test_format_error_message(self):
        """Test the format_error_message function"""
        # Test with simple exception
        message = format_error_message(self.simple_error, verbose=False)
        assert "Test error message" in message
        
        # Test with context
        context = {"context_key": "context_value"}
        message = format_error_message(self.simple_error, context, verbose=False)
        assert "Test error message" in message
        assert "context_key=context_value" in message
        
        # Test with CLI exception
        message = format_error_message(self.cli_error, verbose=False)
        assert "CLI test error" in message
        
        # Test with nested exception (cause)
        message = format_error_message(self.nested_error, verbose=True)
        assert "Outer error" in message
        assert "Inner error" in message

    def test_error_handler_class(self):
        """Test the ErrorHandler class"""
        # Create error handler with exit_on_error=False to prevent test termination
        handler = ErrorHandler(verbose=True, exit_on_error=False)
        
        # Test handle method
        exit_code = handler.handle(self.simple_error)
        assert exit_code > 0  # Should return non-zero exit code
        
        # Test with context
        context = {"context_key": "context_value"}
        exit_code = handler.handle(self.cli_error, context)
        assert exit_code > 0
        
        # Test decorator method
        @handler.decorator()
        def test_func():
            raise self.simple_error
        
        # Should handle the error and not propagate it
        test_func()


class TestConfigHelpers:
    """Test class for configuration utility functions"""

    def setup_method(self, method):
        """Set up test fixtures for each test method"""
        # Create temporary directory for config files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Sample configurations for testing
        self.sample_config = {
            "log_level": "INFO",
            "output_dir": str(self.temp_path / "output"),
            "command": {
                "param1": "value1",
                "param2": 123
            }
        }

    def teardown_method(self, method):
        """Clean up after each test method"""
        self.temp_dir.cleanup()

    def test_config_helper_class(self):
        """Test the ConfigHelper class"""
        # Create a ConfigHelper with sample configuration
        helper = ConfigHelper(self.sample_config)
        
        # Test get_value with existing keys
        assert helper.get_value("log_level") == "INFO"
        assert helper.get_value("command.param1") == "value1"
        assert helper.get_value("command.param2") == 123
        
        # Test get_value with non-existent keys
        assert helper.get_value("nonexistent") is None
        assert helper.get_value("nonexistent", default="default") == "default"
        assert helper.get_value("command.nonexistent") is None
        
        # Test set_value for new keys
        helper.set_value("new_key", "new_value")
        assert helper.get_value("new_key") == "new_value"
        
        # Test set_value for existing keys
        helper.set_value("log_level", "DEBUG")
        assert helper.get_value("log_level") == "DEBUG"
        
        # Test set_value for nested keys
        helper.set_value("command.param3", "value3")
        assert helper.get_value("command.param3") == "value3"

    def test_merge_configs(self):
        """Test the merge_configs function"""
        # Test merging empty dictionaries
        result = merge_configs([{}, {}])
        assert result == {}
        
        # Test merging non-overlapping dictionaries
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        result = merge_configs([dict1, dict2])
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}
        
        # Test merging with overlapping keys
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        result = merge_configs([dict1, dict2])
        assert result == {"a": 1, "b": 3, "c": 4}
        
        # Test merging nested dictionaries
        dict1 = {"a": {"x": 1, "y": 2}, "b": 3}
        dict2 = {"a": {"y": 3, "z": 4}, "c": 5}
        result = merge_configs([dict1, dict2])
        assert result == {"a": {"x": 1, "y": 3, "z": 4}, "b": 3, "c": 5}


class TestProgressBars:
    """Test class for progress bar utility functions"""

    def setup_method(self, method):
        """Set up test fixtures for each test method"""
        # Mock stdout for capturing progress bar output
        self.stdout_mock = io.StringIO()

    def test_progress_bar_class(self):
        """Test the ProgressBar class"""
        # Create a progress bar with total=100
        progress_bar = ProgressBar(total=100, desc="Test Progress", disable=True)
        
        # Test update method
        progress_bar.update(10)
        progress_bar.update(20, desc="Updated Description")
        
        # Test close method
        progress_bar.close()
        
        # Test reset method
        progress_bar.reset(total=50)
        progress_bar.update(25)
        
        # Test as context manager
        with ProgressBar(total=10, disable=True) as pbar:
            for i in range(10):
                pbar.update(1)

    def test_indeterminate_spinner_class(self):
        """Test the IndeterminateSpinner class"""
        # Create a spinner with disabled output for testing
        spinner = IndeterminateSpinner("Test Spinner", disable=True)
        
        # Test update method
        spinner.update("Updated message")
        
        # Test close method
        spinner.close("Completed")
        
        # Test context manager
        with IndeterminateSpinner("Testing...", disable=True) as sp:
            sp.update("Still testing...")


class TestOutputHandlers:
    """Test class for output handling utility functions"""

    def setup_method(self, method):
        """Set up test fixtures for each test method"""
        # Create sample output data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.command_result = {
            "command": "fetch-data",
            "status": "success",
            "data": {
                "timestamp": "2023-07-15 12:00",
                "node_id": "HB_NORTH",
                "price": 45.75
            },
            "message": "Data fetched successfully"
        }
        
        self.forecast_result = {
            "title": "RTLMP Spike Forecast",
            "forecast": [
                {
                    "target_timestamp": "2023-07-16 14:00",
                    "threshold_value": 100.0,
                    "spike_probability": 0.65
                },
                {
                    "target_timestamp": "2023-07-16 15:00",
                    "threshold_value": 100.0,
                    "spike_probability": 0.75
                }
            ]
        }
        
        self.sample_df = pd.DataFrame({
            "timestamp": ["2023-07-15 12:00", "2023-07-15 12:05"],
            "node_id": ["HB_NORTH", "HB_NORTH"],
            "price": [45.75, 48.50],
            "probability": [0.65, 0.70]
        })

    def teardown_method(self, method):
        """Clean up test fixtures after each test method"""
        self.temp_dir.cleanup()

    def test_format_command_result(self):
        """Test the format_command_result function"""
        # Test with text format
        text_output = format_command_result(self.command_result, output_format="text")
        assert "Data fetched successfully" in text_output
        
        # Test with JSON format
        json_output = format_command_result(self.command_result, output_format="json")
        assert "\"command\": \"fetch-data\"" in json_output
        assert "\"message\": \"Data fetched successfully\"" in json_output
        
        # Test with CSV format (may not work well with nested dictionaries)
        csv_output = format_command_result(self.command_result, output_format="csv")
        assert isinstance(csv_output, str)

    def test_export_to_file(self):
        """Test the export_to_file function"""
        # Test exporting to JSON file
        json_path = self.temp_path / "output.json"
        result = export_to_file(self.command_result, json_path)
        assert result is True
        assert json_path.exists()
        
        # Test exporting to CSV file
        csv_path = self.temp_path / "output.csv"
        result = export_to_file(self.forecast_result, csv_path)
        assert result is True
        assert csv_path.exists()
        
        # Test exporting to text file
        text_path = self.temp_path / "output.txt"
        result = export_to_file(self.command_result, text_path)
        assert result is True
        assert text_path.exists()

    def test_output_handler_class(self):
        """Test the OutputHandler class"""
        # Create OutputHandler instances for testing
        handler = OutputHandler(
            output_path=self.temp_path / "test_output.json",
            output_format="json",
            use_colors=False
        )
        
        # Save original print function and replace it with a no-op for testing
        original_print = print
        import builtins
        builtins.print = lambda *args, **kwargs: None
        
        try:
            # Test handle_result method
            result = handler.handle_result(self.command_result)
            assert result is True
            assert (self.temp_path / "test_output.json").exists()
            
            # Test export_dataframe method
            result = handler.export_dataframe(self.sample_df, custom_path=self.temp_path / "test_df.csv")
            assert result is True
            assert (self.temp_path / "test_df.csv").exists()
        finally:
            # Restore print function
            builtins.print = original_print


# Individual test functions for utility modules

def test_format_price():
    """Tests the format_price function with various inputs"""
    # Test with positive price value
    assert format_price(100.0) == "$100.00/MWh"
    
    # Test with zero price value
    assert format_price(0.0) == "$0.00/MWh"
    
    # Test with negative price value
    assert format_price(-10.5) == "$-10.50/MWh"
    
    # Test with None value
    assert format_price(None) == "N/A"
    
    # Test with string value that can be converted to float
    assert format_price("75.25") == "$75.25/MWh"
    
    # Test with color formatting enabled
    colored_price = format_price(120.0, use_colors=True)
    assert "$120.00/MWh" in colored_price

def test_format_probability():
    """Tests the format_probability function with various inputs"""
    # Test with probability value between 0 and 1
    assert format_probability(0.75) == "75.0%"
    
    # Test with 0 probability value
    assert format_probability(0) == "0.0%"
    
    # Test with 1 probability value
    assert format_probability(1) == "100.0%"
    
    # Test with None value
    assert format_probability(None) == "N/A"
    
    # Test with string value that can be converted to float
    assert format_probability("0.5") == "50.0%"
    
    # Test with color formatting enabled
    colored_prob = format_probability(0.85, use_colors=True)
    assert "85.0%" in colored_prob

def test_format_date():
    """Tests the format_date function with various inputs"""
    # Test with date object
    date_obj = datetime.date(2023, 7, 15)
    assert format_date(date_obj) == "2023-07-15"
    
    # Test with datetime object
    dt_obj = datetime.datetime(2023, 7, 15, 12, 30)
    assert format_date(dt_obj) == "2023-07-15"
    
    # Test with string in valid date format
    assert format_date("2023-07-15") == "2023-07-15"
    
    # Test with None value
    assert format_date(None) == "N/A"

def test_format_datetime_str():
    """Tests the format_datetime_str function with various inputs"""
    # Test with datetime object
    dt_obj = datetime.datetime(2023, 7, 15, 12, 30)
    assert format_datetime_str(dt_obj) == "2023-07-15 12:30"
    
    # Test with string in valid datetime format
    assert format_datetime_str("2023-07-15T12:30:00") == "2023-07-15 12:30"
    
    # Test with None value
    assert format_datetime_str(None) == "N/A"

def test_truncate_string():
    """Tests the truncate_string function"""
    # Test with string shorter than max length
    assert truncate_string("short", 10) == "short"
    
    # Test with string equal to max length
    assert truncate_string("exactlength", 11) == "exactlength"
    
    # Test with string longer than max length
    assert truncate_string("thisisalongstring", 10) == "thisisal..."
    
    # Test with None value
    assert truncate_string(None, 10) == ""
    
    # Test with custom suffix
    assert truncate_string("longstring", 7, suffix="..") == "longst.."

def test_align_text():
    """Tests the align_text function"""
    # Test with left alignment
    assert align_text("text", 10, "left") == "text      "
    
    # Test with right alignment
    assert align_text("text", 10, "right") == "      text"
    
    # Test with center alignment
    assert align_text("text", 10, "center") == "   text   "
    
    # Test with string longer than width
    assert align_text("longerthanwidth", 10, "left") == "longerthanwidth"
    
    # Test with None value
    assert align_text(None, 10, "left") == ""

def test_format_rtlmp_data():
    """Tests the format_rtlmp_data function"""
    # Create sample RTLMP data dictionary
    rtlmp_data = {
        "timestamp": datetime.datetime(2023, 7, 15, 12, 0),
        "node_id": "HB_NORTH",
        "price": 45.75,
        "congestion_price": 10.25,
        "loss_price": 2.50,
        "energy_price": 33.00
    }
    
    # Format the data
    formatted = format_rtlmp_data(rtlmp_data)
    
    # Check that all fields are formatted correctly
    assert formatted["timestamp"] == "2023-07-15 12:00"
    assert formatted["price"] == "$45.75/MWh"
    assert formatted["congestion_price"] == "$10.25/MWh"
    assert formatted["loss_price"] == "$2.50/MWh"
    assert formatted["energy_price"] == "$33.00/MWh"
    
    # Test with color formatting
    colored = format_rtlmp_data(rtlmp_data, use_colors=True)
    assert "$45.75/MWh" in colored["price"]
    
    # Test with missing fields
    incomplete_data = {"timestamp": datetime.datetime(2023, 7, 15, 12, 0), "price": 45.75}
    formatted = format_rtlmp_data(incomplete_data)
    assert formatted["timestamp"] == "2023-07-15 12:00"
    assert formatted["price"] == "$45.75/MWh"
    assert "congestion_price" not in formatted

def test_format_forecast_data():
    """Tests the format_forecast_data function"""
    # Create sample forecast data dictionary
    forecast_data = {
        "forecast_timestamp": datetime.datetime(2023, 7, 15, 6, 0),
        "target_timestamp": datetime.datetime(2023, 7, 16, 14, 0),
        "threshold_value": 100.0,
        "spike_probability": 0.65,
        "confidence_interval_lower": 0.55,
        "confidence_interval_upper": 0.75,
        "model_version": "v1.2.3"
    }
    
    # Format the data
    formatted = format_forecast_data(forecast_data)
    
    # Check that all fields are formatted correctly
    assert formatted["forecast_timestamp"] == "2023-07-15 06:00"
    assert formatted["target_timestamp"] == "2023-07-16 14:00"
    assert formatted["threshold_value"] == "$100.00/MWh"
    assert formatted["spike_probability"] == "65.0%"
    assert formatted["confidence_interval_lower"] == "55.0%"
    assert formatted["confidence_interval_upper"] == "75.0%"
    assert formatted["model_version"] == "v1.2.3"
    
    # Test with color formatting
    colored = format_forecast_data(forecast_data, use_colors=True)
    assert "65.0%" in colored["spike_probability"]
    
    # Test with missing fields
    incomplete_data = {
        "forecast_timestamp": datetime.datetime(2023, 7, 15, 6, 0),
        "spike_probability": 0.65
    }
    formatted = format_forecast_data(incomplete_data)
    assert formatted["forecast_timestamp"] == "2023-07-15 06:00"
    assert formatted["spike_probability"] == "65.0%"
    assert "target_timestamp" not in formatted

def test_validate_command_type():
    """Tests the validate_command_type function with various inputs"""
    # Test with valid command types
    assert validate_command_type("fetch-data") == "fetch-data"
    assert validate_command_type("train") == "train"
    assert validate_command_type("predict") == "predict"
    assert validate_command_type("backtest") == "backtest"
    assert validate_command_type("evaluate") == "evaluate"
    assert validate_command_type("visualize") == "visualize"
    
    # Test with invalid command type
    with pytest.raises(ValidationError):
        validate_command_type("invalid-command")

def test_validate_threshold_value():
    """Tests the validate_threshold_value function with various inputs"""
    # Test with valid threshold value
    assert validate_threshold_value(100.0) == 100.0
    assert validate_threshold_value(500.0) == 500.0
    
    # Test with threshold value as string
    assert validate_threshold_value("100.0") == 100.0
    
    # Test with threshold below minimum
    with pytest.raises(ValidationError):
        validate_threshold_value(5.0)
    
    # Test with threshold above maximum
    with pytest.raises(ValidationError):
        validate_threshold_value(12000.0)
    
    # Test with non-numeric value
    with pytest.raises(ValidationError):
        validate_threshold_value("not_a_number")

def test_validate_date():
    """Tests the validate_date function with various inputs"""
    # Test with valid date object
    valid_date = datetime.date(2023, 7, 15)
    assert validate_date(valid_date) == valid_date
    
    # Test with valid date string
    assert validate_date("2023-07-15") == datetime.date(2023, 7, 15)
    
    # Test with date before minimum date
    with pytest.raises(ValidationError):
        validate_date(datetime.date(2000, 1, 1))
    
    # Test with date after maximum date
    with pytest.raises(ValidationError):
        validate_date(datetime.date(2060, 1, 1))
    
    # Test with invalid date format
    with pytest.raises(ValidationError):
        validate_date("not-a-date")

def test_validate_date_range():
    """Tests the validate_date_range function with various inputs"""
    # Test with valid date range (start < end)
    start_date = datetime.date(2023, 7, 1)
    end_date = datetime.date(2023, 7, 31)
    result = validate_date_range(start_date, end_date)
    assert result == (start_date, end_date)
    
    # Test with valid date range (start = end)
    start_date = end_date = datetime.date(2023, 7, 15)
    result = validate_date_range(start_date, end_date)
    assert result == (start_date, end_date)
    
    # Test with invalid date range (start > end)
    start_date = datetime.date(2023, 7, 31)
    end_date = datetime.date(2023, 7, 1)
    with pytest.raises(ValidationError):
        validate_date_range(start_date, end_date)

def test_validation_decorator():
    """Tests the ValidationDecorator class"""
    # Define validators for testing
    validators = {
        "value": lambda x: x * 2 if x > 0 else ValidationError("Value must be positive", "value", x)
    }
    
    # Create a decorated test function
    @ValidationDecorator(validators)
    def test_func(value, other=None):
        return value
    
    # Test with valid parameters
    assert test_func(5) == 10
    
    # Test with invalid parameters
    with pytest.raises(ValidationError):
        test_func(-1)
    
    # Test with raise_errors=False option
    decorator = ValidationDecorator(validators, raise_errors=False)
    
    @decorator
    def test_func2(value, other=None):
        return value
    
    # Should not raise, just return the input value
    assert test_func2(-1) == -1

def test_error_handler():
    """Tests the ErrorHandler class"""
    # Create error handler with exit_on_error=False to prevent test termination
    handler = ErrorHandler(verbose=True, exit_on_error=False)
    
    # Test handle method with different exception types
    simple_error = ValueError("Test error message")
    exit_code = handler.handle(simple_error)
    assert exit_code > 0  # Should return non-zero exit code
    
    cli_error = CLIException("CLI test error", {"param": "value"})
    exit_code = handler.handle(cli_error)
    assert exit_code > 0
    
    # Test context method for context manager functionality
    with handler.context({"test": "context"}):
        # No exception here, should not raise
        pass
    
    # Test decorator method for function decoration
    @handler.decorator()
    def test_func():
        raise simple_error
    
    # Should handle the error and not propagate it
    test_func()

def test_error_handling_context():
    """Tests the ErrorHandlingContext class"""
    # Create context with exit_on_error=False to prevent test termination
    context_mgr = ErrorHandlingContext(
        {"test": "context"}, 
        verbose=True, 
        exit_on_error=False
    )
    
    # Test with no exception
    with context_mgr:
        pass  # No exception raised
    
    # Test with exception raised inside context
    try:
        with context_mgr:
            raise ValueError("Test error inside context")
    except:
        pytest.fail("Exception should have been handled by context")

def test_format_error_message():
    """Tests the format_error_message function"""
    # Test with simple exception
    error = ValueError("Test error message")
    message = format_error_message(error, verbose=False)
    assert "Test error message" in message
    
    # Test with exception and context
    context = {"context_key": "context_value"}
    message = format_error_message(error, context, verbose=False)
    assert "Test error message" in message
    assert "context_key=context_value" in message
    
    # Test with nested exception (cause)
    nested_error = CLIException(
        "Outer error", 
        {"outer_param": "outer_value"}, 
        cause=ValueError("Inner error")
    )
    message = format_error_message(nested_error, verbose=True)
    assert "Outer error" in message
    assert "Inner error" in message
    
    # Test with verbose option
    detailed_message = format_error_message(error, verbose=True)
    assert "Test error message" in detailed_message
    
    # Test with color formatting
    colored_message = format_error_message(error, use_colors=True)
    assert "Test error message" in colored_message

def test_config_helper():
    """Tests the ConfigHelper class"""
    # Create sample configuration for testing
    sample_config = {
        "log_level": "INFO",
        "output_dir": "/tmp/output",
        "command": {
            "param1": "value1",
            "param2": 123
        }
    }
    
    # Create a ConfigHelper with sample configuration
    helper = ConfigHelper(sample_config)
    
    # Test get_value with existing keys
    assert helper.get_value("log_level") == "INFO"
    assert helper.get_value("command.param1") == "value1"
    assert helper.get_value("command.param2") == 123
    
    # Test get_value with non-existent keys
    assert helper.get_value("nonexistent") is None
    assert helper.get_value("nonexistent", default="default") == "default"
    assert helper.get_value("command.nonexistent") is None
    
    # Test set_value for new keys
    helper.set_value("new_key", "new_value")
    assert helper.get_value("new_key") == "new_value"
    
    # Test set_value for existing keys
    helper.set_value("log_level", "DEBUG")
    assert helper.get_value("log_level") == "DEBUG"
    
    # Test set_value for nested keys
    helper.set_value("command.param3", "value3")
    assert helper.get_value("command.param3") == "value3"
    
    # Test get_config() returns a copy of the config
    config = helper.get_config()
    assert config["log_level"] == "DEBUG"
    config["log_level"] = "CHANGED"
    assert helper.get_value("log_level") == "DEBUG"  # Original unchanged
    
    # Test update_config()
    helper.update_config({"new_section": {"key": "value"}})
    assert helper.get_value("new_section.key") == "value"

def test_merge_configs():
    """Tests the merge_configs function"""
    # Test merging empty dictionaries
    result = merge_configs([{}, {}])
    assert result == {}
    
    # Test merging dictionaries with non-overlapping keys
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    result = merge_configs([dict1, dict2])
    assert result == {"a": 1, "b": 2, "c": 3, "d": 4}
    
    # Test merging dictionaries with overlapping keys
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    result = merge_configs([dict1, dict2])
    assert result == {"a": 1, "b": 3, "c": 4}
    
    # Test merging nested dictionaries
    dict1 = {"a": {"x": 1, "y": 2}, "b": 3}
    dict2 = {"a": {"y": 3, "z": 4}, "c": 5}
    result = merge_configs([dict1, dict2])
    assert result == {"a": {"x": 1, "y": 3, "z": 4}, "b": 3, "c": 5}
    
    # Test merging lists (should replace, not merge)
    dict1 = {"a": [1, 2, 3]}
    dict2 = {"a": [4, 5, 6]}
    result = merge_configs([dict1, dict2])
    assert result == {"a": [4, 5, 6]}

def test_load_config_from_file():
    """Tests the load_config_from_file function"""
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create YAML file with sample config
        yaml_path = temp_path / "config.yaml"
        with open(yaml_path, 'w') as f:
            f.write("""
log_level: INFO
output_dir: /tmp/output
command:
  param1: value1
  param2: 123
""")
        
        # Create JSON file with sample config
        json_path = temp_path / "config.json"
        with open(json_path, 'w') as f:
            f.write("""
{
  "log_level": "DEBUG",
  "output_dir": "/tmp/output",
  "command": {
    "param1": "value1",
    "param2": 456
  }
}
""")
        
        # Test loading from YAML file
        yaml_config = load_config_from_file(yaml_path)
        assert yaml_config["log_level"] == "INFO"
        assert yaml_config["command"]["param2"] == 123
        
        # Test loading from JSON file
        json_config = load_config_from_file(json_path)
        assert json_config["log_level"] == "DEBUG"
        assert json_config["command"]["param2"] == 456
        
        # Test with non-existent file
        with pytest.raises(FileError):
            load_config_from_file(temp_path / "nonexistent.yaml")
        
        # Test with invalid file format
        invalid_path = temp_path / "invalid.txt"
        invalid_path.touch()
        with pytest.raises(FileError):
            load_config_from_file(invalid_path)

def test_save_config_to_file():
    """Tests the save_config_to_file function"""
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Sample configuration to save
        sample_config = {
            "log_level": "INFO",
            "output_dir": "/tmp/output",
            "command": {
                "param1": "value1",
                "param2": 123
            }
        }
        
        # Test saving to YAML file
        yaml_path = temp_path / "output.yaml"
        result = save_config_to_file(sample_config, yaml_path)
        assert result is True
        assert yaml_path.exists()
        
        # Verify content by loading it back
        loaded_config = load_config_from_file(yaml_path)
        assert loaded_config["log_level"] == sample_config["log_level"]
        assert loaded_config["command"]["param1"] == sample_config["command"]["param1"]
        
        # Test saving to JSON file
        json_path = temp_path / "output.json"
        result = save_config_to_file(sample_config, json_path)
        assert result is True
        assert json_path.exists()
        
        # Verify content by loading it back
        loaded_config = load_config_from_file(json_path)
        assert loaded_config["log_level"] == sample_config["log_level"]
        
        # Test saving to file with non-existent directory (should create it)
        deep_path = temp_path / "new_dir" / "config.yaml"
        result = save_config_to_file(sample_config, deep_path)
        assert result is True
        assert deep_path.exists()

def test_progress_bar():
    """Tests the ProgressBar class"""
    # Create a progress bar with total=100
    progress_bar = ProgressBar(total=100, desc="Test Progress", disable=True)
    
    # Test update method with different increments
    progress_bar.update(10)
    progress_bar.update(n=20)
    
    # Test update method with description changes
    progress_bar.update(desc="Updated Description")
    
    # Test reset method
    progress_bar.reset(total=50)
    
    # Test close method
    progress_bar.close()
    
    # Test context manager protocol (__enter__, __exit__)
    with ProgressBar(total=10, disable=True) as pbar:
        pbar.update(5)
        pbar.update(5)
    
    # Test get_callback method
    callback = progress_bar.get_callback()
    assert callable(callback)

def test_indeterminate_spinner():
    """Tests the IndeterminateSpinner class"""
    # Create a spinner
    spinner = IndeterminateSpinner(desc="Test Spinner", disable=True)
    
    # Test update method with new description
    spinner.update("Updated Spinner")
    
    # Test close method with final message
    spinner.close(final_message="Completed")
    
    # Test context manager protocol (__enter__, __exit__)
    with IndeterminateSpinner(desc="Test Spinner", disable=True) as sp:
        sp.update("Working...")
    
    # Test context manager with exception handling
    try:
        with IndeterminateSpinner(desc="Test Spinner", disable=True) as sp:
            raise ValueError("Test error")
    except ValueError:
        pass  # Exception should be reraised

def test_progress_bar_context():
    """Tests the progress_bar_context function"""
    # Use progress_bar_context with different parameters
    with progress_bar_context(total=100, desc="Test", disable=True) as pbar:
        pbar.update(50)
        pbar.update(50)
    
    # Test normal operation within context
    with progress_bar_context(total=10, disable=True) as pbar:
        for i in range(10):
            pbar.update(1)
    
    # Test exception handling within context
    try:
        with progress_bar_context(total=10, disable=True) as pbar:
            pbar.update(5)
            raise ValueError("Test error")
    except ValueError:
        pass  # Exception should be reraised
    
    # Verify progress bar is properly closed after context exit
    with progress_bar_context(total=10, disable=True) as pbar:
        pass  # Empty block

def test_with_progress_bar():
    """Tests the with_progress_bar decorator"""
    # Create a test function decorated with with_progress_bar
    @with_progress_bar(total=10, desc="Test Function", disable=True)
    def test_function(a, b, progress_bar=None):
        assert progress_bar is not None
        progress_bar.update(5)
        progress_bar.update(5)
        return a + b
    
    # Test function execution with different parameters
    result = test_function(3, 4)
    assert result == 7
    
    # Verify progress bar is passed to the function
    @with_progress_bar(total=5, disable=True)
    def check_progress_bar(progress_bar=None):
        assert progress_bar is not None
        assert hasattr(progress_bar, 'update')
        assert hasattr(progress_bar, 'close')
        return True
    
    assert check_progress_bar() is True

def test_format_command_result():
    """Tests the format_command_result function"""
    # Create sample command result dictionary
    command_result = {
        "command": "fetch-data",
        "status": "success",
        "data": {
            "timestamp": "2023-07-15 12:00",
            "node_id": "HB_NORTH",
            "price": 45.75
        },
        "message": "Data fetched successfully"
    }
    
    # Test formatting as text
    text_output = format_command_result(command_result, output_format="text")
    assert "Data fetched successfully" in text_output
    
    # Test formatting as JSON
    json_output = format_command_result(command_result, output_format="json")
    assert "\"command\": \"fetch-data\"" in json_output
    assert "\"message\": \"Data fetched successfully\"" in json_output
    
    # Test formatting as CSV
    csv_output = format_command_result(command_result, output_format="csv")
    assert isinstance(csv_output, str)
    
    # Test with color formatting enabled
    colored_output = format_command_result(command_result, use_colors=True)
    assert "Data fetched successfully" in colored_output

def test_format_forecast_result():
    """Tests the format_forecast_result function"""
    # Create sample forecast result dictionary
    forecast_result = {
        "title": "RTLMP Spike Forecast",
        "forecast": [
            {
                "target_timestamp": "2023-07-16 14:00",
                "threshold_value": 100.0,
                "spike_probability": 0.65
            },
            {
                "target_timestamp": "2023-07-16 15:00",
                "threshold_value": 100.0,
                "spike_probability": 0.75
            }
        ]
    }
    
    # Test formatting as text
    text_output = format_forecast_result(forecast_result, output_format="text")
    assert "RTLMP Spike Forecast" in text_output
    assert "65.0%" in text_output
    assert "75.0%" in text_output
    
    # Test formatting as JSON
    json_output = format_forecast_result(forecast_result, output_format="json")
    assert "\"title\": \"RTLMP Spike Forecast\"" in json_output
    assert "\"spike_probability\": 0.65" in json_output
    
    # Test formatting as CSV
    csv_output = format_forecast_result(forecast_result, output_format="csv")
    assert "target_timestamp" in csv_output
    assert "threshold_value" in csv_output
    assert "spike_probability" in csv_output
    
    # Test with color formatting enabled
    colored_output = format_forecast_result(forecast_result, use_colors=True)
    assert "RTLMP Spike Forecast" in colored_output

def test_format_metrics_result():
    """Tests the format_metrics_result function"""
    # Create sample metrics result dictionary
    metrics_result = {
        "title": "Model Performance Metrics",
        "metrics": {
            "auc": 0.85,
            "precision": 0.76,
            "recall": 0.72,
            "f1_score": 0.74,
            "brier_score": 0.12
        }
    }
    
    # Test formatting as text
    text_output = format_metrics_result(metrics_result, output_format="text")
    assert "Model Performance Metrics" in text_output
    assert "85.0%" in text_output  # AUC
    assert "76.0%" in text_output  # Precision
    
    # Test formatting as JSON
    json_output = format_metrics_result(metrics_result, output_format="json")
    assert "\"title\": \"Model Performance Metrics\"" in json_output
    assert "\"auc\": 0.85" in json_output
    
    # Test formatting as CSV
    csv_output = format_metrics_result(metrics_result, output_format="csv")
    assert "auc" in csv_output
    assert "precision" in csv_output
    
    # Test with color formatting enabled
    colored_output = format_metrics_result(metrics_result, use_colors=True)
    assert "Model Performance Metrics" in colored_output

def test_dict_to_csv():
    """Tests the dict_to_csv function"""
    # Create sample dictionary
    data = {"a": 1, "b": 2, "c": 3}
    
    # Test conversion to CSV with default delimiter
    csv_output = dict_to_csv(data)
    assert "a,b,c" in csv_output
    assert "1,2,3" in csv_output
    
    # Test conversion to CSV with custom delimiter
    csv_output = dict_to_csv(data, delimiter=";")
    assert "a;b;c" in csv_output
    assert "1;2;3" in csv_output
    
    # Test with empty dictionary
    empty_output = dict_to_csv({})
    assert empty_output == ""

def test_list_of_dicts_to_csv():
    """Tests the list_of_dicts_to_csv function"""
    # Create sample list of dictionaries
    data = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6},
        {"a": 7, "b": 8, "d": 9}  # Note 'd' instead of 'c'
    ]
    
    # Test conversion to CSV with default delimiter
    csv_output = list_of_dicts_to_csv(data)
    assert "a,b,c,d" in csv_output
    assert "1,2,3," in csv_output
    assert "7,8,,9" in csv_output
    
    # Test conversion to CSV with custom delimiter
    custom_output = list_of_dicts_to_csv(data, delimiter=";")
    assert "a;b;c;d" in custom_output
    
    # Test with empty list
    empty_output = list_of_dicts_to_csv([])
    assert empty_output == ""
    
    # Test with dictionaries having different keys
    diverse_data = [
        {"x": 1, "y": 2},
        {"y": 3, "z": 4}
    ]
    diverse_output = list_of_dicts_to_csv(diverse_data)
    assert "x,y,z" in diverse_output
    assert "1,2," in diverse_output
    assert ",3,4" in diverse_output

def test_export_to_file():
    """Tests the export_to_file function"""
    # Create sample data dictionary
    data = {
        "command": "test",
        "status": "success",
        "values": [1, 2, 3]
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test exporting to JSON file
        json_path = temp_path / "output.json"
        result = export_to_file(data, json_path)
        assert result is True
        assert json_path.exists()
        
        # Test exporting to CSV file
        csv_path = temp_path / "output.csv"
        result = export_to_file(data, csv_path)
        assert result is True
        assert csv_path.exists()
        
        # Test exporting to text file
        txt_path = temp_path / "output.txt"
        result = export_to_file(data, txt_path)
        assert result is True
        assert txt_path.exists()
        
        # Test with non-existent directory (should create it)
        nested_path = temp_path / "nested" / "output.json"
        result = export_to_file(data, nested_path)
        assert result is True
        assert nested_path.exists()

def test_output_handler():
    """Tests the OutputHandler class"""
    # Create an OutputHandler instance with different output options
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        handler = OutputHandler(
            output_path=temp_path / "output.json",
            output_format="json",
            use_colors=False
        )
        
        # Mock print to avoid cluttering test output
        import builtins
        original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None
        
        try:
            # Test handle_result method with sample data
            sample_data = {"command": "test", "status": "success"}
            result = handler.handle_result(sample_data)
            assert result is True
            assert (temp_path / "output.json").exists()
            
            # Test handle_forecast method with sample forecast data
            forecast_data = {
                "title": "Forecast",
                "forecast": [
                    {"timestamp": "2023-07-15", "probability": 0.65}
                ]
            }
            forecast_handler = OutputHandler(
                output_path=temp_path / "forecast.json",
                output_format="json"
            )
            result = forecast_handler.handle_forecast(forecast_data)
            assert result is True
            assert (temp_path / "forecast.json").exists()
            
            # Test handle_metrics method with sample metrics data
            metrics_data = {
                "title": "Metrics",
                "metrics": {"accuracy": 0.85}
            }
            metrics_handler = OutputHandler(
                output_path=temp_path / "metrics.json",
                output_format="json"
            )
            result = metrics_handler.handle_metrics(metrics_data)
            assert result is True
            assert (temp_path / "metrics.json").exists()
            
            # Test export_dataframe method with sample DataFrame
            df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
            df_handler = OutputHandler(
                output_path=temp_path / "data.csv",
                output_format="csv"
            )
            result = df_handler.export_dataframe(df)
            assert result is True
            assert (temp_path / "data.csv").exists()
            
        finally:
            # Restore original print function
            builtins.print = original_print