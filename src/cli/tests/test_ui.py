import unittest
from unittest.mock import patch, MagicMock, ANY
import io
import os
import sys
import pandas as pd
import numpy as np

# Import color functions
from src.cli.ui.colors import (
    colorize, bold, italic, underline, color_by_status, color_by_level,
    color_by_value, color_by_probability, supports_color, strip_color,
    get_color_safe_length
)

# Import formatter functions
from src.cli.ui.formatters import (
    format_header, format_subheader, format_title, format_paragraph,
    format_bullet_list, format_numbered_list, format_key_value, format_key_value_list,
    format_section, format_subsection, format_box, get_terminal_size,
    wrap_text, truncate_text, center_text, align_text
)

# Import table functions
from src.cli.ui.tables import (
    create_table, create_simple_table, create_markdown_table, create_dataframe_table,
    create_key_value_table, create_metrics_table, create_comparison_table,
    create_forecast_table, create_feature_importance_table, create_confusion_matrix_table
)

# Import chart functions
from src.cli.ui.charts import (
    create_bar_chart, create_horizontal_bar_chart, create_vertical_bar_chart,
    create_line_chart, create_multi_line_chart, create_histogram, create_sparkline,
    create_probability_sparkline, create_heatmap, create_confusion_matrix,
    create_roc_curve, create_calibration_curve, create_feature_importance_chart,
    create_metrics_chart, create_probability_timeline, create_threshold_comparison,
    create_from_dataframe, Chart, BarChart, LineChart, MultiLineChart
)

# Import spinner components
from src.cli.ui.spinners import (
    Spinner, create_spinner, spinner_context, with_spinner, get_spinner_frames
)

# Import sample config for testing
from src.cli.tests.fixtures.sample_configs import SAMPLE_CLI_CONFIG
from src.cli.tests.fixtures.mock_responses import get_mock_response


class TestColors(unittest.TestCase):
    """Test case for color formatting functions in colors.py"""
    
    def setUp(self):
        """Set up test environment before each test"""
        pass
        
    def tearDown(self):
        """Clean up test environment after each test"""
        pass
        
    def test_colorize(self):
        """Test the colorize function with various parameters"""
        # Test with color
        colored_text = colorize("test", color="red")
        self.assertIn("\033[31m", colored_text)
        self.assertIn("test", colored_text)
        self.assertIn("\033[0m", colored_text)
        
        # Test with background
        bg_colored_text = colorize("test", background="blue")
        self.assertIn("\033[44m", bg_colored_text)
        
        # Test with style
        styled_text = colorize("test", style="bold")
        self.assertIn("\033[1m", styled_text)
        
        # Test with all parameters
        full_text = colorize("test", color="green", background="red", style="underline")
        self.assertIn("\033[32;41;4m", full_text)
        
        # Test with use_colors=False
        no_color_text = colorize("test", color="red", use_colors=False)
        self.assertEqual("test", no_color_text)
        
    def test_bold_italic_underline(self):
        """Test the bold, italic, and underline functions"""
        # Test bold
        bold_text = bold("test")
        self.assertIn("\033[1m", bold_text)
        
        # Test italic
        italic_text = italic("test")
        self.assertIn("\033[3m", italic_text)
        
        # Test underline
        underline_text = underline("test")
        self.assertIn("\033[4m", underline_text)
        
    def test_color_by_status(self):
        """Test the color_by_status function with different statuses"""
        # Test success status
        success_text = color_by_status("Success", "success")
        self.assertIn("\033[32m", success_text)  # green
        
        # Test warning status
        warning_text = color_by_status("Warning", "warning")
        self.assertIn("\033[33m", warning_text)  # yellow
        
        # Test error status
        error_text = color_by_status("Error", "error")
        self.assertIn("\033[31m", error_text)  # red
        
        # Test info status
        info_text = color_by_status("Info", "info")
        self.assertIn("\033[36m", info_text)  # cyan
        
        # Test invalid status
        invalid_text = color_by_status("Invalid", "invalid")
        self.assertEqual("Invalid", invalid_text)
        
    def test_color_by_level(self):
        """Test the color_by_level function with different log levels"""
        # Test DEBUG level
        debug_text = color_by_level("Debug", "DEBUG")
        self.assertIn("\033[90m", debug_text)  # bright_black
        
        # Test INFO level
        info_text = color_by_level("Info", "INFO")
        self.assertIn("\033[36m", info_text)  # cyan
        
        # Test WARNING level
        warning_text = color_by_level("Warning", "WARNING")
        self.assertIn("\033[33m", warning_text)  # yellow
        
        # Test ERROR level
        error_text = color_by_level("Error", "ERROR")
        self.assertIn("\033[31m", error_text)  # red
        
        # Test CRITICAL level
        critical_text = color_by_level("Critical", "CRITICAL")
        self.assertIn("\033[91m", critical_text)  # bright_red
        
        # Test invalid level
        invalid_text = color_by_level("Invalid", "INVALID")
        self.assertEqual("Invalid", invalid_text)
        
    def test_color_by_value(self):
        """Test the color_by_value function with different numeric values"""
        # Test low value
        low_value = color_by_value(25.0)
        self.assertIn("\033[32m", low_value)  # green
        
        # Test medium value
        med_value = color_by_value(75.0)
        self.assertIn("\033[33m", med_value)  # yellow
        
        # Test high value
        high_value = color_by_value(150.0)
        self.assertIn("\033[31m", high_value)  # red
        
        # Test with string value that can be converted to float
        str_value = color_by_value("25.0")
        self.assertIn("\033[32m", str_value)  # green
        
        # Test with custom thresholds
        custom_value = color_by_value(15.0, low_threshold=10.0, high_threshold=20.0)
        self.assertIn("\033[33m", custom_value)  # yellow
        
    def test_color_by_probability(self):
        """Test the color_by_probability function with different probability values"""
        # Test low probability
        low_prob = color_by_probability(0.1)
        self.assertIn("\033[32m", low_prob)  # green
        
        # Test medium probability
        med_prob = color_by_probability(0.5)
        self.assertIn("\033[33m", med_prob)  # yellow
        
        # Test high probability
        high_prob = color_by_probability(0.9)
        self.assertIn("\033[31m", high_prob)  # red
        
        # Test with string value that can be converted to float
        str_prob = color_by_probability("0.1")
        self.assertIn("\033[32m", str_prob)  # green
        
        # Test with custom thresholds
        custom_prob = color_by_probability(0.3, low_threshold=0.2, high_threshold=0.4)
        self.assertIn("\033[33m", custom_prob)  # yellow
        
    @patch('src.cli.ui.colors.os.environ')
    @patch('src.cli.ui.colors.sys.stdout')
    def test_supports_color(self, mock_stdout, mock_environ):
        """Test the supports_color function under different conditions"""
        # Test with NO_COLOR set
        mock_environ.get.return_value = "1"
        self.assertFalse(supports_color())
        
        # Test with FORCE_COLOR set
        mock_environ.get.side_effect = lambda x: "1" if x == "FORCE_COLOR" else None
        self.assertTrue(supports_color())
        
        # Test with TTY stdout
        mock_environ.get.return_value = None
        mock_stdout.isatty.return_value = True
        self.assertTrue(supports_color())
        
        # Test with non-TTY stdout
        mock_stdout.isatty.return_value = False
        self.assertFalse(supports_color())
        
    def test_strip_color(self):
        """Test the strip_color function with colored text"""
        colored_text = "\033[31mRed Text\033[0m"
        stripped_text = strip_color(colored_text)
        self.assertEqual("Red Text", stripped_text)
        
    def test_get_color_safe_length(self):
        """Test the get_color_safe_length function with colored text"""
        colored_text = "\033[31mRed Text\033[0m"
        length = get_color_safe_length(colored_text)
        self.assertEqual(8, length)  # "Red Text" has 8 characters


class TestFormatters(unittest.TestCase):
    """Test case for text formatting functions in formatters.py"""
    
    def setUp(self):
        """Set up test environment before each test"""
        pass
        
    def tearDown(self):
        """Clean up test environment after each test"""
        pass
        
    def test_format_header(self):
        """Test the format_header function"""
        # Test with default parameters
        header = format_header("Test Header")
        self.assertIn("Test Header", header)
        self.assertIn("===========", header)  # Underline of same length
        
        # Test with custom character
        custom_header = format_header("Test Header", char="-")
        self.assertIn("Test Header", custom_header)
        self.assertIn("-----------", custom_header)  # Underline with custom char
        
        # Test with colors
        colored_header = format_header("Test Header", use_colors=True)
        self.assertIn("\033[1m", colored_header)  # Bold formatting
        
    def test_format_subheader(self):
        """Test the format_subheader function"""
        # Test with default parameters
        subheader = format_subheader("Test Subheader")
        self.assertIn("Test Subheader", subheader)
        self.assertIn("--------------", subheader)  # Underline of same length
        
        # Test with custom character
        custom_subheader = format_subheader("Test Subheader", char="~")
        self.assertIn("Test Subheader", custom_subheader)
        self.assertIn("~~~~~~~~~~~~~~", custom_subheader)  # Underline with custom char
        
        # Test with colors
        colored_subheader = format_subheader("Test Subheader", use_colors=True)
        self.assertIn("\033[1m", colored_subheader)  # Bold formatting
        
    @patch('src.cli.ui.formatters.get_terminal_size')
    def test_format_title(self, mock_get_terminal_size):
        """Test the format_title function"""
        mock_get_terminal_size.return_value = (80, 24)
        
        # Test with default parameters
        title = format_title("Test Title")
        self.assertIn("Test Title", title)
        
        # Test with custom width
        custom_width_title = format_title("Test Title", width=40)
        self.assertIn("Test Title", custom_width_title)
        
        # Test with colors
        colored_title = format_title("Test Title", use_colors=True)
        self.assertIn("\033[1m", colored_title)  # Bold formatting
        
    @patch('src.cli.ui.formatters.get_terminal_size')
    def test_format_paragraph(self, mock_get_terminal_size):
        """Test the format_paragraph function"""
        mock_get_terminal_size.return_value = (80, 24)
        
        long_text = "This is a long paragraph that should be wrapped to fit within the specified width. " * 3
        
        # Test with default parameters
        paragraph = format_paragraph(long_text)
        self.assertIn("This is a long paragraph", paragraph)
        
        # Test with custom width
        custom_width_paragraph = format_paragraph(long_text, width=40)
        lines = custom_width_paragraph.split("\n")
        for line in lines:
            self.assertLessEqual(len(line), 40)
            
        # Test with custom indent
        indented_paragraph = format_paragraph(long_text, indent=4)
        lines = indented_paragraph.split("\n")
        for line in lines:
            self.assertTrue(line.startswith("    "))
            
    def test_format_bullet_list(self):
        """Test the format_bullet_list function"""
        items = ["First item", "Second item", "Third item"]
        
        # Test with default parameters
        bullet_list = format_bullet_list(items)
        for item in items:
            self.assertIn(item, bullet_list)
        self.assertIn("• ", bullet_list)  # Default bullet character
        
        # Test with custom bullet character
        custom_bullet_list = format_bullet_list(items, bullet="*")
        self.assertIn("* ", custom_bullet_list)
        
        # Test with custom indent
        indented_bullet_list = format_bullet_list(items, indent=4)
        lines = indented_bullet_list.split("\n")
        for line in lines:
            self.assertTrue(line.startswith("    "))
            
        # Test with custom width
        width_bullet_list = format_bullet_list(items, width=40)
        lines = width_bullet_list.split("\n")
        for line in lines:
            self.assertLessEqual(len(line), 40)
            
    def test_format_numbered_list(self):
        """Test the format_numbered_list function"""
        items = ["First item", "Second item", "Third item"]
        
        # Test with default parameters
        numbered_list = format_numbered_list(items)
        for i, item in enumerate(items, 1):
            self.assertIn(f"{i}. {item}", numbered_list)
            
        # Test with custom start number
        custom_start_list = format_numbered_list(items, start=5)
        for i, item in enumerate(items, 5):
            self.assertIn(f"{i}. {item}", custom_start_list)
            
        # Test with custom indent
        indented_list = format_numbered_list(items, indent=4)
        lines = indented_list.split("\n")
        for line in lines:
            self.assertTrue(line.startswith("    "))
            
        # Test with custom width
        width_list = format_numbered_list(items, width=40)
        lines = width_list.split("\n")
        for line in lines:
            self.assertLessEqual(len(line), 40)
            
    def test_format_key_value(self):
        """Test the format_key_value function"""
        # Test with string value
        kv_pair = format_key_value("Key", "Value")
        self.assertIn("Key", kv_pair)
        self.assertIn("Value", kv_pair)
        
        # Test with numeric value
        kv_num_pair = format_key_value("Count", 42)
        self.assertIn("Count", kv_num_pair)
        self.assertIn("42", kv_num_pair)
        
        # Test with custom key_width
        custom_width_pair = format_key_value("Key", "Value", key_width=10)
        self.assertIn("Key", custom_width_pair)
        
        # Test with use_colors
        colored_pair = format_key_value("Key", "Value", use_colors=True)
        self.assertIn("\033[1m", colored_pair)  # Bold formatting for key
        
    def test_format_key_value_list(self):
        """Test the format_key_value_list function"""
        data = {"Key1": "Value1", "Key2": 42, "LongerKey": "LongerValue"}
        
        # Test with default parameters
        kv_list = format_key_value_list(data)
        for key, value in data.items():
            self.assertIn(key, kv_list)
            self.assertIn(str(value), kv_list)
            
        # Test with custom key_width
        custom_width_list = format_key_value_list(data, key_width=15)
        self.assertIn("Key1", custom_width_list)
        
        # Test with use_colors
        colored_list = format_key_value_list(data, use_colors=True)
        self.assertIn("\033[1m", colored_list)  # Bold formatting for keys
        
    def test_format_section(self):
        """Test the format_section function"""
        # Test with default parameters
        section = format_section("Section Title", "Section content goes here")
        self.assertIn("Section Title", section)
        self.assertIn("=============", section)  # Underline
        self.assertIn("Section content goes here", section)
        
        # Test with use_colors
        colored_section = format_section("Section Title", "Section content", use_colors=True)
        self.assertIn("\033[1m", colored_section)  # Bold formatting for title
        
    def test_format_subsection(self):
        """Test the format_subsection function"""
        # Test with default parameters
        subsection = format_subsection("Subsection Title", "Subsection content goes here")
        self.assertIn("Subsection Title", subsection)
        self.assertIn("----------------", subsection)  # Underline
        self.assertIn("Subsection content goes here", subsection)
        
        # Test with use_colors
        colored_subsection = format_subsection("Subsection Title", "Subsection content", use_colors=True)
        self.assertIn("\033[1m", colored_subsection)  # Bold formatting for title
        
    def test_format_box(self):
        """Test the format_box function"""
        # Test with default parameters
        box = format_box("Text in a box")
        self.assertIn("Text in a box", box)
        
        # Test with custom width
        custom_width_box = format_box("Text in a box", width=30)
        lines = custom_width_box.split("\n")
        for line in lines:
            self.assertLessEqual(len(line), 30)
            
        # Test with custom border character
        custom_border_box = format_box("Text in a box", border_char="#")
        self.assertIn("####", custom_border_box)
        
        # Test with use_colors
        colored_box = format_box("Text in a box", use_colors=True)
        self.assertIn("\033[36m", colored_box)  # Cyan formatting for border
        
    @patch('src.cli.ui.formatters.shutil.get_terminal_size')
    def test_get_terminal_size(self, mock_shutil_get_terminal_size):
        """Test the get_terminal_size function"""
        # Test normal operation
        mock_shutil_get_terminal_size.return_value = (100, 30)
        width, height = get_terminal_size()
        self.assertEqual(width, 100)
        self.assertEqual(height, 30)
        
        # Test with exception (fallback to defaults)
        mock_shutil_get_terminal_size.side_effect = OSError("Mocked error")
        width, height = get_terminal_size()
        self.assertEqual(width, 80)  # Default width
        self.assertEqual(height, 24)  # Default height
        
    def test_wrap_text(self):
        """Test the wrap_text function"""
        long_text = "This is a long text that should be wrapped to fit within the specified width. " * 2
        
        # Test with default parameters
        wrapped_text = wrap_text(long_text)
        self.assertIn("This is a long text", wrapped_text)
        
        # Test with custom width
        custom_width_text = wrap_text(long_text, width=30)
        lines = custom_width_text.split("\n")
        for line in lines:
            self.assertLessEqual(len(line), 30)
            
        # Test with custom indent
        indented_text = wrap_text(long_text, indent=4)
        lines = indented_text.split("\n")
        self.assertTrue(lines[0].startswith("    "))
        
        # Test with subsequent_indent
        subsequent_indent_text = wrap_text(long_text, indent=4, subsequent_indent=True)
        lines = subsequent_indent_text.split("\n")
        for line in lines:
            self.assertTrue(line.startswith("    "))
            
    def test_truncate_text(self):
        """Test the truncate_text function"""
        long_text = "This is a very long text that needs to be truncated"
        
        # Test with text shorter than max_length
        short_text = "Short"
        truncated_short = truncate_text(short_text, 10)
        self.assertEqual(short_text, truncated_short)
        
        # Test with text longer than max_length
        truncated_long = truncate_text(long_text, 20)
        self.assertEqual(len(truncated_long), 20)
        self.assertTrue(truncated_long.endswith("..."))
        
        # Test with custom suffix
        custom_suffix = truncate_text(long_text, 20, suffix="[...]")
        self.assertTrue(custom_suffix.endswith("[...]"))
        
    def test_center_text(self):
        """Test the center_text function"""
        text = "Centered Text"
        
        # Test with default parameters
        centered = center_text(text, 30)
        # The text should be in the middle of the string
        self.assertTrue(centered.startswith(" ") and centered.endswith(" "))
        self.assertEqual(len(centered), 30)
        
        # Test with width less than text length
        narrow_centered = center_text(text, 5)
        self.assertEqual(narrow_centered, text)
        
        # Test with colored text
        colored_text = f"\033[31m{text}\033[0m"
        colored_centered = center_text(colored_text, 30)
        self.assertEqual(len(strip_color(colored_centered)), 30)
        
    def test_align_text(self):
        """Test the align_text function"""
        text = "Aligned Text"
        width = 30
        
        # Test left alignment
        left_aligned = align_text(text, width, "left")
        self.assertTrue(left_aligned.startswith(text))
        self.assertEqual(len(left_aligned), width)
        
        # Test right alignment
        right_aligned = align_text(text, width, "right")
        self.assertTrue(right_aligned.endswith(text))
        self.assertEqual(len(right_aligned), width)
        
        # Test center alignment
        center_aligned = align_text(text, width, "center")
        self.assertEqual(len(center_aligned), width)
        # The text should not be at the start or end
        self.assertTrue(center_aligned.startswith(" ") and center_aligned.endswith(" "))
        
        # Test with colored text
        colored_text = f"\033[31m{text}\033[0m"
        colored_aligned = align_text(colored_text, width, "right")
        self.assertEqual(len(strip_color(colored_aligned)), width)


class TestTables(unittest.TestCase):
    """Test case for table creation functions in tables.py"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Sample data for testing tables
        self.headers = ["Name", "Age", "Location"]
        self.data = [
            ["Alice", 30, "New York"],
            ["Bob", 25, "San Francisco"],
            ["Charlie", 35, "Chicago"]
        ]
        
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [30, 25, 35],
            "Location": ["New York", "San Francisco", "Chicago"],
            "Probability": [0.75, 0.45, 0.9]
        })
        
    def tearDown(self):
        """Clean up test environment after each test"""
        pass
        
    def test_create_table(self):
        """Test the create_table function"""
        # Test with default parameters
        table = create_table(self.data, self.headers)
        for header in self.headers:
            self.assertIn(header, table)
        for row in self.data:
            for item in row:
                self.assertIn(str(item), table)
                
        # Test with custom tablefmt
        simple_table = create_table(self.data, self.headers, tablefmt="simple")
        self.assertIn("Name", simple_table)
        
        # Test with use_colors
        colored_table = create_table(self.data, self.headers, use_colors=True)
        self.assertIn("\033[1m", colored_table)  # Bold headers
        
        # Test with max_width
        narrow_table = create_table(self.data, self.headers, max_width=30)
        self.assertIn("Name", narrow_table)
        
    def test_create_simple_table(self):
        """Test the create_simple_table function"""
        table = create_simple_table(self.data, self.headers)
        for header in self.headers:
            self.assertIn(header, table)
        for row in self.data:
            for item in row:
                self.assertIn(str(item), table)
                
        # Test with use_colors
        colored_table = create_simple_table(self.data, self.headers, use_colors=True)
        self.assertIn("\033[1m", colored_table)  # Bold headers
        
    def test_create_markdown_table(self):
        """Test the create_markdown_table function"""
        table = create_markdown_table(self.data, self.headers)
        for header in self.headers:
            self.assertIn(header, table)
        for row in self.data:
            for item in row:
                self.assertIn(str(item), table)
        self.assertIn("|", table)  # Markdown table uses pipe character
        
    def test_create_dataframe_table(self):
        """Test the create_dataframe_table function"""
        # Test with default parameters
        table = create_dataframe_table(self.df)
        for col in self.df.columns:
            self.assertIn(col, table)
            
        # Test with custom columns
        columns_table = create_dataframe_table(self.df, columns=["Name", "Age"])
        self.assertIn("Name", columns_table)
        self.assertIn("Age", columns_table)
        self.assertNotIn("Location", columns_table)
        
        # Test with max_rows and max_cols
        limited_table = create_dataframe_table(self.df, max_rows=2, max_cols=2)
        self.assertIn("...", limited_table)  # Truncation indicator
        
        # Test with custom tablefmt
        simple_table = create_dataframe_table(self.df, tablefmt="simple")
        self.assertIn("Name", simple_table)
        
        # Test with use_colors
        colored_table = create_dataframe_table(self.df, use_colors=True)
        # Check for some color codes in the output
        self.assertTrue("\033[" in colored_table)
        
        # Test with column_formats
        formats_table = create_dataframe_table(
            self.df, 
            column_formats={"Probability": "probability"}
        )
        self.assertIn("%", formats_table)  # Probability formatted as percentage
        
    def test_create_key_value_table(self):
        """Test the create_key_value_table function"""
        data = {"Key1": "Value1", "Key2": 42, "Key3": "Value3"}
        
        # Test with default parameters
        table = create_key_value_table(data)
        for key, value in data.items():
            self.assertIn(key, table)
            self.assertIn(str(value), table)
            
        # Test with custom key_header and value_header
        custom_headers_table = create_key_value_table(
            data, 
            key_header="Parameter", 
            value_header="Setting"
        )
        self.assertIn("Parameter", custom_headers_table)
        self.assertIn("Setting", custom_headers_table)
        
        # Test with custom tablefmt
        simple_table = create_key_value_table(data, tablefmt="simple")
        self.assertIn("Key1", simple_table)
        
        # Test with use_colors
        colored_table = create_key_value_table(data, use_colors=True)
        self.assertIn("\033[1m", colored_table)  # Bold headers
        
    def test_create_metrics_table(self):
        """Test the create_metrics_table function"""
        metrics = {
            "accuracy": 0.85,
            "precision": 0.78,
            "recall": 0.81,
            "f1": 0.79,
            "brier_score": 0.12,
            "training_time": 123.45
        }
        
        # Test with default parameters
        table = create_metrics_table(metrics)
        for metric, value in metrics.items():
            self.assertIn(metric, table)
            
        # Test with custom title
        titled_table = create_metrics_table(metrics, title="Model Performance")
        self.assertIn("Model Performance", titled_table)
        
        # Test with custom tablefmt
        simple_table = create_metrics_table(metrics, tablefmt="simple")
        self.assertIn("accuracy", simple_table)
        
        # Test with use_colors
        colored_table = create_metrics_table(metrics, use_colors=True)
        self.assertIn("\033[", colored_table)  # Color codes
        
    def test_create_comparison_table(self):
        """Test the create_comparison_table function"""
        comparison_data = {
            "Model A": {
                "accuracy": 0.85,
                "precision": 0.78,
                "recall": 0.81
            },
            "Model B": {
                "accuracy": 0.82,
                "precision": 0.76,
                "recall": 0.83
            }
        }
        
        # Test with default parameters
        table = create_comparison_table(comparison_data)
        for model in comparison_data.keys():
            self.assertIn(model, table)
        for metric in comparison_data["Model A"].keys():
            self.assertIn(metric, table)
            
        # Test with metrics_to_include
        filtered_table = create_comparison_table(
            comparison_data,
            metrics_to_include=["accuracy", "precision"]
        )
        self.assertIn("accuracy", filtered_table)
        self.assertIn("precision", filtered_table)
        self.assertNotIn("recall", filtered_table)
        
        # Test with custom title
        titled_table = create_comparison_table(comparison_data, title="Model Comparison")
        self.assertIn("Model Comparison", titled_table)
        
        # Test with custom tablefmt
        simple_table = create_comparison_table(comparison_data, tablefmt="simple")
        self.assertIn("Model A", simple_table)
        
        # Test with use_colors
        colored_table = create_comparison_table(comparison_data, use_colors=True)
        self.assertIn("\033[", colored_table)  # Color codes
        
    def test_create_forecast_table(self):
        """Test the create_forecast_table function"""
        forecasts = [
            {
                "target_timestamp": "2023-07-15 14:00",
                "threshold_value": 100.0,
                "spike_probability": 0.85,
                "confidence_interval_lower": 0.75,
                "confidence_interval_upper": 0.95
            },
            {
                "target_timestamp": "2023-07-15 15:00",
                "threshold_value": 100.0,
                "spike_probability": 0.65,
                "confidence_interval_lower": 0.55,
                "confidence_interval_upper": 0.75
            }
        ]
        
        # Test with default parameters
        table = create_forecast_table(forecasts)
        for field in ["target_timestamp", "threshold_value", "spike_probability"]:
            self.assertIn(field, table)
            
        # Test with custom columns
        columns_table = create_forecast_table(
            forecasts,
            columns=["target_timestamp", "spike_probability"]
        )
        self.assertIn("target_timestamp", columns_table)
        self.assertIn("spike_probability", columns_table)
        self.assertNotIn("threshold_value", columns_table)
        
        # Test with custom title
        titled_table = create_forecast_table(forecasts, title="Forecast Results")
        self.assertIn("Forecast Results", titled_table)
        
        # Test with custom tablefmt
        simple_table = create_forecast_table(forecasts, tablefmt="simple")
        self.assertIn("target_timestamp", simple_table)
        
        # Test with use_colors
        colored_table = create_forecast_table(forecasts, use_colors=True)
        self.assertIn("\033[", colored_table)  # Color codes
        
    def test_create_feature_importance_table(self):
        """Test the create_feature_importance_table function"""
        feature_importance = {
            "feature1": 0.35,
            "feature2": 0.25,
            "feature3": 0.20,
            "feature4": 0.15,
            "feature5": 0.05
        }
        
        # Test with default parameters
        table = create_feature_importance_table(feature_importance)
        for feature in feature_importance.keys():
            self.assertIn(feature, table)
            
        # Test with max_features
        limited_table = create_feature_importance_table(feature_importance, max_features=3)
        self.assertIn("feature1", limited_table)
        self.assertIn("feature2", limited_table)
        self.assertIn("feature3", limited_table)
        self.assertNotIn("feature5", limited_table)
        
        # Test with custom title
        titled_table = create_feature_importance_table(feature_importance, title="Feature Importance")
        self.assertIn("Feature Importance", titled_table)
        
        # Test with custom tablefmt
        simple_table = create_feature_importance_table(feature_importance, tablefmt="simple")
        self.assertIn("feature1", simple_table)
        
        # Test with use_colors
        colored_table = create_feature_importance_table(feature_importance, use_colors=True)
        self.assertIn("\033[", colored_table)  # Color codes
        
    def test_create_confusion_matrix_table(self):
        """Test the create_confusion_matrix_table function"""
        matrix = [
            [45, 5],
            [8, 42]
        ]
        
        # Test with default parameters
        table = create_confusion_matrix_table(matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                self.assertIn(str(matrix[i][j]), table)
                
        # Test with custom class_labels
        labeled_table = create_confusion_matrix_table(
            matrix,
            class_labels=["Negative", "Positive"]
        )
        self.assertIn("Negative", labeled_table)
        self.assertIn("Positive", labeled_table)
        
        # Test with custom title
        titled_table = create_confusion_matrix_table(matrix, title="Confusion Matrix")
        self.assertIn("Confusion Matrix", titled_table)
        
        # Test with custom tablefmt
        simple_table = create_confusion_matrix_table(matrix, tablefmt="simple")
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                self.assertIn(str(matrix[i][j]), simple_table)
                
        # Test with use_colors
        colored_table = create_confusion_matrix_table(matrix, use_colors=True)
        self.assertIn("\033[", colored_table)  # Color codes


class TestCharts(unittest.TestCase):
    """Test case for chart creation functions in charts.py"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Sample data for testing charts
        self.numeric_data = [10, 25, 15, 30, 20, 35]
        self.timestamps = ["2023-07-01", "2023-07-02", "2023-07-03", "2023-07-04", "2023-07-05", "2023-07-06"]
        self.categorical_data = {"Category A": 35, "Category B": 25, "Category C": 15, "Category D": 25}
        
    def tearDown(self):
        """Clean up test environment after each test"""
        pass
        
    def test_create_bar_chart(self):
        """Test the create_bar_chart function"""
        # Test with default parameters
        chart = create_bar_chart(self.categorical_data)
        for category in self.categorical_data.keys():
            self.assertIn(category, chart)
            
        # Test with custom width
        narrow_chart = create_bar_chart(self.categorical_data, width=40)
        self.assertTrue(len(max(narrow_chart.split('\n'), key=len)) <= 40)
        
        # Test with custom max_label_width
        label_chart = create_bar_chart(self.categorical_data, max_label_width=15)
        self.assertIn("Category A", label_chart)
        
        # Test with use_colors
        colored_chart = create_bar_chart(self.categorical_data, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
        # Test with custom title
        titled_chart = create_bar_chart(self.categorical_data, title="Bar Chart")
        self.assertIn("Bar Chart", titled_chart)
        
    def test_create_horizontal_bar_chart(self):
        """Test the create_horizontal_bar_chart function"""
        # This is essentially the same as create_bar_chart
        chart = create_horizontal_bar_chart(self.categorical_data)
        for category in self.categorical_data.keys():
            self.assertIn(category, chart)
            
        # Test with custom width
        narrow_chart = create_horizontal_bar_chart(self.categorical_data, width=40)
        self.assertTrue(len(max(narrow_chart.split('\n'), key=len)) <= 40)
        
        # Test with use_colors
        colored_chart = create_horizontal_bar_chart(self.categorical_data, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
        # Test with custom title
        titled_chart = create_horizontal_bar_chart(self.categorical_data, title="Bar Chart")
        self.assertIn("Bar Chart", titled_chart)
        
        # Test with custom color_scheme
        scheme_chart = create_horizontal_bar_chart(
            self.categorical_data,
            color_scheme="probability"
        )
        self.assertIn("Category A", scheme_chart)
        
    def test_create_vertical_bar_chart(self):
        """Test the create_vertical_bar_chart function"""
        chart = create_vertical_bar_chart(self.categorical_data)
        for category in self.categorical_data.keys():
            self.assertIn(category, chart)
            
        # Test with custom width
        narrow_chart = create_vertical_bar_chart(self.categorical_data, width=40)
        self.assertTrue(len(max(narrow_chart.split('\n'), key=len)) <= 40)
        
        # Test with custom height
        short_chart = create_vertical_bar_chart(self.categorical_data, height=10)
        self.assertLessEqual(len(short_chart.split('\n')), 10)
        
        # Test with use_colors
        colored_chart = create_vertical_bar_chart(self.categorical_data, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
        # Test with custom title
        titled_chart = create_vertical_bar_chart(self.categorical_data, title="Bar Chart")
        self.assertIn("Bar Chart", titled_chart)
        
    def test_create_line_chart(self):
        """Test the create_line_chart function"""
        # Test with default parameters
        chart = create_line_chart(self.numeric_data)
        
        # Test with labels
        labeled_chart = create_line_chart(self.numeric_data, labels=self.timestamps)
        for timestamp in self.timestamps:
            # Check for timestamp in chart, but be aware that they might be truncated
            # or not all included depending on chart width
            # Just verify the first date appears
            self.assertIn("2023-07-01", labeled_chart)
            
        # Test with custom width and height
        custom_size_chart = create_line_chart(self.numeric_data, width=40, height=10)
        self.assertTrue(len(max(custom_size_chart.split('\n'), key=len)) <= 40)
        self.assertLessEqual(len(custom_size_chart.split('\n')), 10)
        
        # Test with use_colors
        colored_chart = create_line_chart(self.numeric_data, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
        # Test with custom title
        titled_chart = create_line_chart(self.numeric_data, title="Line Chart")
        self.assertIn("Line Chart", titled_chart)
        
        # Test with custom min_value and max_value
        scaled_chart = create_line_chart(
            self.numeric_data,
            min_value=0,
            max_value=50
        )
        # No specific assertion as we're just testing that it executes without error
        
    def test_create_multi_line_chart(self):
        """Test the create_multi_line_chart function"""
        # Create sample multi-series data
        multi_data = {
            "Series A": [10, 20, 15, 25, 30, 20],
            "Series B": [15, 10, 25, 20, 15, 30]
        }
        
        # Test with default parameters
        chart = create_multi_line_chart(multi_data)
        for series in multi_data.keys():
            self.assertIn(series, chart)
            
        # Test with labels
        labeled_chart = create_multi_line_chart(multi_data, labels=self.timestamps)
        self.assertIn("2023-07-01", labeled_chart)
        
        # Test with custom width and height
        custom_size_chart = create_multi_line_chart(
            multi_data,
            width=40,
            height=10
        )
        self.assertTrue(len(max(custom_size_chart.split('\n'), key=len)) <= 40)
        self.assertLessEqual(len(custom_size_chart.split('\n')), 15)  # Account for legend
        
        # Test with use_colors
        colored_chart = create_multi_line_chart(multi_data, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
        # Test with custom title
        titled_chart = create_multi_line_chart(multi_data, title="Multi-Line Chart")
        self.assertIn("Multi-Line Chart", titled_chart)
        
        # Test with custom colors
        custom_colors = {"Series A": "blue", "Series B": "red"}
        colored_series_chart = create_multi_line_chart(
            multi_data,
            colors=custom_colors,
            use_colors=True
        )
        self.assertIn("Series A", colored_series_chart)
        
    def test_create_histogram(self):
        """Test the create_histogram function"""
        data = np.random.normal(0, 1, 100)
        
        # Test with default parameters
        chart = create_histogram(data.tolist())
        self.assertIn("Histogram", chart)
        
        # Test with custom bins
        binned_chart = create_histogram(data.tolist(), bins=5)
        
        # Test with custom width and height
        custom_size_chart = create_histogram(
            data.tolist(),
            width=40,
            height=10
        )
        self.assertTrue(len(max(custom_size_chart.split('\n'), key=len)) <= 40)
        self.assertLessEqual(len(custom_size_chart.split('\n')), 15)
        
        # Test with use_colors
        colored_chart = create_histogram(data.tolist(), use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
        # Test with custom title
        titled_chart = create_histogram(data.tolist(), title="Distribution")
        self.assertIn("Distribution", titled_chart)
        
    def test_create_sparkline(self):
        """Test the create_sparkline function"""
        # Test with default parameters
        sparkline = create_sparkline(self.numeric_data)
        # Should be a line of special characters
        self.assertTrue(len(sparkline) == len(self.numeric_data))
        
        # Test with custom width
        wide_sparkline = create_sparkline(self.numeric_data, width=10)
        self.assertEqual(len(wide_sparkline), 10)
        
        # Test with use_colors
        colored_sparkline = create_sparkline(self.numeric_data, use_colors=True)
        self.assertIn("\033[", colored_sparkline)  # Color codes
        
        # Test with custom min_value and max_value
        scaled_sparkline = create_sparkline(
            self.numeric_data,
            min_value=0,
            max_value=50
        )
        self.assertEqual(len(scaled_sparkline), len(self.numeric_data))
        
    def test_create_probability_sparkline(self):
        """Test the create_probability_sparkline function"""
        probabilities = [0.1, 0.5, 0.9, 0.3, 0.7, 0.4]
        
        # Test with default parameters
        sparkline = create_probability_sparkline(probabilities)
        # Should be a line of special characters
        self.assertEqual(len(sparkline), len(probabilities))
        
        # Test with custom width
        wide_sparkline = create_probability_sparkline(probabilities, width=10)
        self.assertEqual(len(strip_color(wide_sparkline)), 10)
        
        # Test with use_colors
        colored_sparkline = create_probability_sparkline(probabilities, use_colors=True)
        self.assertIn("\033[", colored_sparkline)  # Color codes
        
    def test_create_heatmap(self):
        """Test the create_heatmap function"""
        # Create sample 2D data
        data = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        # Test with default parameters
        heatmap = create_heatmap(data)
        for row in data:
            for value in row:
                # Check that the string representation of each value is in the heatmap
                self.assertIn(str(value)[0:3], heatmap)
                
        # Test with row and column labels
        row_labels = ["Row 1", "Row 2", "Row 3"]
        col_labels = ["Col A", "Col B", "Col C"]
        labeled_heatmap = create_heatmap(data, row_labels, col_labels)
        for label in row_labels + col_labels:
            self.assertIn(label, labeled_heatmap)
            
        # Test with custom width and cell_width
        custom_width_heatmap = create_heatmap(data, width=40, cell_width=5)
        self.assertTrue(len(max(custom_width_heatmap.split('\n'), key=len)) <= 40)
        
        # Test with use_colors
        colored_heatmap = create_heatmap(data, use_colors=True)
        self.assertIn("\033[", colored_heatmap)  # Color codes
        
        # Test with custom title
        titled_heatmap = create_heatmap(data, title="Heatmap")
        self.assertIn("Heatmap", titled_heatmap)
        
    def test_create_confusion_matrix(self):
        """Test the create_confusion_matrix function"""
        # Create sample confusion matrix data
        confusion_matrix = [
            [45, 5],
            [8, 42]
        ]
        
        # Test with default parameters
        matrix = create_confusion_matrix(confusion_matrix)
        for row in confusion_matrix:
            for value in row:
                self.assertIn(str(value), matrix)
                
        # Test with custom class_labels
        class_labels = ["Negative", "Positive"]
        labeled_matrix = create_confusion_matrix(
            confusion_matrix,
            class_labels=class_labels
        )
        for label in class_labels:
            self.assertIn(label, labeled_matrix)
            
        # Test with custom cell_width
        custom_width_matrix = create_confusion_matrix(
            confusion_matrix,
            cell_width=5
        )
        # No specific assertion as we're just testing that it executes without error
        
        # Test with use_colors
        colored_matrix = create_confusion_matrix(confusion_matrix, use_colors=True)
        self.assertIn("\033[", colored_matrix)  # Color codes
        
        # Test with custom title
        titled_matrix = create_confusion_matrix(confusion_matrix, title="Confusion Matrix")
        self.assertIn("Confusion Matrix", titled_matrix)
        
    def test_create_roc_curve(self):
        """Test the create_roc_curve function"""
        # Create sample FPR and TPR data
        fpr = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        tpr = [0, 0.4, 0.7, 0.8, 0.9, 0.95, 1.0]
        auc = 0.82
        
        # Test with default parameters
        curve = create_roc_curve(fpr, tpr, auc)
        self.assertIn(str(auc), curve)
        self.assertIn("ROC Curve", curve)
        
        # Test with custom width and height
        custom_size_curve = create_roc_curve(fpr, tpr, auc, width=40, height=10)
        self.assertTrue(len(max(custom_size_curve.split('\n'), key=len)) <= 40)
        self.assertLessEqual(len(custom_size_curve.split('\n')), 15)
        
        # Test with use_colors
        colored_curve = create_roc_curve(fpr, tpr, auc, use_colors=True)
        self.assertIn("\033[", colored_curve)  # Color codes
        
    def test_create_calibration_curve(self):
        """Test the create_calibration_curve function"""
        # Create sample predicted and true probability data
        pred_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        true_probs = [0.15, 0.35, 0.45, 0.65, 0.85]
        brier_score = 0.02
        
        # Test with default parameters
        curve = create_calibration_curve(pred_probs, true_probs)
        self.assertIn("Calibration Curve", curve)
        
        # Test with brier_score
        scored_curve = create_calibration_curve(pred_probs, true_probs, brier_score)
        self.assertIn(str(brier_score), scored_curve)
        
        # Test with custom width and height
        custom_size_curve = create_calibration_curve(
            pred_probs,
            true_probs,
            brier_score,
            width=40,
            height=10
        )
        self.assertTrue(len(max(custom_size_curve.split('\n'), key=len)) <= 40)
        self.assertLessEqual(len(custom_size_curve.split('\n')), 15)
        
        # Test with use_colors
        colored_curve = create_calibration_curve(
            pred_probs,
            true_probs,
            brier_score,
            use_colors=True
        )
        self.assertIn("\033[", colored_curve)  # Color codes
        
    def test_create_feature_importance_chart(self):
        """Test the create_feature_importance_chart function"""
        # Create sample feature importance data
        feature_importance = {
            "feature1": 0.35,
            "feature2": 0.25,
            "feature3": 0.20,
            "feature4": 0.15,
            "feature5": 0.05
        }
        
        # Test with default parameters
        chart = create_feature_importance_chart(feature_importance)
        for feature in feature_importance.keys():
            self.assertIn(feature, chart)
            
        # Test with max_features
        limited_chart = create_feature_importance_chart(feature_importance, max_features=3)
        self.assertIn("feature1", limited_chart)
        self.assertIn("feature2", limited_chart)
        self.assertIn("feature3", limited_chart)
        self.assertNotIn("feature5", limited_chart)
        
        # Test with custom width
        narrow_chart = create_feature_importance_chart(feature_importance, width=40)
        self.assertTrue(len(max(narrow_chart.split('\n'), key=len)) <= 40)
        
        # Test with use_colors
        colored_chart = create_feature_importance_chart(feature_importance, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
    def test_create_metrics_chart(self):
        """Test the create_metrics_chart function"""
        # Create sample metrics data
        metrics = {
            "accuracy": 0.85,
            "precision": 0.78,
            "recall": 0.81,
            "f1": 0.79,
            "brier_score": 0.12
        }
        
        # Test with default parameters
        chart = create_metrics_chart(metrics)
        for metric in metrics.keys():
            self.assertIn(metric, chart)
            
        # Test with custom width
        narrow_chart = create_metrics_chart(metrics, width=40)
        self.assertTrue(len(max(narrow_chart.split('\n'), key=len)) <= 40)
        
        # Test with use_colors
        colored_chart = create_metrics_chart(metrics, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
    def test_create_probability_timeline(self):
        """Test the create_probability_timeline function"""
        # Create sample probability data
        probabilities = [0.1, 0.5, 0.9, 0.3, 0.7, 0.4]
        
        # Test with default parameters
        chart = create_probability_timeline(probabilities)
        self.assertIn("Spike Probability Forecast", chart)
        
        # Test with timestamps
        timestamps = self.timestamps
        timeseries_chart = create_probability_timeline(probabilities, timestamps)
        self.assertIn("2023-07-01", timeseries_chart)
        
        # Test with custom width and height
        custom_size_chart = create_probability_timeline(
            probabilities,
            width=40,
            height=10
        )
        self.assertTrue(len(max(custom_size_chart.split('\n'), key=len)) <= 40)
        self.assertLessEqual(len(custom_size_chart.split('\n')), 15)
        
        # Test with use_colors
        colored_chart = create_probability_timeline(probabilities, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
        # Test with threshold_label
        threshold_chart = create_probability_timeline(
            probabilities,
            threshold_label="100 $/MWh"
        )
        self.assertIn("100 $/MWh", threshold_chart)
        
    def test_create_threshold_comparison(self):
        """Test the create_threshold_comparison function"""
        # Create sample threshold comparison data
        threshold_data = {
            "50 $/MWh": [0.5, 0.6, 0.7, 0.5, 0.6, 0.5],
            "100 $/MWh": [0.3, 0.4, 0.5, 0.3, 0.4, 0.3],
            "200 $/MWh": [0.1, 0.2, 0.3, 0.1, 0.2, 0.1]
        }
        
        # Test with default parameters
        chart = create_threshold_comparison(threshold_data)
        for threshold in threshold_data.keys():
            self.assertIn(threshold, chart)
            
        # Test with timestamps
        timestamps = self.timestamps
        timeseries_chart = create_threshold_comparison(threshold_data, timestamps)
        self.assertIn("2023-07-01", timeseries_chart)
        
        # Test with custom width and height
        custom_size_chart = create_threshold_comparison(
            threshold_data,
            width=40,
            height=10
        )
        self.assertTrue(len(max(custom_size_chart.split('\n'), key=len)) <= 40)
        self.assertLessEqual(len(custom_size_chart.split('\n')), 20)  # Account for legend
        
        # Test with use_colors
        colored_chart = create_threshold_comparison(threshold_data, use_colors=True)
        self.assertIn("\033[", colored_chart)  # Color codes
        
    def test_create_from_dataframe(self):
        """Test the create_from_dataframe function"""
        # Create sample DataFrame
        df = pd.DataFrame({
            "Date": pd.date_range(start="2023-07-01", periods=6),
            "Value": [10, 20, 15, 25, 30, 20],
            "Category": ["A", "B", "A", "B", "A", "B"]
        })
        
        # Test with bar chart type
        bar_chart = create_from_dataframe(df, "bar", x_column="Category", y_column="Value")
        self.assertIn("A", bar_chart)
        self.assertIn("B", bar_chart)
        
        # Test with line chart type
        line_chart = create_from_dataframe(df, "line", y_column="Value")
        # Check that it returns something (no specific content to check)
        self.assertTrue(len(line_chart) > 0)
        
        # Test with multi_line chart type
        multi_line_chart = create_from_dataframe(
            df,
            "multi_line",
            y_column="Value",
            group_column="Category"
        )
        self.assertIn("A", multi_line_chart)
        self.assertIn("B", multi_line_chart)
        
        # Test with histogram chart type
        histogram = create_from_dataframe(df, "histogram", y_column="Value")
        self.assertIn("Histogram", histogram)
        
        # Test with custom parameters
        custom_chart = create_from_dataframe(
            df,
            "line",
            y_column="Value",
            width=40,
            height=10,
            use_colors=True,
            title="Custom Chart"
        )
        self.assertIn("Custom Chart", custom_chart)
        self.assertIn("\033[", custom_chart)  # Color codes
        
    def test_chart_classes(self):
        """Test the Chart, BarChart, LineChart, and MultiLineChart classes"""
        # Test base Chart class
        chart = Chart(width=40, height=10, title="Base Chart")
        self.assertEqual(chart._width, 40)
        self.assertEqual(chart._height, 10)
        self.assertEqual(chart._title, "Base Chart")
        chart.add_title("Added Title")
        chart.add_line("Line 1")
        chart.add_line("Line 2")
        rendered = chart.render()
        self.assertIn("Added Title", rendered)
        self.assertIn("Line 1", rendered)
        self.assertIn("Line 2", rendered)
        
        # Test BarChart class
        bar_chart = BarChart(self.categorical_data, width=40, title="Bar Chart")
        self.assertEqual(bar_chart._width, 40)
        self.assertEqual(bar_chart._title, "Bar Chart")
        rendered_bar = bar_chart.render()
        for category in self.categorical_data.keys():
            self.assertIn(category, rendered_bar)
            
        # Test BarChart.horizontal factory method
        horiz_bar = BarChart.horizontal(
            self.categorical_data,
            width=40,
            title="Horizontal Bar"
        )
        rendered_horiz = horiz_bar.render()
        for category in self.categorical_data.keys():
            self.assertIn(category, rendered_horiz)
            
        # Test LineChart class
        line_chart = LineChart(
            self.numeric_data,
            labels=self.timestamps,
            width=40,
            height=10,
            title="Line Chart"
        )
        rendered_line = line_chart.render()
        self.assertIn("Line Chart", rendered_line)
        
        # Test LineChart.probability_timeline factory method
        prob_timeline = LineChart.probability_timeline(
            [0.1, 0.5, 0.9, 0.3, 0.7, 0.4],
            self.timestamps,
            width=40,
            height=10,
            threshold_label="100 $/MWh"
        )
        rendered_timeline = prob_timeline.render()
        self.assertIn("100 $/MWh", rendered_timeline)
        
        # Test MultiLineChart class
        multi_data = {
            "Series A": [10, 20, 15, 25, 30, 20],
            "Series B": [15, 10, 25, 20, 15, 30]
        }
        multi_chart = MultiLineChart(
            multi_data,
            labels=self.timestamps,
            width=40,
            height=10,
            title="Multi-Line Chart"
        )
        rendered_multi = multi_chart.render()
        self.assertIn("Multi-Line Chart", rendered_multi)
        self.assertIn("Series A", rendered_multi)
        self.assertIn("Series B", rendered_multi)
        
        # Test MultiLineChart.threshold_comparison factory method
        threshold_data = {
            "50 $/MWh": [0.5, 0.6, 0.7, 0.5, 0.6, 0.5],
            "100 $/MWh": [0.3, 0.4, 0.5, 0.3, 0.4, 0.3]
        }
        thresh_comp = MultiLineChart.threshold_comparison(
            threshold_data,
            self.timestamps,
            width=40,
            height=10
        )
        rendered_thresh = thresh_comp.render()
        self.assertIn("Threshold Comparison", rendered_thresh)
        self.assertIn("50 $/MWh", rendered_thresh)
        self.assertIn("100 $/MWh", rendered_thresh)


class TestSpinners(unittest.TestCase):
    """Test case for spinner components in spinners.py"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Mock stdout to capture spinner output
        self.stdout_patcher = patch('sys.stdout', new_callable=io.StringIO)
        self.mock_stdout = self.stdout_patcher.start()
        
    def tearDown(self):
        """Clean up test environment after each test"""
        self.stdout_patcher.stop()
        
    def test_spinner_initialization(self):
        """Test Spinner class initialization"""
        # Test with default parameters
        spinner = Spinner("Loading...")
        self.assertEqual(spinner._message, "Loading...")
        self.assertEqual(spinner._interval, 0.1)
        self.assertEqual(spinner._color, "cyan")
        self.assertFalse(spinner._running)
        
        # Test with custom frames
        custom_frames = [".", "..", "..."]
        custom_spinner = Spinner("Loading...", frames=custom_frames)
        self.assertEqual(custom_spinner._frames, custom_frames)
        
        # Test with custom color
        color_spinner = Spinner("Loading...", color="green")
        self.assertEqual(color_spinner._color, "green")
        
        # Test with custom interval
        interval_spinner = Spinner("Loading...", interval=0.2)
        self.assertEqual(interval_spinner._interval, 0.2)
        
        # Test with disable=True
        disabled_spinner = Spinner("Loading...", disable=True)
        self.assertTrue(disabled_spinner._disable)
        
    @patch('threading.Thread')
    def test_spinner_start_stop(self, mock_thread):
        """Test Spinner start and stop methods"""
        # Test with disable=False
        spinner = Spinner("Loading...", disable=False)
        spinner.start()
        mock_thread.assert_called_once()
        self.assertTrue(spinner._running)
        
        # Test stop method
        spinner.stop()
        self.assertFalse(spinner._running)
        
        # Reset mock and test with disable=True
        mock_thread.reset_mock()
        disabled_spinner = Spinner("Loading...", disable=True)
        disabled_spinner.start()
        mock_thread.assert_not_called()
        self.assertFalse(disabled_spinner._running)
        
    def test_spinner_update(self):
        """Test Spinner update method"""
        spinner = Spinner("Initial message")
        spinner.update("Updated message")
        self.assertEqual(spinner._message, "Updated message")
        
        # Test with disable=True
        disabled_spinner = Spinner("Initial message", disable=True)
        disabled_spinner.update("Updated message")
        self.assertEqual(disabled_spinner._message, "Updated message")
        
    @patch('src.cli.ui.spinners.Spinner.stop')
    def test_spinner_succeed_fail_warn(self, mock_stop):
        """Test Spinner succeed, fail, and warn methods"""
        # Test succeed method
        spinner = Spinner("Working...")
        spinner.succeed()
        mock_stop.assert_called_once()
        self.assertIn("✓", mock_stop.call_args[0][0])
        
        # Reset mock and test fail method
        mock_stop.reset_mock()
        spinner.fail()
        mock_stop.assert_called_once()
        self.assertIn("✗", mock_stop.call_args[0][0])
        
        # Reset mock and test warn method
        mock_stop.reset_mock()
        spinner.warn()
        mock_stop.assert_called_once()
        self.assertIn("⚠", mock_stop.call_args[0][0])
        
        # Test with custom messages
        mock_stop.reset_mock()
        spinner.succeed("Custom success")
        self.assertIn("Custom success", mock_stop.call_args[0][0])
        
        mock_stop.reset_mock()
        spinner.fail("Custom failure")
        self.assertIn("Custom failure", mock_stop.call_args[0][0])
        
        mock_stop.reset_mock()
        spinner.warn("Custom warning")
        self.assertIn("Custom warning", mock_stop.call_args[0][0])
        
    @patch('src.cli.ui.spinners.Spinner.start')
    @patch('src.cli.ui.spinners.Spinner.succeed')
    @patch('src.cli.ui.spinners.Spinner.fail')
    def test_spinner_context_manager(self, mock_fail, mock_succeed, mock_start):
        """Test Spinner as a context manager"""
        # Test normal exit
        with Spinner("Working...") as spinner:
            self.assertIsInstance(spinner, Spinner)
        
        mock_start.assert_called_once()
        mock_succeed.assert_called_once()
        mock_fail.assert_not_called()
        
        # Reset mocks and test exception case
        mock_start.reset_mock()
        mock_succeed.reset_mock()
        
        # Test with exception
        try:
            with Spinner("Working..."):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        mock_start.assert_called_once()
        mock_succeed.assert_not_called()
        mock_fail.assert_called_once()
        
    def test_create_spinner(self):
        """Test create_spinner function"""
        # Test with default parameters
        spinner = create_spinner("Loading...")
        self.assertEqual(spinner._message, "Loading...")
        self.assertEqual(spinner._color, "cyan")
        self.assertEqual(spinner._interval, 0.1)
        
        # Test with custom spinner_type
        dots_spinner = create_spinner("Loading...", spinner_type="dots")
        self.assertEqual(dots_spinner._frames, get_spinner_frames("dots"))
        
        # Test with custom color
        green_spinner = create_spinner("Loading...", color="green")
        self.assertEqual(green_spinner._color, "green")
        
        # Test with custom interval
        slow_spinner = create_spinner("Loading...", interval=0.5)
        self.assertEqual(slow_spinner._interval, 0.5)
        
        # Test with disable=True
        disabled_spinner = create_spinner("Loading...", disable=True)
        self.assertTrue(disabled_spinner._disable)
        
    @patch('src.cli.ui.spinners.Spinner')
    def test_spinner_context(self, mock_spinner_class):
        """Test spinner_context function"""
        mock_spinner = MagicMock()
        mock_spinner_class.return_value = mock_spinner
        
        # Test normal exit
        with spinner_context("Working...") as spinner:
            self.assertEqual(spinner, mock_spinner)
        
        mock_spinner_class.assert_called_once_with(
            "Working...", ANY, ANY, ANY, ANY
        )
        mock_spinner.start.assert_called_once()
        mock_spinner.succeed.assert_called_once()
        mock_spinner.fail.assert_not_called()
        
        # Reset mocks and test exception case
        mock_spinner_class.reset_mock()
        mock_spinner.reset_mock()
        
        # Test with exception
        try:
            with spinner_context("Working..."):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        mock_spinner.start.assert_called_once()
        mock_spinner.succeed.assert_not_called()
        mock_spinner.fail.assert_called_once()
        
    @patch('src.cli.ui.spinners.Spinner')
    def test_with_spinner_decorator(self, mock_spinner_class):
        """Test with_spinner decorator"""
        mock_spinner = MagicMock()
        mock_spinner_class.return_value = mock_spinner
        
        # Define a test function with the decorator
        @with_spinner("Working...")
        def test_function(param):
            return f"Result: {param}"
        
        # Test normal execution
        result = test_function("test")
        
        self.assertEqual(result, "Result: test")
        mock_spinner_class.assert_called_once()
        mock_spinner.start.assert_called_once()
        mock_spinner.succeed.assert_called_once()
        mock_spinner.fail.assert_not_called()
        
        # Reset mocks and test exception case
        mock_spinner_class.reset_mock()
        mock_spinner.reset_mock()
        
        # Test with exception
        try:
            @with_spinner("Working...")
            def failing_function():
                raise ValueError("Test exception")
            
            failing_function()
        except ValueError:
            pass
        
        mock_spinner.start.assert_called_once()
        mock_spinner.succeed.assert_not_called()
        mock_spinner.fail.assert_called_once()
        
    def test_get_spinner_frames(self):
        """Test get_spinner_frames function"""
        # Test with default
        default_frames = get_spinner_frames()
        self.assertEqual(default_frames, get_spinner_frames("dots"))
        
        # Test with specific types
        dots_frames = get_spinner_frames("dots")
        self.assertIsInstance(dots_frames, list)
        self.assertGreater(len(dots_frames), 0)
        
        line_frames = get_spinner_frames("line")
        self.assertEqual(line_frames, ["|", "/", "-", "\\"])
        
        arrows_frames = get_spinner_frames("arrows")
        self.assertGreater(len(arrows_frames), 0)
        
        bouncing_frames = get_spinner_frames("bouncing")
        self.assertGreater(len(bouncing_frames), 0)
        
        # Test with invalid type (should return default)
        invalid_frames = get_spinner_frames("invalid")
        self.assertEqual(invalid_frames, default_frames)