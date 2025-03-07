"""
Color formatting utilities for the CLI user interface of the ERCOT RTLMP spike prediction system.

This module provides functions for consistent color styling of terminal output
to enhance readability and visual distinction of different types of information.
It also handles terminal color support detection and accessibility considerations.
"""

import os
import re
import sys
from typing import Dict, Optional, Union

# ANSI color codes
COLORS: Dict[str, str] = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "bright_black": "90",
    "bright_red": "91",
    "bright_green": "92",
    "bright_yellow": "93",
    "bright_blue": "94",
    "bright_magenta": "95",
    "bright_cyan": "96",
    "bright_white": "97"
}

# ANSI background color codes
BACKGROUNDS: Dict[str, str] = {
    "black": "40",
    "red": "41",
    "green": "42",
    "yellow": "43",
    "blue": "44",
    "magenta": "45",
    "cyan": "46",
    "white": "47",
    "bright_black": "100",
    "bright_red": "101",
    "bright_green": "102",
    "bright_yellow": "103",
    "bright_blue": "104",
    "bright_magenta": "105",
    "bright_cyan": "106",
    "bright_white": "107"
}

# ANSI style codes
STYLES: Dict[str, str] = {
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "underline": "4",
    "blink": "5",
    "reverse": "7",
    "hidden": "8",
    "strikethrough": "9"
}

# ANSI reset code
RESET: str = "\033[0m"

# Environment variables
NO_COLOR_ENV_VAR: str = "NO_COLOR"
FORCE_COLOR_ENV_VAR: str = "FORCE_COLOR"

# Color mappings for different statuses
STATUS_COLORS: Dict[str, str] = {
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "cyan"
}

# Color mappings for log levels
LOG_LEVEL_COLORS: Dict[str, str] = {
    "DEBUG": "bright_black",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bright_red"
}

# Color mappings for price thresholds
PRICE_THRESHOLD_COLORS: Dict[str, str] = {
    "low": "green",
    "medium": "yellow",
    "high": "red"
}

# Color mappings for probability thresholds
PROBABILITY_THRESHOLD_COLORS: Dict[str, str] = {
    "low": "green",
    "medium": "yellow",
    "high": "red"
}


def supports_color() -> bool:
    """
    Determines if the current terminal supports color output.

    Returns:
        bool: True if color is supported, False otherwise.
    """
    # Check if NO_COLOR environment variable is set
    if os.environ.get(NO_COLOR_ENV_VAR) is not None:
        return False

    # Check if FORCE_COLOR environment variable is set
    if os.environ.get(FORCE_COLOR_ENV_VAR) is not None:
        return True

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check for known terminals that support color
    if "COLORTERM" in os.environ:
        return True

    term = os.environ.get("TERM", "")
    supported_terms = [
        "xterm", "xterm-color", "xterm-256color",
        "screen", "screen-256color", "tmux", "tmux-256color",
        "linux", "cygwin", "ansi"
    ]
    
    if term in supported_terms:
        return True

    return False


def colorize(
    text: str,
    color: Optional[str] = None,
    background: Optional[str] = None,
    style: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Applies color and style formatting to text.

    Args:
        text: The text to format.
        color: The color name from COLORS.
        background: The background color name from BACKGROUNDS.
        style: The style name from STYLES.
        use_colors: Override to force enable/disable colors.

    Returns:
        str: Formatted text with ANSI color codes.
    """
    # Determine if we should use colors
    should_use_colors = use_colors if use_colors is not None else supports_color()
    
    if not should_use_colors:
        return text

    # Build the format codes
    codes = []
    
    if color and color in COLORS:
        codes.append(COLORS[color])
    
    if background and background in BACKGROUNDS:
        codes.append(BACKGROUNDS[background])
    
    if style and style in STYLES:
        codes.append(STYLES[style])
    
    if not codes:
        return text
    
    # Combine codes and create the ANSI escape sequence
    code_str = ";".join(codes)
    return f"\033[{code_str}m{text}{RESET}"


def bold(text: str, use_colors: Optional[bool] = None) -> str:
    """
    Makes text bold.

    Args:
        text: The text to format.
        use_colors: Override to force enable/disable colors.

    Returns:
        str: Bold text.
    """
    return colorize(text, style="bold", use_colors=use_colors)


def italic(text: str, use_colors: Optional[bool] = None) -> str:
    """
    Makes text italic.

    Args:
        text: The text to format.
        use_colors: Override to force enable/disable colors.

    Returns:
        str: Italic text.
    """
    return colorize(text, style="italic", use_colors=use_colors)


def underline(text: str, use_colors: Optional[bool] = None) -> str:
    """
    Underlines text.

    Args:
        text: The text to format.
        use_colors: Override to force enable/disable colors.

    Returns:
        str: Underlined text.
    """
    return colorize(text, style="underline", use_colors=use_colors)


def color_by_status(text: str, status: str, use_colors: Optional[bool] = None) -> str:
    """
    Colors text based on status type (success, warning, error, info).

    Args:
        text: The text to format.
        status: The status type.
        use_colors: Override to force enable/disable colors.

    Returns:
        str: Colored text based on status.
    """
    color = STATUS_COLORS.get(status.lower())
    if color:
        return colorize(text, color=color, use_colors=use_colors)
    return text


def color_by_level(text: str, level: str, use_colors: Optional[bool] = None) -> str:
    """
    Colors text based on log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Args:
        text: The text to format.
        level: The log level.
        use_colors: Override to force enable/disable colors.

    Returns:
        str: Colored text based on log level.
    """
    color = LOG_LEVEL_COLORS.get(level.upper())
    if color:
        return colorize(text, color=color, use_colors=use_colors)
    return text


def color_by_value(
    value: Union[float, str],
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Colors a numeric value based on thresholds.

    Args:
        value: The numeric value to format.
        low_threshold: The lower threshold (default: 50).
        high_threshold: The higher threshold (default: 100).
        use_colors: Override to force enable/disable colors.

    Returns:
        str: Colored value based on thresholds.
    """
    # Convert value to float if it's a string
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        return str(value)

    # Set default thresholds if not provided
    if low_threshold is None:
        low_threshold = 50.0
    if high_threshold is None:
        high_threshold = 100.0

    # Determine color based on value
    if float_value < low_threshold:
        color = PRICE_THRESHOLD_COLORS['low']
    elif float_value < high_threshold:
        color = PRICE_THRESHOLD_COLORS['medium']
    else:
        color = PRICE_THRESHOLD_COLORS['high']

    return colorize(str(value), color=color, use_colors=use_colors)


def color_by_probability(
    probability: Union[float, str],
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Colors a probability value based on standard thresholds.

    Args:
        probability: The probability value (0-1) to format.
        low_threshold: The lower threshold (default: 0.25).
        high_threshold: The higher threshold (default: 0.75).
        use_colors: Override to force enable/disable colors.

    Returns:
        str: Colored probability based on thresholds.
    """
    # Convert probability to float if it's a string
    try:
        float_prob = float(probability)
    except (ValueError, TypeError):
        return str(probability)

    # Ensure probability is between 0 and 1
    float_prob = max(0.0, min(1.0, float_prob))

    # Set default thresholds if not provided
    if low_threshold is None:
        low_threshold = 0.25
    if high_threshold is None:
        high_threshold = 0.75

    # Determine color based on probability
    if float_prob < low_threshold:
        color = PROBABILITY_THRESHOLD_COLORS['low']
    elif float_prob < high_threshold:
        color = PROBABILITY_THRESHOLD_COLORS['medium']
    else:
        color = PROBABILITY_THRESHOLD_COLORS['high']

    return colorize(str(probability), color=color, use_colors=use_colors)


def strip_color(text: str) -> str:
    """
    Removes ANSI color codes from a string.

    Args:
        text: The text to clean.

    Returns:
        str: Text with color codes removed.
    """
    # Regular expression to match ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def get_color_safe_length(text: str) -> int:
    """
    Gets the visible length of a string, ignoring ANSI color codes.

    Args:
        text: The text to measure.

    Returns:
        int: Visible length of the text.
    """
    return len(strip_color(text))