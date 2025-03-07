"""
Text formatting utilities for the CLI user interface of the ERCOT RTLMP spike prediction system.

This module provides functions for formatting various types of text output including
headers, titles, paragraphs, and specialized formatting for different data types
in the command-line interface.
"""

import re
import shutil
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union

from .colors import (
    bold, 
    colorize, 
    get_color_safe_length, 
    italic, 
    supports_color, 
    underline
)

# Constants
DEFAULT_TERMINAL_WIDTH = 80
HEADER_CHAR = "="
SUBHEADER_CHAR = "-"
BULLET_CHAR = "•"
INDENT_SPACES = 2


def format_header(text: str, char: Optional[str] = None, use_colors: Optional[bool] = None) -> str:
    """
    Formats text as a header with underline characters.

    Args:
        text: The text to format as a header.
        char: The character to use for the underline. Defaults to HEADER_CHAR.
        use_colors: Whether to use colors. If None, auto-detected.

    Returns:
        Formatted header text with underline.
    """
    if char is None:
        char = HEADER_CHAR
    
    if use_colors is True:
        formatted_text = bold(text, use_colors=use_colors)
    else:
        formatted_text = text
    
    # Get the visible length of the text (ignoring color codes)
    text_length = get_color_safe_length(formatted_text)
    
    # Create underline
    underline_text = char * text_length
    
    return f"{formatted_text}\n{underline_text}"


def format_subheader(text: str, char: Optional[str] = None, use_colors: Optional[bool] = None) -> str:
    """
    Formats text as a subheader with underline characters.

    Args:
        text: The text to format as a subheader.
        char: The character to use for the underline. Defaults to SUBHEADER_CHAR.
        use_colors: Whether to use colors. If None, auto-detected.

    Returns:
        Formatted subheader text with underline.
    """
    if char is None:
        char = SUBHEADER_CHAR
    
    return format_header(text, char, use_colors)


def format_title(text: str, width: Optional[int] = None, use_colors: Optional[bool] = None) -> str:
    """
    Formats text as a centered title.

    Args:
        text: The text to format as a title.
        width: The width to center within. If None, uses terminal width.
        use_colors: Whether to use colors. If None, auto-detected.

    Returns:
        Centered title text.
    """
    if width is None:
        term_width, _ = get_terminal_size()
        width = term_width
    
    if use_colors is True:
        formatted_text = bold(text, use_colors=use_colors)
    else:
        formatted_text = text
    
    return center_text(formatted_text, width)


def format_paragraph(text: str, width: Optional[int] = None, indent: Optional[int] = None) -> str:
    """
    Formats text as a wrapped paragraph.

    Args:
        text: The text to format as a paragraph.
        width: The width to wrap to. If None, uses terminal width.
        indent: The number of spaces to indent. If None, no indentation.

    Returns:
        Wrapped paragraph text.
    """
    if width is None:
        term_width, _ = get_terminal_size()
        width = term_width
    
    if indent is None:
        indent = 0
    
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=' ' * indent,
        subsequent_indent=' ' * indent
    )
    
    return wrapper.fill(text)


def format_bullet_list(
    items: List[str], 
    bullet: Optional[str] = None, 
    indent: Optional[int] = None,
    width: Optional[int] = None
) -> str:
    """
    Formats a list of items as a bullet list.

    Args:
        items: The list of items to format.
        bullet: The bullet character to use. Defaults to BULLET_CHAR.
        indent: The number of spaces to indent each item. Defaults to INDENT_SPACES.
        width: The width to wrap each item to. If None, uses terminal width.

    Returns:
        Formatted bullet list as a string.
    """
    if bullet is None:
        bullet = BULLET_CHAR
    
    if indent is None:
        indent = INDENT_SPACES
    
    if width is None:
        term_width, _ = get_terminal_size()
        width = term_width
    
    # Initial bullet and space
    bullet_prefix = f"{' ' * indent}{bullet} "
    bullet_indent = ' ' * (indent + len(bullet) + 1)
    
    formatted_items = []
    for item in items:
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=bullet_prefix,
            subsequent_indent=bullet_indent
        )
        formatted_items.append(wrapper.fill(item))
    
    return '\n'.join(formatted_items)


def format_numbered_list(
    items: List[str], 
    start: Optional[int] = None,
    indent: Optional[int] = None,
    width: Optional[int] = None
) -> str:
    """
    Formats a list of items as a numbered list.

    Args:
        items: The list of items to format.
        start: The number to start from. Defaults to 1.
        indent: The number of spaces to indent each item. Defaults to INDENT_SPACES.
        width: The width to wrap each item to. If None, uses terminal width.

    Returns:
        Formatted numbered list as a string.
    """
    if start is None:
        start = 1
    
    if indent is None:
        indent = INDENT_SPACES
    
    if width is None:
        term_width, _ = get_terminal_size()
        width = term_width
    
    formatted_items = []
    for i, item in enumerate(items, start=start):
        # Calculate the width of the number prefix
        number_prefix = f"{' ' * indent}{i}. "
        number_indent = ' ' * len(number_prefix)
        
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=number_prefix,
            subsequent_indent=number_indent
        )
        formatted_items.append(wrapper.fill(item))
    
    return '\n'.join(formatted_items)


def format_key_value(
    key: str, 
    value: Any, 
    key_width: Optional[int] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats a key-value pair with alignment.

    Args:
        key: The key to format.
        value: The value to format.
        key_width: The width to right-align the key. If None, uses a default width.
        use_colors: Whether to use colors. If None, auto-detected.

    Returns:
        Formatted key-value pair.
    """
    if key_width is None:
        key_width = 20
    
    if use_colors is True:
        formatted_key = bold(key, use_colors=use_colors)
    else:
        formatted_key = key
    
    # Calculate needed spacing with color-safe length
    visible_key_length = get_color_safe_length(formatted_key)
    padding = max(0, key_width - visible_key_length)
    padded_key = ' ' * padding + formatted_key
    
    # Convert value to string if it's not already
    value_str = str(value)
    
    return f"{padded_key}: {value_str}"


def format_key_value_list(
    data: Dict[str, Any], 
    key_width: Optional[int] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats a dictionary as a list of key-value pairs.

    Args:
        data: The dictionary to format.
        key_width: The width to right-align the keys. If None, calculated from longest key.
        use_colors: Whether to use colors. If None, auto-detected.

    Returns:
        Formatted list of key-value pairs.
    """
    if key_width is None:
        # Find the length of the longest key
        key_width = max((get_color_safe_length(k) for k in data.keys()), default=0) + 2
    
    formatted_entries = []
    for key, value in data.items():
        formatted_entries.append(format_key_value(key, value, key_width, use_colors))
    
    return '\n'.join(formatted_entries)


def format_section(title: str, content: str, use_colors: Optional[bool] = None) -> str:
    """
    Formats a section with a header and content.

    Args:
        title: The section title.
        content: The section content.
        use_colors: Whether to use colors. If None, auto-detected.

    Returns:
        Formatted section with header and content.
    """
    header = format_header(title, use_colors=use_colors)
    return f"{header}\n\n{content}"


def format_subsection(title: str, content: str, use_colors: Optional[bool] = None) -> str:
    """
    Formats a subsection with a subheader and content.

    Args:
        title: The subsection title.
        content: The subsection content.
        use_colors: Whether to use colors. If None, auto-detected.

    Returns:
        Formatted subsection with subheader and content.
    """
    subheader = format_subheader(title, use_colors=use_colors)
    return f"{subheader}\n\n{content}"


def format_box(
    text: str, 
    width: Optional[int] = None,
    border_char: Optional[str] = None,
    use_colors: Optional[bool] = None
) -> str:
    """
    Formats text within a box of characters.

    Args:
        text: The text to put in the box.
        width: The width of the box. If None, uses terminal width.
        border_char: The character to use for the border. Defaults to "+".
        use_colors: Whether to use colors. If None, auto-detected.

    Returns:
        Text formatted within a box.
    """
    if width is None:
        term_width, _ = get_terminal_size()
        width = term_width
    
    if border_char is None:
        border_char = "+"
    
    # The usable width for text is the box width minus 4 (for borders and spaces)
    text_width = width - 4
    
    # Create top and bottom borders
    horizontal_border = border_char * width
    
    # Wrap the text to fit in the box
    wrapper = textwrap.TextWrapper(width=text_width)
    wrapped_lines = wrapper.wrap(text)
    
    # Format each line with side borders
    formatted_lines = [f"| {line}{' ' * (text_width - get_color_safe_length(line))} |" for line in wrapped_lines]
    
    # If use_colors is True, color the borders
    if use_colors is True:
        horizontal_border = colorize(horizontal_border, color="cyan", use_colors=use_colors)
        formatted_lines = [
            f"{colorize('|', color='cyan', use_colors=use_colors)} {line}{' ' * (text_width - get_color_safe_length(line))} {colorize('|', color='cyan', use_colors=use_colors)}"
            for line in wrapped_lines
        ]
    
    # Combine all parts
    result = [horizontal_border]
    result.extend(formatted_lines)
    result.append(horizontal_border)
    
    return '\n'.join(result)


def get_terminal_size() -> Tuple[int, int]:
    """
    Gets the current terminal size.

    Returns:
        A tuple of (width, height) representing the terminal dimensions in characters.
    """
    try:
        columns, lines = shutil.get_terminal_size()
        return columns, lines
    except (AttributeError, OSError):
        # Fallback to default size if terminal size cannot be determined
        return DEFAULT_TERMINAL_WIDTH, 24


def wrap_text(
    text: str, 
    width: Optional[int] = None, 
    indent: Optional[int] = None,
    subsequent_indent: Optional[bool] = None
) -> str:
    """
    Wraps text to fit within a specified width.

    Args:
        text: The text to wrap.
        width: The width to wrap to. If None, uses terminal width.
        indent: The number of spaces to indent. If None, no indentation.
        subsequent_indent: Whether to indent all lines (True) or just the first line (False).
            If None, defaults to False.

    Returns:
        Wrapped text.
    """
    if width is None:
        term_width, _ = get_terminal_size()
        width = term_width
    
    if indent is None:
        indent = 0
    
    initial_indent = ' ' * indent
    subsequent = ' ' * indent if subsequent_indent else ''
    
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent
    )
    
    return wrapper.fill(text)


def truncate_text(text: str, max_length: int, suffix: Optional[str] = None) -> str:
    """
    Truncates text to a specified length with an ellipsis.

    Args:
        text: The text to truncate.
        max_length: The maximum length (including suffix).
        suffix: The suffix to add to truncated text. Defaults to "...".

    Returns:
        Truncated text.
    """
    if suffix is None:
        suffix = "..."
    
    if len(text) <= max_length:
        return text
    
    # Calculate truncation point
    trunc_length = max_length - len(suffix)
    
    # Ensure we don't have a negative truncation point
    if trunc_length <= 0:
        return suffix[:max_length]
    
    return text[:trunc_length] + suffix


def center_text(text: str, width: Optional[int] = None) -> str:
    """
    Centers text within a specified width.

    Args:
        text: The text to center.
        width: The width to center within. If None, uses terminal width.

    Returns:
        Centered text.
    """
    if width is None:
        term_width, _ = get_terminal_size()
        width = term_width
    
    # Calculate visible length (ignoring color codes)
    visible_length = get_color_safe_length(text)
    
    # Calculate padding needed on each side
    total_padding = max(0, width - visible_length)
    left_padding = total_padding // 2
    
    # Return centered text
    return ' ' * left_padding + text


def align_text(text: str, width: Optional[int] = None, alignment: Optional[str] = None) -> str:
    """
    Aligns text to left, right, or center within a specified width.

    Args:
        text: The text to align.
        width: The width to align within. If None, uses terminal width.
        alignment: The alignment type: 'left', 'right', or 'center'. Defaults to 'left'.

    Returns:
        Aligned text.
    """
    if width is None:
        term_width, _ = get_terminal_size()
        width = term_width
    
    if alignment is None:
        alignment = 'left'
    
    # Calculate visible length (ignoring color codes)
    visible_length = get_color_safe_length(text)
    
    # Apply alignment
    if alignment.lower() == 'left':
        return text
    elif alignment.lower() == 'right':
        padding = max(0, width - visible_length)
        return ' ' * padding + text
    elif alignment.lower() == 'center':
        return center_text(text, width)
    else:
        # Default to left alignment for unknown alignment types
        return text