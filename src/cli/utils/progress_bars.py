"""
Provides progress bar utilities for the CLI application of the ERCOT RTLMP spike prediction system.

This module implements progress indicators for long-running operations such as data fetching,
model training, and inference, enhancing user experience by providing visual feedback on
operation progress.
"""

import sys
import time
import contextlib
import functools
from typing import Optional, Callable, Dict, Any

import tqdm  # version 4.64+

from ..logger import get_cli_logger
from ..ui.colors import colorize, supports_color
from ..ui.spinners import Spinner

# Configure logger for progress bar operations
PROGRESS_LOGGER = get_cli_logger('progress_bars')

# Default progress bar settings
DEFAULT_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
DEFAULT_UNIT = "items"
DEFAULT_COLOR = "cyan"


class ProgressBar:
    """Class for creating and managing progress bars."""

    def __init__(
        self,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: Optional[str] = None,
        color: Optional[str] = None,
        disable: bool = False
    ):
        """
        Initialize a ProgressBar instance with the specified parameters.

        Args:
            total: Total number of items to process
            desc: Description of the progress bar
            unit: Unit name for the items being processed
            color: Color name for the progress bar
            disable: Whether to disable the progress bar
        """
        self._desc = desc
        self._total = total
        self._color = color if color is not None else DEFAULT_COLOR
        # Disable progress bar if explicitly set or if stdout is not a terminal
        self._disable = disable or not sys.stdout.isatty()
        
        # Apply color to description if enabled and supported
        colored_desc = None
        if desc and not self._disable and self._color and supports_color():
            colored_desc = colorize(desc, color=self._color)
        
        # Create tqdm instance with appropriate settings
        self._tqdm = tqdm.tqdm(
            total=total,
            desc=colored_desc or desc,
            unit=unit or DEFAULT_UNIT,
            bar_format=DEFAULT_BAR_FORMAT,
            file=sys.stdout,
            disable=self._disable
        )

    def update(self, n: Optional[int] = None, desc: Optional[str] = None) -> None:
        """
        Update the progress bar with the specified increment and description.

        Args:
            n: Amount to increment the progress bar (default: 1)
            desc: New description for the progress bar
        """
        if self._disable:
            return
        
        # Update description if provided
        if desc is not None:
            self._desc = desc
            colored_desc = desc
            if self._color and supports_color():
                colored_desc = colorize(desc, color=self._color)
            self._tqdm.set_description(colored_desc)
        
        # Update progress
        if n is not None:
            self._tqdm.update(n)
        else:
            self._tqdm.update(1)

    def close(self) -> None:
        """
        Close the progress bar.
        """
        if self._disable:
            return
        
        self._tqdm.close()

    def reset(self, total: Optional[int] = None) -> None:
        """
        Reset the progress bar to its initial state.

        Args:
            total: New total for the progress bar (if None, uses the original total)
        """
        if self._disable:
            return
        
        if total is not None:
            self._total = total
            self._tqdm.total = total
        
        self._tqdm.reset()

    def get_callback(self, desc_template: Optional[str] = None) -> Callable[[int, Optional[Dict[str, Any]]], None]:
        """
        Get a callback function for updating this progress bar.

        Args:
            desc_template: Template string for formatting descriptions with context

        Returns:
            Callable: Callback function for updating the progress bar
        """
        return progress_callback(self, desc_template)

    def __enter__(self) -> 'ProgressBar':
        """
        Enter the context, returning the ProgressBar instance.
        
        Returns:
            ProgressBar: Self for use within a with statement
        """
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> bool:
        """
        Exit the context, closing the progress bar.
        
        Args:
            exc_type: Type of exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Exception traceback, if any
            
        Returns:
            bool: False to propagate exceptions
        """
        self.close()
        return False


class IndeterminateSpinner:
    """Class for creating and managing spinners for operations with unknown duration."""

    def __init__(self, desc: str, color: Optional[str] = None, disable: bool = False):
        """
        Initialize an IndeterminateSpinner instance with the specified parameters.

        Args:
            desc: Description of the operation
            color: Color name for the spinner
            disable: Whether to disable the spinner
        """
        self._desc = desc
        self._color = color if color is not None else DEFAULT_COLOR
        # Disable spinner if explicitly set or if stdout is not a terminal
        self._disable = disable or not sys.stdout.isatty()
        
        # Create spinner instance
        self._spinner = Spinner(
            message=desc,
            color=self._color,
            disable=self._disable
        )
        
        # Start the spinner immediately
        if not self._disable:
            self._spinner.start()

    def update(self, desc: str) -> None:
        """
        Update the spinner message.

        Args:
            desc: New description for the spinner
        """
        if self._disable:
            return
        
        self._desc = desc
        self._spinner.update(desc)

    def close(self, final_message: Optional[str] = None) -> None:
        """
        Stop the spinner and clean up.

        Args:
            final_message: Optional message to display after stopping the spinner
        """
        if self._disable:
            return
        
        if final_message and self._color and supports_color():
            final_message = colorize(final_message, color=self._color)
        
        self._spinner.stop(final_message)

    def __enter__(self) -> 'IndeterminateSpinner':
        """
        Enter the context, returning the IndeterminateSpinner instance.
        
        Returns:
            IndeterminateSpinner: Self for use within a with statement
        """
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> bool:
        """
        Exit the context, stopping the spinner.
        
        Args:
            exc_type: Type of exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Exception traceback, if any
            
        Returns:
            bool: False to propagate exceptions
        """
        if exc_type is not None:
            # If an exception occurred, show an error message
            self._spinner.fail()
        else:
            # Operation completed successfully
            self._spinner.succeed()
        
        return False


def create_progress_bar(
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: Optional[str] = None,
    color: Optional[str] = None,
    disable: bool = False
) -> ProgressBar:
    """
    Creates a tqdm progress bar with consistent styling.

    Args:
        total: Total number of items to process
        desc: Description of the progress bar
        unit: Unit name for the items being processed
        color: Color name for the progress bar
        disable: Whether to disable the progress bar

    Returns:
        ProgressBar: Configured progress bar instance
    """
    return ProgressBar(total, desc, unit, color, disable)


@contextlib.contextmanager
def progress_bar_context(
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: Optional[str] = None,
    color: Optional[str] = None,
    disable: bool = False
) -> Any:  # Using Any as the return type for broader compatibility
    """
    Context manager for creating and managing a progress bar.

    Args:
        total: Total number of items to process
        desc: Description of the progress bar
        unit: Unit name for the items being processed
        color: Color name for the progress bar
        disable: Whether to disable the progress bar

    Returns:
        ContextManager[ProgressBar]: Context manager yielding a ProgressBar instance
    """
    progress_bar = create_progress_bar(total, desc, unit, color, disable)
    try:
        yield progress_bar
    finally:
        progress_bar.close()


def update_progress_bar(
    progress_bar: ProgressBar,
    increment: Optional[int] = None,
    desc: Optional[str] = None
) -> None:
    """
    Updates a progress bar with the specified increment and description.

    Args:
        progress_bar: The progress bar to update
        increment: Amount to increment the progress bar (default: 1)
        desc: New description for the progress bar
    """
    if progress_bar is None:
        return
    
    progress_bar.update(increment, desc)


def create_indeterminate_spinner(
    desc: str,
    color: Optional[str] = None,
    disable: bool = False
) -> IndeterminateSpinner:
    """
    Creates a spinner for operations with unknown duration.

    Args:
        desc: Description of the operation
        color: Color name for the spinner
        disable: Whether to disable the spinner

    Returns:
        IndeterminateSpinner: Configured spinner instance
    """
    return IndeterminateSpinner(desc, color, disable)


def with_progress_bar(
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: Optional[str] = None,
    color: Optional[str] = None,
    disable: bool = False
) -> Callable:
    """
    Decorator that wraps a function with a progress bar.

    Args:
        total: Total number of items to process
        desc: Description of the progress bar
        unit: Unit name for the items being processed
        color: Color name for the progress bar
        disable: Whether to disable the progress bar

    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with progress_bar_context(total, desc, unit, color, disable) as pbar:
                # Add progress bar to kwargs
                kwargs['progress_bar'] = pbar
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def progress_callback(
    progress_bar: ProgressBar,
    desc_template: Optional[str] = None
) -> Callable[[int, Optional[Dict[str, Any]]], None]:
    """
    Creates a callback function for updating a progress bar.

    Args:
        progress_bar: The progress bar to update
        desc_template: Template string for formatting descriptions with context

    Returns:
        Callable: Callback function for updating the progress bar
    """
    def callback(increment: int = 1, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the progress bar with the given increment and context.

        Args:
            increment: Amount to increment the progress bar
            context: Context variables for description formatting
        """
        description = None
        if desc_template is not None and context is not None:
            try:
                description = desc_template.format(**context)
            except KeyError as e:
                PROGRESS_LOGGER.warning(f"Missing key in progress bar description template: {e}")
        
        update_progress_bar(progress_bar, increment, description)
    
    return callback


def format_progress_message(
    message: str,
    current: int,
    total: int,
    elapsed_time: Optional[float] = None,
    color: Optional[str] = None
) -> str:
    """
    Formats a progress message with current status information.

    Args:
        message: Base message to format
        current: Current progress value
        total: Total expected value
        elapsed_time: Elapsed time in seconds
        color: Color name for the message

    Returns:
        str: Formatted progress message
    """
    # Calculate percentage complete
    percentage = (current / total) * 100 if total > 0 else 0
    
    # Format basic progress message
    formatted_message = f"{message} [{current}/{total}, {percentage:.1f}%]"
    
    # Add ETA if elapsed time is provided
    if elapsed_time is not None and elapsed_time > 0 and current > 0 and current < total:
        items_per_second = current / elapsed_time
        remaining_items = total - current
        eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
        
        # Format ETA in a human-readable way
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.1f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds / 60:.1f}m"
        else:
            eta_str = f"{eta_seconds / 3600:.1f}h"
        
        formatted_message += f" ETA: {eta_str}"
    
    # Apply color if specified and supported
    if color and supports_color():
        formatted_message = colorize(formatted_message, color=color)
    
    return formatted_message