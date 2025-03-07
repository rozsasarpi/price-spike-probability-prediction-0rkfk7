"""
Provides spinner components for the CLI application of the ERCOT RTLMP spike prediction system.

This module implements animated terminal spinners to indicate progress for operations
with unknown duration, enhancing user experience by providing visual feedback during
long-running tasks.
"""

import sys
import time
import threading
import contextlib
from typing import Any, Callable, Dict, List, Optional, ContextManager

from .colors import colorize, supports_color
from ..logger import get_cli_logger

# Configure module logger
SPINNER_LOGGER = get_cli_logger('spinners')

# Default spinner configuration
DEFAULT_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
ALTERNATIVE_SPINNER_FRAMES = ["|", "/", "-", "\\"]
SPINNER_TYPES = {
    "dots": DEFAULT_SPINNER_FRAMES,
    "line": ALTERNATIVE_SPINNER_FRAMES,
    "arrows": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    "bouncing": ["⠁", "⠂", "⠄", "⠠", "⠐", "⠈"]
}
DEFAULT_SPINNER_TYPE = "dots"
DEFAULT_SPINNER_COLOR = "cyan"
DEFAULT_SPINNER_INTERVAL = 0.1


class Spinner:
    """Class for creating and managing animated terminal spinners."""

    def __init__(
        self,
        message: str,
        frames: Optional[List[str]] = None,
        color: Optional[str] = None,
        interval: Optional[float] = None,
        disable: bool = False
    ):
        """
        Initialize a Spinner instance with the specified parameters.

        Args:
            message: Text to display next to the spinner
            frames: List of characters for spinner animation
            color: Color name for the spinner text
            interval: Time interval between spinner frames in seconds
            disable: Whether to disable the spinner (useful for non-interactive sessions)
        """
        self._message = message
        self._frames = frames if frames is not None else DEFAULT_SPINNER_FRAMES
        self._interval = interval if interval is not None else DEFAULT_SPINNER_INTERVAL
        self._color = color if color is not None else DEFAULT_SPINNER_COLOR
        # Disable spinner if explicitly set or if stdout is not a terminal
        self._disable = disable or not sys.stdout.isatty()
        self._running = False
        self._thread = None
        self._frame_index = 0
        self._final_message = None

    def start(self) -> None:
        """
        Start the spinner animation in a separate thread.
        """
        if self._disable:
            return
        
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._spin)
        self._thread.daemon = True
        self._thread.start()

    def stop(self, final_message: Optional[str] = None) -> None:
        """
        Stop the spinner animation and display final message.

        Args:
            final_message: Optional message to display after stopping the spinner
        """
        if self._disable:
            return
        
        if not self._running:
            return
        
        self._running = False
        self._final_message = final_message
        
        if self._thread:
            self._thread.join()
        
        self._clear_line()
        
        if self._final_message:
            print(self._final_message)
        
        self._thread = None

    def update(self, message: str) -> None:
        """
        Update the spinner message.

        Args:
            message: New message to display
        """
        if self._disable:
            return
        
        self._message = message
        
        if not self._running:
            SPINNER_LOGGER.warning("Attempted to update a spinner that is not running")

    def succeed(self, message: Optional[str] = None) -> None:
        """
        Stop the spinner with a success message.

        Args:
            message: Success message to display
        """
        if message is None:
            if self._message.endswith('...'):
                message = self._message.rstrip('...') + ' completed'
            else:
                message = 'Done'
        
        formatted_message = f"✓ {message}"
        if supports_color():
            formatted_message = colorize(formatted_message, color="green")
        
        self.stop(formatted_message)

    def fail(self, message: Optional[str] = None) -> None:
        """
        Stop the spinner with a failure message.

        Args:
            message: Failure message to display
        """
        if message is None:
            if self._message.endswith('...'):
                message = self._message.rstrip('...') + ' failed'
            else:
                message = 'Failed'
        
        formatted_message = f"✗ {message}"
        if supports_color():
            formatted_message = colorize(formatted_message, color="red")
        
        self.stop(formatted_message)

    def warn(self, message: Optional[str] = None) -> None:
        """
        Stop the spinner with a warning message.

        Args:
            message: Warning message to display
        """
        if message is None:
            if self._message.endswith('...'):
                message = self._message.rstrip('...') + ' warning'
            else:
                message = 'Warning'
        
        formatted_message = f"⚠ {message}"
        if supports_color():
            formatted_message = colorize(formatted_message, color="yellow")
        
        self.stop(formatted_message)

    def _spin(self) -> None:
        """
        Internal method that performs the spinner animation.
        """
        while self._running:
            frame = self._frames[self._frame_index]
            spinner_line = f"\r{frame} {self._message}"
            
            if supports_color() and self._color:
                frame = colorize(frame, color=self._color)
                spinner_line = f"\r{frame} {self._message}"
            
            sys.stdout.write(spinner_line)
            sys.stdout.flush()
            
            self._frame_index = (self._frame_index + 1) % len(self._frames)
            time.sleep(self._interval)
            self._clear_line()

    def _clear_line(self) -> None:
        """
        Clear the current terminal line.
        """
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()

    def __enter__(self) -> 'Spinner':
        """
        Enter the context, starting the spinner and returning the instance.
        
        Returns:
            Spinner: Self for use within a with statement
        """
        self.start()
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
            self.fail()
        else:
            self.succeed()
        
        return False


def create_spinner(
    message: str,
    spinner_type: Optional[str] = None,
    color: Optional[str] = None,
    interval: Optional[float] = None,
    disable: bool = False
) -> Spinner:
    """
    Creates a spinner instance with the specified parameters.
    
    Args:
        message: Text to display next to the spinner
        spinner_type: Type of spinner animation to use
        color: Color name for the spinner text
        interval: Time interval between spinner frames in seconds
        disable: Whether to disable the spinner
        
    Returns:
        Spinner: Configured spinner instance
    """
    frames = get_spinner_frames(spinner_type)
    return Spinner(message, frames, color, interval, disable)


@contextlib.contextmanager
def spinner_context(
    message: str,
    spinner_type: Optional[str] = None,
    color: Optional[str] = None,
    interval: Optional[float] = None,
    disable: bool = False
) -> ContextManager[Spinner]:
    """
    Context manager for creating and managing a spinner.
    
    Args:
        message: Text to display next to the spinner
        spinner_type: Type of spinner animation to use
        color: Color name for the spinner text
        interval: Time interval between spinner frames in seconds
        disable: Whether to disable the spinner
        
    Returns:
        ContextManager[Spinner]: Context manager yielding a Spinner instance
    """
    spinner = create_spinner(message, spinner_type, color, interval, disable)
    spinner.start()
    try:
        yield spinner
    except Exception:
        spinner.fail()
        raise
    finally:
        if spinner._running:
            spinner.stop()


def with_spinner(
    message: str,
    spinner_type: Optional[str] = None,
    color: Optional[str] = None,
    interval: Optional[float] = None,
    disable: bool = False
) -> Callable:
    """
    Decorator that wraps a function with a spinner.
    
    Args:
        message: Text to display next to the spinner
        spinner_type: Type of spinner animation to use
        color: Color name for the spinner text
        interval: Time interval between spinner frames in seconds
        disable: Whether to disable the spinner
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            spinner = create_spinner(message, spinner_type, color, interval, disable)
            spinner.start()
            try:
                result = func(*args, **kwargs)
                spinner.succeed()
                return result
            except Exception:
                spinner.fail()
                raise
            
        return wrapper
    
    return decorator


def get_spinner_frames(spinner_type: Optional[str] = None) -> List[str]:
    """
    Gets the frames for the specified spinner type.
    
    Args:
        spinner_type: Type of spinner animation to use
        
    Returns:
        List[str]: List of spinner frames
    """
    if spinner_type is None:
        spinner_type = DEFAULT_SPINNER_TYPE
    
    return SPINNER_TYPES.get(spinner_type, DEFAULT_SPINNER_FRAMES)