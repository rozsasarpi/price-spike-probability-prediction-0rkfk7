"""
Utility functions for date and time operations in the ERCOT RTLMP spike prediction system.

This module provides standardized date handling, timezone conversions, and time-related 
helper functions to ensure consistent datetime processing across the application.
"""

import datetime
from typing import Dict, Optional, Union

import pandas as pd
import pytz  # version: 2023.3+

# Constants for timezone handling
ERCOT_TIMEZONE = pytz.timezone('US/Central')
UTC_TIMEZONE = pytz.UTC

# Standard datetime formats
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ERCOT_API_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

# ERCOT market parameters
DAY_AHEAD_MARKET_CLOSURE_HOUR = 10


def localize_datetime(dt: datetime.datetime, is_dst: bool = None) -> datetime.datetime:
    """
    Localizes a naive datetime object to the ERCOT timezone.
    
    Args:
        dt: The datetime object to localize
        is_dst: Boolean flag indicating whether daylight saving time is in effect.
                If None, pytz will attempt to determine it or raise AmbiguousTimeError.
    
    Returns:
        Timezone-aware datetime in ERCOT timezone
    """
    if dt.tzinfo is not None:
        # If already timezone-aware, convert to ERCOT timezone
        return dt.astimezone(ERCOT_TIMEZONE)
    
    # Localize naive datetime
    return ERCOT_TIMEZONE.localize(dt, is_dst=is_dst)


def convert_to_utc(dt: datetime.datetime) -> datetime.datetime:
    """
    Converts a datetime object to UTC timezone.
    
    Args:
        dt: The datetime object to convert
    
    Returns:
        Timezone-aware datetime in UTC
    """
    if dt.tzinfo is None:
        # If naive, assume it's in ERCOT timezone
        dt = localize_datetime(dt)
    
    # Convert to UTC
    return dt.astimezone(UTC_TIMEZONE)


def format_datetime(dt: datetime.datetime, format_string: str = DEFAULT_DATETIME_FORMAT) -> str:
    """
    Formats a datetime object as a string using the specified format.
    
    Args:
        dt: The datetime object to format
        format_string: The format string to use (default: DEFAULT_DATETIME_FORMAT)
    
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_string)


def parse_datetime(datetime_str: str, format_string: str = DEFAULT_DATETIME_FORMAT, 
                   localize: bool = True) -> datetime.datetime:
    """
    Parses a datetime string into a datetime object.
    
    Args:
        datetime_str: The datetime string to parse
        format_string: The format string to use (default: DEFAULT_DATETIME_FORMAT)
        localize: Whether to localize the datetime to ERCOT timezone (default: True)
    
    Returns:
        Parsed datetime object
    """
    dt = datetime.datetime.strptime(datetime_str, format_string)
    
    if localize:
        dt = localize_datetime(dt)
    
    return dt


def get_current_time() -> datetime.datetime:
    """
    Returns the current time in the ERCOT timezone.
    
    Returns:
        Current datetime in ERCOT timezone
    """
    return datetime.datetime.now(UTC_TIMEZONE).astimezone(ERCOT_TIMEZONE)


def is_dst(dt: datetime.datetime) -> bool:
    """
    Checks if a given datetime is during Daylight Saving Time in the ERCOT timezone.
    
    Args:
        dt: The datetime to check
    
    Returns:
        True if datetime is during DST, False otherwise
    """
    if dt.tzinfo is None or dt.tzinfo.zone != ERCOT_TIMEZONE.zone:
        dt = dt.astimezone(ERCOT_TIMEZONE) if dt.tzinfo else localize_datetime(dt)
    
    return dt.dst() != datetime.timedelta(0)


def create_date_range(start_date: datetime.datetime, end_date: datetime.datetime, 
                      freq: str = 'H') -> pd.DatetimeIndex:
    """
    Creates a pandas DatetimeIndex with the specified frequency between start and end dates.
    
    Args:
        start_date: The start date of the range
        end_date: The end date of the range
        freq: The frequency of the range (default: 'H' for hourly)
    
    Returns:
        DatetimeIndex with the specified frequency
    """
    # Ensure both dates are timezone-aware and in the same timezone
    if start_date.tzinfo is None:
        start_date = localize_datetime(start_date)
    
    if end_date.tzinfo is None:
        end_date = localize_datetime(end_date)
    
    # Ensure both dates are in the same timezone
    if start_date.tzinfo != end_date.tzinfo:
        end_date = end_date.astimezone(start_date.tzinfo)
    
    return pd.date_range(start=start_date, end=end_date, freq=freq, tz=start_date.tzinfo)


def round_datetime(dt: datetime.datetime, freq: str = 'H') -> datetime.datetime:
    """
    Rounds a datetime to the nearest specified frequency.
    
    Args:
        dt: The datetime to round
        freq: The frequency to round to (default: 'H' for hourly)
    
    Returns:
        Rounded datetime
    """
    # Preserve timezone information
    tz = dt.tzinfo
    
    # Convert to pandas Timestamp, round, and convert back to datetime
    rounded = pd.Timestamp(dt).round(freq).to_pydatetime()
    
    # Ensure the timezone is preserved
    if tz is not None and rounded.tzinfo is None:
        rounded = tz.localize(rounded)
    elif tz is not None and rounded.tzinfo != tz:
        rounded = rounded.astimezone(tz)
    
    return rounded


def floor_datetime(dt: datetime.datetime, freq: str = 'H') -> datetime.datetime:
    """
    Floors a datetime to the specified frequency.
    
    Args:
        dt: The datetime to floor
        freq: The frequency to floor to (default: 'H' for hourly)
    
    Returns:
        Floored datetime
    """
    # Preserve timezone information
    tz = dt.tzinfo
    
    # Convert to pandas Timestamp, floor, and convert back to datetime
    floored = pd.Timestamp(dt).floor(freq).to_pydatetime()
    
    # Ensure the timezone is preserved
    if tz is not None and floored.tzinfo is None:
        floored = tz.localize(floored)
    elif tz is not None and floored.tzinfo != tz:
        floored = floored.astimezone(tz)
    
    return floored


def ceil_datetime(dt: datetime.datetime, freq: str = 'H') -> datetime.datetime:
    """
    Ceils a datetime to the specified frequency.
    
    Args:
        dt: The datetime to ceil
        freq: The frequency to ceil to (default: 'H' for hourly)
    
    Returns:
        Ceiled datetime
    """
    # Preserve timezone information
    tz = dt.tzinfo
    
    # Convert to pandas Timestamp, ceil, and convert back to datetime
    ceiled = pd.Timestamp(dt).ceil(freq).to_pydatetime()
    
    # Ensure the timezone is preserved
    if tz is not None and ceiled.tzinfo is None:
        ceiled = tz.localize(ceiled)
    elif tz is not None and ceiled.tzinfo != tz:
        ceiled = ceiled.astimezone(tz)
    
    return ceiled


def validate_date_range(start_date: datetime.datetime, end_date: datetime.datetime, 
                        max_days: Optional[int] = None) -> bool:
    """
    Validates that a date range is valid and within acceptable bounds.
    
    Args:
        start_date: The start date of the range
        end_date: The end date of the range
        max_days: Maximum allowed days between start and end dates (optional)
    
    Returns:
        True if date range is valid, False otherwise
    """
    # Ensure datetimes are comparable (same timezone)
    if start_date.tzinfo is None:
        start_date = localize_datetime(start_date)
    
    if end_date.tzinfo is None:
        end_date = localize_datetime(end_date)
    
    if start_date.tzinfo != end_date.tzinfo:
        end_date = end_date.astimezone(start_date.tzinfo)
    
    # Get current time in the same timezone
    current_time = datetime.datetime.now(start_date.tzinfo)
    
    # Check that start_date is not in the future
    if start_date > current_time:
        return False
    
    # Check that end_date is not in the future
    if end_date > current_time:
        return False
    
    # Check that start_date is before end_date
    if start_date >= end_date:
        return False
    
    # Check max_days constraint if provided
    if max_days is not None:
        if (end_date - start_date).days > max_days:
            return False
    
    return True


def get_day_ahead_market_closure(target_date: datetime.datetime) -> datetime.datetime:
    """
    Returns the day-ahead market closure datetime for a given date.
    
    Args:
        target_date: The date to get the DAM closure for
    
    Returns:
        Day-ahead market closure datetime
    """
    # Ensure date is timezone-aware in ERCOT timezone
    if target_date.tzinfo is None:
        target_date = localize_datetime(target_date)
    elif target_date.tzinfo != ERCOT_TIMEZONE:
        target_date = target_date.astimezone(ERCOT_TIMEZONE)
    
    # Create a new datetime with the DAM closure hour
    closure_datetime = datetime.datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=DAY_AHEAD_MARKET_CLOSURE_HOUR,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=ERCOT_TIMEZONE
    )
    
    return closure_datetime


def is_before_day_ahead_market_closure(dt: datetime.datetime, 
                                       reference_date: Optional[datetime.datetime] = None) -> bool:
    """
    Checks if a given datetime is before the day-ahead market closure.
    
    Args:
        dt: The datetime to check
        reference_date: The reference date for the DAM closure (default: current date)
    
    Returns:
        True if datetime is before DAM closure, False otherwise
    """
    # Ensure datetime is timezone-aware in ERCOT timezone
    if dt.tzinfo is None:
        dt = localize_datetime(dt)
    elif dt.tzinfo != ERCOT_TIMEZONE:
        dt = dt.astimezone(ERCOT_TIMEZONE)
    
    # If reference_date is not provided, use the current date
    if reference_date is None:
        reference_date = get_current_time()
    
    # Ensure reference_date is timezone-aware in ERCOT timezone
    if reference_date.tzinfo is None:
        reference_date = localize_datetime(reference_date)
    elif reference_date.tzinfo != ERCOT_TIMEZONE:
        reference_date = reference_date.astimezone(ERCOT_TIMEZONE)
    
    # Get the DAM closure datetime for the reference date
    dam_closure = get_day_ahead_market_closure(reference_date)
    
    # Check if the provided datetime is before the DAM closure
    return dt < dam_closure


def get_forecast_horizon_end(reference_date: datetime.datetime, horizon_hours: int = 72) -> datetime.datetime:
    """
    Calculates the end datetime for a forecast horizon starting from a reference date.
    
    Args:
        reference_date: The starting point for the forecast horizon
        horizon_hours: Number of hours in the forecast horizon (default: 72)
    
    Returns:
        End datetime of the forecast horizon
    """
    # Ensure reference_date is timezone-aware in ERCOT timezone
    if reference_date.tzinfo is None:
        reference_date = localize_datetime(reference_date)
    elif reference_date.tzinfo != ERCOT_TIMEZONE:
        reference_date = reference_date.astimezone(ERCOT_TIMEZONE)
    
    # Add the specified number of hours
    return reference_date + datetime.timedelta(hours=horizon_hours)


def get_datetime_components(dt: datetime.datetime) -> Dict[str, int]:
    """
    Extracts various components from a datetime object for feature engineering.
    
    Args:
        dt: The datetime object to extract components from
    
    Returns:
        Dictionary of datetime components including hour, day of week, etc.
    """
    # Ensure datetime is timezone-aware in ERCOT timezone
    if dt.tzinfo is None:
        dt = localize_datetime(dt)
    elif dt.tzinfo != ERCOT_TIMEZONE:
        dt = dt.astimezone(ERCOT_TIMEZONE)
    
    # Extract components
    components = {
        'hour': dt.hour,
        'day_of_week': dt.weekday(),  # 0-6, Monday is 0
        'day_of_month': dt.day,
        'month': dt.month,
        'year': dt.year,
        'quarter': (dt.month - 1) // 3 + 1,
        'is_weekend': 1 if dt.weekday() >= 5 else 0,  # 5=Saturday, 6=Sunday
        'is_business_hour': 1 if 8 <= dt.hour < 18 and dt.weekday() < 5 else 0,
        'is_dst': 1 if is_dst(dt) else 0
    }
    
    return components