"""
Provides mock API responses for testing the data fetchers in the ERCOT RTLMP spike prediction system.

This module contains functions that generate realistic JSON responses mimicking the structure
and content of ERCOT API, weather API, and grid conditions API responses, enabling unit tests
to run without external dependencies.
"""

import pandas as pd  # version 2.0+
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import random

from ...utils.type_definitions import RTLMPDataDict, WeatherDataDict, GridConditionDict
from .sample_data import (
    get_sample_rtlmp_data,
    get_sample_weather_data,
    get_sample_grid_condition_data,
    SAMPLE_NODES,
    SAMPLE_START_DATE,
    SAMPLE_END_DATE,
    DEFAULT_LOCATIONS
)

# Constants for API response status codes
ERCOT_API_SUCCESS_STATUS = '200'
ERCOT_API_ERROR_STATUS = '400'
WEATHER_API_SUCCESS_STATUS = '200'
WEATHER_API_ERROR_STATUS = '400'

# Types of errors that can be simulated
ERROR_TYPES = ['connection', 'rate_limit', 'data_format', 'missing_data', 'authentication']


def get_mock_rtlmp_response(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    nodes: Optional[List[str]] = None,
    success: bool = True,
    error_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a mock RTLMP API response based on sample data.
    
    Args:
        start_date: Optional start date for the data range
        end_date: Optional end date for the data range
        nodes: Optional list of node IDs
        success: Whether to generate a successful response
        error_type: Type of error to simulate if success is False
    
    Returns:
        Mock RTLMP API response in JSON format
    """
    # Set default values if not provided
    start_date = start_date or SAMPLE_START_DATE
    end_date = end_date or SAMPLE_END_DATE
    nodes = nodes or SAMPLE_NODES
    
    # If not successful, generate an error response
    if not success:
        return generate_error_response(error_type or random.choice(ERROR_TYPES), 'ercot')
    
    # Get sample RTLMP data
    rtlmp_df = get_sample_rtlmp_data(start_date, end_date, nodes)
    
    # Convert DataFrame to list of dictionaries
    rtlmp_records = rtlmp_df.to_dict('records')
    
    # Format the data according to ERCOT API structure
    response = format_rtlmp_data(rtlmp_records)
    
    return response


def get_mock_grid_conditions_response(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    success: bool = True,
    error_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a mock grid conditions API response based on sample data.
    
    Args:
        start_date: Optional start date for the data range
        end_date: Optional end date for the data range
        success: Whether to generate a successful response
        error_type: Type of error to simulate if success is False
    
    Returns:
        Mock grid conditions API response in JSON format
    """
    # Set default values if not provided
    start_date = start_date or SAMPLE_START_DATE
    end_date = end_date or SAMPLE_END_DATE
    
    # If not successful, generate an error response
    if not success:
        return generate_error_response(error_type or random.choice(ERROR_TYPES), 'grid')
    
    # Get sample grid condition data
    grid_df = get_sample_grid_condition_data(start_date, end_date)
    
    # Convert DataFrame to list of dictionaries
    grid_records = grid_df.to_dict('records')
    
    # Format the data according to ERCOT API structure
    response = format_grid_conditions_data(grid_records)
    
    return response


def get_mock_weather_response(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    locations: Optional[List[str]] = None,
    is_forecast: bool = False,
    success: bool = True,
    error_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a mock weather API response based on sample data.
    
    Args:
        start_date: Optional start date for the data range
        end_date: Optional end date for the data range
        locations: Optional list of location IDs
        is_forecast: Whether to structure the response as a forecast
        success: Whether to generate a successful response
        error_type: Type of error to simulate if success is False
    
    Returns:
        Mock weather API response in JSON format
    """
    # Set default values if not provided
    start_date = start_date or SAMPLE_START_DATE
    end_date = end_date or SAMPLE_END_DATE
    locations = locations or DEFAULT_LOCATIONS
    
    # If not successful, generate an error response
    if not success:
        return generate_error_response(error_type or random.choice(ERROR_TYPES), 'weather')
    
    # Get sample weather data
    weather_df = get_sample_weather_data(start_date, end_date, locations)
    
    # Prepare response with data for each location
    response = {
        "status": WEATHER_API_SUCCESS_STATUS,
        "timestamp": datetime.now().isoformat(),
        "data": []
    }
    
    for location in locations:
        # Filter data for this location
        location_data = weather_df[weather_df['location_id'] == location]
        if location_data.empty:
            continue
            
        # Convert DataFrame to list of dictionaries
        weather_records = location_data.to_dict('records')
        
        # Format the data according to weather API structure
        formatted_data = format_weather_data(weather_records, location, is_forecast)
        response["data"].append(formatted_data)
    
    return response


def generate_error_response(error_type: str, api_type: str) -> Dict[str, Any]:
    """
    Generates a mock error response for API testing.
    
    Args:
        error_type: Type of error to simulate
        api_type: Type of API (ercot, weather, grid)
    
    Returns:
        Mock error response in JSON format
    """
    # Base error response
    error_response = {
        "status": ERCOT_API_ERROR_STATUS if api_type == 'ercot' or api_type == 'grid' else WEATHER_API_ERROR_STATUS,
        "timestamp": datetime.now().isoformat(),
        "error": {}
    }
    
    # Add specific error details based on error_type
    if error_type == 'connection':
        error_response["error"] = {
            "code": "ERR_NETWORK",
            "message": f"Network connection error while accessing {api_type.upper()} API",
            "details": "Could not establish connection to the server"
        }
    elif error_type == 'rate_limit':
        error_response["error"] = {
            "code": "ERR_RATE_LIMIT",
            "message": f"Rate limit exceeded for {api_type.upper()} API",
            "details": "Too many requests in the last minute, please try again later"
        }
    elif error_type == 'data_format':
        error_response["error"] = {
            "code": "ERR_DATA_FORMAT",
            "message": f"Invalid data format in {api_type.upper()} API request",
            "details": "The request parameters are not in the expected format"
        }
    elif error_type == 'missing_data':
        error_response["error"] = {
            "code": "ERR_MISSING_DATA",
            "message": f"Missing data in {api_type.upper()} API response",
            "details": "The requested data is not available for the specified parameters"
        }
    elif error_type == 'authentication':
        error_response["error"] = {
            "code": "ERR_AUTH",
            "message": f"Authentication failed for {api_type.upper()} API",
            "details": "Invalid API key or credentials"
        }
    else:
        error_response["error"] = {
            "code": "ERR_UNKNOWN",
            "message": f"Unknown error occurred in {api_type.upper()} API",
            "details": "An unexpected error occurred"
        }
    
    return error_response


def format_rtlmp_data(data_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Formats RTLMP data into the structure expected by the ERCOT API response.
    
    Args:
        data_records: List of RTLMP data records
    
    Returns:
        Formatted RTLMP data structure
    """
    response = {
        "status": ERCOT_API_SUCCESS_STATUS,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "data_type": "RTLMP",
            "version": "1.0",
            "units": "$/MWh"
        },
        "data": []
    }
    
    for record in data_records:
        formatted_record = {
            "settlement_point": record["node_id"],
            "timestamp": record["timestamp"].isoformat(),
            "price_components": {
                "total": record["price"],
                "congestion": record["congestion_price"],
                "loss": record["loss_price"],
                "energy": record["energy_price"]
            }
        }
        response["data"].append(formatted_record)
    
    return response


def format_grid_conditions_data(data_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Formats grid condition data into the structure expected by the ERCOT API response.
    
    Args:
        data_records: List of grid condition data records
    
    Returns:
        Formatted grid conditions data structure
    """
    response = {
        "status": ERCOT_API_SUCCESS_STATUS,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "data_type": "GRID_CONDITIONS",
            "version": "1.0",
            "units": {
                "load": "MW",
                "capacity": "MW",
                "generation": "MW"
            }
        },
        "data": []
    }
    
    for record in data_records:
        formatted_record = {
            "timestamp": record["timestamp"].isoformat(),
            "grid_metrics": {
                "total_load": record["total_load"],
                "available_capacity": record["available_capacity"],
                "generation": {
                    "wind": record["wind_generation"],
                    "solar": record["solar_generation"]
                }
            }
        }
        response["data"].append(formatted_record)
    
    return response


def format_weather_data(
    data_records: List[Dict[str, Any]],
    location_id: str,
    is_forecast: bool
) -> Dict[str, Any]:
    """
    Formats weather data into the structure expected by the weather API response.
    
    Args:
        data_records: List of weather data records
        location_id: Location ID for the weather data
        is_forecast: Whether to structure the data as a forecast
    
    Returns:
        Formatted weather data structure
    """
    location_metadata = {
        "id": location_id,
        "name": location_id.replace("_", " ").title(),
        "type": "Region"
    }
    
    formatted_data = {
        "location": location_metadata,
        "data_type": "forecast" if is_forecast else "historical",
        "units": {
            "temperature": "C",
            "wind_speed": "m/s",
            "solar_irradiance": "W/m2",
            "humidity": "%"
        }
    }
    
    # Different structure for historical vs forecast data
    if is_forecast:
        # Group by date for forecast
        forecast_days = {}
        for record in data_records:
            date_str = record["timestamp"].date().isoformat()
            hour = record["timestamp"].hour
            
            if date_str not in forecast_days:
                forecast_days[date_str] = {
                    "date": date_str,
                    "hourly_forecast": []
                }
            
            hour_data = {
                "hour": hour,
                "temperature": record["temperature"],
                "wind_speed": record["wind_speed"],
                "solar_irradiance": record["solar_irradiance"],
                "humidity": record["humidity"]
            }
            forecast_days[date_str]["hourly_forecast"].append(hour_data)
        
        formatted_data["forecast"] = list(forecast_days.values())
    else:
        # List all records for historical data
        historical_records = []
        for record in data_records:
            hist_record = {
                "timestamp": record["timestamp"].isoformat(),
                "measurements": {
                    "temperature": record["temperature"],
                    "wind_speed": record["wind_speed"],
                    "solar_irradiance": record["solar_irradiance"],
                    "humidity": record["humidity"]
                }
            }
            historical_records.append(hist_record)
        
        formatted_data["historical"] = historical_records
    
    return formatted_data