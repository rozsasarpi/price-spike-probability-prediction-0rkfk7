"""
Defines JSON schema definitions for validating data structures in the ERCOT RTLMP spike prediction system.

This module provides schema implementations using JSON Schema to define data structure,
type constraints, and validation rules for RTLMP data, weather data, grid condition data,
feature data, and forecast data. These schemas complement the Pandera schemas by providing
validation for non-DataFrame data formats.
"""

import json
import jsonschema  # version 4.17+
from typing import Dict, List, Any, Optional, Union, Callable
import datetime

from ...utils.type_definitions import (
    RTLMPDataDict, WeatherDataDict, GridConditionDict, FeatureDict, ForecastDict
)
from ...utils.logging import get_logger
from ...utils.validation import validate_dataframe_schema

# Set up logger
logger = get_logger(__name__)

# JSON Schema definitions
RTLMP_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "node_id": {"type": "string"},
        "price": {"type": "number"},
        "congestion_price": {"type": "number"},
        "loss_price": {"type": "number"},
        "energy_price": {"type": "number"}
    },
    "required": ["timestamp", "node_id", "price", "congestion_price", "loss_price", "energy_price"],
    "additionalProperties": False
}

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "location_id": {"type": "string"},
        "temperature": {"type": "number"},
        "wind_speed": {"type": "number"},
        "solar_irradiance": {"type": "number"},
        "humidity": {"type": "number"},
        "cloud_cover": {"type": "number", "nullable": True}
    },
    "required": ["timestamp", "location_id", "temperature", "wind_speed", "solar_irradiance", "humidity"],
    "additionalProperties": False
}

GRID_CONDITION_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "total_load": {"type": "number"},
        "available_capacity": {"type": "number"},
        "wind_generation": {"type": "number"},
        "solar_generation": {"type": "number"},
        "reserve_margin": {"type": "number", "nullable": True}
    },
    "required": ["timestamp", "total_load", "available_capacity", "wind_generation", "solar_generation"],
    "additionalProperties": False
}

FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "feature_id": {"type": "string"},
        "feature_name": {"type": "string"},
        "feature_group": {"type": "string"},
        "data_type": {"type": "string"},
        "metadata": {"type": "object", "nullable": True},
        "dependencies": {"type": "array", "items": {"type": "string"}, "nullable": True}
    },
    "required": ["feature_id", "feature_name", "feature_group", "data_type"],
    "additionalProperties": False
}

FORECAST_SCHEMA = {
    "type": "object",
    "properties": {
        "forecast_timestamp": {"type": "string", "format": "date-time"},
        "target_timestamp": {"type": "string", "format": "date-time"},
        "threshold_value": {"type": "number"},
        "spike_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence_interval_lower": {"type": "number", "minimum": 0, "maximum": 1, "nullable": True},
        "confidence_interval_upper": {"type": "number", "minimum": 0, "maximum": 1, "nullable": True},
        "model_version": {"type": "string"},
        "node_id": {"type": "string"}
    },
    "required": ["forecast_timestamp", "target_timestamp", "threshold_value", "spike_probability", "model_version", "node_id"],
    "additionalProperties": False
}

def validate_json_schema(data: dict, schema: dict, raise_error: bool = True) -> bool:
    """
    Validates a JSON object against a JSON schema
    
    Args:
        data: Data to validate
        schema: JSON schema to validate against
        raise_error: Whether to raise an error on validation failure
        
    Returns:
        True if validation succeeds, False otherwise
        
    Raises:
        ValidationError: If validation fails and raise_error is True
    """
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        if raise_error:
            raise ValidationError(
                message=f"JSON schema validation failed: {str(e)}",
                errors=e,
                data=data,
                schema=schema
            )
        logger.warning(f"JSON schema validation failed: {str(e)}")
        return False

def validate_rtlmp_json(data: dict, raise_error: bool = True) -> bool:
    """
    Validates RTLMP data in JSON format against the RTLMP schema
    
    Args:
        data: RTLMP data to validate
        raise_error: Whether to raise an error on validation failure
        
    Returns:
        True if validation succeeds, False otherwise
    """
    return validate_json_schema(data, RTLMP_SCHEMA, raise_error)

def validate_weather_json(data: dict, raise_error: bool = True) -> bool:
    """
    Validates weather data in JSON format against the weather schema
    
    Args:
        data: Weather data to validate
        raise_error: Whether to raise an error on validation failure
        
    Returns:
        True if validation succeeds, False otherwise
    """
    return validate_json_schema(data, WEATHER_SCHEMA, raise_error)

def validate_grid_condition_json(data: dict, raise_error: bool = True) -> bool:
    """
    Validates grid condition data in JSON format against the grid condition schema
    
    Args:
        data: Grid condition data to validate
        raise_error: Whether to raise an error on validation failure
        
    Returns:
        True if validation succeeds, False otherwise
    """
    return validate_json_schema(data, GRID_CONDITION_SCHEMA, raise_error)

def validate_feature_json(data: dict, raise_error: bool = True) -> bool:
    """
    Validates feature data in JSON format against the feature schema
    
    Args:
        data: Feature data to validate
        raise_error: Whether to raise an error on validation failure
        
    Returns:
        True if validation succeeds, False otherwise
    """
    return validate_json_schema(data, FEATURE_SCHEMA, raise_error)

def validate_forecast_json(data: dict, raise_error: bool = True) -> bool:
    """
    Validates forecast data in JSON format against the forecast schema
    
    Args:
        data: Forecast data to validate
        raise_error: Whether to raise an error on validation failure
        
    Returns:
        True if validation succeeds, False otherwise
    """
    return validate_json_schema(data, FORECAST_SCHEMA, raise_error)

def convert_to_json_schema(type_hint: Any) -> dict:
    """
    Converts a Python type hint or TypedDict to a JSON schema
    
    Args:
        type_hint: Python type hint to convert
        
    Returns:
        JSON schema representation of the type hint
    """
    # This is a simplified implementation
    # A full implementation would need to handle all typing constructs
    
    schema = {"type": "object", "properties": {}, "required": []}
    
    # Handle TypedDict
    if hasattr(type_hint, "__annotations__"):
        for field_name, field_type in type_hint.__annotations__.items():
            # Map Python types to JSON schema types
            if field_type == str:
                schema["properties"][field_name] = {"type": "string"}
            elif field_type == int:
                schema["properties"][field_name] = {"type": "integer"}
            elif field_type == float:
                schema["properties"][field_name] = {"type": "number"}
            elif field_type == bool:
                schema["properties"][field_name] = {"type": "boolean"}
            elif field_type == datetime.datetime:
                schema["properties"][field_name] = {"type": "string", "format": "date-time"}
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                item_type = field_type.__args__[0]
                schema["properties"][field_name] = {
                    "type": "array",
                    "items": convert_to_json_schema(item_type)
                }
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is dict:
                schema["properties"][field_name] = {"type": "object"}
            else:
                # Default to object for complex types
                schema["properties"][field_name] = {"type": "object"}
            
            # Add to required fields if not Optional
            if not (hasattr(field_type, "__origin__") and field_type.__origin__ is Union and type(None) in field_type.__args__):
                schema["required"].append(field_name)
    
    return schema

def datetime_to_str(obj: Any) -> Any:
    """
    Converts datetime objects to ISO format strings for JSON serialization
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with datetime objects converted to strings
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: datetime_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [datetime_to_str(item) for item in obj]
    return obj

def str_to_datetime(obj: Any) -> Any:
    """
    Converts ISO format strings to datetime objects for JSON deserialization
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with ISO format strings converted to datetime objects
    """
    if isinstance(obj, str):
        try:
            return datetime.datetime.fromisoformat(obj)
        except ValueError:
            return obj
    elif isinstance(obj, dict):
        return {k: str_to_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [str_to_datetime(item) for item in obj]
    return obj

class ValidationError(Exception):
    """
    Custom exception for JSON schema validation errors
    """
    
    def __init__(self, message: str, errors: dict, data: dict, schema: dict):
        """
        Initialize the ValidationError with error details
        
        Args:
            message: Error message
            errors: Validation errors
            data: Data that failed validation
            schema: Schema that was used for validation
        """
        super().__init__(message)
        self.message = message
        self.errors = errors
        self.data = data
        self.schema = schema
    
    def to_dict(self) -> dict:
        """
        Converts the validation error to a dictionary
        
        Returns:
            Dictionary representation of the validation error
        """
        return {
            "message": self.message,
            "errors": self.errors,
            "schema": self.schema
        }

class JSONSchemaValidator:
    """
    Class that provides methods for JSON schema validation
    """
    
    def __init__(self):
        """
        Initialize the JSONSchemaValidator with default schemas
        """
        self._schemas = {
            "rtlmp": RTLMP_SCHEMA,
            "weather": WEATHER_SCHEMA,
            "grid_condition": GRID_CONDITION_SCHEMA,
            "feature": FEATURE_SCHEMA,
            "forecast": FORECAST_SCHEMA
        }
        self._logger = get_logger(__name__)
    
    def add_schema(self, schema_name: str, schema: dict) -> None:
        """
        Adds a custom schema for validation
        
        Args:
            schema_name: Name of the schema
            schema: JSON schema
        """
        self._schemas[schema_name] = schema
        self._logger.info(f"Added new JSON schema: {schema_name}")
    
    def validate(self, data: dict, schema_name: str, raise_error: bool = True) -> bool:
        """
        Validates data against a named schema
        
        Args:
            data: Data to validate
            schema_name: Name of the schema to validate against
            raise_error: Whether to raise an error on validation failure
            
        Returns:
            True if validation succeeds, False otherwise
        """
        if schema_name not in self._schemas:
            self._logger.error(f"Schema {schema_name} not found")
            return False
        
        schema = self._schemas[schema_name]
        return validate_json_schema(data, schema, raise_error)
    
    def validate_batch(self, data_list: list, schema_name: str, raise_error: bool = True) -> dict:
        """
        Validates a list of data objects against a named schema
        
        Args:
            data_list: List of data objects to validate
            schema_name: Name of the schema to validate against
            raise_error: Whether to raise an error on validation failure
            
        Returns:
            Dictionary with validation results for each item
        """
        results = {}
        failed_count = 0
        
        for i, data in enumerate(data_list):
            try:
                is_valid = self.validate(data, schema_name, raise_error=False)
                results[i] = is_valid
                if not is_valid:
                    failed_count += 1
            except Exception as e:
                self._logger.error(f"Error validating item {i}: {str(e)}")
                results[i] = False
                failed_count += 1
        
        if failed_count > 0 and raise_error:
            raise ValidationError(
                message=f"Batch validation failed: {failed_count} of {len(data_list)} items failed validation",
                errors={"failed_count": failed_count, "results": results},
                data={"count": len(data_list)},
                schema=self._schemas[schema_name]
            )
        
        return results
    
    def get_schema(self, schema_name: str) -> dict:
        """
        Retrieves a schema by name
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            The requested schema
        """
        if schema_name not in self._schemas:
            self._logger.error(f"Schema {schema_name} not found")
            return None
        
        return self._schemas[schema_name]