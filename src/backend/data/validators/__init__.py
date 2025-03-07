"""
Initializes the validators module and exposes key validation functionality for the ERCOT RTLMP spike prediction system.
This module serves as the entry point for data validation, providing access to JSON schema validation,
Pandera DataFrame validation, and data quality assessment tools.
"""

from typing import Any

# Import JSON schema validation functionality
from .schemas import (
    RTLMP_SCHEMA, WEATHER_SCHEMA, GRID_CONDITION_SCHEMA, FEATURE_SCHEMA, FORECAST_SCHEMA,
    validate_json_schema, validate_rtlmp_json, validate_weather_json, validate_grid_condition_json,
    validate_feature_json, validate_forecast_json, JSONSchemaValidator, ValidationError,
    convert_to_json_schema, datetime_to_str, str_to_datetime
)

# Import Pandera schema validation functionality
from .pandera_schemas import (
    RTLMPSchema, WeatherSchema, GridConditionSchema, FeatureSchema, ForecastSchema,
    SchemaRegistry, validate_with_rtlmp_schema, validate_with_weather_schema,
    validate_with_grid_condition_schema, validate_with_feature_schema, validate_with_forecast_schema,
    get_schema_for_data_type, RTLMP_VALUE_RANGES, WEATHER_VALUE_RANGES, GRID_VALUE_RANGES
)

# Import data quality assessment functionality
from .data_quality import (
    check_rtlmp_data_quality, check_weather_data_quality, check_grid_condition_data_quality,
    detect_rtlmp_anomalies, check_data_consistency, calculate_data_quality_metrics,
    impute_missing_values, validate_forecast_quality, DataQualityChecker, QualityReport,
    DEFAULT_COMPLETENESS_THRESHOLD, DEFAULT_OUTLIER_THRESHOLD, DEFAULT_TEMPORAL_FREQUENCY
)

# Import logger
from ...utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)


def validate_data(data: Any, data_type: str, strict: bool = True) -> Any:
    """
    Unified function to validate data using appropriate schema based on data type.
    
    Args:
        data: Data to validate (can be DataFrame or dictionary/JSON)
        data_type: Type of data ('rtlmp', 'weather', 'grid_condition', 'feature', 'forecast')
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Validated data if successful
        
    Raises:
        DataFormatError: If validation fails and strict is True
        ValueError: If data_type is not supported
    """
    logger.debug(f"Validating {data_type} data, strict mode: {strict}")
    
    # Determine if data is a DataFrame or dictionary/JSON
    import pandas as pd
    
    if isinstance(data, pd.DataFrame):
        # Use Pandera schema validation for DataFrames
        if data_type == 'rtlmp':
            return validate_with_rtlmp_schema(data, strict=strict)
        elif data_type == 'weather':
            return validate_with_weather_schema(data, strict=strict)
        elif data_type == 'grid_condition':
            return validate_with_grid_condition_schema(data, strict=strict)
        elif data_type == 'feature':
            return validate_with_feature_schema(data, strict=strict)
        elif data_type == 'forecast':
            return validate_with_forecast_schema(data, strict=strict)
        else:
            msg = f"Unsupported data type: {data_type}"
            logger.error(msg)
            if strict:
                raise ValueError(msg)
            return data
    
    else:
        # Assume dictionary/JSON format and use JSON schema validation
        if data_type == 'rtlmp':
            valid = validate_rtlmp_json(data, raise_error=strict)
        elif data_type == 'weather':
            valid = validate_weather_json(data, raise_error=strict)
        elif data_type == 'grid_condition':
            valid = validate_grid_condition_json(data, raise_error=strict)
        elif data_type == 'feature':
            valid = validate_feature_json(data, raise_error=strict)
        elif data_type == 'forecast':
            valid = validate_forecast_json(data, raise_error=strict)
        else:
            msg = f"Unsupported data type: {data_type}"
            logger.error(msg)
            if strict:
                raise ValueError(msg)
            return data
        
        if valid or not strict:
            return data
        
        # If we get here, validation failed but no exception was raised
        # (this shouldn't happen with strict=True, but adding as a safeguard)
        return None