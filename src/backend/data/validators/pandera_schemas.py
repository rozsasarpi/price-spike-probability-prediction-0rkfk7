"""
Pandera schema definitions for validating DataFrame structures in the ERCOT RTLMP spike prediction system.

This module provides strongly-typed schema implementations using Pandera to enforce data structure,
type constraints, and validation rules for RTLMP data, weather data, grid condition data,
feature data, and forecast data.
"""

import pandas as pd  # version 2.0+
import pandera as pa  # version 0.15+
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

from ...utils.type_definitions import (
    RTLMPDataDict, WeatherDataDict, GridConditionDict, FeatureDict, ForecastResultDict
)
from ...utils.logging import get_logger
from ...utils.validation import validate_dataframe_schema
from ...utils.error_handling import DataFormatError

# Set up logger
logger = get_logger(__name__)

# Define value ranges for different data types
RTLMP_VALUE_RANGES = {
    'price': (-50.0, 5000.0),  # in $/MWh
    'congestion_price': (-1000.0, 1000.0),  # in $/MWh
    'loss_price': (-100.0, 100.0),  # in $/MWh
    'energy_price': (-50.0, 5000.0)  # in $/MWh
}

WEATHER_VALUE_RANGES = {
    'temperature': (-20.0, 50.0),  # in Celsius
    'wind_speed': (0.0, 50.0),  # in m/s
    'solar_irradiance': (0.0, 1500.0),  # in W/m²
    'humidity': (0.0, 100.0),  # in %
    'cloud_cover': (0.0, 100.0)  # in %
}

GRID_VALUE_RANGES = {
    'total_load': (0.0, 100000.0),  # in MW
    'available_capacity': (0.0, 120000.0),  # in MW
    'wind_generation': (0.0, 30000.0),  # in MW
    'solar_generation': (0.0, 10000.0),  # in MW
    'reserve_margin': (-10000.0, 50000.0)  # in MW
}


@pa.schema_model
class RTLMPSchema:
    """Pandera schema for RTLMP data validation."""
    
    timestamp: pa.Field[datetime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Time at which the price was recorded"
    )
    
    node_id: pa.Field[str] = pa.Field(
        nullable=False,
        description="Identifier of the ERCOT node"
    )
    
    price: pa.Field[float] = pa.Field(
        gt=RTLMP_VALUE_RANGES['price'][0],
        lt=RTLMP_VALUE_RANGES['price'][1],
        nullable=False,
        description="Real-Time LMP in $/MWh"
    )
    
    congestion_price: pa.Field[float] = pa.Field(
        gt=RTLMP_VALUE_RANGES['congestion_price'][0],
        lt=RTLMP_VALUE_RANGES['congestion_price'][1],
        nullable=False,
        description="Congestion component of LMP in $/MWh"
    )
    
    loss_price: pa.Field[float] = pa.Field(
        gt=RTLMP_VALUE_RANGES['loss_price'][0],
        lt=RTLMP_VALUE_RANGES['loss_price'][1],
        nullable=False,
        description="Loss component of LMP in $/MWh"
    )
    
    energy_price: pa.Field[float] = pa.Field(
        gt=RTLMP_VALUE_RANGES['energy_price'][0],
        lt=RTLMP_VALUE_RANGES['energy_price'][1],
        nullable=False,
        description="Energy component of LMP in $/MWh"
    )
    
    @pa.check
    def price_consistency(cls, price: pd.Series, energy_price: pd.Series, 
                          congestion_price: pd.Series, loss_price: pd.Series) -> pd.Series:
        """
        Validates that price equals the sum of energy_price, congestion_price, and loss_price.
        
        Returns:
            Boolean series indicating if the constraint is satisfied
        """
        # Allow for small floating-point differences (0.001 $/MWh)
        return (price - (energy_price + congestion_price + loss_price)).abs() < 0.001


@pa.schema_model
class WeatherSchema:
    """Pandera schema for weather data validation."""
    
    timestamp: pa.Field[datetime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Time at which the weather data was recorded or forecasted"
    )
    
    location_id: pa.Field[str] = pa.Field(
        nullable=False,
        description="Identifier of the location"
    )
    
    temperature: pa.Field[float] = pa.Field(
        gt=WEATHER_VALUE_RANGES['temperature'][0],
        lt=WEATHER_VALUE_RANGES['temperature'][1],
        nullable=False,
        description="Temperature in Celsius"
    )
    
    wind_speed: pa.Field[float] = pa.Field(
        gt=WEATHER_VALUE_RANGES['wind_speed'][0],
        lt=WEATHER_VALUE_RANGES['wind_speed'][1],
        nullable=False,
        description="Wind speed in m/s"
    )
    
    solar_irradiance: pa.Field[float] = pa.Field(
        gt=WEATHER_VALUE_RANGES['solar_irradiance'][0],
        lt=WEATHER_VALUE_RANGES['solar_irradiance'][1],
        nullable=False,
        description="Solar irradiance in W/m²"
    )
    
    humidity: pa.Field[float] = pa.Field(
        gt=WEATHER_VALUE_RANGES['humidity'][0],
        lt=WEATHER_VALUE_RANGES['humidity'][1],
        nullable=False,
        description="Humidity percentage"
    )
    
    cloud_cover: pa.Field[float] = pa.Field(
        gt=WEATHER_VALUE_RANGES['cloud_cover'][0],
        lt=WEATHER_VALUE_RANGES['cloud_cover'][1],
        nullable=True,  # Can be null if not available
        description="Cloud cover percentage"
    )


@pa.schema_model
class GridConditionSchema:
    """Pandera schema for grid condition data validation."""
    
    timestamp: pa.Field[datetime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Time at which the grid condition was recorded or forecasted"
    )
    
    total_load: pa.Field[float] = pa.Field(
        gt=GRID_VALUE_RANGES['total_load'][0],
        lt=GRID_VALUE_RANGES['total_load'][1],
        nullable=False,
        description="Total load in MW"
    )
    
    available_capacity: pa.Field[float] = pa.Field(
        gt=GRID_VALUE_RANGES['available_capacity'][0],
        lt=GRID_VALUE_RANGES['available_capacity'][1],
        nullable=False,
        description="Available generation capacity in MW"
    )
    
    wind_generation: pa.Field[float] = pa.Field(
        gt=GRID_VALUE_RANGES['wind_generation'][0],
        lt=GRID_VALUE_RANGES['wind_generation'][1],
        nullable=False,
        description="Wind generation in MW"
    )
    
    solar_generation: pa.Field[float] = pa.Field(
        gt=GRID_VALUE_RANGES['solar_generation'][0],
        lt=GRID_VALUE_RANGES['solar_generation'][1],
        nullable=False,
        description="Solar generation in MW"
    )
    
    reserve_margin: pa.Field[float] = pa.Field(
        gt=GRID_VALUE_RANGES['reserve_margin'][0],
        lt=GRID_VALUE_RANGES['reserve_margin'][1],
        nullable=True,  # Can be null if not calculated
        description="Reserve margin in MW"
    )
    
    @pa.check
    def capacity_consistency(cls, available_capacity: pd.Series, total_load: pd.Series) -> pd.Series:
        """
        Validates that available_capacity is greater than or equal to total_load.
        
        Returns:
            Boolean series indicating if the constraint is satisfied
        """
        return available_capacity >= total_load


@pa.schema_model
class FeatureSchema:
    """Pandera schema for feature data validation."""
    
    feature_id: pa.Field[str] = pa.Field(
        nullable=False,
        description="Unique identifier for the feature"
    )
    
    feature_name: pa.Field[str] = pa.Field(
        nullable=False,
        description="Human-readable name of the feature"
    )
    
    feature_group: pa.Field[str] = pa.Field(
        isin=['time', 'statistical', 'weather', 'market'],
        nullable=False,
        description="Group categorization of the feature"
    )
    
    data_type: pa.Field[str] = pa.Field(
        isin=['numeric', 'categorical', 'boolean', 'datetime'],
        nullable=False,
        description="Data type of the feature"
    )


@pa.schema_model
class ForecastSchema:
    """Pandera schema for forecast data validation."""
    
    forecast_timestamp: pa.Field[datetime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Time at which the forecast was generated"
    )
    
    target_timestamp: pa.Field[datetime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Target time for which the forecast is made"
    )
    
    threshold_value: pa.Field[float] = pa.Field(
        gt=0.0,  # Must be positive
        nullable=False,
        description="Price threshold for spike definition in $/MWh"
    )
    
    spike_probability: pa.Field[float] = pa.Field(
        ge=0.0,
        le=1.0,
        nullable=False,
        description="Probability of price exceeding threshold"
    )
    
    confidence_interval_lower: pa.Field[float] = pa.Field(
        ge=0.0,
        le=1.0,
        nullable=True,  # Can be null if confidence intervals are not provided
        description="Lower bound of confidence interval"
    )
    
    confidence_interval_upper: pa.Field[float] = pa.Field(
        ge=0.0,
        le=1.0,
        nullable=True,  # Can be null if confidence intervals are not provided
        description="Upper bound of confidence interval"
    )
    
    model_version: pa.Field[str] = pa.Field(
        nullable=False,
        description="Version of the model used for prediction"
    )
    
    node_id: pa.Field[str] = pa.Field(
        nullable=False,
        description="Identifier of the ERCOT node"
    )
    
    @pa.check
    def confidence_interval_consistency(
        cls, spike_probability: pd.Series, 
        confidence_interval_lower: pd.Series,
        confidence_interval_upper: pd.Series
    ) -> pd.Series:
        """
        Validates that confidence intervals contain the probability value when present.
        
        Returns:
            Boolean series indicating if the constraint is satisfied
        """
        # Create a mask for rows where both intervals are present (not null)
        has_intervals = confidence_interval_lower.notna() & confidence_interval_upper.notna()
        
        # For rows with intervals, check if: lower <= probability <= upper
        valid_intervals = pd.Series(True, index=spike_probability.index)
        valid_intervals[has_intervals] = (
            (confidence_interval_lower[has_intervals] <= spike_probability[has_intervals]) & 
            (confidence_interval_upper[has_intervals] >= spike_probability[has_intervals])
        )
        
        return valid_intervals


def validate_with_rtlmp_schema(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Validates a DataFrame against the RTLMP schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Validated DataFrame if successful
        
    Raises:
        DataFormatError: If validation fails and strict is True
    """
    try:
        return RTLMPSchema.validate(df, lazy=not strict)
    except pa.errors.SchemaError as e:
        if strict:
            logger.error(f"RTLMP schema validation failed: {str(e)}")
            raise DataFormatError(f"RTLMP schema validation failed: {str(e)}")
        else:
            logger.warning(f"RTLMP schema validation issues: {str(e)}")
            return df


def validate_with_weather_schema(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Validates a DataFrame against the Weather schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Validated DataFrame if successful
        
    Raises:
        DataFormatError: If validation fails and strict is True
    """
    try:
        return WeatherSchema.validate(df, lazy=not strict)
    except pa.errors.SchemaError as e:
        if strict:
            logger.error(f"Weather schema validation failed: {str(e)}")
            raise DataFormatError(f"Weather schema validation failed: {str(e)}")
        else:
            logger.warning(f"Weather schema validation issues: {str(e)}")
            return df


def validate_with_grid_condition_schema(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Validates a DataFrame against the GridCondition schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Validated DataFrame if successful
        
    Raises:
        DataFormatError: If validation fails and strict is True
    """
    try:
        return GridConditionSchema.validate(df, lazy=not strict)
    except pa.errors.SchemaError as e:
        if strict:
            logger.error(f"Grid condition schema validation failed: {str(e)}")
            raise DataFormatError(f"Grid condition schema validation failed: {str(e)}")
        else:
            logger.warning(f"Grid condition schema validation issues: {str(e)}")
            return df


def validate_with_feature_schema(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Validates a DataFrame against the Feature schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Validated DataFrame if successful
        
    Raises:
        DataFormatError: If validation fails and strict is True
    """
    try:
        return FeatureSchema.validate(df, lazy=not strict)
    except pa.errors.SchemaError as e:
        if strict:
            logger.error(f"Feature schema validation failed: {str(e)}")
            raise DataFormatError(f"Feature schema validation failed: {str(e)}")
        else:
            logger.warning(f"Feature schema validation issues: {str(e)}")
            return df


def validate_with_forecast_schema(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Validates a DataFrame against the Forecast schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Validated DataFrame if successful
        
    Raises:
        DataFormatError: If validation fails and strict is True
    """
    try:
        return ForecastSchema.validate(df, lazy=not strict)
    except pa.errors.SchemaError as e:
        if strict:
            logger.error(f"Forecast schema validation failed: {str(e)}")
            raise DataFormatError(f"Forecast schema validation failed: {str(e)}")
        else:
            logger.warning(f"Forecast schema validation issues: {str(e)}")
            return df


def get_schema_for_data_type(data_type: str) -> pa.SchemaModel:
    """
    Returns the appropriate schema class for a given data type.
    
    Args:
        data_type: Name of the data type
    
    Returns:
        Schema class for the specified data type
        
    Raises:
        ValueError: If no matching schema is found
    """
    schema_map = {
        'rtlmp': RTLMPSchema,
        'weather': WeatherSchema,
        'grid_condition': GridConditionSchema,
        'feature': FeatureSchema,
        'forecast': ForecastSchema
    }
    
    if data_type not in schema_map:
        valid_types = list(schema_map.keys())
        raise ValueError(f"No schema found for data type '{data_type}'. Valid types: {valid_types}")
    
    return schema_map[data_type]


class SchemaRegistry:
    """
    Registry class for managing and accessing schema definitions.
    """
    
    def __init__(self):
        """
        Initialize the SchemaRegistry with default schemas.
        """
        self._schemas = {
            'rtlmp': RTLMPSchema,
            'weather': WeatherSchema,
            'grid_condition': GridConditionSchema,
            'feature': FeatureSchema,
            'forecast': ForecastSchema
        }
        self._logger = get_logger(__name__)
    
    def register_schema(self, schema_name: str, schema_class: pa.SchemaModel) -> None:
        """
        Registers a schema class with the registry.
        
        Args:
            schema_name: Name to register the schema under
            schema_class: Pandera SchemaModel class
        """
        self._schemas[schema_name] = schema_class
        self._logger.debug(f"Registered schema '{schema_name}'")
    
    def get_schema(self, schema_name: str) -> pa.SchemaModel:
        """
        Retrieves a schema class by name.
        
        Args:
            schema_name: Name of the schema to retrieve
            
        Returns:
            SchemaModel class
            
        Raises:
            KeyError: If schema_name is not found in the registry
        """
        if schema_name not in self._schemas:
            self._logger.error(f"Schema '{schema_name}' not found in registry")
            raise KeyError(f"Schema '{schema_name}' not found in registry")
        
        return self._schemas[schema_name]
    
    def validate(self, df: pd.DataFrame, schema_name: str, strict: bool = True) -> pd.DataFrame:
        """
        Validates data against a named schema.
        
        Args:
            df: DataFrame to validate
            schema_name: Name of the schema to validate against
            strict: If True, raises an error for validation failures; otherwise, logs warnings
            
        Returns:
            Validated DataFrame if successful
            
        Raises:
            DataFormatError: If validation fails and strict is True
            KeyError: If schema_name is not found in the registry
        """
        schema_class = self.get_schema(schema_name)
        
        try:
            return schema_class.validate(df, lazy=not strict)
        except pa.errors.SchemaError as e:
            if strict:
                self._logger.error(f"Schema validation failed for '{schema_name}': {str(e)}")
                raise DataFormatError(f"Schema validation failed for '{schema_name}': {str(e)}")
            else:
                self._logger.warning(f"Schema validation issues for '{schema_name}': {str(e)}")
                return df