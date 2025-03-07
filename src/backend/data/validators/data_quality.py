"""
Comprehensive data quality validation for the ERCOT RTLMP spike prediction system.

This module provides functions and classes to assess data quality beyond basic schema
validation, including completeness checks, outlier detection, temporal consistency validation,
and statistical quality metrics for RTLMP data, weather data, and grid condition data.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from typing import Dict, List, Optional, Union, Tuple, Any
import datetime
from scipy.stats import zscore  # version 1.10+

from .pandera_schemas import (
    RTLMPSchema, WeatherSchema, GridConditionSchema, ForecastSchema,
    RTLMP_VALUE_RANGES, WEATHER_VALUE_RANGES, GRID_VALUE_RANGES
)
from ...utils.validation import (
    validate_dataframe_schema, validate_data_completeness, validate_value_ranges,
    validate_temporal_consistency, detect_outliers
)
from ...utils.type_definitions import DataFrameType, SeriesType, RTLMPDataDict, WeatherDataDict, GridConditionDict
from ...utils.logging import get_logger
from ...utils.error_handling import DataFormatError, MissingDataError, handle_errors

# Set up logger
logger = get_logger(__name__)

# Default thresholds
DEFAULT_COMPLETENESS_THRESHOLD = 0.95
DEFAULT_OUTLIER_THRESHOLD = 3.0
DEFAULT_TEMPORAL_FREQUENCY = '5min'

# Required columns for different data types
RTLMP_REQUIRED_COLUMNS = ['timestamp', 'node_id', 'price', 'congestion_price', 'loss_price', 'energy_price']
WEATHER_REQUIRED_COLUMNS = ['timestamp', 'location_id', 'temperature', 'wind_speed', 'solar_irradiance', 'humidity']
GRID_REQUIRED_COLUMNS = ['timestamp', 'total_load', 'available_capacity', 'wind_generation', 'solar_generation']


@handle_errors(exceptions=(DataFormatError, MissingDataError), reraise=True)
def check_rtlmp_data_quality(df: DataFrameType, strict: bool = True) -> dict:
    """
    Performs comprehensive quality checks on RTLMP data.
    
    Args:
        df: DataFrame containing RTLMP data
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Dictionary with quality assessment results, including metrics and issues
        
    Raises:
        DataFormatError: If validation fails and strict is True
        MissingDataError: If required data is missing and strict is True
    """
    logger.debug(f"Checking RTLMP data quality, strict mode: {strict}")
    
    # Check required columns
    if not all(col in df.columns for col in RTLMP_REQUIRED_COLUMNS):
        missing_cols = [col for col in RTLMP_REQUIRED_COLUMNS if col not in df.columns]
        msg = f"Missing required columns in RTLMP data: {missing_cols}"
        logger.error(msg)
        if strict:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Check data completeness
    completeness_result = validate_data_completeness(
        df, threshold=DEFAULT_COMPLETENESS_THRESHOLD, raise_error=strict
    )
    
    # Check value ranges
    value_ranges_result = validate_value_ranges(
        df, value_ranges=RTLMP_VALUE_RANGES, raise_error=strict
    )
    
    # Check temporal consistency
    temporal_result = validate_temporal_consistency(
        df, timestamp_column='timestamp', expected_frequency=DEFAULT_TEMPORAL_FREQUENCY, 
        raise_error=strict
    )
    
    # Detect outliers in price components
    price_outliers = {}
    for col in ['price', 'congestion_price', 'loss_price', 'energy_price']:
        if col in df.columns:
            price_outliers[col] = detect_outliers(
                df[col], method='zscore', threshold=DEFAULT_OUTLIER_THRESHOLD
            ).sum()
    
    # Check price component consistency
    price_consistency = {}
    if all(col in df.columns for col in ['price', 'congestion_price', 'loss_price', 'energy_price']):
        # Allow for small floating-point differences
        diff = df['price'] - (df['energy_price'] + df['congestion_price'] + df['loss_price'])
        inconsistent = (diff.abs() > 0.001)
        price_consistency = {
            "valid": not inconsistent.any(),
            "inconsistent_count": inconsistent.sum(),
            "max_discrepancy": diff.abs().max() if inconsistent.any() else 0.0
        }
    
    # Compile results
    results = {
        "valid": (
            completeness_result.get("valid", False) and 
            value_ranges_result.get("valid", False) and 
            temporal_result.get("valid", False) and 
            price_consistency.get("valid", True)
        ),
        "completeness": completeness_result,
        "value_ranges": value_ranges_result,
        "temporal_consistency": temporal_result,
        "outliers": price_outliers,
        "price_consistency": price_consistency
    }
    
    if not results["valid"]:
        issues = []
        if not completeness_result.get("valid", False):
            issues.append("data completeness below threshold")
        if not value_ranges_result.get("valid", False):
            issues.append("values outside expected ranges")
        if not temporal_result.get("valid", False):
            issues.append("temporal inconsistencies found")
        if not price_consistency.get("valid", True):
            issues.append("price component inconsistencies found")
        
        logger.warning(f"RTLMP data quality issues found: {', '.join(issues)}")
    else:
        logger.info("RTLMP data quality checks passed")
    
    return results


@handle_errors(exceptions=(DataFormatError, MissingDataError), reraise=True)
def check_weather_data_quality(df: DataFrameType, strict: bool = True) -> dict:
    """
    Performs comprehensive quality checks on weather data.
    
    Args:
        df: DataFrame containing weather data
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Dictionary with quality assessment results, including metrics and issues
        
    Raises:
        DataFormatError: If validation fails and strict is True
        MissingDataError: If required data is missing and strict is True
    """
    logger.debug(f"Checking weather data quality, strict mode: {strict}")
    
    # Check required columns
    if not all(col in df.columns for col in WEATHER_REQUIRED_COLUMNS):
        missing_cols = [col for col in WEATHER_REQUIRED_COLUMNS if col not in df.columns]
        msg = f"Missing required columns in weather data: {missing_cols}"
        logger.error(msg)
        if strict:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Check data completeness
    completeness_result = validate_data_completeness(
        df, threshold=DEFAULT_COMPLETENESS_THRESHOLD, raise_error=strict
    )
    
    # Check value ranges
    value_ranges_result = validate_value_ranges(
        df, value_ranges=WEATHER_VALUE_RANGES, raise_error=strict
    )
    
    # Check temporal consistency (weather data typically hourly)
    temporal_result = validate_temporal_consistency(
        df, timestamp_column='timestamp', expected_frequency='1H', 
        raise_error=strict
    )
    
    # Detect outliers in weather metrics
    weather_outliers = {}
    for col in ['temperature', 'wind_speed', 'solar_irradiance', 'humidity']:
        if col in df.columns:
            weather_outliers[col] = detect_outliers(
                df[col], method='zscore', threshold=DEFAULT_OUTLIER_THRESHOLD
            ).sum()
    
    # Check for physically impossible combinations
    physical_consistency = {"valid": True, "issues": []}
    
    # Check: High solar irradiance with high humidity is unlikely
    if 'solar_irradiance' in df.columns and 'humidity' in df.columns:
        high_solar = df['solar_irradiance'] > 800  # W/m²
        high_humidity = df['humidity'] > 90  # %
        impossible_combo = high_solar & high_humidity
        
        if impossible_combo.any():
            physical_consistency["valid"] = False
            physical_consistency["issues"].append("high solar irradiance with high humidity")
            physical_consistency["solar_humidity_inconsistencies"] = impossible_combo.sum()
    
    # Compile results
    results = {
        "valid": (
            completeness_result.get("valid", False) and 
            value_ranges_result.get("valid", False) and 
            temporal_result.get("valid", False) and 
            physical_consistency.get("valid", True)
        ),
        "completeness": completeness_result,
        "value_ranges": value_ranges_result,
        "temporal_consistency": temporal_result,
        "outliers": weather_outliers,
        "physical_consistency": physical_consistency
    }
    
    if not results["valid"]:
        issues = []
        if not completeness_result.get("valid", False):
            issues.append("data completeness below threshold")
        if not value_ranges_result.get("valid", False):
            issues.append("values outside expected ranges")
        if not temporal_result.get("valid", False):
            issues.append("temporal inconsistencies found")
        if not physical_consistency.get("valid", True):
            issues.append("physically impossible combinations found")
        
        logger.warning(f"Weather data quality issues found: {', '.join(issues)}")
    else:
        logger.info("Weather data quality checks passed")
    
    return results


@handle_errors(exceptions=(DataFormatError, MissingDataError), reraise=True)
def check_grid_condition_data_quality(df: DataFrameType, strict: bool = True) -> dict:
    """
    Performs comprehensive quality checks on grid condition data.
    
    Args:
        df: DataFrame containing grid condition data
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Dictionary with quality assessment results, including metrics and issues
        
    Raises:
        DataFormatError: If validation fails and strict is True
        MissingDataError: If required data is missing and strict is True
    """
    logger.debug(f"Checking grid condition data quality, strict mode: {strict}")
    
    # Check required columns
    if not all(col in df.columns for col in GRID_REQUIRED_COLUMNS):
        missing_cols = [col for col in GRID_REQUIRED_COLUMNS if col not in df.columns]
        msg = f"Missing required columns in grid condition data: {missing_cols}"
        logger.error(msg)
        if strict:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Check data completeness
    completeness_result = validate_data_completeness(
        df, threshold=DEFAULT_COMPLETENESS_THRESHOLD, raise_error=strict
    )
    
    # Check value ranges
    value_ranges_result = validate_value_ranges(
        df, value_ranges=GRID_VALUE_RANGES, raise_error=strict
    )
    
    # Check temporal consistency (grid data typically hourly)
    temporal_result = validate_temporal_consistency(
        df, timestamp_column='timestamp', expected_frequency='1H', 
        raise_error=strict
    )
    
    # Detect outliers in grid metrics
    grid_outliers = {}
    for col in ['total_load', 'available_capacity', 'wind_generation', 'solar_generation']:
        if col in df.columns:
            grid_outliers[col] = detect_outliers(
                df[col], method='zscore', threshold=DEFAULT_OUTLIER_THRESHOLD
            ).sum()
    
    # Check grid consistency constraints
    grid_consistency = {"valid": True, "issues": []}
    
    # Check: Available capacity should be >= total load
    if 'available_capacity' in df.columns and 'total_load' in df.columns:
        capacity_violation = df['available_capacity'] < df['total_load']
        
        if capacity_violation.any():
            grid_consistency["valid"] = False
            grid_consistency["issues"].append("available capacity less than total load")
            grid_consistency["capacity_violations"] = capacity_violation.sum()
    
    # Check: Wind + Solar generation should typically be <= total load
    if all(col in df.columns for col in ['wind_generation', 'solar_generation', 'total_load']):
        renewables = df['wind_generation'] + df['solar_generation']
        generation_violation = renewables > df['total_load'] * 1.1  # Allow 10% margin for exports
        
        if generation_violation.any():
            grid_consistency["valid"] = False
            grid_consistency["issues"].append("renewable generation exceeds total load")
            grid_consistency["generation_violations"] = generation_violation.sum()
    
    # Compile results
    results = {
        "valid": (
            completeness_result.get("valid", False) and 
            value_ranges_result.get("valid", False) and 
            temporal_result.get("valid", False) and 
            grid_consistency.get("valid", True)
        ),
        "completeness": completeness_result,
        "value_ranges": value_ranges_result,
        "temporal_consistency": temporal_result,
        "outliers": grid_outliers,
        "grid_consistency": grid_consistency
    }
    
    if not results["valid"]:
        issues = []
        if not completeness_result.get("valid", False):
            issues.append("data completeness below threshold")
        if not value_ranges_result.get("valid", False):
            issues.append("values outside expected ranges")
        if not temporal_result.get("valid", False):
            issues.append("temporal inconsistencies found")
        if not grid_consistency.get("valid", True):
            issues.append("grid constraint violations found")
        
        logger.warning(f"Grid condition data quality issues found: {', '.join(issues)}")
    else:
        logger.info("Grid condition data quality checks passed")
    
    return results


def detect_rtlmp_anomalies(df: DataFrameType, method: str = 'zscore', threshold: float = DEFAULT_OUTLIER_THRESHOLD) -> DataFrameType:
    """
    Detects anomalies in RTLMP data using statistical methods.
    
    Args:
        df: DataFrame containing RTLMP data
        method: Method to use for anomaly detection ('zscore', 'iqr', or 'isolation_forest')
        threshold: Threshold for anomaly detection
        
    Returns:
        DataFrame with anomaly flags added
    """
    logger.debug(f"Detecting RTLMP anomalies using method: {method}")
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Price columns to check for anomalies
    price_columns = ['price', 'congestion_price', 'loss_price', 'energy_price']
    price_columns = [col for col in price_columns if col in df.columns]
    
    if method == 'zscore':
        # Z-score method
        for col in price_columns:
            # Calculate z-scores
            z_scores = np.abs(zscore(df[col].dropna()))
            
            # Create a Series with the right index, defaulting to False
            anomaly_flag = pd.Series(False, index=df.index)
            
            # Set True for rows where z-score exceeds threshold
            valid_indices = df[col].dropna().index
            anomaly_flag.loc[valid_indices] = z_scores > threshold
            
            # Add the flag to the result DataFrame
            flag_col = f"{col}_anomaly"
            result_df[flag_col] = anomaly_flag
            
            logger.info(f"Detected {anomaly_flag.sum()} anomalies in {col} using z-score method")
    
    elif method == 'iqr':
        # IQR method
        for col in price_columns:
            # Calculate IQR bounds
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Identify outliers
            anomaly_flag = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # Add the flag to the result DataFrame
            flag_col = f"{col}_anomaly"
            result_df[flag_col] = anomaly_flag
            
            logger.info(f"Detected {anomaly_flag.sum()} anomalies in {col} using IQR method")
    
    elif method == 'isolation_forest':
        try:
            from sklearn.ensemble import IsolationForest
            
            # Prepare data for Isolation Forest
            X = df[price_columns].fillna(df[price_columns].mean())
            
            # Train Isolation Forest model
            model = IsolationForest(contamination=0.05, random_state=42)
            predictions = model.fit_predict(X)
            
            # Convert predictions to anomaly flags (1 is normal, -1 is anomaly)
            anomaly_flag = pd.Series(predictions == -1, index=df.index)
            
            # Add the flag to the result DataFrame
            result_df['price_anomaly'] = anomaly_flag
            
            logger.info(f"Detected {anomaly_flag.sum()} anomalies using Isolation Forest method")
        
        except ImportError:
            msg = "scikit-learn is required for Isolation Forest method"
            logger.error(msg)
            raise ValueError(msg)
    
    else:
        msg = f"Unsupported anomaly detection method: {method}"
        logger.error(msg)
        raise ValueError(msg)
    
    return result_df


def check_data_consistency(rtlmp_df: DataFrameType, weather_df: DataFrameType, grid_df: DataFrameType) -> dict:
    """
    Checks consistency between different data sources.
    
    Args:
        rtlmp_df: DataFrame containing RTLMP data
        weather_df: DataFrame containing weather data
        grid_df: DataFrame containing grid condition data
        
    Returns:
        Dictionary with consistency check results
    """
    logger.debug("Checking consistency between data sources")
    
    # Ensure all DataFrames have timestamp column
    for df_name, df in [("RTLMP", rtlmp_df), ("Weather", weather_df), ("Grid", grid_df)]:
        if 'timestamp' not in df.columns:
            msg = f"Missing timestamp column in {df_name} DataFrame"
            logger.error(msg)
            raise DataFormatError(msg)
    
    # Convert timestamps to datetime if needed
    for df in [rtlmp_df, weather_df, grid_df]:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get the overlapping time range
    start_time = max(
        rtlmp_df['timestamp'].min(),
        weather_df['timestamp'].min(),
        grid_df['timestamp'].min()
    )
    
    end_time = min(
        rtlmp_df['timestamp'].max(),
        weather_df['timestamp'].max(),
        grid_df['timestamp'].max()
    )
    
    if start_time > end_time:
        logger.warning("No overlapping time range between data sources")
        return {
            "valid": False,
            "message": "No overlapping time range between data sources"
        }
    
    # Filter data to the overlapping time range
    rtlmp_overlap = rtlmp_df[(rtlmp_df['timestamp'] >= start_time) & (rtlmp_df['timestamp'] <= end_time)]
    weather_overlap = weather_df[(weather_df['timestamp'] >= start_time) & (weather_df['timestamp'] <= end_time)]
    grid_overlap = grid_df[(grid_df['timestamp'] >= start_time) & (grid_df['timestamp'] <= end_time)]
    
    # Check for missing timestamps
    all_timestamps = sorted(set(
        pd.unique(rtlmp_overlap['timestamp']).tolist() +
        pd.unique(weather_overlap['timestamp']).tolist() +
        pd.unique(grid_overlap['timestamp']).tolist()
    ))
    
    rtlmp_missing = [ts for ts in all_timestamps if ts not in set(rtlmp_overlap['timestamp'])]
    weather_missing = [ts for ts in all_timestamps if ts not in set(weather_overlap['timestamp'])]
    grid_missing = [ts for ts in all_timestamps if ts not in set(grid_overlap['timestamp'])]
    
    # Evaluate relationships between weather conditions and grid metrics
    weather_grid_relationships = []
    
    # For this check, we need to resample data to a common frequency (hourly)
    if not weather_overlap.empty and not grid_overlap.empty:
        # Group data by hour
        weather_hourly = weather_overlap.groupby(pd.Grouper(key='timestamp', freq='1H')).mean()
        grid_hourly = grid_overlap.groupby(pd.Grouper(key='timestamp', freq='1H')).mean()
        
        # Merge on timestamp index
        merged = pd.merge(weather_hourly.reset_index(), grid_hourly.reset_index(), on='timestamp')
        
        if not merged.empty:
            # Check: Solar generation should correlate with solar irradiance
            if 'solar_irradiance' in merged.columns and 'solar_generation' in merged.columns:
                corr = merged['solar_irradiance'].corr(merged['solar_generation'])
                weather_grid_relationships.append({
                    "relationship": "solar_irradiance_to_solar_generation",
                    "correlation": corr,
                    "valid": corr > 0.5  # Expected to be positively correlated
                })
            
            # Check: Wind generation should correlate with wind speed
            if 'wind_speed' in merged.columns and 'wind_generation' in merged.columns:
                corr = merged['wind_speed'].corr(merged['wind_generation'])
                weather_grid_relationships.append({
                    "relationship": "wind_speed_to_wind_generation",
                    "correlation": corr,
                    "valid": corr > 0.3  # Expected to be positively correlated
                })
    
    # Evaluate relationships between grid conditions and price components
    grid_price_relationships = []
    
    if not rtlmp_overlap.empty and not grid_overlap.empty:
        # Group RTLMP data by hour
        rtlmp_hourly = rtlmp_overlap.groupby([pd.Grouper(key='timestamp', freq='1H'), 'node_id']).mean().reset_index()
        grid_hourly = grid_overlap.groupby(pd.Grouper(key='timestamp', freq='1H')).mean().reset_index()
        
        # Merge on timestamp
        merged = pd.merge(rtlmp_hourly, grid_hourly, on='timestamp')
        
        if not merged.empty:
            # Check: Price should increase with load when close to capacity
            if 'price' in merged.columns and 'total_load' in merged.columns and 'available_capacity' in merged.columns:
                # Calculate reserve margin
                merged['reserve_margin'] = merged['available_capacity'] - merged['total_load']
                
                # Filter to periods of low reserve margin
                low_margin = merged[merged['reserve_margin'] < merged['reserve_margin'].quantile(0.25)]
                
                if not low_margin.empty:
                    corr = low_margin['total_load'].corr(low_margin['price'])
                    grid_price_relationships.append({
                        "relationship": "load_to_price_at_low_margin",
                        "correlation": corr,
                        "valid": corr > 0.3  # Expected to be positively correlated
                    })
    
    # Compile results
    results = {
        "valid": (
            len(rtlmp_missing) == 0 and
            len(weather_missing) == 0 and
            len(grid_missing) == 0 and
            all(rel.get("valid", False) for rel in weather_grid_relationships) and
            all(rel.get("valid", False) for rel in grid_price_relationships)
        ),
        "overlapping_time_range": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "hours": (end_time - start_time).total_seconds() / 3600
        },
        "missing_timestamps": {
            "rtlmp": [ts.isoformat() for ts in rtlmp_missing[:10]],  # Limit to 10 examples
            "weather": [ts.isoformat() for ts in weather_missing[:10]],
            "grid": [ts.isoformat() for ts in grid_missing[:10]],
            "rtlmp_count": len(rtlmp_missing),
            "weather_count": len(weather_missing),
            "grid_count": len(grid_missing)
        },
        "weather_grid_relationships": weather_grid_relationships,
        "grid_price_relationships": grid_price_relationships
    }
    
    if not results["valid"]:
        issues = []
        if len(rtlmp_missing) > 0:
            issues.append(f"missing {len(rtlmp_missing)} RTLMP timestamps")
        if len(weather_missing) > 0:
            issues.append(f"missing {len(weather_missing)} weather timestamps")
        if len(grid_missing) > 0:
            issues.append(f"missing {len(grid_missing)} grid timestamps")
        
        invalid_relationships = [
            rel["relationship"] for rel in weather_grid_relationships + grid_price_relationships
            if not rel.get("valid", False)
        ]
        
        if invalid_relationships:
            issues.append(f"invalid relationships: {', '.join(invalid_relationships)}")
        
        logger.warning(f"Data consistency issues found: {', '.join(issues)}")
    else:
        logger.info("Data consistency checks passed")
    
    return results


def calculate_data_quality_metrics(df: DataFrameType, data_type: str) -> dict:
    """
    Calculates comprehensive data quality metrics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        data_type: Type of data in the DataFrame ('rtlmp', 'weather', 'grid_condition', 'forecast')
        
    Returns:
        Dictionary of data quality metrics
    """
    logger.debug(f"Calculating data quality metrics for {data_type} data")
    
    metrics = {}
    
    # 1. Completeness metrics
    non_null_pct = 1 - df.isna().mean()
    metrics["completeness"] = {
        "overall": non_null_pct.mean(),
        "by_column": non_null_pct.to_dict()
    }
    
    # 2. Uniqueness metrics
    uniqueness = {}
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            unique_pct = df[col].nunique() / df[col].count() if df[col].count() > 0 else 0
            uniqueness[col] = unique_pct
        else:
            uniqueness[col] = None
    
    metrics["uniqueness"] = {
        "by_column": {k: v for k, v in uniqueness.items() if v is not None}
    }
    
    # 3. Consistency metrics based on data type
    if data_type == 'rtlmp':
        # Check price component consistency
        if all(col in df.columns for col in ['price', 'congestion_price', 'loss_price', 'energy_price']):
            diff = df['price'] - (df['energy_price'] + df['congestion_price'] + df['loss_price'])
            metrics["consistency"] = {
                "price_components": {
                    "mean_absolute_error": diff.abs().mean(),
                    "max_absolute_error": diff.abs().max(),
                    "within_tolerance_pct": (diff.abs() < 0.001).mean() * 100
                }
            }
    
    elif data_type == 'weather':
        # No specific consistency metrics for weather data
        metrics["consistency"] = {}
    
    elif data_type == 'grid_condition':
        # Check capacity vs load consistency
        if 'available_capacity' in df.columns and 'total_load' in df.columns:
            margin = df['available_capacity'] - df['total_load']
            metrics["consistency"] = {
                "reserve_margin": {
                    "mean": margin.mean(),
                    "min": margin.min(),
                    "negative_count": (margin < 0).sum(),
                    "negative_pct": (margin < 0).mean() * 100 if len(margin) > 0 else 0
                }
            }
    
    elif data_type == 'forecast':
        # Check probability values consistency
        if 'spike_probability' in df.columns:
            valid_probs = (df['spike_probability'] >= 0) & (df['spike_probability'] <= 1)
            metrics["consistency"] = {
                "probability_values": {
                    "valid_pct": valid_probs.mean() * 100 if len(valid_probs) > 0 else 0,
                    "invalid_count": (~valid_probs).sum()
                }
            }
    
    # 4. Timeliness metrics (recency of data)
    if 'timestamp' in df.columns:
        current_time = pd.Timestamp.now()
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            most_recent = df['timestamp'].max()
            oldest = df['timestamp'].min()
            
            metrics["timeliness"] = {
                "most_recent": most_recent.isoformat(),
                "oldest": oldest.isoformat(),
                "age_hours": (current_time - most_recent).total_seconds() / 3600,
                "span_days": (most_recent - oldest).total_seconds() / (24 * 3600)
            }
    
    # 5. Accuracy metrics where applicable
    # These are generally domain-specific and would require reference data
    metrics["accuracy"] = {}
    
    # 6. Calculate statistics for numeric columns
    numeric_stats = {}
    for col in df.select_dtypes(include=['number']).columns:
        col_stats = {
            "count": df[col].count(),
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "25%": df[col].quantile(0.25),
            "50%": df[col].median(),
            "75%": df[col].quantile(0.75),
            "max": df[col].max()
        }
        numeric_stats[col] = col_stats
    
    metrics["statistics"] = numeric_stats
    
    # 7. Overall quality score (simple average of completeness and consistency)
    quality_scores = [metrics["completeness"]["overall"]]
    
    if data_type == 'rtlmp' and "consistency" in metrics and "price_components" in metrics["consistency"]:
        quality_scores.append(metrics["consistency"]["price_components"]["within_tolerance_pct"] / 100)
    
    if data_type == 'grid_condition' and "consistency" in metrics and "reserve_margin" in metrics["consistency"]:
        quality_scores.append(1 - metrics["consistency"]["reserve_margin"]["negative_pct"] / 100)
    
    if data_type == 'forecast' and "consistency" in metrics and "probability_values" in metrics["consistency"]:
        quality_scores.append(metrics["consistency"]["probability_values"]["valid_pct"] / 100)
    
    metrics["overall_quality_score"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    logger.info(f"Calculated data quality metrics for {data_type} data, overall score: {metrics['overall_quality_score']:.2f}")
    
    return metrics


def impute_missing_values(df: DataFrameType, method: str = 'mean', columns: List[str] = None) -> DataFrameType:
    """
    Imputes missing values in data using specified method.
    
    Args:
        df: DataFrame with missing values
        method: Imputation method ('mean', 'median', 'mode', 'interpolate', 'knn')
        columns: List of columns to impute (if None, all columns with missing values)
        
    Returns:
        DataFrame with imputed values
    """
    logger.debug(f"Imputing missing values using method: {method}")
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, use all columns with missing values
    if columns is None:
        columns = [col for col in df.columns if df[col].isna().any()]
    
    # Filter to only columns that exist and have missing values
    columns = [col for col in columns if col in df.columns and df[col].isna().any()]
    
    if not columns:
        logger.info("No missing values to impute")
        return result_df
    
    if method == 'mean':
        # Impute with column mean
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                result_df[col].fillna(mean_val, inplace=True)
                logger.debug(f"Imputed {df[col].isna().sum()} missing values in {col} with mean: {mean_val}")
    
    elif method == 'median':
        # Impute with column median
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                result_df[col].fillna(median_val, inplace=True)
                logger.debug(f"Imputed {df[col].isna().sum()} missing values in {col} with median: {median_val}")
    
    elif method == 'mode':
        # Impute with column mode
        for col in columns:
            mode_val = df[col].mode()[0]  # Use first mode if multiple exist
            result_df[col].fillna(mode_val, inplace=True)
            logger.debug(f"Imputed {df[col].isna().sum()} missing values in {col} with mode: {mode_val}")
    
    elif method == 'interpolate':
        # Impute using interpolation (for time series)
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Ensure DataFrame is sorted if using time index
                if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    result_df = result_df.sort_values('timestamp')
                    result_df[col] = result_df[col].interpolate(method='time')
                else:
                    result_df[col] = result_df[col].interpolate(method='linear')
                
                logger.debug(f"Imputed {df[col].isna().sum()} missing values in {col} using interpolation")
    
    elif method == 'knn':
        try:
            from sklearn.impute import KNNImputer
            
            # Identify numeric columns for KNN imputation
            numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if numeric_columns:
                # Create imputer
                imputer = KNNImputer(n_neighbors=5)
                
                # Impute numeric columns
                numeric_df = df[numeric_columns]
                imputed_values = imputer.fit_transform(numeric_df)
                
                # Update result DataFrame
                result_df[numeric_columns] = imputed_values
                
                logger.debug(f"Imputed missing values in {len(numeric_columns)} columns using KNN")
            
        except ImportError:
            msg = "scikit-learn is required for KNN imputation"
            logger.error(msg)
            raise ValueError(msg)
    
    else:
        msg = f"Unsupported imputation method: {method}"
        logger.error(msg)
        raise ValueError(msg)
    
    # Count imputed values
    imputed_counts = {col: df[col].isna().sum() for col in columns}
    total_imputed = sum(imputed_counts.values())
    
    logger.info(f"Imputed {total_imputed} missing values across {len(columns)} columns using {method} method")
    
    return result_df


@handle_errors(exceptions=(DataFormatError, MissingDataError), reraise=True)
def validate_forecast_quality(forecast_df: DataFrameType, strict: bool = True) -> dict:
    """
    Validates the quality of forecast data.
    
    Args:
        forecast_df: DataFrame containing forecast data
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        Dictionary with forecast quality assessment results
        
    Raises:
        DataFormatError: If validation fails and strict is True
        MissingDataError: If required data is missing and strict is True
    """
    logger.debug(f"Validating forecast quality, strict mode: {strict}")
    
    # Required columns for forecast data
    required_columns = [
        'forecast_timestamp', 'target_timestamp', 'threshold_value', 
        'spike_probability', 'model_version', 'node_id'
    ]
    
    # Check required columns
    if not all(col in forecast_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in forecast_df.columns]
        msg = f"Missing required columns in forecast data: {missing_cols}"
        logger.error(msg)
        if strict:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Check data completeness
    completeness_result = validate_data_completeness(
        forecast_df, threshold=DEFAULT_COMPLETENESS_THRESHOLD, raise_error=strict
    )
    
    # Validate probability values
    probability_validation = {}
    if 'spike_probability' in forecast_df.columns:
        invalid_probs = ~((forecast_df['spike_probability'] >= 0) & (forecast_df['spike_probability'] <= 1))
        probability_validation = {
            "valid": not invalid_probs.any(),
            "invalid_count": invalid_probs.sum(),
            "invalid_examples": forecast_df.loc[invalid_probs, 'spike_probability'].head(5).tolist() if invalid_probs.any() else []
        }
    
    # If confidence intervals are present, validate them
    confidence_interval_valid = True
    confidence_interval_issues = []
    
    if all(col in forecast_df.columns for col in ['confidence_interval_lower', 'confidence_interval_upper']):
        # Check if intervals are valid probabilities
        invalid_lower = ~((forecast_df['confidence_interval_lower'] >= 0) & 
                         (forecast_df['confidence_interval_lower'] <= 1))
        invalid_upper = ~((forecast_df['confidence_interval_upper'] >= 0) & 
                         (forecast_df['confidence_interval_upper'] <= 1))
        
        if invalid_lower.any() or invalid_upper.any():
            confidence_interval_valid = False
            confidence_interval_issues.append("invalid probability values in confidence intervals")
        
        # Check if probability is within confidence interval
        has_intervals = forecast_df['confidence_interval_lower'].notna() & forecast_df['confidence_interval_upper'].notna()
        
        if has_intervals.any():
            within_interval = (
                (forecast_df['spike_probability'] >= forecast_df['confidence_interval_lower']) & 
                (forecast_df['spike_probability'] <= forecast_df['confidence_interval_upper'])
            )
            
            violations = has_intervals & ~within_interval
            
            if violations.any():
                confidence_interval_valid = False
                confidence_interval_issues.append("probability outside confidence interval")
    
    # Check temporal consistency of forecast timestamps
    forecast_temporal_result = None
    if 'forecast_timestamp' in forecast_df.columns:
        forecast_temporal_result = validate_temporal_consistency(
            forecast_df, timestamp_column='forecast_timestamp', expected_frequency=None, 
            raise_error=False
        )
    
    # Check temporal consistency of target timestamps
    target_temporal_result = None
    if 'target_timestamp' in forecast_df.columns:
        target_temporal_result = validate_temporal_consistency(
            forecast_df, timestamp_column='target_timestamp', expected_frequency='1H', 
            raise_error=False
        )
    
    # Check forecast horizon
    horizon_check = {}
    if all(col in forecast_df.columns for col in ['forecast_timestamp', 'target_timestamp']):
        # Convert to datetime if needed
        for col in ['forecast_timestamp', 'target_timestamp']:
            if not pd.api.types.is_datetime64_any_dtype(forecast_df[col]):
                forecast_df[col] = pd.to_datetime(forecast_df[col])
        
        # Calculate horizon in hours
        forecast_df['horizon_hours'] = (
            forecast_df['target_timestamp'] - forecast_df['forecast_timestamp']
        ).dt.total_seconds() / 3600
        
        horizon_hours = forecast_df['horizon_hours']
        
        horizon_check = {
            "min_hours": horizon_hours.min(),
            "max_hours": horizon_hours.max(),
            "negative_horizons": (horizon_hours < 0).sum(),
            "valid": (horizon_hours >= 0).all()
        }
    
    # Compile results
    results = {
        "valid": (
            completeness_result.get("valid", False) and 
            probability_validation.get("valid", True) and
            confidence_interval_valid and
            (forecast_temporal_result.get("valid", True) if forecast_temporal_result else True) and
            (target_temporal_result.get("valid", True) if target_temporal_result else True) and
            horizon_check.get("valid", True)
        ),
        "completeness": completeness_result,
        "probability_validation": probability_validation,
        "confidence_intervals": {
            "valid": confidence_interval_valid,
            "issues": confidence_interval_issues
        },
        "forecast_timestamp_consistency": forecast_temporal_result,
        "target_timestamp_consistency": target_temporal_result,
        "horizon_check": horizon_check
    }
    
    if not results["valid"]:
        issues = []
        if not completeness_result.get("valid", False):
            issues.append("data completeness below threshold")
        if not probability_validation.get("valid", True):
            issues.append("invalid probability values")
        if not confidence_interval_valid:
            issues.append(f"confidence interval issues: {', '.join(confidence_interval_issues)}")
        if forecast_temporal_result and not forecast_temporal_result.get("valid", True):
            issues.append("forecast timestamp inconsistencies")
        if target_temporal_result and not target_temporal_result.get("valid", True):
            issues.append("target timestamp inconsistencies")
        if not horizon_check.get("valid", True):
            issues.append("invalid forecast horizons")
        
        logger.warning(f"Forecast quality issues found: {', '.join(issues)}")
    else:
        logger.info("Forecast quality checks passed")
    
    return results


class DataQualityChecker:
    """
    Class that provides methods for comprehensive data quality assessment.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the DataQualityChecker with default settings.
        
        Args:
            strict_mode: If True, raises exceptions for validation failures
        """
        # Dictionary to store quality check functions for different data types
        self._quality_checks = {}
        
        # Dictionary to store quality thresholds
        self._thresholds = {
            "completeness": DEFAULT_COMPLETENESS_THRESHOLD,
            "outlier": DEFAULT_OUTLIER_THRESHOLD
        }
        
        # Flag for strict mode
        self._strict_mode = strict_mode
        
        # Register default quality check functions
        self._register_default_quality_checks()
        
        # Set up logger
        self._logger = get_logger(__name__)
    
    def _register_default_quality_checks(self):
        """Registers default quality check functions for common data types."""
        self.register_quality_check("rtlmp", "data_quality", check_rtlmp_data_quality)
        self.register_quality_check("weather", "data_quality", check_weather_data_quality)
        self.register_quality_check("grid_condition", "data_quality", check_grid_condition_data_quality)
        self.register_quality_check("forecast", "data_quality", validate_forecast_quality)
    
    def register_quality_check(self, data_type: str, check_name: str, check_function: Callable) -> None:
        """
        Registers a quality check function for a data type.
        
        Args:
            data_type: Type of data this check applies to
            check_name: Name for the quality check
            check_function: Function that implements the check
        """
        if data_type not in self._quality_checks:
            self._quality_checks[data_type] = {}
        
        self._quality_checks[data_type][check_name] = check_function
        
        self._logger.debug(f"Registered quality check '{check_name}' for data type '{data_type}'")
    
    def set_threshold(self, metric_name: str, threshold_value: float) -> None:
        """
        Sets a quality threshold for a specific metric.
        
        Args:
            metric_name: Name of the metric
            threshold_value: Threshold value
        """
        self._thresholds[metric_name] = threshold_value
        self._logger.debug(f"Set threshold for '{metric_name}' to {threshold_value}")
    
    def check_quality(self, df: DataFrameType, data_type: str) -> dict:
        """
        Performs all registered quality checks for a data type.
        
        Args:
            df: DataFrame to check
            data_type: Type of data in the DataFrame
            
        Returns:
            Dictionary with quality assessment results
            
        Raises:
            DataFormatError: If checks fail and strict mode is enabled
        """
        if data_type not in self._quality_checks:
            self._logger.warning(f"No quality checks registered for data type '{data_type}'")
            return {"valid": True, "message": f"No quality checks registered for data type '{data_type}'"}
        
        results = {}
        overall_valid = True
        
        # Run all registered quality checks for this data type
        for check_name, check_function in self._quality_checks[data_type].items():
            try:
                check_result = check_function(df, strict=False)  # Run in non-strict mode to collect all issues
                results[check_name] = check_result
                
                # Update overall validity
                if isinstance(check_result, dict) and "valid" in check_result:
                    overall_valid = overall_valid and check_result["valid"]
            
            except Exception as e:
                self._logger.error(f"Error running quality check '{check_name}' for '{data_type}': {e}")
                results[check_name] = {"valid": False, "error": str(e)}
                overall_valid = False
        
        # Calculate overall quality metrics
        metrics = calculate_data_quality_metrics(df, data_type)
        results["metrics"] = metrics
        
        # Calculate overall quality score
        results["overall_score"] = metrics.get("overall_quality_score", 0)
        results["valid"] = overall_valid
        
        # Raise exception if in strict mode and quality checks failed
        if self._strict_mode and not overall_valid:
            msg = f"Quality checks failed for {data_type} data"
            self._logger.error(msg)
            raise DataFormatError(msg)
        
        return results
    
    def generate_quality_report(self, df: DataFrameType, data_type: str) -> dict:
        """
        Generates a detailed quality report for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            data_type: Type of data in the DataFrame
            
        Returns:
            Detailed quality report dictionary
        """
        # Run quality checks
        check_results = self.check_quality(df, data_type)
        
        # Extract issues from check results
        issues = []
        
        for check_name, result in check_results.items():
            if check_name == "metrics" or check_name == "overall_score" or check_name == "valid":
                continue
                
            if isinstance(result, dict) and not result.get("valid", True):
                # Extract specific issues from the result
                if "message" in result:
                    issues.append(result["message"])
                elif "issues" in result:
                    issues.extend(result["issues"])
                else:
                    issues.append(f"Failed check: {check_name}")
        
        # Generate recommendations based on issues
        recommendations = []
        
        for issue in issues:
            if "missing" in issue.lower() and "column" in issue.lower():
                recommendations.append("Ensure all required columns are present in the data")
            
            if "completeness" in issue.lower():
                recommendations.append("Impute missing values or filter incomplete records")
            
            if "range" in issue.lower() or "outside" in issue.lower():
                recommendations.append("Investigate and correct out-of-range values")
            
            if "temporal" in issue.lower() or "timestamp" in issue.lower():
                recommendations.append("Ensure regular time intervals and no missing timestamps")
            
            if "consistency" in issue.lower():
                recommendations.append("Review data consistency constraints and correct violations")
        
        # Remove duplicate recommendations
        recommendations = list(set(recommendations))
        
        # Format report
        report = {
            "data_type": data_type,
            "timestamp": pd.Timestamp.now().isoformat(),
            "overall_valid": check_results.get("valid", False),
            "overall_score": check_results.get("overall_score", 0),
            "metrics": check_results.get("metrics", {}),
            "issues": issues,
            "recommendations": recommendations,
            "check_results": {k: v for k, v in check_results.items() 
                             if k not in ["metrics", "overall_score", "valid"]}
        }
        
        self._logger.info(
            f"Generated quality report for {data_type} data, "
            f"overall score: {report['overall_score']:.2f}, "
            f"issues: {len(issues)}, "
            f"recommendations: {len(recommendations)}"
        )
        
        return report
    
    def set_strict_mode(self, strict_mode: bool) -> None:
        """
        Sets the strict mode flag for quality checks.
        
        Args:
            strict_mode: If True, raises exceptions for validation failures
        """
        self._strict_mode = strict_mode
        self._logger.debug(f"Set strict mode to {strict_mode}")


class QualityReport:
    """
    Class that represents a data quality assessment report.
    """
    
    def __init__(self, data_type: str, metrics: dict, issues: dict, overall_score: float):
        """
        Initialize the QualityReport with assessment results.
        
        Args:
            data_type: Type of data that was assessed
            metrics: Dictionary of quality metrics
            issues: Dictionary of identified issues
            overall_score: Overall quality score (0-1)
        """
        self.data_type = data_type
        self.metrics = metrics
        self.issues = issues
        self.overall_score = overall_score
        self.recommendations = []
        
        # Generate recommendations based on issues
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generates recommendations based on identified issues."""
        if not self.issues:
            return
        
        for issue_type, details in self.issues.items():
            if issue_type == "completeness" and not details.get("valid", True):
                self.add_recommendation("Impute missing values or filter incomplete records")
            
            elif issue_type == "value_ranges" and not details.get("valid", True):
                self.add_recommendation("Investigate and correct out-of-range values")
            
            elif issue_type == "temporal_consistency" and not details.get("valid", True):
                self.add_recommendation("Ensure regular time intervals and no missing timestamps")
            
            elif "consistency" in issue_type and not details.get("valid", True):
                self.add_recommendation("Review data consistency constraints and correct violations")
            
            elif issue_type == "outliers" and sum(details.values()) > 0:
                self.add_recommendation("Investigate outliers that may represent data quality issues")
    
    def add_issue(self, issue_type: str, description: str, details: dict = None) -> None:
        """
        Adds an issue to the quality report.
        
        Args:
            issue_type: Type of issue
            description: Description of the issue
            details: Additional details about the issue
        """
        if issue_type not in self.issues:
            self.issues[issue_type] = []
        
        issue = {
            "description": description,
            "details": details or {}
        }
        
        self.issues[issue_type].append(issue)
        
        # Adjust overall score based on new issue
        self.overall_score = max(0, self.overall_score - 0.05)
        
        # Generate recommendation for this issue
        if "missing" in description.lower():
            self.add_recommendation("Ensure all required data is present")
        elif "range" in description.lower():
            self.add_recommendation("Investigate values outside expected ranges")
        elif "inconsistent" in description.lower():
            self.add_recommendation("Review data consistency constraints")
    
    def add_recommendation(self, recommendation: str) -> None:
        """
        Adds a recommendation to the quality report.
        
        Args:
            recommendation: Recommendation text
        """
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
    
    def to_dict(self) -> dict:
        """
        Converts the quality report to a dictionary.
        
        Returns:
            Dictionary representation of the quality report
        """
        return {
            "data_type": self.data_type,
            "metrics": self.metrics,
            "issues": self.issues,
            "overall_score": self.overall_score,
            "recommendations": self.recommendations
        }
    
    def to_dataframe(self) -> DataFrameType:
        """
        Converts the quality report to a DataFrame.
        
        Returns:
            DataFrame representation of the quality report
        """
        # Extract metrics into a flat dictionary for DataFrame
        flat_metrics = {}
        
        for category, metrics in self.metrics.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float, str, bool)):
                        flat_metrics[f"{category}_{metric_name}"] = value
            else:
                flat_metrics[category] = metrics
        
        # Add overall score and issue counts
        flat_metrics["overall_score"] = self.overall_score
        flat_metrics["issue_count"] = sum(len(issues) for issues in self.issues.values())
        
        # Create DataFrame with one row
        return pd.DataFrame([flat_metrics])
    
    def merge(self, other: 'QualityReport') -> 'QualityReport':
        """
        Merges another quality report into this one.
        
        Args:
            other: Another QualityReport instance
            
        Returns:
            Self reference for method chaining
        """
        # Update metrics by merging dictionaries
        for category, metrics in other.metrics.items():
            if category in self.metrics:
                if isinstance(self.metrics[category], dict) and isinstance(metrics, dict):
                    self.metrics[category].update(metrics)
                else:
                    self.metrics[category] = metrics
            else:
                self.metrics[category] = metrics
        
        # Merge issues
        for issue_type, issues in other.issues.items():
            if issue_type in self.issues:
                self.issues[issue_type].extend(issues)
            else:
                self.issues[issue_type] = issues
        
        # Recalculate overall score as average
        self.overall_score = (self.overall_score + other.overall_score) / 2
        
        # Merge recommendations
        for recommendation in other.recommendations:
            self.add_recommendation(recommendation)
        
        return self