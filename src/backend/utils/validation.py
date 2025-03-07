"""
Core validation utilities for the ERCOT RTLMP spike prediction system.

Provides generic data validation functions, schema validation helpers, 
and validation decorators to ensure data integrity throughout the system.
"""

from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import datetime

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+

from .type_definitions import DataFrameType, SeriesType
from .logging import get_logger
from .error_handling import DataFormatError, MissingDataError, handle_errors

# Set up logger
logger = get_logger(__name__)

# Default thresholds
DEFAULT_COMPLETENESS_THRESHOLD = 0.95
DEFAULT_OUTLIER_THRESHOLD = 3.0


@handle_errors(DataFormatError, reraise=True)
def validate_dataframe_schema(df: DataFrameType, schema: Any, strict: bool = True) -> DataFrameType:
    """
    Validates a pandas DataFrame against a schema definition.
    
    Args:
        df: DataFrame to validate
        schema: Schema definition to validate against
        strict: If True, raises an error for validation failures; otherwise, logs warnings
        
    Returns:
        The validated DataFrame if successful
        
    Raises:
        DataFormatError: If validation fails and strict is True
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        raise DataFormatError(msg)
    
    try:
        # The schema could be a pandera schema, a dict of dtypes, or another validation mechanism
        # Here we implement a simple validation approach that can be replaced with pandera in actual use
        if isinstance(schema, dict):
            # Schema is a dict of column names and expected dtypes
            for col, dtype in schema.items():
                if col not in df.columns:
                    msg = f"Required column {col} not found in DataFrame"
                    if strict:
                        logger.error(msg)
                        raise DataFormatError(msg)
                    else:
                        logger.warning(msg)
                
                try:
                    # Skip dtype check for columns with null values if not strict
                    if not strict and df[col].isna().any():
                        logger.warning(f"Column {col} contains null values, skipping dtype check")
                        continue
                    
                    # Check column type
                    if not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                        msg = f"Column {col} has dtype {df[col].dtype}, expected {dtype}"
                        if strict:
                            logger.error(msg)
                            raise DataFormatError(msg)
                        else:
                            logger.warning(msg)
                except Exception as e:
                    if strict:
                        raise DataFormatError(f"Error validating column {col}: {str(e)}")
                    else:
                        logger.warning(f"Error validating column {col}: {str(e)}")
        else:
            # Assume schema is a callable validator
            schema(df)
        
        logger.debug(f"DataFrame validation successful with schema {schema}")
        return df
    except Exception as e:
        msg = f"Schema validation failed: {str(e)}"
        logger.error(msg)
        if strict:
            raise DataFormatError(msg)
        else:
            logger.warning(msg)
            return df


def validate_data_completeness(
    df: DataFrameType,
    threshold: float = DEFAULT_COMPLETENESS_THRESHOLD,
    raise_error: bool = False
) -> Dict[str, Any]:
    """
    Checks the completeness of data and identifies missing values.
    
    Args:
        df: DataFrame to check
        threshold: Minimum required completeness percentage (0-1)
        raise_error: Whether to raise an error if completeness is below threshold
        
    Returns:
        Dictionary with completeness metrics and missing data information
        
    Raises:
        MissingDataError: If raise_error is True and completeness is below threshold
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Calculate completeness for each column
    completeness = (1 - df.isna().mean()).to_dict()
    
    # Identify columns with too many missing values
    incomplete_columns = {col: compl for col, compl in completeness.items() if compl < threshold}
    
    # Calculate overall completeness
    overall_completeness = 1 - df.isna().values.mean()
    
    # Build result dictionary
    result = {
        "valid": overall_completeness >= threshold,
        "overall_completeness": overall_completeness,
        "column_completeness": completeness,
        "incomplete_columns": incomplete_columns,
        "missing_count": df.isna().sum().to_dict()
    }
    
    if not result["valid"]:
        msg = (f"Data completeness below threshold: {overall_completeness:.2f} < {threshold}. "
               f"Columns with too many missing values: {list(incomplete_columns.keys())}")
        logger.warning(msg)
        if raise_error:
            raise MissingDataError(msg)
    else:
        logger.debug(f"Data completeness: {overall_completeness:.2f} (threshold: {threshold})")
    
    return result


def validate_value_ranges(
    df: DataFrameType,
    value_ranges: Dict[str, Tuple[float, float]],
    raise_error: bool = False
) -> Dict[str, Any]:
    """
    Validates that values in specified columns fall within expected ranges.
    
    Args:
        df: DataFrame to validate
        value_ranges: Dictionary mapping column names to (min, max) value tuples
        raise_error: Whether to raise an error if values are out of range
        
    Returns:
        Dictionary with validation results for each column
        
    Raises:
        DataFormatError: If raise_error is True and values are out of range
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Check each column against its value range
    results = {}
    out_of_range_columns = []
    
    for column, (min_val, max_val) in value_ranges.items():
        if column not in df.columns:
            msg = f"Column {column} not found in DataFrame"
            logger.warning(msg)
            results[column] = {"valid": False, "message": msg}
            out_of_range_columns.append(column)
            continue
        
        # Filter out NA values for the check
        col_data = df[column].dropna()
        
        # Check range
        min_violation = col_data < min_val
        max_violation = col_data > max_val
        violations = min_violation | max_violation
        
        if violations.any():
            # Get violation details
            violation_count = violations.sum()
            violation_pct = 100 * violation_count / len(col_data)
            
            # Create result dictionary
            results[column] = {
                "valid": False,
                "min_violations": min_violation.sum(),
                "max_violations": max_violation.sum(),
                "total_violations": violation_count,
                "violation_percentage": violation_pct,
                "example_violations": df.loc[violations, column].head(5).tolist()
            }
            
            msg = (f"Column {column} has {violation_count} values ({violation_pct:.2f}%) "
                   f"outside range [{min_val}, {max_val}]")
            logger.warning(msg)
            out_of_range_columns.append(column)
        else:
            results[column] = {"valid": True}
            logger.debug(f"Column {column} values within range [{min_val}, {max_val}]")
    
    # Overall result
    result = {
        "valid": len(out_of_range_columns) == 0,
        "column_results": results,
        "out_of_range_columns": out_of_range_columns
    }
    
    if not result["valid"] and raise_error:
        msg = f"Value range validation failed for columns: {out_of_range_columns}"
        logger.error(msg)
        raise DataFormatError(msg)
    
    return result


def validate_temporal_consistency(
    df: DataFrameType,
    timestamp_column: str,
    expected_frequency: str,
    raise_error: bool = False
) -> Dict[str, Any]:
    """
    Validates the temporal consistency of time series data.
    
    Args:
        df: DataFrame to validate
        timestamp_column: Name of the column containing timestamps
        expected_frequency: Expected time series frequency (e.g., '5min', '1H')
        raise_error: Whether to raise an error if inconsistencies are found
        
    Returns:
        Dictionary with temporal consistency validation results
        
    Raises:
        DataFormatError: If raise_error is True and inconsistencies are found
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    if timestamp_column not in df.columns:
        msg = f"Timestamp column {timestamp_column} not found in DataFrame"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Ensure timestamp column is datetime
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    except Exception as e:
        msg = f"Failed to convert {timestamp_column} to datetime: {str(e)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Sort by timestamp
    df_sorted = df.sort_values(timestamp_column)
    
    # Check if timestamps are in ascending order
    is_sorted = (df[timestamp_column] == df_sorted[timestamp_column]).all()
    
    # Check for duplicates
    has_duplicates = df[timestamp_column].duplicated().any()
    duplicate_timestamps = df[df[timestamp_column].duplicated(keep=False)][timestamp_column].unique()
    
    # Check for gaps
    timestamps = df[timestamp_column].sort_values()
    ideal_range = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq=expected_frequency
    )
    
    missing_timestamps = set(ideal_range) - set(timestamps)
    gaps_found = len(missing_timestamps) > 0
    
    # Calculate results
    valid = is_sorted and not has_duplicates and not gaps_found
    
    result = {
        "valid": valid,
        "is_sorted": is_sorted,
        "has_duplicates": has_duplicates,
        "duplicate_count": df[timestamp_column].duplicated().sum(),
        "duplicate_timestamps": [ts.isoformat() for ts in duplicate_timestamps][:10] if has_duplicates else [],
        "gaps_found": gaps_found,
        "missing_timestamps_count": len(missing_timestamps),
        "missing_timestamps": [ts.isoformat() for ts in sorted(missing_timestamps)[:10]] if gaps_found else [],
    }
    
    if not valid:
        issues = []
        if not is_sorted:
            issues.append("timestamps not in ascending order")
        if has_duplicates:
            issues.append(f"{result['duplicate_count']} duplicate timestamps found")
        if gaps_found:
            issues.append(f"{result['missing_timestamps_count']} gaps found in time series")
        
        msg = f"Temporal consistency validation failed: {', '.join(issues)}"
        logger.warning(msg)
        if raise_error:
            raise DataFormatError(msg)
    else:
        logger.debug("Temporal consistency validation passed")
    
    return result


def validate_required_columns(
    df: DataFrameType,
    required_columns: List[str],
    raise_error: bool = False
) -> bool:
    """
    Validates that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        raise_error: Whether to raise an error if columns are missing
        
    Returns:
        True if all required columns are present, False otherwise
        
    Raises:
        DataFormatError: If raise_error is True and columns are missing
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return False
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        msg = f"Required columns missing: {missing_columns}"
        logger.warning(msg)
        if raise_error:
            raise DataFormatError(msg)
        return False
    
    logger.debug(f"All required columns present: {required_columns}")
    return True


def validate_unique_values(
    df: DataFrameType,
    columns: List[str],
    raise_error: bool = False
) -> Dict[str, Any]:
    """
    Validates that specified columns contain unique values.
    
    Args:
        df: DataFrame to validate
        columns: List of column names to check for uniqueness
        raise_error: Whether to raise an error if duplicate values are found
        
    Returns:
        Dictionary with uniqueness validation results for each column
        
    Raises:
        DataFormatError: If raise_error is True and duplicate values are found
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Check each column for uniqueness
    results = {}
    columns_with_duplicates = []
    
    for column in columns:
        if column not in df.columns:
            msg = f"Column {column} not found in DataFrame"
            logger.warning(msg)
            results[column] = {"valid": False, "message": msg}
            columns_with_duplicates.append(column)
            continue
        
        # Check for duplicates
        duplicated = df[column].duplicated()
        has_duplicates = duplicated.any()
        
        if has_duplicates:
            # Get duplicate values
            duplicate_values = df.loc[df[column].duplicated(keep=False), column].unique()
            duplicate_count = duplicated.sum()
            
            # Create result dictionary
            results[column] = {
                "valid": False,
                "duplicate_count": duplicate_count,
                "duplicate_values": duplicate_values[:10].tolist()  # Limit to 10 examples
            }
            
            msg = f"Column {column} has {duplicate_count} duplicate values"
            logger.warning(msg)
            columns_with_duplicates.append(column)
        else:
            results[column] = {"valid": True}
            logger.debug(f"Column {column} values are unique")
    
    # Overall result
    result = {
        "valid": len(columns_with_duplicates) == 0,
        "column_results": results,
        "columns_with_duplicates": columns_with_duplicates
    }
    
    if not result["valid"] and raise_error:
        msg = f"Uniqueness validation failed for columns: {columns_with_duplicates}"
        logger.error(msg)
        raise DataFormatError(msg)
    
    return result


def validate_data_types(
    df: DataFrameType,
    expected_types: Dict[str, Type],
    raise_error: bool = False
) -> Dict[str, Any]:
    """
    Validates that columns have the expected data types.
    
    Args:
        df: DataFrame to validate
        expected_types: Dictionary mapping column names to expected types
        raise_error: Whether to raise an error if type mismatches are found
        
    Returns:
        Dictionary with type validation results for each column
        
    Raises:
        DataFormatError: If raise_error is True and type mismatches are found
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Check each column against its expected type
    results = {}
    type_mismatch_columns = []
    
    for column, expected_type in expected_types.items():
        if column not in df.columns:
            msg = f"Column {column} not found in DataFrame"
            logger.warning(msg)
            results[column] = {"valid": False, "message": msg}
            type_mismatch_columns.append(column)
            continue
        
        # Check type
        column_type = df[column].dtype
        type_matches = False
        
        # Handle special cases for pandas/numpy types
        if expected_type == float:
            type_matches = pd.api.types.is_float_dtype(column_type)
        elif expected_type == int:
            type_matches = pd.api.types.is_integer_dtype(column_type)
        elif expected_type == bool:
            type_matches = pd.api.types.is_bool_dtype(column_type)
        elif expected_type == str:
            type_matches = pd.api.types.is_string_dtype(column_type)
        elif expected_type == datetime.datetime:
            type_matches = pd.api.types.is_datetime64_any_dtype(column_type)
        else:
            # Fallback for other types
            try:
                # Try converting a sample for type checking
                sample = df[column].dropna().iloc[0] if not df[column].empty else None
                type_matches = isinstance(sample, expected_type)
            except (IndexError, TypeError):
                type_matches = False
        
        if not type_matches:
            results[column] = {
                "valid": False,
                "found_type": str(column_type),
                "expected_type": str(expected_type)
            }
            
            msg = f"Column {column} has type {column_type}, expected {expected_type}"
            logger.warning(msg)
            type_mismatch_columns.append(column)
        else:
            results[column] = {"valid": True}
            logger.debug(f"Column {column} has expected type {expected_type}")
    
    # Overall result
    result = {
        "valid": len(type_mismatch_columns) == 0,
        "column_results": results,
        "type_mismatch_columns": type_mismatch_columns
    }
    
    if not result["valid"] and raise_error:
        msg = f"Data type validation failed for columns: {type_mismatch_columns}"
        logger.error(msg)
        raise DataFormatError(msg)
    
    return result


def validate_probability_values(
    df: DataFrameType,
    probability_columns: List[str],
    raise_error: bool = False
) -> Dict[str, Any]:
    """
    Validates that values in specified columns are valid probabilities (between 0 and 1).
    
    Args:
        df: DataFrame to validate
        probability_columns: List of column names containing probability values
        raise_error: Whether to raise an error if invalid probabilities are found
        
    Returns:
        Dictionary with probability validation results for each column
        
    Raises:
        DataFormatError: If raise_error is True and invalid probabilities are found
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Check each column for probability values
    results = {}
    invalid_probability_columns = []
    
    for column in probability_columns:
        if column not in df.columns:
            msg = f"Column {column} not found in DataFrame"
            logger.warning(msg)
            results[column] = {"valid": False, "message": msg}
            invalid_probability_columns.append(column)
            continue
        
        # Filter out NA values for the check
        col_data = df[column].dropna()
        
        # Check if values are between 0 and 1
        below_zero = col_data < 0
        above_one = col_data > 1
        violations = below_zero | above_one
        
        if violations.any():
            # Get violation details
            violation_count = violations.sum()
            violation_pct = 100 * violation_count / len(col_data)
            
            # Create result dictionary
            results[column] = {
                "valid": False,
                "below_zero_count": below_zero.sum(),
                "above_one_count": above_one.sum(),
                "total_violations": violation_count,
                "violation_percentage": violation_pct,
                "example_violations": df.loc[violations, column].head(5).tolist()
            }
            
            msg = (f"Column {column} has {violation_count} values ({violation_pct:.2f}%) "
                   f"outside probability range [0, 1]")
            logger.warning(msg)
            invalid_probability_columns.append(column)
        else:
            results[column] = {"valid": True}
            logger.debug(f"Column {column} contains valid probability values")
    
    # Overall result
    result = {
        "valid": len(invalid_probability_columns) == 0,
        "column_results": results,
        "invalid_probability_columns": invalid_probability_columns
    }
    
    if not result["valid"] and raise_error:
        msg = f"Probability validation failed for columns: {invalid_probability_columns}"
        logger.error(msg)
        raise DataFormatError(msg)
    
    return result


def validate_input_data(validation_rules: Dict[str, Any]) -> Callable:
    """
    Decorator that validates function input data against specified criteria.
    
    Args:
        validation_rules: Dictionary mapping parameter names to validation rules
        
    Returns:
        Decorated function with input validation
        
    Example:
        >>> @validate_input_data({
        ...     'df': {
        ...         'validator': validate_required_columns,
        ...         'args': [['timestamp', 'price']],
        ...         'kwargs': {'raise_error': True}
        ...     }
        ... })
        ... def process_data(df):
        ...     return df
    """
    def decorator(func: Callable) -> Callable:
        sig = signature(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to parameter names
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Apply validation rules
            for param_name, rule in validation_rules.items():
                if param_name in bound_args.arguments:
                    param_value = bound_args.arguments[param_name]
                    
                    validator = rule.get('validator')
                    validator_args = rule.get('args', [])
                    validator_kwargs = rule.get('kwargs', {})
                    
                    if validator:
                        if isinstance(validator, Callable):
                            # Call the validator with the parameter value and any additional args/kwargs
                            validator(param_value, *validator_args, **validator_kwargs)
                        else:
                            logger.warning(f"Invalid validator for parameter {param_name}: not callable")
            
            # Call the original function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def validate_output_data(validation_rules: Dict[str, Any]) -> Callable:
    """
    Decorator that validates function output data against specified criteria.
    
    Args:
        validation_rules: Dictionary with validation rules for the output
        
    Returns:
        Decorated function with output validation
        
    Example:
        >>> @validate_output_data({
        ...     'validator': validate_data_completeness,
        ...     'kwargs': {'threshold': 0.9, 'raise_error': True}
        ... })
        ... def get_data():
        ...     return pd.DataFrame(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            result = func(*args, **kwargs)
            
            # Apply validation rules to the result
            validator = validation_rules.get('validator')
            validator_args = validation_rules.get('args', [])
            validator_kwargs = validation_rules.get('kwargs', {})
            
            if validator and isinstance(validator, Callable):
                # Call the validator with the result and any additional args/kwargs
                validator(result, *validator_args, **validator_kwargs)
            
            return result
        
        return wrapper
    
    return decorator


def detect_outliers(
    series: SeriesType,
    method: str = 'zscore',
    threshold: float = DEFAULT_OUTLIER_THRESHOLD
) -> SeriesType:
    """
    Detects outliers in a DataFrame column using specified method.
    
    Args:
        series: Series to check for outliers
        method: Method to use for outlier detection ('zscore', 'iqr', or 'percentile')
        threshold: Threshold for outlier detection (units depend on method)
        
    Returns:
        Boolean mask where True indicates an outlier
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pandas Series, got {type(series)}")
    
    # Filter out NA values
    data = series.dropna()
    
    # Initialize result series with False
    outliers = pd.Series(False, index=series.index)
    
    if len(data) == 0:
        logger.warning("Series contains no non-null values, cannot detect outliers")
        return outliers
    
    if method == 'zscore':
        # Z-score method: mark values more than threshold standard deviations from mean
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            logger.warning("Standard deviation is zero, cannot detect outliers using z-score method")
            return outliers
        
        z_scores = (data - mean) / std
        outliers.loc[data.index] = abs(z_scores) > threshold
    
    elif method == 'iqr':
        # IQR method: mark values more than threshold * IQR from quartiles
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            logger.warning("IQR is zero, cannot detect outliers using IQR method")
            return outliers
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers.loc[data.index] = (data < lower_bound) | (data > upper_bound)
    
    elif method == 'percentile':
        # Percentile method: mark values outside specified percentile range
        # Threshold is interpreted as the percentile range (0-100)
        if threshold < 0 or threshold > 100:
            raise ValueError(f"Percentile threshold must be between 0 and 100, got {threshold}")
        
        percentile_range = threshold / 2
        lower_bound = data.quantile(percentile_range / 100)
        upper_bound = data.quantile(1 - percentile_range / 100)
        
        outliers.loc[data.index] = (data < lower_bound) | (data > upper_bound)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Log outlier information
    outlier_count = outliers.sum()
    if outlier_count > 0:
        outlier_pct = 100 * outlier_count / len(series)
        logger.debug(f"Detected {outlier_count} outliers ({outlier_pct:.2f}%) using {method} method")
    
    return outliers


def validate_consistency(
    df: DataFrameType,
    consistency_rules: Dict[str, Callable],
    raise_error: bool = False
) -> Dict[str, Any]:
    """
    Validates data consistency based on custom rules.
    
    Args:
        df: DataFrame to validate
        consistency_rules: Dictionary mapping rule names to rule functions
        raise_error: Whether to raise an error if inconsistencies are found
        
    Returns:
        Dictionary with consistency validation results
        
    Raises:
        DataFormatError: If raise_error is True and inconsistencies are found
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame, got {type(df)}"
        logger.error(msg)
        if raise_error:
            raise DataFormatError(msg)
        return {"valid": False, "message": msg}
    
    # Apply each consistency rule
    results = {}
    failed_rules = []
    
    for rule_name, rule_func in consistency_rules.items():
        try:
            # Rule functions should return True for valid data, False or raise exception otherwise
            if callable(rule_func):
                rule_result = rule_func(df)
                
                if isinstance(rule_result, bool):
                    results[rule_name] = {"valid": rule_result}
                    
                    if not rule_result:
                        msg = f"Consistency rule '{rule_name}' failed"
                        logger.warning(msg)
                        failed_rules.append(rule_name)
                    else:
                        logger.debug(f"Consistency rule '{rule_name}' passed")
                else:
                    # Assume the function returned detailed results
                    results[rule_name] = rule_result
                    if not rule_result.get("valid", False):
                        failed_rules.append(rule_name)
            else:
                msg = f"Consistency rule '{rule_name}' is not callable"
                logger.warning(msg)
                results[rule_name] = {"valid": False, "message": msg}
                failed_rules.append(rule_name)
        
        except Exception as e:
            msg = f"Error applying consistency rule '{rule_name}': {str(e)}"
            logger.error(msg)
            results[rule_name] = {"valid": False, "message": msg, "error": str(e)}
            failed_rules.append(rule_name)
    
    # Overall result
    result = {
        "valid": len(failed_rules) == 0,
        "rule_results": results,
        "failed_rules": failed_rules
    }
    
    if not result["valid"] and raise_error:
        msg = f"Consistency validation failed for rules: {failed_rules}"
        logger.error(msg)
        raise DataFormatError(msg)
    
    return result


class ValidationResult:
    """
    Class that represents the result of a validation operation.
    """
    
    def __init__(
        self,
        is_valid: bool = True,
        errors: List[str] = None,
        warnings: List[str] = None,
        details: Dict[str, Any] = None
    ):
        """
        Initialize the ValidationResult with validation status.
        
        Args:
            is_valid: Whether the validation passed
            errors: List of error messages
            warnings: List of warning messages
            details: Dictionary with detailed validation results
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.details = details or {}
    
    def add_error(self, error_message: str, error_details: Dict[str, Any] = None) -> None:
        """
        Adds an error to the validation result.
        
        Args:
            error_message: Error message
            error_details: Additional details about the error
        """
        self.errors.append(error_message)
        self.is_valid = False
        
        if error_details:
            if "errors" not in self.details:
                self.details["errors"] = []
            self.details["errors"].append(error_details)
    
    def add_warning(self, warning_message: str, warning_details: Dict[str, Any] = None) -> None:
        """
        Adds a warning to the validation result.
        
        Args:
            warning_message: Warning message
            warning_details: Additional details about the warning
        """
        self.warnings.append(warning_message)
        
        if warning_details:
            if "warnings" not in self.details:
                self.details["warnings"] = []
            self.details["warnings"].append(warning_details)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the validation result to a dictionary.
        
        Returns:
            Dictionary representation of the validation result
        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details
        }
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """
        Merges another validation result into this one.
        
        Args:
            other: Another ValidationResult to merge
            
        Returns:
            Self reference with merged results
        """
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        
        # Merge details
        for key, value in other.details.items():
            if key in self.details:
                if isinstance(self.details[key], list) and isinstance(value, list):
                    self.details[key].extend(value)
                elif isinstance(self.details[key], dict) and isinstance(value, dict):
                    self.details[key].update(value)
                else:
                    self.details[key] = value
            else:
                self.details[key] = value
        
        return self


class DataValidator:
    """
    Class that provides methods for comprehensive data validation.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the DataValidator with default settings.
        
        Args:
            strict_mode: Whether to raise exceptions for validation failures
        """
        self._validation_rules = {}
        self._schema_validators = {}
        self._strict_mode = strict_mode
        self._logger = get_logger(__name__)
    
    def add_validation_rule(
        self,
        data_type: str,
        rule_name: str,
        rule_function: Callable,
        rule_params: Dict[str, Any] = None
    ) -> None:
        """
        Adds a validation rule for a specific data type.
        
        Args:
            data_type: Type of data this rule applies to
            rule_name: Name of the rule
            rule_function: Function that implements the rule
            rule_params: Parameters to pass to the rule function
        """
        if data_type not in self._validation_rules:
            self._validation_rules[data_type] = {}
        
        self._validation_rules[data_type][rule_name] = {
            "function": rule_function,
            "params": rule_params or {}
        }
        
        self._logger.debug(f"Added validation rule '{rule_name}' for data type '{data_type}'")
    
    def add_schema_validator(self, data_type: str, schema: Any) -> None:
        """
        Adds a schema validator for a specific data type.
        
        Args:
            data_type: Type of data this schema applies to
            schema: Schema to validate against
        """
        self._schema_validators[data_type] = schema
        self._logger.debug(f"Added schema validator for data type '{data_type}'")
    
    def validate(self, df: DataFrameType, data_type: str) -> Dict[str, Any]:
        """
        Validates data against all applicable rules and schemas.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            Validation results dictionary
            
        Raises:
            DataFormatError: If validation fails and strict mode is enabled
        """
        results = ValidationResult()
        
        # Validate schema if available
        if data_type in self._schema_validators:
            try:
                validate_dataframe_schema(df, self._schema_validators[data_type], strict=self._strict_mode)
                self._logger.debug(f"Schema validation passed for data type '{data_type}'")
            except Exception as e:
                msg = f"Schema validation failed for data type '{data_type}': {str(e)}"
                if self._strict_mode:
                    self._logger.error(msg)
                    raise DataFormatError(msg)
                else:
                    self._logger.warning(msg)
                    results.add_error(msg)
        
        # Apply all applicable validation rules
        if data_type in self._validation_rules:
            for rule_name, rule_info in self._validation_rules[data_type].items():
                rule_function = rule_info["function"]
                rule_params = rule_info["params"]
                
                try:
                    # Apply the rule with parameter dictionary
                    rule_result = rule_function(df, **rule_params)
                    
                    # Interpret the result
                    if isinstance(rule_result, dict) and "valid" in rule_result:
                        if not rule_result["valid"]:
                            msg = f"Validation rule '{rule_name}' failed for data type '{data_type}'"
                            if "message" in rule_result:
                                msg += f": {rule_result['message']}"
                            
                            if self._strict_mode:
                                self._logger.error(msg)
                                raise DataFormatError(msg)
                            else:
                                self._logger.warning(msg)
                                results.add_error(msg, rule_result)
                        else:
                            self._logger.debug(f"Validation rule '{rule_name}' passed for data type '{data_type}'")
                    
                    # Add to details regardless of success
                    results.details[rule_name] = rule_result
                
                except Exception as e:
                    msg = f"Error applying validation rule '{rule_name}' for data type '{data_type}': {str(e)}"
                    if self._strict_mode:
                        self._logger.error(msg)
                        raise DataFormatError(msg)
                    else:
                        self._logger.warning(msg)
                        results.add_error(msg)
        
        return results.to_dict()
    
    def validate_with_rules(self, df: DataFrameType, rule_names: List[str]) -> Dict[str, Any]:
        """
        Validates data against specific validation rules.
        
        Args:
            df: DataFrame to validate
            rule_names: List of rule names to apply
            
        Returns:
            Validation results dictionary
            
        Raises:
            DataFormatError: If validation fails and strict mode is enabled
        """
        results = ValidationResult()
        
        # Collect all matching rules from all data types
        matching_rules = []
        for data_type, rules in self._validation_rules.items():
            for rule_name, rule_info in rules.items():
                if rule_name in rule_names:
                    matching_rules.append((data_type, rule_name, rule_info))
        
        # Apply all matching rules
        for data_type, rule_name, rule_info in matching_rules:
            rule_function = rule_info["function"]
            rule_params = rule_info["params"]
            
            try:
                # Apply the rule with parameter dictionary
                rule_result = rule_function(df, **rule_params)
                
                # Interpret the result
                if isinstance(rule_result, dict) and "valid" in rule_result:
                    if not rule_result["valid"]:
                        msg = f"Validation rule '{rule_name}' failed for data type '{data_type}'"
                        if "message" in rule_result:
                            msg += f": {rule_result['message']}"
                        
                        if self._strict_mode:
                            self._logger.error(msg)
                            raise DataFormatError(msg)
                        else:
                            self._logger.warning(msg)
                            results.add_error(msg, rule_result)
                    else:
                        self._logger.debug(f"Validation rule '{rule_name}' passed for data type '{data_type}'")
                
                # Add to details regardless of success
                results.details[rule_name] = rule_result
            
            except Exception as e:
                msg = f"Error applying validation rule '{rule_name}' for data type '{data_type}': {str(e)}"
                if self._strict_mode:
                    self._logger.error(msg)
                    raise DataFormatError(msg)
                else:
                    self._logger.warning(msg)
                    results.add_error(msg)
        
        return results.to_dict()
    
    def set_strict_mode(self, strict_mode: bool) -> None:
        """
        Sets the strict mode flag for validation.
        
        Args:
            strict_mode: Whether to raise exceptions for validation failures
        """
        self._strict_mode = strict_mode
        self._logger.debug(f"Strict mode {'enabled' if strict_mode else 'disabled'}")