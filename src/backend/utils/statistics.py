"""
Utility module providing statistical functions for time series data analysis in the ERCOT RTLMP spike prediction system.

This module implements various statistical calculations including rolling statistics, volatility metrics,
quantile analysis, and spike frequency calculations that are used in feature engineering and model evaluation.
"""

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
from scipy import stats  # version 1.10+
from typing import List, Dict, Optional, Union, Tuple, Any, Callable

from .type_definitions import DataFrameType, SeriesType, ArrayType
from .validation import validate_dataframe_schema
from .logging import get_logger, log_execution_time

# Set up logger
logger = get_logger(__name__)

# Default configuration values
DEFAULT_ROLLING_WINDOWS = [1, 6, 12, 24, 48, 72, 168]  # Default window sizes in hours
DEFAULT_QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]  # Default quantile values
DEFAULT_STATISTICS = ['mean', 'std', 'min', 'max', 'median']  # Default statistical measures

@log_execution_time(logger, 'INFO')
def calculate_rolling_statistics(
    series: SeriesType,
    windows: Optional[List[int]] = None,
    statistics: Optional[List[str]] = None
) -> DataFrameType:
    """
    Calculate rolling statistical measures for a time series.
    
    Args:
        series: Time series data as pandas Series
        windows: List of window sizes in hours, defaults to DEFAULT_ROLLING_WINDOWS
        statistics: List of statistics to calculate, defaults to DEFAULT_STATISTICS
    
    Returns:
        DataFrame with rolling statistics for each window and statistic
    """
    # Validate input series is not empty
    if series.empty:
        logger.warning("Empty series provided to calculate_rolling_statistics")
        return pd.DataFrame(index=series.index)
    
    # Use default values if not provided
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    
    if statistics is None:
        statistics = DEFAULT_STATISTICS
    
    # Initialize an empty DataFrame to store results
    result = pd.DataFrame(index=series.index)
    
    # Calculate rolling statistics for each window size
    for window in windows:
        rolling = series.rolling(window=window, min_periods=1)
        
        # Calculate each requested statistic
        if 'mean' in statistics:
            result[f'rolling_mean_{window}h'] = rolling.mean()
        
        if 'std' in statistics:
            result[f'rolling_std_{window}h'] = rolling.std()
        
        if 'min' in statistics:
            result[f'rolling_min_{window}h'] = rolling.min()
        
        if 'max' in statistics:
            result[f'rolling_max_{window}h'] = rolling.max()
        
        if 'median' in statistics:
            result[f'rolling_median_{window}h'] = rolling.median()
    
    logger.debug(f"Calculated {len(statistics)} rolling statistics for {len(windows)} window sizes")
    return result

@log_execution_time(logger, 'INFO')
def calculate_price_volatility(
    series: SeriesType,
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Calculate price volatility metrics for a time series.
    
    Args:
        series: Time series data as pandas Series
        windows: List of window sizes in hours, defaults to DEFAULT_ROLLING_WINDOWS
    
    Returns:
        DataFrame with volatility metrics for each window
    """
    # Validate input series is not empty
    if series.empty:
        logger.warning("Empty series provided to calculate_price_volatility")
        return pd.DataFrame(index=series.index)
    
    # Use default windows if not provided
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    
    # Initialize an empty DataFrame to store results
    result = pd.DataFrame(index=series.index)
    
    # Calculate percentage changes in the series
    pct_change = series.pct_change().fillna(0)
    
    # Calculate volatility metrics for each window
    for window in windows:
        # Calculate rolling standard deviation of percentage changes (volatility)
        result[f'volatility_{window}h'] = pct_change.rolling(window=window, min_periods=1).std()
        
        # Calculate rolling range (max-min) normalized by mean
        rolling = series.rolling(window=window, min_periods=1)
        rolling_range = rolling.max() - rolling.min()
        rolling_mean = rolling.mean()
        # Avoid division by zero
        result[f'normalized_range_{window}h'] = rolling_range / rolling_mean.replace(0, np.nan)
        
        # Calculate rolling coefficient of variation (std/mean)
        rolling_std = rolling.std()
        # Avoid division by zero
        result[f'cv_{window}h'] = rolling_std / rolling_mean.replace(0, np.nan)
    
    # Replace NaN values with 0 for better compatibility
    result = result.fillna(0)
    
    logger.debug(f"Calculated volatility metrics for {len(windows)} window sizes")
    return result

@log_execution_time(logger, 'INFO')
def calculate_spike_frequency(
    series: SeriesType,
    thresholds: List[float],
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Calculate frequency of price spikes above specified thresholds.
    
    Args:
        series: Time series data as pandas Series
        thresholds: List of threshold values to define spikes
        windows: List of window sizes in hours, defaults to DEFAULT_ROLLING_WINDOWS
    
    Returns:
        DataFrame with spike frequencies for each threshold and window
    """
    # Validate input series is not empty
    if series.empty:
        logger.warning("Empty series provided to calculate_spike_frequency")
        return pd.DataFrame(index=series.index)
    
    # Use default windows if not provided
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    
    # Initialize an empty DataFrame to store results
    result = pd.DataFrame(index=series.index)
    
    # Calculate spike frequencies for each threshold
    for threshold in thresholds:
        # Create binary indicator series (1 where value > threshold, 0 otherwise)
        spike_indicator = (series > threshold).astype(int)
        
        # Calculate rolling sum and frequency for each window
        for window in windows:
            # Calculate rolling sum (count of spikes)
            spike_count = spike_indicator.rolling(window=window, min_periods=1).sum()
            
            # Calculate rolling frequency (count / window size)
            # Use max(1, window) to avoid division by zero
            window_size = max(1, window)
            result[f'spike_freq_{threshold}_{window}h'] = spike_count / window_size
    
    logger.debug(f"Calculated spike frequencies for {len(thresholds)} thresholds and {len(windows)} window sizes")
    return result

@log_execution_time(logger, 'INFO')
def calculate_quantiles(
    series: SeriesType,
    quantiles: Optional[List[float]] = None,
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Calculate quantile values for a series over rolling windows.
    
    Args:
        series: Time series data as pandas Series
        quantiles: List of quantile values to calculate, defaults to DEFAULT_QUANTILES
        windows: List of window sizes in hours, defaults to DEFAULT_ROLLING_WINDOWS
    
    Returns:
        DataFrame with quantile values for each quantile and window
    """
    # Validate input series is not empty
    if series.empty:
        logger.warning("Empty series provided to calculate_quantiles")
        return pd.DataFrame(index=series.index)
    
    # Use default values if not provided
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES
    
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    
    # Initialize an empty DataFrame to store results
    result = pd.DataFrame(index=series.index)
    
    # Calculate quantiles for each window size
    for window in windows:
        rolling = series.rolling(window=window, min_periods=1)
        
        # Calculate each requested quantile
        for q in quantiles:
            # Use 'quantile' method for rolling window
            result[f'quantile_{q}_{window}h'] = rolling.quantile(q)
    
    logger.debug(f"Calculated {len(quantiles)} quantiles for {len(windows)} window sizes")
    return result

@log_execution_time(logger, 'INFO')
def calculate_distribution_metrics(
    series: SeriesType,
    windows: Optional[List[int]] = None
) -> DataFrameType:
    """
    Calculate distribution metrics (skewness, kurtosis) for a series.
    
    Args:
        series: Time series data as pandas Series
        windows: List of window sizes in hours, defaults to DEFAULT_ROLLING_WINDOWS
    
    Returns:
        DataFrame with distribution metrics for each window
    """
    # Validate input series is not empty
    if series.empty:
        logger.warning("Empty series provided to calculate_distribution_metrics")
        return pd.DataFrame(index=series.index)
    
    # Use default windows if not provided
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS
    
    # Initialize an empty DataFrame to store results
    result = pd.DataFrame(index=series.index)
    
    # Calculate distribution metrics for each window size
    for window in windows:
        # Require at least 4 points for meaningful skewness and kurtosis calculations
        min_periods = min(4, window)
        rolling = series.rolling(window=window, min_periods=min_periods)
        
        # Calculate skewness using scipy.stats
        def rolling_skewness(x):
            if len(x) < min_periods:
                return np.nan
            return stats.skew(x, nan_policy='omit')
        
        result[f'skewness_{window}h'] = rolling.apply(rolling_skewness, raw=True)
        
        # Calculate kurtosis using scipy.stats
        def rolling_kurtosis(x):
            if len(x) < min_periods:
                return np.nan
            return stats.kurtosis(x, nan_policy='omit')
        
        result[f'kurtosis_{window}h'] = rolling.apply(rolling_kurtosis, raw=True)
    
    # Fill NaN values with 0 for better compatibility
    result = result.fillna(0)
    
    logger.debug(f"Calculated distribution metrics for {len(windows)} window sizes")
    return result

@log_execution_time(logger, 'INFO')
def calculate_autocorrelation(
    series: SeriesType,
    lags: Optional[List[int]] = None
) -> DataFrameType:
    """
    Calculate autocorrelation of a series at specified lags.
    
    Args:
        series: Time series data as pandas Series
        lags: List of lag values to calculate, defaults to [1, 2, 3, 6, 12, 24, 48, 72]
    
    Returns:
        DataFrame with autocorrelation values for each lag
    """
    # Validate input series is not empty
    if series.empty:
        logger.warning("Empty series provided to calculate_autocorrelation")
        return pd.DataFrame()
    
    # Use default lags if not provided
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 72]
    
    # Initialize an empty DataFrame to store results
    result = pd.DataFrame(index=[0])  # Single row for autocorrelation values
    
    # Calculate autocorrelation for each lag
    for lag in lags:
        try:
            # Use pandas autocorr method
            autocorr_value = series.autocorr(lag=lag)
            result[f'autocorr_{lag}'] = autocorr_value
        except Exception as e:
            logger.warning(f"Error calculating autocorrelation at lag {lag}: {str(e)}")
            result[f'autocorr_{lag}'] = np.nan
    
    logger.debug(f"Calculated autocorrelation for {len(lags)} lags")
    return result

@log_execution_time(logger, 'INFO')
def calculate_crosscorrelation(
    series1: SeriesType,
    series2: SeriesType,
    lags: Optional[List[int]] = None
) -> DataFrameType:
    """
    Calculate cross-correlation between two series at specified lags.
    
    Args:
        series1: First time series data as pandas Series
        series2: Second time series data as pandas Series
        lags: List of lag values to calculate, defaults to [-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24]
    
    Returns:
        DataFrame with cross-correlation values for each lag
    """
    # Validate input series are not empty
    if series1.empty or series2.empty:
        logger.warning("Empty series provided to calculate_crosscorrelation")
        return pd.DataFrame()
    
    # Validate that series have the same index
    if not series1.index.equals(series2.index):
        logger.warning("Series have different indices in calculate_crosscorrelation")
        # Reindex series to use the intersection of indices
        common_index = series1.index.intersection(series2.index)
        series1 = series1.loc[common_index]
        series2 = series2.loc[common_index]
    
    # Use default lags if not provided
    if lags is None:
        lags = [-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24]
    
    # Initialize an empty DataFrame to store results
    result = pd.DataFrame(index=[0])  # Single row for correlation values
    
    # Calculate cross-correlation for each lag
    for lag in lags:
        try:
            # Shift series2 by the lag value
            shifted_series = series2.shift(periods=-lag)
            
            # Calculate correlation between series1 and shifted series2
            # Use dropna to handle missing values at the beginning/end after shifting
            valid_indices = series1.notna() & shifted_series.notna()
            s1_valid = series1[valid_indices]
            s2_valid = shifted_series[valid_indices]
            
            if len(s1_valid) > 1:  # Need at least 2 points for correlation
                corr_value = s1_valid.corr(s2_valid)
                result[f'crosscorr_{lag}'] = corr_value
            else:
                result[f'crosscorr_{lag}'] = np.nan
        except Exception as e:
            logger.warning(f"Error calculating cross-correlation at lag {lag}: {str(e)}")
            result[f'crosscorr_{lag}'] = np.nan
    
    logger.debug(f"Calculated cross-correlation for {len(lags)} lags")
    return result

@log_execution_time(logger, 'INFO')
def calculate_hourly_statistics(
    df: DataFrameType,
    value_column: str,
    timestamp_column: str
) -> DataFrameType:
    """
    Aggregate 5-minute data to hourly statistics.
    
    Args:
        df: DataFrame containing 5-minute data
        value_column: Name of the column containing values to aggregate
        timestamp_column: Name of the column containing timestamps
    
    Returns:
        DataFrame with hourly statistics
    """
    # Validate input DataFrame has required columns
    if value_column not in df.columns or timestamp_column not in df.columns:
        logger.warning(f"Required columns {value_column} or {timestamp_column} not found in DataFrame")
        return pd.DataFrame()
    
    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        logger.info(f"Converting {timestamp_column} to datetime type")
        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Create hour start timestamps by flooring to the hour
    df['hour_start'] = df[timestamp_column].dt.floor('H')
    
    # Group data by hour start timestamps
    grouped = df.groupby('hour_start')
    
    # Calculate hourly statistics
    hourly_stats = pd.DataFrame({
        'mean': grouped[value_column].mean(),
        'min': grouped[value_column].min(),
        'max': grouped[value_column].max(),
        'std': grouped[value_column].std(),
        'range': grouped[value_column].max() - grouped[value_column].min(),
        'count': grouped[value_column].count()
    })
    
    # Reset index to convert groupby result back to DataFrame
    hourly_stats.reset_index(inplace=True)
    
    logger.debug(f"Aggregated {len(df)} records to {len(hourly_stats)} hourly statistics")
    return hourly_stats

def calculate_binary_classification_metrics(
    y_true: ArrayType,
    y_pred: ArrayType
) -> Dict[str, float]:
    """
    Calculate binary classification metrics from actual and predicted values.
    
    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted binary labels
    
    Returns:
        Dictionary of classification metrics
    """
    # Validate input arrays have the same shape
    if y_true.shape != y_pred.shape:
        logger.error(f"Input arrays have different shapes: {y_true.shape} vs {y_pred.shape}")
        return {}
    
    # Calculate confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    # Avoid division by zero
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Assemble metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    logger.debug(f"Calculated binary classification metrics: accuracy={accuracy:.4f}, "
                 f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    return metrics

def calculate_probability_metrics(
    y_true: ArrayType,
    y_prob: ArrayType
) -> Dict[str, float]:
    """
    Calculate metrics for probability predictions.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
    
    Returns:
        Dictionary of probability-based metrics
    """
    # Validate input arrays have the same shape
    if y_true.shape != y_prob.shape:
        logger.error(f"Input arrays have different shapes: {y_true.shape} vs {y_prob.shape}")
        return {}
    
    # Calculate Brier score: mean squared difference between predicted probabilities and outcomes
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    # Calculate log loss (negative log likelihood)
    # Clip probabilities to avoid log(0) or log(1)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    log_loss = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
    
    # Calculate AUC-ROC if scikit-learn is available
    auc_roc = None
    try:
        from sklearn.metrics import roc_auc_score  # scikit-learn version 1.2+
        auc_roc = roc_auc_score(y_true, y_prob)
    except ImportError:
        logger.warning("scikit-learn not available, AUC-ROC not calculated")
    except Exception as e:
        logger.warning(f"Error calculating AUC-ROC: {str(e)}")
    
    # Assemble metrics dictionary
    metrics = {
        'brier_score': brier_score,
        'log_loss': log_loss
    }
    
    if auc_roc is not None:
        metrics['auc_roc'] = auc_roc
    
    logger.debug(f"Calculated probability metrics: brier_score={brier_score:.4f}, log_loss={log_loss:.4f}")
    return metrics

def calculate_confidence_intervals(
    series: SeriesType,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence intervals for a series of values.
    
    Args:
        series: Series of values
        confidence_level: Confidence level (0 to 1)
    
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    # Validate input series is not empty
    if series.empty:
        logger.warning("Empty series provided to calculate_confidence_intervals")
        return (np.nan, np.nan)
    
    # Validate confidence level is between 0 and 1
    if confidence_level <= 0 or confidence_level >= 1:
        logger.error(f"Confidence level must be between 0 and 1, got {confidence_level}")
        return (np.nan, np.nan)
    
    # Calculate mean and standard error
    mean = series.mean()
    std_err = series.std() / np.sqrt(len(series))
    
    # Calculate critical value from t-distribution
    alpha = 1 - confidence_level
    degrees_freedom = max(1, len(series) - 1)  # Ensure at least 1 degree of freedom
    t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
    
    # Calculate margin of error
    margin_of_error = t_critical * std_err
    
    # Calculate lower and upper bounds of confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    logger.debug(f"Calculated {confidence_level*100:.1f}% confidence interval: "
                 f"({lower_bound:.4f}, {upper_bound:.4f})")
    return (lower_bound, upper_bound)