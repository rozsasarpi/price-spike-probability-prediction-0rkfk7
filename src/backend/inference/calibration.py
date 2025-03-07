"""
Implements probability calibration functionality for the ERCOT RTLMP spike prediction system.

This module provides tools to calibrate raw probability predictions from machine learning models
to ensure they accurately reflect the true likelihood of price spike events.
"""

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
from sklearn.calibration import calibration_curve, CalibratedClassifierCV  # version 1.2+
from sklearn.isotonic import IsotonicRegression  # version 1.2+
import matplotlib.pyplot as plt  # version 3.7+
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

# Internal imports
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ModelType, ThresholdValue
from ..utils.logging import get_logger, log_execution_time
from ..utils.validation import validate_probability_values
from ..utils.statistics import calculate_binary_classification_metrics, calculate_probability_metrics
from .threshold_config import ThresholdConfig

# Set up logger
logger = get_logger(__name__)

# Constants
CALIBRATION_METHODS = ['isotonic', 'sigmoid', 'beta']
DEFAULT_N_BINS = 10
DEFAULT_CALIBRATION_METHOD = 'isotonic'


@log_execution_time(logger, 'INFO')
def calibrate_probabilities(
    y_prob: ArrayType,
    y_true: ArrayType,
    y_prob_new: ArrayType,
    method: Optional[str] = None
) -> ArrayType:
    """
    Calibrates raw probability predictions using the specified method.
    
    Args:
        y_prob: Array of predicted probabilities for training
        y_true: Array of true binary labels for training
        y_prob_new: Array of new predicted probabilities to calibrate
        method: Calibration method to use ('isotonic', 'sigmoid', or 'beta')
        
    Returns:
        Calibrated probability predictions
    """
    # Validate inputs
    if not isinstance(y_prob, np.ndarray) or not isinstance(y_true, np.ndarray) or not isinstance(y_prob_new, np.ndarray):
        msg = f"Inputs must be numpy arrays, got: {type(y_prob)}, {type(y_true)}, {type(y_prob_new)}"
        logger.error(msg)
        raise TypeError(msg)
    
    if y_prob.shape != y_true.shape:
        msg = f"y_prob and y_true must have the same shape, got: {y_prob.shape}, {y_true.shape}"
        logger.error(msg)
        raise ValueError(msg)
    
    # Use default method if none specified
    if method is None:
        method = DEFAULT_CALIBRATION_METHOD
    
    # Validate method
    if method not in CALIBRATION_METHODS:
        msg = f"Calibration method must be one of {CALIBRATION_METHODS}, got: {method}"
        logger.error(msg)
        raise ValueError(msg)
    
    # Reshape inputs to 1D arrays if needed
    y_prob_flat = y_prob.ravel()
    y_true_flat = y_true.ravel()
    y_prob_new_flat = y_prob_new.ravel()
    
    try:
        # Perform calibration
        if method == 'isotonic':
            # Use Isotonic Regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_prob_flat, y_true_flat)
            calibrated_probs = calibrator.predict(y_prob_new_flat)
        
        elif method == 'sigmoid':
            # Use Platt scaling via CalibratedClassifierCV
            # We need a dummy classifier that just returns the probabilities
            class DummyClassifier:
                def __init__(self, probs):
                    self.probs = probs
                def predict_proba(self, X):
                    return np.vstack([1-self.probs, self.probs]).T
            
            dummy = DummyClassifier(y_prob_flat)
            calibrator = CalibratedClassifierCV(dummy, cv='prefit', method='sigmoid')
            calibrator.fit(y_prob_flat.reshape(-1, 1), y_true_flat)
            calibrated_probs = calibrator.predict_proba(y_prob_new_flat.reshape(-1, 1))[:, 1]
        
        elif method == 'beta':
            # Use beta distribution-based calibration
            calibrated_probs = beta_calibration(y_prob_flat, y_true_flat, y_prob_new_flat)
        
        else:
            # This should never happen due to validation above
            logger.error(f"Unexpected calibration method: {method}")
            return y_prob_new_flat
        
        # Ensure probabilities are within [0, 1]
        calibrated_probs = np.clip(calibrated_probs, 0, 1)
        
        # Reshape to match input shape if needed
        if y_prob_new.shape != y_prob_new_flat.shape:
            calibrated_probs = calibrated_probs.reshape(y_prob_new.shape)
        
        logger.debug(f"Calibrated probabilities using {method} method")
        return calibrated_probs
    
    except Exception as e:
        logger.error(f"Error calibrating probabilities with method {method}: {str(e)}")
        raise


@log_execution_time(logger, 'INFO')
def evaluate_calibration(
    y_true: ArrayType,
    y_prob: ArrayType,
    n_bins: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluates the calibration quality of probability predictions.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        n_bins: Number of bins for calibration curve calculation
        
    Returns:
        Dictionary with calibration metrics and curve data
    """
    # Validate inputs
    if not isinstance(y_prob, np.ndarray) or not isinstance(y_true, np.ndarray):
        msg = f"Inputs must be numpy arrays, got: {type(y_prob)}, {type(y_true)}"
        logger.error(msg)
        raise TypeError(msg)
    
    if y_prob.shape != y_true.shape:
        msg = f"y_prob and y_true must have the same shape, got: {y_prob.shape}, {y_true.shape}"
        logger.error(msg)
        raise ValueError(msg)
    
    # Use default number of bins if not specified
    if n_bins is None:
        n_bins = DEFAULT_N_BINS
    
    # Calculate Brier score and other probability metrics
    metrics = calculate_probability_metrics(y_true, y_prob)
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Calculate Expected Calibration Error (ECE)
    ece = calculate_expected_calibration_error(y_true, y_prob, n_bins)
    
    # Calculate Maximum Calibration Error (MCE)
    mce = calculate_maximum_calibration_error(y_true, y_prob, n_bins)
    
    # Return dictionary with all metrics
    result = {
        'brier_score': metrics.get('brier_score', None),
        'log_loss': metrics.get('log_loss', None),
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce,
        'calibration_curve': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
    }
    
    if 'auc_roc' in metrics:
        result['auc_roc'] = metrics['auc_roc']
    
    logger.debug(f"Calibration evaluation complete: brier_score={metrics.get('brier_score'):.4f}, ece={ece:.4f}")
    return result


def plot_calibration_curve(
    y_true: ArrayType,
    y_prob: ArrayType,
    n_bins: Optional[int] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the calibration curve (reliability diagram) for probability predictions.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        n_bins: Number of bins for calibration curve calculation
        fig: Optional figure object to plot on
        ax: Optional axes object to plot on
        
    Returns:
        Figure and axes objects
    """
    # Calculate calibration metrics
    cal_metrics = evaluate_calibration(y_true, y_prob, n_bins)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract calibration curve data
    prob_true = np.array(cal_metrics['calibration_curve']['prob_true'])
    prob_pred = np.array(cal_metrics['calibration_curve']['prob_pred'])
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 's-', label='Calibration curve')
    
    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    
    # Set axis labels and title
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    
    # Add metrics to the title
    title = (f"Calibration Curve\nBrier Score: {cal_metrics['brier_score']:.4f}, "
             f"ECE: {cal_metrics['expected_calibration_error']:.4f}")
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid for readability
    ax.grid(True)
    
    return fig, ax


def beta_calibration(
    y_prob: ArrayType,
    y_true: ArrayType,
    y_prob_new: ArrayType
) -> ArrayType:
    """
    Implements beta distribution-based calibration for probability predictions.
    
    Args:
        y_prob: Array of predicted probabilities for training
        y_true: Array of true binary labels for training
        y_prob_new: Array of new predicted probabilities to calibrate
        
    Returns:
        Calibrated probability predictions
    """
    # This is a simplified implementation of Beta calibration
    # Divide predictions into bins
    bin_count = min(10, len(y_prob) // 10)  # Ensure enough samples per bin
    bins = np.linspace(0, 1, bin_count + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)
    
    # Calculate observed frequencies in each bin
    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=bin_count)
    bin_counts = np.bincount(bin_indices, minlength=bin_count)
    bin_means = np.zeros(bin_count)
    
    # Avoid division by zero
    for i in range(bin_count):
        if bin_counts[i] > 0:
            bin_means[i] = bin_sums[i] / bin_counts[i]
        else:
            bin_means[i] = bins[i]
    
    # Use bin midpoints as reference points
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    
    # Fit a mapping function from predicted probabilities to observed frequencies
    # We'll use a simplified approach with interpolation
    from scipy.interpolate import interp1d
    
    # Ensure monotonicity
    for i in range(1, bin_count):
        if bin_means[i] < bin_means[i-1]:
            bin_means[i] = bin_means[i-1]
    
    # Create interpolation function
    # Use 'linear' for simplicity, but 'cubic' might give smoother results
    calibration_func = interp1d(
        bin_midpoints, bin_means, 
        kind='linear', bounds_error=False, 
        fill_value=(bin_means[0], bin_means[-1])
    )
    
    # Apply calibration function to new probabilities
    calibrated_probs = calibration_func(y_prob_new)
    
    # Ensure probabilities are within [0, 1]
    calibrated_probs = np.clip(calibrated_probs, 0, 1)
    
    return calibrated_probs


def calculate_expected_calibration_error(
    y_true: ArrayType,
    y_prob: ArrayType,
    n_bins: int
) -> float:
    """
    Calculates the Expected Calibration Error (ECE) for probability predictions.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        n_bins: Number of bins for calibration error calculation
        
    Returns:
        Expected Calibration Error value
    """
    # Create bins and bin the data
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate bin counts
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Calculate the average predicted probability in each bin
    bin_probs = np.zeros(n_bins)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_probs[i] = np.mean(y_prob[bin_indices == i])
        else:
            bin_probs[i] = (bin_edges[i] + bin_edges[i+1]) / 2
    
    # Calculate the observed frequency of positives in each bin
    bin_freqs = np.zeros(n_bins)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_freqs[i] = np.mean(y_true[bin_indices == i])
        else:
            bin_freqs[i] = 0
    
    # Calculate the absolute difference between predicted probabilities and observed frequencies
    abs_diff = np.abs(bin_probs - bin_freqs)
    
    # Weight each bin's calibration error by the fraction of samples in the bin
    total_samples = len(y_true)
    weighted_errors = abs_diff * (bin_counts / total_samples)
    
    # ECE is the sum of weighted errors
    ece = np.sum(weighted_errors)
    
    return ece


def calculate_maximum_calibration_error(
    y_true: ArrayType,
    y_prob: ArrayType,
    n_bins: int
) -> float:
    """
    Calculates the Maximum Calibration Error (MCE) for probability predictions.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        n_bins: Number of bins for calibration error calculation
        
    Returns:
        Maximum Calibration Error value
    """
    # Create bins and bin the data
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate bin counts
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Calculate the average predicted probability in each bin
    bin_probs = np.zeros(n_bins)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_probs[i] = np.mean(y_prob[bin_indices == i])
        else:
            bin_probs[i] = (bin_edges[i] + bin_edges[i+1]) / 2
    
    # Calculate the observed frequency of positives in each bin
    bin_freqs = np.zeros(n_bins)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_freqs[i] = np.mean(y_true[bin_indices == i])
        else:
            bin_freqs[i] = 0
    
    # Calculate the absolute difference between predicted probabilities and observed frequencies
    abs_diff = np.abs(bin_probs - bin_freqs)
    
    # MCE is the maximum absolute difference
    # Ignore bins with no samples
    mce = 0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            mce = max(mce, abs_diff[i])
    
    return mce


class ProbabilityCalibrator:
    """
    Class for calibrating probability predictions from machine learning models.
    """
    
    def __init__(self, method: Optional[str] = None):
        """
        Initializes the ProbabilityCalibrator with the specified method.
        
        Args:
            method: Calibration method to use ('isotonic', 'sigmoid', or 'beta')
        """
        # Use default method if none specified
        if method is None:
            method = DEFAULT_CALIBRATION_METHOD
        
        # Validate method
        if method not in CALIBRATION_METHODS:
            msg = f"Calibration method must be one of {CALIBRATION_METHODS}, got: {method}"
            logger.error(msg)
            raise ValueError(msg)
        
        self._method = method
        self._calibration_model = None
        self._is_fitted = False
        self._threshold_calibrators = {}  # For threshold-specific calibrators
    
    def fit(self, y_prob: ArrayType, y_true: ArrayType, threshold: Optional[ThresholdValue] = None) -> bool:
        """
        Fits the calibration model on historical predictions and actual outcomes.
        
        Args:
            y_prob: Array of predicted probabilities
            y_true: Array of true binary labels
            threshold: Optional threshold value for threshold-specific calibration
            
        Returns:
            True if fitting was successful, False otherwise
        """
        try:
            # Validate inputs
            if not isinstance(y_prob, np.ndarray) or not isinstance(y_true, np.ndarray):
                logger.error(f"Inputs must be numpy arrays, got: {type(y_prob)}, {type(y_true)}")
                return False
            
            if y_prob.shape != y_true.shape:
                logger.error(f"y_prob and y_true must have the same shape, got: {y_prob.shape}, {y_true.shape}")
                return False
            
            # If threshold is provided, create/update a threshold-specific calibrator
            if threshold is not None:
                # Create a new calibrator for this threshold
                threshold_calibrator = None
                
                if self._method == 'isotonic':
                    threshold_calibrator = IsotonicRegression(out_of_bounds='clip')
                    threshold_calibrator.fit(y_prob.ravel(), y_true.ravel())
                
                elif self._method == 'sigmoid':
                    # We need a dummy classifier that just returns the probabilities
                    class DummyClassifier:
                        def __init__(self, probs):
                            self.probs = probs
                        def predict_proba(self, X):
                            return np.vstack([1-self.probs, self.probs]).T
                    
                    dummy = DummyClassifier(y_prob.ravel())
                    threshold_calibrator = CalibratedClassifierCV(dummy, cv='prefit', method='sigmoid')
                    threshold_calibrator.fit(y_prob.reshape(-1, 1), y_true.ravel())
                
                elif self._method == 'beta':
                    # For beta calibration, store the training data
                    threshold_calibrator = {
                        'y_prob': y_prob.copy(),
                        'y_true': y_true.copy()
                    }
                
                # Store the calibrator for this threshold
                self._threshold_calibrators[threshold] = threshold_calibrator
                logger.debug(f"Fitted calibrator for threshold {threshold} using {self._method} method")
                
            else:
                # Fit the general calibration model
                if self._method == 'isotonic':
                    self._calibration_model = IsotonicRegression(out_of_bounds='clip')
                    self._calibration_model.fit(y_prob.ravel(), y_true.ravel())
                
                elif self._method == 'sigmoid':
                    # We need a dummy classifier that just returns the probabilities
                    class DummyClassifier:
                        def __init__(self, probs):
                            self.probs = probs
                        def predict_proba(self, X):
                            return np.vstack([1-self.probs, self.probs]).T
                    
                    dummy = DummyClassifier(y_prob.ravel())
                    self._calibration_model = CalibratedClassifierCV(dummy, cv='prefit', method='sigmoid')
                    self._calibration_model.fit(y_prob.reshape(-1, 1), y_true.ravel())
                
                elif self._method == 'beta':
                    # For beta calibration, store the training data
                    self._calibration_model = {
                        'y_prob': y_prob.copy(),
                        'y_true': y_true.copy()
                    }
                
                self._is_fitted = True
                logger.debug(f"Fitted general calibrator using {self._method} method")
            
            return True
        
        except Exception as e:
            logger.error(f"Error fitting calibrator: {str(e)}")
            return False
    
    def calibrate(self, y_prob: ArrayType, threshold: Optional[ThresholdValue] = None) -> ArrayType:
        """
        Calibrates new probability predictions using the fitted model.
        
        Args:
            y_prob: Array of predicted probabilities to calibrate
            threshold: Optional threshold value for threshold-specific calibration
            
        Returns:
            Calibrated probability predictions
        """
        # Validate input
        if not isinstance(y_prob, np.ndarray):
            msg = f"Input must be a numpy array, got: {type(y_prob)}"
            logger.error(msg)
            raise TypeError(msg)
        
        # Check if the calibrator is fitted
        if threshold is not None and threshold in self._threshold_calibrators:
            # Use threshold-specific calibrator
            calibrator = self._threshold_calibrators[threshold]
            logger.debug(f"Using threshold-specific calibrator for threshold {threshold}")
        elif self._is_fitted:
            # Use general calibrator
            calibrator = self._calibration_model
            logger.debug("Using general calibrator")
        else:
            msg = "Calibrator is not fitted yet. Call fit() before calibrate()."
            logger.error(msg)
            raise RuntimeError(msg)
        
        try:
            # Apply calibration
            if self._method == 'isotonic':
                calibrated_probs = calibrator.predict(y_prob.ravel())
                # Reshape to match input shape if needed
                if calibrated_probs.shape != y_prob.shape:
                    calibrated_probs = calibrated_probs.reshape(y_prob.shape)
            
            elif self._method == 'sigmoid':
                # Reshape input for predict_proba if needed
                if len(y_prob.shape) == 1:
                    y_prob_reshaped = y_prob.reshape(-1, 1)
                else:
                    y_prob_reshaped = y_prob.ravel().reshape(-1, 1)
                
                calibrated_probs = calibrator.predict_proba(y_prob_reshaped)[:, 1]
                
                # Reshape to match input shape if needed
                if calibrated_probs.shape != y_prob.shape:
                    calibrated_probs = calibrated_probs.reshape(y_prob.shape)
            
            elif self._method == 'beta':
                # Use the beta_calibration function with stored training data
                if threshold is not None and threshold in self._threshold_calibrators:
                    y_prob_train = self._threshold_calibrators[threshold]['y_prob']
                    y_true_train = self._threshold_calibrators[threshold]['y_true']
                else:
                    y_prob_train = self._calibration_model['y_prob']
                    y_true_train = self._calibration_model['y_true']
                
                calibrated_probs = beta_calibration(
                    y_prob_train.ravel(),
                    y_true_train.ravel(),
                    y_prob.ravel()
                )
                
                # Reshape to match input shape if needed
                if calibrated_probs.shape != y_prob.shape:
                    calibrated_probs = calibrated_probs.reshape(y_prob.shape)
            
            # Ensure probabilities are within [0, 1]
            calibrated_probs = np.clip(calibrated_probs, 0, 1)
            
            return calibrated_probs
        
        except Exception as e:
            logger.error(f"Error calibrating probabilities: {str(e)}")
            raise
    
    def calibrate_dataframe(
        self,
        df: DataFrameType,
        probability_columns: List[str],
        column_thresholds: Optional[Dict[str, ThresholdValue]] = None
    ) -> DataFrameType:
        """
        Calibrates probability columns in a DataFrame.
        
        Args:
            df: DataFrame containing probability columns
            probability_columns: List of column names to calibrate
            column_thresholds: Optional dictionary mapping column names to threshold values
            
        Returns:
            DataFrame with calibrated probability columns
        """
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            msg = f"Input must be a pandas DataFrame, got: {type(df)}"
            logger.error(msg)
            raise TypeError(msg)
        
        # Check that all probability columns exist in the DataFrame
        missing_columns = [col for col in probability_columns if col not in df.columns]
        if missing_columns:
            msg = f"Columns not found in DataFrame: {missing_columns}"
            logger.error(msg)
            raise ValueError(msg)
        
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Process each probability column
        for col in probability_columns:
            try:
                # Extract column values as array
                y_prob = df[col].values
                
                # Determine threshold for this column
                threshold = None
                if column_thresholds is not None and col in column_thresholds:
                    threshold = column_thresholds[col]
                
                # Calibrate probabilities
                calibrated_probs = self.calibrate(y_prob, threshold)
                
                # Replace column with calibrated values
                result_df[col] = calibrated_probs
                
                logger.debug(f"Calibrated column {col} with threshold {threshold}")
            
            except Exception as e:
                logger.error(f"Error calibrating column {col}: {str(e)}")
                raise
        
        return result_df
    
    def evaluate(
        self,
        y_prob: ArrayType,
        y_true: ArrayType,
        n_bins: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the calibration quality of probability predictions.
        
        Args:
            y_prob: Array of predicted probabilities
            y_true: Array of true binary labels
            n_bins: Number of bins for calibration curve calculation
            
        Returns:
            Dictionary with calibration metrics
        """
        return evaluate_calibration(y_true, y_prob, n_bins)
    
    def plot_calibration(
        self,
        y_prob: ArrayType,
        y_true: ArrayType,
        n_bins: Optional[int] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the calibration curve for probability predictions.
        
        Args:
            y_prob: Array of predicted probabilities
            y_true: Array of true binary labels
            n_bins: Number of bins for calibration curve calculation
            fig: Optional figure object to plot on
            ax: Optional axes object to plot on
            
        Returns:
            Figure and axes objects
        """
        return plot_calibration_curve(y_true, y_prob, n_bins, fig, ax)
    
    def is_fitted(self, threshold: Optional[ThresholdValue] = None) -> bool:
        """
        Checks if the calibrator has been fitted.
        
        Args:
            threshold: Optional threshold value to check
            
        Returns:
            True if fitted, False otherwise
        """
        if threshold is not None:
            return threshold in self._threshold_calibrators
        else:
            return self._is_fitted
    
    def get_calibration_method(self) -> str:
        """
        Returns the calibration method being used.
        
        Returns:
            Calibration method name
        """
        return self._method
    
    @staticmethod
    def get_supported_methods() -> List[str]:
        """
        Returns a list of supported calibration methods.
        
        Returns:
            List of supported calibration method names
        """
        return CALIBRATION_METHODS


class CalibrationEvaluator:
    """
    Class for evaluating the calibration quality of probability predictions.
    """
    
    def __init__(self, n_bins: Optional[int] = None):
        """
        Initializes the CalibrationEvaluator.
        
        Args:
            n_bins: Number of bins for calibration curve calculation
        """
        # Use default number of bins if not specified
        if n_bins is None:
            self._n_bins = DEFAULT_N_BINS
        else:
            self._n_bins = n_bins
        
        # Dictionary to store evaluation results
        self._evaluation_results = {}
    
    def evaluate(
        self,
        y_prob: ArrayType,
        y_true: ArrayType,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the calibration quality of probability predictions.
        
        Args:
            y_prob: Array of predicted probabilities
            y_true: Array of true binary labels
            model_id: Optional identifier for the model being evaluated
            
        Returns:
            Dictionary with calibration metrics
        """
        # Calculate calibration metrics
        metrics = evaluate_calibration(y_true, y_prob, self._n_bins)
        
        # Store results if model_id is provided
        if model_id is not None:
            self._evaluation_results[model_id] = metrics
            logger.debug(f"Stored calibration evaluation results for model {model_id}")
        
        return metrics
    
    def compare_calibration(
        self,
        model_predictions: Dict[str, Tuple[ArrayType, ArrayType]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compares calibration quality across multiple models.
        
        Args:
            model_predictions: Dictionary mapping model IDs to tuples of (y_prob, y_true)
            
        Returns:
            Dictionary with calibration metrics for each model
        """
        results = {}
        
        # Evaluate each model
        for model_id, (y_prob, y_true) in model_predictions.items():
            results[model_id] = self.evaluate(y_prob, y_true, model_id)
        
        logger.debug(f"Compared calibration across {len(model_predictions)} models")
        return results
    
    def plot_calibration_comparison(
        self,
        model_predictions: Dict[str, Tuple[ArrayType, ArrayType]],
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots calibration curves for multiple models on the same axes.
        
        Args:
            model_predictions: Dictionary mapping model IDs to tuples of (y_prob, y_true)
            fig: Optional figure object to plot on
            ax: Optional axes object to plot on
            
        Returns:
            Figure and axes objects
        """
        # Create figure and axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Add perfect calibration reference line
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
        
        # Plot calibration curve for each model with different colors
        for model_id, (y_prob, y_true) in model_predictions.items():
            # Calculate calibration metrics
            metrics = evaluate_calibration(y_true, y_prob, self._n_bins)
            
            # Extract calibration curve data
            prob_true = np.array(metrics['calibration_curve']['prob_true'])
            prob_pred = np.array(metrics['calibration_curve']['prob_pred'])
            
            # Plot calibration curve for this model
            ax.plot(
                prob_pred, prob_true, 'o-',
                label=f"{model_id} (Brier: {metrics['brier_score']:.4f}, ECE: {metrics['expected_calibration_error']:.4f})"
            )
        
        # Set axis labels and title
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Curve Comparison')
        
        # Add legend
        ax.legend(loc='best')
        
        # Add grid for readability
        ax.grid(True)
        
        return fig, ax
    
    def get_results(self, model_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Returns stored evaluation results.
        
        Args:
            model_id: Optional identifier for a specific model's results
            
        Returns:
            Dictionary with evaluation results
        """
        if model_id is not None:
            # Return results for a specific model if it exists
            if model_id in self._evaluation_results:
                return {model_id: self._evaluation_results[model_id]}
            else:
                logger.warning(f"No evaluation results found for model {model_id}")
                return {}
        else:
            # Return all evaluation results
            return self._evaluation_results
    
    def set_n_bins(self, n_bins: int) -> None:
        """
        Sets the number of bins for calibration evaluation.
        
        Args:
            n_bins: Number of bins for calibration curve calculation
        """
        if n_bins <= 0:
            msg = f"Number of bins must be positive, got: {n_bins}"
            logger.error(msg)
            raise ValueError(msg)
        
        self._n_bins = n_bins
        logger.debug(f"Set number of bins to {n_bins}")