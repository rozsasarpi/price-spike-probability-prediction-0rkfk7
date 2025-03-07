"""
Module for evaluating model performance in the ERCOT RTLMP spike prediction system.

Provides comprehensive metrics calculation, model validation, and performance assessment 
functionality to ensure model quality and reliability.
"""

# Standard library imports
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

# External imports
import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,
    log_loss, confusion_matrix
)  # version 1.2+
import matplotlib.pyplot as plt  # version 3.7+

# Internal imports
from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, ThresholdValue, ModelType
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import ModelError, handle_errors
from ..utils.statistics import calculate_binary_classification_metrics, calculate_probability_metrics
from ..utils.validation import validate_probability_values
from .base_model import BaseModel
from ..inference.calibration import evaluate_calibration

# Initialize logger
logger = get_logger(__name__)

# Default values
DEFAULT_CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
DEFAULT_PROBABILITY_METRICS = ["brier_score", "log_loss"]
DEFAULT_CALIBRATION_METRICS = ["expected_calibration_error", "maximum_calibration_error"]
DEFAULT_THRESHOLD = 0.5


@log_execution_time(logger, 'INFO')
@handle_errors(ModelError, reraise=True)
def evaluate_model_performance(
    model: BaseModel,
    features: DataFrameType,
    targets: SeriesType,
    metrics: Optional[List[str]] = None,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluates model performance using various metrics.
    
    Args:
        model: Trained model to evaluate
        features: Feature DataFrame for evaluation
        targets: True target values
        metrics: List of metrics to calculate, defaults to classification and probability metrics
        threshold: Probability threshold for binary classification, defaults to 0.5
        
    Returns:
        Dictionary of performance metrics
    
    Raises:
        ModelError: If model evaluation fails
    """
    # Validate that model is trained
    if not model.is_trained():
        raise ModelError("Model must be trained before evaluation")
    
    # Validate that features contain required columns
    if not model.validate_features(features):
        raise ModelError("Features do not contain required columns for this model")
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS + DEFAULT_PROBABILITY_METRICS
    
    # Use default threshold if none provided
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    
    # Generate probability predictions
    y_prob = model.predict_proba(features)
    
    # Ensure y_prob is 1D array
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        # For multi-class, use the positive class probability (class 1)
        y_prob = y_prob[:, 1]
    
    # Generate binary predictions using threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Convert targets to numpy array if it's a Series
    y_true = targets.values if isinstance(targets, pd.Series) else targets
    
    # Calculate classification metrics
    classification_metrics = calculate_binary_classification_metrics(y_true, y_pred)
    
    # Calculate probability metrics
    probability_metrics = calculate_probability_metrics(y_true, y_prob)
    
    # Calculate calibration metrics
    calibration_metrics = evaluate_calibration(y_true, y_prob)
    
    # Combine all metrics
    all_metrics = {}
    all_metrics.update(classification_metrics)
    all_metrics.update(probability_metrics)
    
    # Add calibration metrics
    if 'expected_calibration_error' in calibration_metrics:
        all_metrics['expected_calibration_error'] = calibration_metrics['expected_calibration_error']
    if 'maximum_calibration_error' in calibration_metrics:
        all_metrics['maximum_calibration_error'] = calibration_metrics['maximum_calibration_error']
    
    # Filter metrics to only those requested
    result_metrics = {k: v for k, v in all_metrics.items() if k in metrics}
    
    # Update model's performance metrics
    model.set_performance_metrics(result_metrics)
    
    return result_metrics


def calculate_confusion_matrix(y_true: SeriesType, y_pred: ArrayType) -> Dict[str, int]:
    """
    Calculates confusion matrix for binary predictions.
    
    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted binary labels
        
    Returns:
        Dictionary with confusion matrix values (TP, FP, TN, FN)
    """
    # Ensure inputs are numpy arrays
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_arr = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
    
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def calculate_threshold_metrics(
    y_true: SeriesType,
    y_prob: ArrayType,
    thresholds: Optional[List[float]] = None
) -> DataFrameType:
    """
    Calculates metrics at different probability thresholds.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        thresholds: List of threshold values to evaluate, defaults to range from 0.05 to 0.95
        
    Returns:
        DataFrame with metrics for each threshold
    """
    # Validate inputs have the same shape
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    if y_true_arr.shape != y_prob.shape:
        raise ValueError(f"Input arrays have different shapes: {y_true_arr.shape} vs {y_prob.shape}")
    
    # Use default thresholds if none provided
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    
    # Initialize lists to store metrics
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = calculate_confusion_matrix(y_true_arr, y_pred)
        tp = cm["true_positives"]
        fp = cm["false_positives"]
        tn = cm["true_negatives"]
        fn = cm["false_negatives"]
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Append to lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision_list,
        'recall': recall_list,
        'f1_score': f1_list,
        'accuracy': accuracy_list
    })
    
    return results


def calculate_roc_curve(y_true: SeriesType, y_prob: ArrayType) -> Tuple[ArrayType, ArrayType, float]:
    """
    Calculates ROC curve points and AUC.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        
    Returns:
        Tuple of (fpr, tpr, auc_value)
    """
    # Ensure inputs are numpy arrays
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob)
    
    # Calculate AUC
    auc_value = roc_auc_score(y_true_arr, y_prob)
    
    return fpr, tpr, auc_value


def calculate_precision_recall_curve(y_true: SeriesType, y_prob: ArrayType) -> Tuple[ArrayType, ArrayType, float]:
    """
    Calculates precision-recall curve points and average precision.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        
    Returns:
        Tuple of (precision, recall, average_precision)
    """
    # Ensure inputs are numpy arrays
    y_true_arr = y_true.values if isinstance(y_true, pd.Series) else y_true
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true_arr, y_prob)
    
    # Calculate average precision
    avg_precision = average_precision_score(y_true_arr, y_prob)
    
    return precision, recall, avg_precision


@log_execution_time(logger, 'INFO')
def evaluate_model_by_threshold(
    model: BaseModel,
    features: DataFrameType,
    targets_by_threshold: DataFrameType,
    price_thresholds: List[ThresholdValue],
    metrics: Optional[List[str]] = None
) -> Dict[ThresholdValue, Dict[str, float]]:
    """
    Evaluates model performance for different price thresholds.
    
    Args:
        model: Trained model to evaluate
        features: Feature DataFrame for evaluation
        targets_by_threshold: DataFrame with target columns for each price threshold
        price_thresholds: List of price threshold values
        metrics: List of metrics to calculate, defaults to classification and probability metrics
        
    Returns:
        Dictionary mapping price thresholds to metric dictionaries
    """
    # Validate that model is trained
    if not model.is_trained():
        raise ModelError("Model must be trained before evaluation")
    
    # Validate that features contain required columns
    if not model.validate_features(features):
        raise ModelError("Features do not contain required columns for this model")
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS + DEFAULT_PROBABILITY_METRICS
    
    # Initialize results dictionary
    results = {}
    
    # Evaluate for each price threshold
    for threshold in price_thresholds:
        # Get target column for this threshold
        # Assume the column name follows a pattern like "spike_occurred_100.0" for threshold 100.0
        target_col = f"spike_occurred_{threshold}"
        
        if target_col not in targets_by_threshold.columns:
            logger.warning(f"Target column {target_col} not found in targets DataFrame")
            continue
        
        # Get target values for this threshold
        targets = targets_by_threshold[target_col]
        
        # Generate probability predictions
        y_prob = model.predict_proba(features)
        
        # Ensure y_prob is 1D array
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            # For multi-class, use the positive class probability (class 1)
            y_prob = y_prob[:, 1]
        
        # Evaluate model performance for this threshold
        threshold_metrics = evaluate_model_performance(model, features, targets, metrics)
        
        # Store results for this threshold
        results[threshold] = threshold_metrics
    
    return results


@log_execution_time(logger, 'INFO')
def evaluate_model_over_time(
    model: BaseModel,
    features: DataFrameType,
    targets: SeriesType,
    time_column: str,
    time_grouping: str,
    metrics: Optional[List[str]] = None
) -> DataFrameType:
    """
    Evaluates model performance over different time periods.
    
    Args:
        model: Trained model to evaluate
        features: Feature DataFrame for evaluation
        targets: True target values
        time_column: Name of the column containing timestamps
        time_grouping: Time period to group by ('hour', 'day', 'month', etc.)
        metrics: List of metrics to calculate, defaults to classification metrics
        
    Returns:
        DataFrame with metrics by time period
    """
    # Validate that model is trained
    if not model.is_trained():
        raise ModelError("Model must be trained before evaluation")
    
    # Validate that features contain required columns
    if not model.validate_features(features):
        raise ModelError("Features do not contain required columns for this model")
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS
    
    # Generate probability predictions
    y_prob = model.predict_proba(features)
    
    # Ensure y_prob is 1D array
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        # For multi-class, use the positive class probability (class 1)
        y_prob = y_prob[:, 1]
    
    # Create DataFrame with predictions, targets, and time column
    eval_df = pd.DataFrame({
        'predictions': y_prob,
        'targets': targets.values if isinstance(targets, pd.Series) else targets,
        'time': features[time_column] if time_column in features.columns else pd.to_datetime(features.index)
    })
    
    # Extract time component based on time_grouping
    if time_grouping == 'hour':
        eval_df['time_group'] = eval_df['time'].dt.hour
        group_name = 'Hour'
    elif time_grouping == 'day':
        eval_df['time_group'] = eval_df['time'].dt.day
        group_name = 'Day'
    elif time_grouping == 'weekday':
        eval_df['time_group'] = eval_df['time'].dt.dayofweek
        group_name = 'Weekday'
    elif time_grouping == 'month':
        eval_df['time_group'] = eval_df['time'].dt.month
        group_name = 'Month'
    elif time_grouping == 'season':
        # Define seasons (Dec-Feb: Winter, Mar-May: Spring, Jun-Aug: Summer, Sep-Nov: Fall)
        month = eval_df['time'].dt.month
        eval_df['time_group'] = np.select(
            [
                (month >= 3) & (month <= 5),
                (month >= 6) & (month <= 8),
                (month >= 9) & (month <= 11),
                (month == 12) | (month <= 2)
            ],
            ['Spring', 'Summer', 'Fall', 'Winter']
        )
        group_name = 'Season'
    else:
        raise ValueError(f"Unsupported time_grouping: {time_grouping}")
    
    # Initialize results list
    results = []
    
    # Group by time group and calculate metrics
    for group_value, group_data in eval_df.groupby('time_group'):
        y_true = group_data['targets'].values
        y_prob = group_data['predictions'].values
        
        # Skip groups with too few samples
        if len(y_true) < 5:
            logger.warning(f"Skipping {group_name} {group_value} with only {len(y_true)} samples")
            continue
        
        # Generate binary predictions with default threshold
        y_pred = (y_prob >= DEFAULT_THRESHOLD).astype(int)
        
        # Calculate classification metrics
        classification_metrics = calculate_binary_classification_metrics(y_true, y_pred)
        
        # Calculate probability metrics
        probability_metrics = calculate_probability_metrics(y_true, y_prob)
        
        # Combine metrics
        all_metrics = {}
        all_metrics.update(classification_metrics)
        all_metrics.update(probability_metrics)
        
        # Filter metrics to only those requested
        group_metrics = {k: v for k, v in all_metrics.items() if k in metrics}
        
        # Add time group information
        group_metrics[group_name.lower()] = group_value
        group_metrics['sample_count'] = len(y_true)
        
        # Append to results
        results.append(group_metrics)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Set time group as index if available
    if group_name.lower() in results_df.columns:
        results_df = results_df.set_index(group_name.lower())
    
    return results_df


@log_execution_time(logger, 'INFO')
def compare_models(
    models: List[BaseModel],
    features: DataFrameType,
    targets: SeriesType,
    metrics: Optional[List[str]] = None
) -> DataFrameType:
    """
    Compares performance of multiple models on the same dataset.
    
    Args:
        models: List of models to compare
        features: Feature DataFrame for evaluation
        targets: True target values
        metrics: List of metrics to calculate, defaults to all metrics
        
    Returns:
        DataFrame with metrics for each model
    """
    # Use default metrics if none provided
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS + DEFAULT_PROBABILITY_METRICS
    
    # Initialize results list
    results = []
    
    # Evaluate each model
    for model in models:
        # Validate that model is trained
        if not model.is_trained():
            logger.warning(f"Model {model.model_id} is not trained, skipping")
            continue
        
        # Validate that features contain required columns
        if not model.validate_features(features):
            logger.warning(f"Features do not contain required columns for model {model.model_id}, skipping")
            continue
        
        # Evaluate model performance
        model_metrics = evaluate_model_performance(model, features, targets, metrics)
        
        # Add model identifier
        model_info = {
            'model_id': model.model_id,
            'model_type': model.model_type,
            'version': model.version if model.version else 'N/A'
        }
        model_info.update(model_metrics)
        
        # Add to results
        results.append(model_info)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Set model_id as index if available
    if 'model_id' in results_df.columns:
        results_df = results_df.set_index('model_id')
    
    return results_df


def plot_roc_curve(
    y_true: SeriesType,
    y_prob: ArrayType,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots ROC curve for model evaluation.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        fig: Optional figure object to plot on
        ax: Optional axes object to plot on
        
    Returns:
        Figure and axes objects
    """
    # Calculate ROC curve
    fpr, tpr, auc_value = calculate_roc_curve(y_true, y_prob)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_value:.3f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Set axis labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic (AUC = {auc_value:.3f})')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    return fig, ax


def plot_precision_recall_curve(
    y_true: SeriesType,
    y_prob: ArrayType,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots precision-recall curve for model evaluation.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        fig: Optional figure object to plot on
        ax: Optional axes object to plot on
        
    Returns:
        Figure and axes objects
    """
    # Calculate precision-recall curve
    precision, recall, avg_precision = calculate_precision_recall_curve(y_true, y_prob)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    
    # Set axis labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (AP = {avg_precision:.3f})')
    ax.legend(loc='lower left')
    ax.grid(True)
    
    return fig, ax


def plot_calibration_curve(
    y_true: SeriesType,
    y_prob: ArrayType,
    n_bins: Optional[int] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots calibration curve (reliability diagram) for model evaluation.
    
    Args:
        y_true: Array of true binary labels
        y_prob: Array of predicted probabilities
        n_bins: Number of bins for calibration curve calculation
        fig: Optional figure object to plot on
        ax: Optional axes object to plot on
        
    Returns:
        Figure and axes objects
    """
    # Set default n_bins if not provided
    if n_bins is None:
        n_bins = 10
    
    # Get calibration curve data
    calibration_data = evaluate_calibration(y_true, y_prob, n_bins)
    prob_true = np.array(calibration_data['calibration_curve']['prob_true'])
    prob_pred = np.array(calibration_data['calibration_curve']['prob_pred'])
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
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
    title = (f"Calibration Curve\nBrier Score: {calibration_data['brier_score']:.4f}, "
             f"ECE: {calibration_data['expected_calibration_error']:.4f}")
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid for readability
    ax.grid(True)
    
    return fig, ax


@log_execution_time(logger, 'INFO')
def generate_evaluation_report(
    model: BaseModel,
    features: DataFrameType,
    targets: SeriesType,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates comprehensive evaluation report for a model.
    
    Args:
        model: Trained model to evaluate
        features: Feature DataFrame for evaluation
        targets: True target values
        output_path: Optional path to save the report as JSON
        
    Returns:
        Dictionary with evaluation results and report metadata
    """
    import json
    import os
    from datetime import datetime
    
    # Validate that model is trained
    if not model.is_trained():
        raise ModelError("Model must be trained before evaluation")
    
    # Calculate performance metrics
    metrics = evaluate_model_performance(model, features, targets)
    
    # Get ROC curve data
    fpr, tpr, auc_value = calculate_roc_curve(targets, model.predict_proba(features))
    
    # Get precision-recall curve data
    precision, recall, avg_precision = calculate_precision_recall_curve(targets, model.predict_proba(features))
    
    # Get calibration metrics
    calibration_data = evaluate_calibration(targets, model.predict_proba(features))
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    
    # Compile report
    report = {
        "model_info": {
            "model_id": model.model_id,
            "model_type": model.model_type,
            "version": model.version,
            "hyperparameters": model.hyperparameters
        },
        "evaluation_time": datetime.now().isoformat(),
        "performance_metrics": metrics,
        "calibration_metrics": {
            "brier_score": calibration_data.get('brier_score'),
            "expected_calibration_error": calibration_data.get('expected_calibration_error'),
            "maximum_calibration_error": calibration_data.get('maximum_calibration_error')
        },
        "curve_data": {
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": auc_value
            },
            "precision_recall_curve": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "average_precision": avg_precision
            }
        },
        "feature_importance": feature_importance
    }
    
    # Save report if output_path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    return report


class ModelEvaluator:
    """
    Class for comprehensive model evaluation.
    
    Provides a unified interface for evaluating models with various metrics,
    across different thresholds, time periods, and comparison between models.
    """
    
    def __init__(self, metrics_config: Optional[Dict[str, List[str]]] = None, threshold: Optional[float] = None):
        """
        Initialize the ModelEvaluator with configuration.
        
        Args:
            metrics_config: Dictionary mapping metric types to lists of metrics to calculate
            threshold: Probability threshold for binary classification
        """
        # Use default metrics configuration if not provided
        if metrics_config is None:
            self._metrics_config = {
                "classification": DEFAULT_CLASSIFICATION_METRICS,
                "probability": DEFAULT_PROBABILITY_METRICS,
                "calibration": DEFAULT_CALIBRATION_METRICS
            }
        else:
            self._metrics_config = metrics_config
        
        # Initialize results dictionary
        self._evaluation_results = {}
        
        # Use default threshold if not provided
        self._threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
    
    @log_execution_time(logger, 'INFO')
    def evaluate(
        self,
        model: BaseModel,
        features: DataFrameType,
        targets: SeriesType,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a model with specified metrics.
        
        Args:
            model: Trained model to evaluate
            features: Feature DataFrame for evaluation
            targets: True target values
            metrics: Optional list of metrics to calculate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Use configured metrics if none provided
        if metrics is None:
            # Flatten metrics configuration
            metrics = []
            for metric_list in self._metrics_config.values():
                metrics.extend(metric_list)
        
        # Evaluate model performance
        evaluation_metrics = evaluate_model_performance(
            model, features, targets, metrics, self._threshold
        )
        
        # Store results with model ID as key
        self._evaluation_results[model.model_id] = evaluation_metrics
        
        return evaluation_metrics
    
    @log_execution_time(logger, 'INFO')
    def evaluate_by_threshold(
        self,
        model: BaseModel,
        features: DataFrameType,
        targets_by_threshold: DataFrameType,
        price_thresholds: List[ThresholdValue]
    ) -> Dict[ThresholdValue, Dict[str, float]]:
        """
        Evaluate a model across different price thresholds.
        
        Args:
            model: Trained model to evaluate
            features: Feature DataFrame for evaluation
            targets_by_threshold: DataFrame with target columns for each price threshold
            price_thresholds: List of price threshold values
            
        Returns:
            Dictionary mapping price thresholds to metric dictionaries
        """
        # Flatten metrics configuration
        metrics = []
        for metric_list in self._metrics_config.values():
            metrics.extend(metric_list)
        
        # Evaluate model by threshold
        threshold_results = evaluate_model_by_threshold(
            model, features, targets_by_threshold, price_thresholds, metrics
        )
        
        # Store results with model ID and threshold as keys
        for threshold, threshold_metrics in threshold_results.items():
            key = f"{model.model_id}_{threshold}"
            self._evaluation_results[key] = threshold_metrics
        
        return threshold_results
    
    @log_execution_time(logger, 'INFO')
    def evaluate_over_time(
        self,
        model: BaseModel,
        features: DataFrameType,
        targets: SeriesType,
        time_column: str,
        time_grouping: str
    ) -> DataFrameType:
        """
        Evaluate a model's performance over different time periods.
        
        Args:
            model: Trained model to evaluate
            features: Feature DataFrame for evaluation
            targets: True target values
            time_column: Name of the column containing timestamps
            time_grouping: Time period to group by ('hour', 'day', 'month', etc.)
            
        Returns:
            DataFrame with metrics by time period
        """
        # Flatten metrics configuration
        metrics = []
        for metric_list in self._metrics_config.values():
            metrics.extend(metric_list)
        
        # Evaluate model over time
        time_results = evaluate_model_over_time(
            model, features, targets, time_column, time_grouping, metrics
        )
        
        # Store results with model ID and time grouping as keys
        key = f"{model.model_id}_{time_grouping}"
        self._evaluation_results[key] = time_results.to_dict()
        
        return time_results
    
    @log_execution_time(logger, 'INFO')
    def compare_models(
        self,
        models: List[BaseModel],
        features: DataFrameType,
        targets: SeriesType
    ) -> DataFrameType:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: List of models to compare
            features: Feature DataFrame for evaluation
            targets: True target values
            
        Returns:
            DataFrame with metrics for each model
        """
        # Flatten metrics configuration
        metrics = []
        for metric_list in self._metrics_config.values():
            metrics.extend(metric_list)
        
        # Compare models
        comparison_results = compare_models(models, features, targets, metrics)
        
        # Store results for each model
        for model in models:
            if model.model_id in comparison_results.index:
                self._evaluation_results[model.model_id] = comparison_results.loc[model.model_id].to_dict()
        
        return comparison_results
    
    @log_execution_time(logger, 'INFO')
    def generate_report(
        self,
        model: BaseModel,
        features: DataFrameType,
        targets: SeriesType,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Trained model to evaluate
            features: Feature DataFrame for evaluation
            targets: True target values
            output_path: Optional path to save the report as JSON
            
        Returns:
            Dictionary with evaluation results
        """
        # Generate report
        report = generate_evaluation_report(model, features, targets, output_path)
        
        # Store report
        self._evaluation_results[f"{model.model_id}_report"] = report
        
        return report
    
    def plot_evaluation_results(
        self,
        model: BaseModel,
        features: DataFrameType,
        targets: SeriesType,
        output_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate plots for model evaluation results.
        
        Args:
            model: Trained model to evaluate
            features: Feature DataFrame for evaluation
            targets: True target values
            output_dir: Optional directory to save plots
            
        Returns:
            Dictionary of generated figures
        """
        import os
        
        # Generate probability predictions
        y_prob = model.predict_proba(features)
        
        # Ensure y_prob is 1D array
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            # For multi-class, use the positive class probability (class 1)
            y_prob = y_prob[:, 1]
        
        # Create plots
        figures = {}
        
        # ROC curve
        fig_roc, ax_roc = plot_roc_curve(targets, y_prob)
        figures['roc_curve'] = fig_roc
        
        # Precision-recall curve
        fig_pr, ax_pr = plot_precision_recall_curve(targets, y_prob)
        figures['precision_recall_curve'] = fig_pr
        
        # Calibration curve
        fig_cal, ax_cal = plot_calibration_curve(targets, y_prob)
        figures['calibration_curve'] = fig_cal
        
        # Save plots if output_dir provided
        if output_dir:
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each figure
            for name, fig in figures.items():
                filename = os.path.join(output_dir, f"{model.model_id}_{name}.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Saved {name} plot to {filename}")
        
        return figures
    
    def get_results(self, model_id: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get stored evaluation results.
        
        Args:
            model_id: Optional model identifier to filter results
            
        Returns:
            Dictionary of evaluation results
        """
        if model_id is not None:
            # Filter results by model_id
            return {k: v for k, v in self._evaluation_results.items() if k.startswith(model_id)}
        else:
            # Return all results
            return self._evaluation_results
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set the probability threshold for binary classification.
        
        Args:
            threshold: Probability threshold between 0 and 1
        """
        # Validate threshold is between 0 and 1
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        self._threshold = threshold
    
    def set_metrics_config(self, metrics_config: Dict[str, List[str]]) -> None:
        """
        Set the metrics configuration.
        
        Args:
            metrics_config: Dictionary mapping metric types to lists of metrics
        """
        # Validate metrics_config structure
        for key, metrics in metrics_config.items():
            if not isinstance(metrics, list):
                raise ValueError(f"Metrics for {key} must be a list, got {type(metrics)}")
        
        self._metrics_config = metrics_config


class ThresholdOptimizer:
    """
    Class for finding optimal probability threshold for binary classification.
    
    Uses various metrics to determine the best threshold for converting
    probability predictions to binary predictions.
    """
    
    def __init__(self, optimization_metric: str = 'f1'):
        """
        Initialize the ThresholdOptimizer.
        
        Args:
            optimization_metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        """
        # Validate optimization_metric
        valid_metrics = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
        if optimization_metric not in valid_metrics:
            raise ValueError(f"Optimization metric must be one of {valid_metrics}, got {optimization_metric}")
        
        self._optimization_metric = optimization_metric
        self._optimal_threshold = None
        self._optimization_results = {}
    
    @log_execution_time(logger, 'INFO')
    def find_optimal_threshold(
        self,
        y_true: SeriesType,
        y_prob: ArrayType,
        thresholds: Optional[List[float]] = None
    ) -> float:
        """
        Find the optimal threshold that maximizes the specified metric.
        
        Args:
            y_true: Array of true binary labels
            y_prob: Array of predicted probabilities
            thresholds: Optional list of thresholds to evaluate
            
        Returns:
            Optimal threshold value
        """
        # Calculate metrics at different thresholds
        threshold_metrics = calculate_threshold_metrics(y_true, y_prob, thresholds)
        
        # Find the threshold that maximizes the optimization metric
        if self._optimization_metric == 'precision':
            best_idx = threshold_metrics['precision'].argmax()
        elif self._optimization_metric == 'recall':
            best_idx = threshold_metrics['recall'].argmax()
        elif self._optimization_metric == 'f1':
            best_idx = threshold_metrics['f1_score'].argmax()
        elif self._optimization_metric == 'accuracy':
            best_idx = threshold_metrics['accuracy'].argmax()
        elif self._optimization_metric == 'balanced_accuracy':
            # Calculate balanced accuracy: (sensitivity + specificity) / 2
            sensitivity = threshold_metrics['recall']
            
            # Compute specificity for each threshold
            tp = y_true.sum()
            fn = y_true.sum() - threshold_metrics['recall'] * tp
            tn = len(y_true) - y_true.sum() - (threshold_metrics['precision'] * threshold_metrics['recall'] * tp / 
                                              (threshold_metrics['precision'] * threshold_metrics['recall'] + 
                                               (1 - threshold_metrics['precision']) * threshold_metrics['recall']))
            fp = len(y_true) - y_true.sum() - tn
            specificity = tn / (tn + fp)
            
            balanced_accuracy = (sensitivity + specificity) / 2
            best_idx = balanced_accuracy.argmax()
        else:
            # Should not reach here due to validation in __init__
            raise ValueError(f"Unsupported optimization metric: {self._optimization_metric}")
        
        # Get the optimal threshold
        optimal_threshold = threshold_metrics.iloc[best_idx]['threshold']
        
        # Store the results
        self._optimal_threshold = optimal_threshold
        self._optimization_results = {
            'threshold_metrics': threshold_metrics.to_dict(),
            'optimal_threshold': optimal_threshold,
            'optimization_metric': self._optimization_metric,
            'best_metric_value': threshold_metrics.iloc[best_idx][self._optimization_metric if self._optimization_metric != 'balanced_accuracy' else 'accuracy']
        }
        
        logger.info(f"Optimal threshold found: {optimal_threshold:.4f} "
                   f"(Optimization metric: {self._optimization_metric})")
        
        return optimal_threshold
    
    @log_execution_time(logger, 'INFO')
    def optimize_for_model(
        self,
        model: BaseModel,
        features: DataFrameType,
        targets: SeriesType,
        thresholds: Optional[List[float]] = None
    ) -> float:
        """
        Find the optimal threshold for a specific model.
        
        Args:
            model: Trained model to optimize threshold for
            features: Feature DataFrame for evaluation
            targets: True target values
            thresholds: Optional list of thresholds to evaluate
            
        Returns:
            Optimal threshold value
        """
        # Validate that model is trained
        if not model.is_trained():
            raise ModelError("Model must be trained before threshold optimization")
        
        # Generate probability predictions
        y_prob = model.predict_proba(features)
        
        # Ensure y_prob is 1D array
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            # For multi-class, use the positive class probability (class 1)
            y_prob = y_prob[:, 1]
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(targets, y_prob, thresholds)
        
        # Store model information
        self._optimization_results['model_id'] = model.model_id
        self._optimization_results['model_type'] = model.model_type
        self._optimization_results['model_version'] = model.version
        
        return optimal_threshold
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """
        Get the results of the threshold optimization.
        
        Returns:
            Dictionary with optimization results
        """
        return self._optimization_results
    
    def plot_optimization_curve(
        self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the optimization metric across different thresholds.
        
        Args:
            fig: Optional figure object to plot on
            ax: Optional axes object to plot on
            
        Returns:
            Figure and axes objects
        """
        # Check if optimization has been performed
        if not self._optimization_results or 'threshold_metrics' not in self._optimization_results:
            raise ValueError("No optimization results available. Call find_optimal_threshold() first.")
        
        # Create figure and axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data from optimization results
        threshold_metrics = self._optimization_results['threshold_metrics']
        thresholds = threshold_metrics['threshold']
        
        # Plot metrics vs thresholds
        for metric in ['precision', 'recall', 'f1_score', 'accuracy']:
            if metric in threshold_metrics:
                ax.plot(thresholds, threshold_metrics[metric], label=metric.capitalize())
        
        # Mark the optimal threshold
        if self._optimal_threshold is not None:
            ax.axvline(
                x=self._optimal_threshold, 
                color='r', 
                linestyle='--', 
                label=f'Optimal Threshold: {self._optimal_threshold:.4f}'
            )
        
        # Set axis labels and title
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'Threshold Optimization (Metric: {self._optimization_metric})')
        ax.legend(loc='best')
        ax.grid(True)
        
        return fig, ax