"""
Module for selecting optimal features for the ERCOT RTLMP spike prediction system.

Implements various feature selection techniques including importance-based, correlation-based,
and model-based selection to identify the most predictive features for RTLMP spike prediction.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, RFECV
)
from xgboost import XGBClassifier  # version 1.7+

from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType
from ..utils.logging import get_logger, log_execution_time
from .feature_registry import get_feature, update_feature_importance

# Set up logger
logger = get_logger(__name__)

# Default parameters
DEFAULT_IMPORTANCE_THRESHOLD = 0.01
DEFAULT_CORRELATION_THRESHOLD = 0.85
DEFAULT_SELECTION_METHODS = ['importance', 'correlation', 'recursive']

@log_execution_time(logger, 'INFO')
def select_features_by_importance(
    df: DataFrameType,
    importance_scores: Optional[Dict[str, float]] = None,
    threshold: float = DEFAULT_IMPORTANCE_THRESHOLD
) -> List[str]:
    """
    Select features based on importance scores from a trained model or registry.
    
    Args:
        df: DataFrame containing features
        importance_scores: Optional dictionary mapping feature names to importance scores
        threshold: Minimum importance score threshold
        
    Returns:
        List[str]: List of selected feature names
    """
    try:
        # If importance_scores not provided, try to get from registry
        if importance_scores is None:
            importance_scores = get_feature_importance_dict()
            if not importance_scores:
                logger.warning("No importance scores provided or found in registry")
                return list(df.columns)
        
        # Filter features based on importance scores
        available_features = set(df.columns)
        selected_features = []
        total_importance = 0.0
        
        for feature, score in sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        ):
            if feature in available_features and score >= threshold:
                selected_features.append(feature)
                total_importance += score
        
        # If no features selected, return all features
        if not selected_features:
            logger.warning(f"No features met importance threshold {threshold}, returning all features")
            return list(df.columns)
        
        logger.info(f"Selected {len(selected_features)} features by importance (threshold={threshold})")
        logger.info(f"Total importance of selected features: {total_importance:.4f}")
        
        return selected_features
    except Exception as e:
        logger.error(f"Error in select_features_by_importance: {str(e)}")
        return list(df.columns)  # Return all features as fallback

@log_execution_time(logger, 'INFO')
def select_features_by_correlation(
    df: DataFrameType,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    importance_scores: Optional[Dict[str, float]] = None
) -> List[str]:
    """
    Select features by removing highly correlated features.
    
    Args:
        df: DataFrame containing features
        threshold: Correlation threshold above which features are considered redundant
        importance_scores: Optional dictionary mapping feature names to importance scores
                          to prioritize which feature to keep from correlated pairs
        
    Returns:
        List[str]: List of selected feature names
    """
    try:
        # Calculate the correlation matrix
        corr_matrix = df.corr().abs()
        
        # Create upper triangle of correlation matrix (excluding diagonal)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = set()
        
        # Get all pairs of correlated features
        correlated_pairs = []
        for i, feature1 in enumerate(upper.columns):
            for feature2 in upper.index[i+1:]:
                if upper.loc[feature2, feature1] > threshold:
                    correlated_pairs.append((feature1, feature2))
        
        # If importance scores provided, use them to decide which feature to keep
        if importance_scores:
            for feature1, feature2 in correlated_pairs:
                # Get importance scores, defaulting to 0 if not present
                score1 = importance_scores.get(feature1, 0)
                score2 = importance_scores.get(feature2, 0)
                
                # Keep feature with higher importance
                if score1 >= score2:
                    to_drop.add(feature2)
                else:
                    to_drop.add(feature1)
        else:
            # Without importance scores, keep first feature alphabetically
            for feature1, feature2 in correlated_pairs:
                if feature1 < feature2:
                    to_drop.add(feature2)
                else:
                    to_drop.add(feature1)
        
        selected_features = [col for col in df.columns if col not in to_drop]
        
        logger.info(f"Removed {len(to_drop)} features due to high correlation (threshold={threshold})")
        logger.info(f"Selected {len(selected_features)} features after correlation filtering")
        
        return selected_features
    except Exception as e:
        logger.error(f"Error in select_features_by_correlation: {str(e)}")
        return list(df.columns)  # Return all features as fallback

@log_execution_time(logger, 'INFO')
def select_features_by_recursive_elimination(
    df: DataFrameType,
    target: SeriesType,
    n_features_to_select: Optional[int] = None,
    model_params: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Select features using recursive feature elimination with cross-validation.
    
    Args:
        df: DataFrame containing features
        target: Series containing target variable
        n_features_to_select: Optional target number of features to select
                             (if None, will use cross-validation to determine optimal)
        model_params: Optional parameters for the base estimator
        
    Returns:
        List[str]: List of selected feature names
    """
    try:
        # If n_features_to_select not provided, use half of features as default
        if n_features_to_select is None:
            n_features_to_select = max(1, len(df.columns) // 2)
        
        # Create base estimator with default or provided parameters
        if model_params is None:
            model_params = {}
        
        estimator = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            **model_params
        )
        
        # Create RFECV
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=5,
            scoring='roc_auc',
            min_features_to_select=n_features_to_select
        )
        
        # Fit selector
        selector.fit(df, target)
        
        # Get selected features
        selected_mask = selector.support_
        selected_features = df.columns[selected_mask].tolist()
        
        logger.info(f"Selected {len(selected_features)} features using recursive elimination")
        logger.info(f"Optimal number of features: {selector.n_features_}")
        logger.info(f"Cross-validation scores: mean={selector.cv_results_['mean_test_score'].mean():.4f}")
        
        return selected_features
    except Exception as e:
        logger.error(f"Error in select_features_by_recursive_elimination: {str(e)}")
        return list(df.columns)  # Return all features as fallback

@log_execution_time(logger, 'INFO')
def select_features_by_mutual_information(
    df: DataFrameType,
    target: SeriesType,
    n_features_to_select: Optional[int] = None
) -> List[str]:
    """
    Select features using mutual information with the target variable.
    
    Args:
        df: DataFrame containing features
        target: Series containing target variable
        n_features_to_select: Number of features to select
        
    Returns:
        List[str]: List of selected feature names
    """
    try:
        # If n_features_to_select not provided, use half of features as default
        if n_features_to_select is None:
            n_features_to_select = max(1, len(df.columns) // 2)
        
        # Create selector
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
        
        # Fit selector
        selector.fit(df, target)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = df.columns[selected_mask].tolist()
        
        # Get scores
        scores = selector.scores_
        feature_scores = dict(zip(df.columns, scores))
        
        # Log selected features and their scores
        logger.info(f"Selected {len(selected_features)} features using mutual information")
        for feature in selected_features:
            logger.debug(f"Feature: {feature}, Score: {feature_scores[feature]:.4f}")
        
        return selected_features
    except Exception as e:
        logger.error(f"Error in select_features_by_mutual_information: {str(e)}")
        return list(df.columns)  # Return all features as fallback

@log_execution_time(logger, 'INFO')
def select_features_pipeline(
    df: DataFrameType,
    target: Optional[SeriesType] = None,
    selection_params: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Apply a pipeline of feature selection methods.
    
    Args:
        df: DataFrame containing features
        target: Optional Series containing target variable (required for some methods)
        selection_params: Optional parameters for the selection process
        
    Returns:
        List[str]: List of selected feature names
    """
    try:
        # Initialize parameters with defaults if not provided
        if selection_params is None:
            selection_params = {}
        
        methods = selection_params.get('methods', DEFAULT_SELECTION_METHODS)
        importance_threshold = selection_params.get('importance_threshold', DEFAULT_IMPORTANCE_THRESHOLD)
        correlation_threshold = selection_params.get('correlation_threshold', DEFAULT_CORRELATION_THRESHOLD)
        importance_scores = selection_params.get('importance_scores', None)
        n_features_recursive = selection_params.get('n_features_recursive', None)
        n_features_mutual_info = selection_params.get('n_features_mutual_info', None)
        model_params = selection_params.get('model_params', None)
        
        # Start with all features
        selected_features = list(df.columns)
        
        # Apply methods in sequence
        if 'importance' in methods:
            selected_features = select_features_by_importance(
                df[selected_features],
                importance_scores=importance_scores,
                threshold=importance_threshold
            )
        
        if 'correlation' in methods:
            selected_features = select_features_by_correlation(
                df[selected_features],
                threshold=correlation_threshold,
                importance_scores=importance_scores
            )
        
        if 'recursive' in methods and target is not None:
            selected_features = select_features_by_recursive_elimination(
                df[selected_features],
                target,
                n_features_to_select=n_features_recursive,
                model_params=model_params
            )
        elif 'recursive' in methods and target is None:
            logger.warning("Target is required for recursive feature elimination, skipping this method")
        
        if 'mutual_info' in methods and target is not None:
            selected_features = select_features_by_mutual_information(
                df[selected_features],
                target,
                n_features_to_select=n_features_mutual_info
            )
        elif 'mutual_info' in methods and target is None:
            logger.warning("Target is required for mutual information selection, skipping this method")
        
        logger.info(f"Final feature selection: {len(selected_features)} features selected")
        
        return selected_features
    except Exception as e:
        logger.error(f"Error in select_features_pipeline: {str(e)}")
        return list(df.columns)  # Return all features as fallback

def update_feature_importance_from_model(
    importance_scores: Dict[str, float],
    model_version: Optional[str] = None
) -> bool:
    """
    Update feature importance scores in the registry from a trained model.
    
    Args:
        importance_scores: Dictionary mapping feature names to importance scores
        model_version: Optional model version that generated these importance scores
        
    Returns:
        bool: True if update was successful
    """
    try:
        success = True
        update_count = 0
        
        for feature, score in importance_scores.items():
            # Update importance in registry
            if update_feature_importance(feature, score, model_version):
                update_count += 1
            else:
                success = False
        
        logger.info(f"Updated importance scores for {update_count} features in registry")
        
        return success
    except Exception as e:
        logger.error(f"Error in update_feature_importance_from_model: {str(e)}")
        return False

def get_feature_importance_dict() -> Dict[str, float]:
    """
    Get feature importance scores from the registry as a dictionary.
    
    Returns:
        Dict[str, float]: Dictionary mapping feature names to importance scores
    """
    try:
        importance_scores = {}
        
        # Get all features from registry
        from .feature_registry import get_all_features
        all_features = get_all_features()
        
        # Extract importance scores
        for feature_id, metadata in all_features.items():
            if 'importance_score' in metadata:
                importance_scores[feature_id] = metadata['importance_score']
        
        return importance_scores
    except Exception as e:
        logger.error(f"Error in get_feature_importance_dict: {str(e)}")
        return {}

class FeatureSelector:
    """
    Class-based interface for feature selection operations.
    """
    
    def __init__(self, selection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a new FeatureSelector instance.
        
        Args:
            selection_params: Optional parameters for the selection process
        """
        # Initialize with default or provided parameters
        self._selection_params = selection_params or {
            'methods': DEFAULT_SELECTION_METHODS,
            'importance_threshold': DEFAULT_IMPORTANCE_THRESHOLD,
            'correlation_threshold': DEFAULT_CORRELATION_THRESHOLD
        }
        self._importance_scores = {}
        self._selected_features = []
    
    def select_features(self, df: DataFrameType, target: Optional[SeriesType] = None) -> List[str]:
        """
        Select features using configured methods.
        
        Args:
            df: DataFrame containing features
            target: Optional Series containing target variable
            
        Returns:
            List[str]: List of selected feature names
        """
        try:
            # Update selection_params with importance_scores if available
            if self._importance_scores:
                self._selection_params['importance_scores'] = self._importance_scores
            
            # Call the pipeline function
            self._selected_features = select_features_pipeline(
                df, target, self._selection_params
            )
            
            return self._selected_features
        except Exception as e:
            logger.error(f"Error in select_features: {str(e)}")
            self._selected_features = list(df.columns)
            return self._selected_features
    
    def set_importance_scores(self, importance_scores: Dict[str, float]) -> None:
        """
        Set feature importance scores for selection.
        
        Args:
            importance_scores: Dictionary mapping feature names to importance scores
        """
        self._importance_scores = importance_scores
        self._selection_params['importance_scores'] = importance_scores
    
    def get_selected_features(self) -> List[str]:
        """
        Get the current list of selected features.
        
        Returns:
            List[str]: List of selected feature names
        """
        return self._selected_features
    
    def update_selection_params(self, params: Dict[str, Any]) -> None:
        """
        Update feature selection parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        self._selection_params.update(params)
    
    def filter_dataframe(self, df: DataFrameType) -> DataFrameType:
        """
        Filter DataFrame to include only selected features.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            DataFrameType: Filtered DataFrame
        """
        try:
            if not self._selected_features:
                logger.warning("No features selected yet, returning original DataFrame")
                return df
            
            # Keep only columns that exist in the DataFrame
            valid_columns = [col for col in self._selected_features if col in df.columns]
            
            if len(valid_columns) < len(self._selected_features):
                logger.warning(f"Some selected features not found in DataFrame. "
                              f"Expected {len(self._selected_features)}, found {len(valid_columns)}")
            
            return df[valid_columns]
        except Exception as e:
            logger.error(f"Error in filter_dataframe: {str(e)}")
            return df  # Return original DataFrame as fallback