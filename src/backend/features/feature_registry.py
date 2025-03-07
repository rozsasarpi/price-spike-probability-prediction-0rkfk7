"""
Central registry for managing feature metadata in the ERCOT RTLMP spike prediction system.

This module maintains a catalog of all available features with their properties,
data types, valid ranges, and relationships, ensuring consistent feature definitions
across training and inference processes.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import pandas as pd  # version 2.0+

from ..utils.type_definitions import DataFrameType, FeatureGroupType
from ..utils.logging import get_logger, log_execution_time

# Set up logger
logger = get_logger(__name__)

# Define valid feature groups
FEATURE_GROUPS = ['time', 'statistical', 'weather', 'market']

# Define default path for registry file
REGISTRY_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'feature_registry.json')

# Initialize empty feature registry
_feature_registry = {}

@log_execution_time(logger, 'INFO')
def register_feature(feature_id: str, metadata: Dict[str, Any]) -> bool:
    """
    Registers a new feature in the registry with its metadata.
    
    Args:
        feature_id: Unique identifier for the feature
        metadata: Dictionary containing feature metadata
        
    Returns:
        bool: True if registration was successful, False otherwise
    """
    # Validate feature_id
    if not feature_id or not isinstance(feature_id, str):
        logger.error(f"Invalid feature_id: {feature_id}")
        return False
    
    # Validate required metadata fields
    required_fields = ['name', 'data_type', 'group']
    if not all(field in metadata for field in required_fields):
        logger.error(f"Missing required metadata fields for feature {feature_id}. Required: {required_fields}")
        return False
    
    # Validate feature group
    if metadata['group'] not in FEATURE_GROUPS:
        logger.error(f"Invalid feature group '{metadata['group']}' for feature {feature_id}. Valid groups: {FEATURE_GROUPS}")
        return False
    
    # Add timestamp to metadata
    metadata['last_updated'] = datetime.now().isoformat()
    if 'created_at' not in metadata:
        metadata['created_at'] = metadata['last_updated']
    
    # Register the feature
    _feature_registry[feature_id] = metadata
    logger.info(f"Registered feature {feature_id} in group {metadata['group']}")
    
    # Save the registry
    save_registry()
    
    return True

def get_feature(feature_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a feature's metadata from the registry by ID.
    
    Args:
        feature_id: Unique identifier for the feature
        
    Returns:
        Optional[Dict[str, Any]]: Feature metadata or None if not found
    """
    if feature_id in _feature_registry:
        return _feature_registry[feature_id]
    
    logger.warning(f"Feature {feature_id} not found in registry")
    return None

def get_features_by_group(group: FeatureGroupType) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves all features belonging to a specific group.
    
    Args:
        group: Feature group category
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of feature IDs and their metadata
    """
    # Validate group
    if group not in FEATURE_GROUPS:
        logger.error(f"Invalid feature group '{group}'. Valid groups: {FEATURE_GROUPS}")
        return {}
    
    # Filter features by group
    features = {
        feature_id: metadata
        for feature_id, metadata in _feature_registry.items()
        if metadata.get('group') == group
    }
    
    return features

def get_all_features() -> Dict[str, Dict[str, Any]]:
    """
    Retrieves all features from the registry.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of all feature IDs and their metadata
    """
    return dict(_feature_registry)  # Return a copy

def delete_feature(feature_id: str) -> bool:
    """
    Removes a feature from the registry.
    
    Args:
        feature_id: Unique identifier for the feature
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    if feature_id not in _feature_registry:
        logger.warning(f"Feature {feature_id} not found in registry, cannot delete")
        return False
    
    # Remove the feature
    del _feature_registry[feature_id]
    logger.info(f"Deleted feature {feature_id} from registry")
    
    # Save the registry
    save_registry()
    
    return True

def update_feature_metadata(feature_id: str, metadata_updates: Dict[str, Any]) -> bool:
    """
    Updates specific metadata fields for an existing feature.
    
    Args:
        feature_id: Unique identifier for the feature
        metadata_updates: Dictionary containing metadata fields to update
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    if feature_id not in _feature_registry:
        logger.warning(f"Feature {feature_id} not found in registry, cannot update metadata")
        return False
    
    # Get existing metadata
    metadata = _feature_registry[feature_id]
    
    # Update fields
    for key, value in metadata_updates.items():
        metadata[key] = value
    
    # Update timestamp
    metadata['last_updated'] = datetime.now().isoformat()
    
    # Save the registry
    save_registry()
    
    logger.info(f"Updated metadata for feature {feature_id}")
    return True

def update_feature_importance(feature_id: str, importance_score: float, model_version: Optional[str] = None) -> bool:
    """
    Updates the importance score for a feature.
    
    Args:
        feature_id: Unique identifier for the feature
        importance_score: New importance score value
        model_version: Optional model version that generated this importance score
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    if feature_id not in _feature_registry:
        logger.warning(f"Feature {feature_id} not found in registry, cannot update importance")
        return False
    
    # Get existing metadata
    metadata = _feature_registry[feature_id]
    
    # Initialize importance_history if it doesn't exist
    if 'importance_history' not in metadata:
        metadata['importance_history'] = []
    
    # Add new importance score to history
    importance_entry = {
        'score': importance_score,
        'timestamp': datetime.now().isoformat(),
        'model_version': model_version
    }
    metadata['importance_history'].append(importance_entry)
    
    # Update current importance score
    metadata['importance_score'] = importance_score
    metadata['last_updated'] = datetime.now().isoformat()
    
    # Save the registry
    save_registry()
    
    logger.info(f"Updated importance for feature {feature_id}: {importance_score}")
    return True

def save_registry(file_path: Optional[str] = None) -> bool:
    """
    Saves the feature registry to disk.
    
    Args:
        file_path: Optional path to save the registry file
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    # Use default path if none provided
    if file_path is None:
        file_path = REGISTRY_FILE_PATH
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        # Convert to JSON
        registry_json = json.dumps(_feature_registry, indent=2)
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(registry_json)
        
        logger.info(f"Saved feature registry to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save feature registry: {str(e)}")
        return False

def load_registry(file_path: Optional[str] = None) -> bool:
    """
    Loads the feature registry from disk.
    
    Args:
        file_path: Optional path to load the registry file from
        
    Returns:
        bool: True if load was successful, False otherwise
    """
    global _feature_registry
    
    # Use default path if none provided
    if file_path is None:
        file_path = REGISTRY_FILE_PATH
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"Feature registry file not found at {file_path}")
        return False
    
    try:
        # Read from file
        with open(file_path, 'r') as f:
            registry_json = f.read()
        
        # Parse JSON
        _feature_registry = json.loads(registry_json)
        
        logger.info(f"Loaded feature registry from {file_path} with {len(_feature_registry)} features")
        return True
    except Exception as e:
        logger.error(f"Failed to load feature registry: {str(e)}")
        return False

@log_execution_time(logger, 'INFO')
def initialize_default_features() -> int:
    """
    Initializes the registry with default features from all categories.
    
    Returns:
        int: Number of features initialized
    """
    # Try to load existing registry
    load_registry()
    
    # TODO: Import feature metadata from feature modules
    # This would be implemented in a production environment
    # from .time_features import TIME_FEATURES
    # from .statistical_features import STATISTICAL_FEATURES
    # from .weather_features import WEATHER_FEATURES
    # from .market_features import MARKET_FEATURES
    
    # For now, use placeholder default features
    time_features = {
        'hour_of_day': {
            'name': 'Hour of Day',
            'data_type': 'int',
            'group': 'time',
            'description': 'Hour of the day (0-23)',
            'valid_range': [0, 23],
            'dependencies': []
        },
        'day_of_week': {
            'name': 'Day of Week',
            'data_type': 'int',
            'group': 'time',
            'description': 'Day of the week (0=Monday, 6=Sunday)',
            'valid_range': [0, 6],
            'dependencies': []
        },
        'is_weekend': {
            'name': 'Is Weekend',
            'data_type': 'bool',
            'group': 'time',
            'description': 'Whether the day is a weekend (True) or weekday (False)',
            'dependencies': ['day_of_week']
        }
    }
    
    statistical_features = {
        'rolling_mean_24h': {
            'name': 'Rolling Mean 24h',
            'data_type': 'float',
            'group': 'statistical',
            'description': '24-hour rolling average of RTLMP',
            'dependencies': []
        },
        'rolling_max_24h': {
            'name': 'Rolling Max 24h',
            'data_type': 'float',
            'group': 'statistical',
            'description': '24-hour rolling maximum of RTLMP',
            'dependencies': []
        }
    }
    
    weather_features = {
        'temperature': {
            'name': 'Temperature',
            'data_type': 'float',
            'group': 'weather',
            'description': 'Temperature in degrees Fahrenheit',
            'dependencies': []
        },
        'wind_speed': {
            'name': 'Wind Speed',
            'data_type': 'float',
            'group': 'weather',
            'description': 'Wind speed in mph',
            'valid_range': [0, float('inf')],
            'dependencies': []
        }
    }
    
    market_features = {
        'load_forecast': {
            'name': 'Load Forecast',
            'data_type': 'float',
            'group': 'market',
            'description': 'Forecasted load in MW',
            'valid_range': [0, float('inf')],
            'dependencies': []
        },
        'generation_mix_wind': {
            'name': 'Wind Generation Mix',
            'data_type': 'float',
            'group': 'market',
            'description': 'Percentage of generation from wind sources',
            'valid_range': [0, 100],
            'dependencies': []
        }
    }
    
    # Register default features
    feature_count = 0
    
    for feature_id, metadata in time_features.items():
        if register_feature(feature_id, metadata):
            feature_count += 1
    
    for feature_id, metadata in statistical_features.items():
        if register_feature(feature_id, metadata):
            feature_count += 1
    
    for feature_id, metadata in weather_features.items():
        if register_feature(feature_id, metadata):
            feature_count += 1
    
    for feature_id, metadata in market_features.items():
        if register_feature(feature_id, metadata):
            feature_count += 1
    
    logger.info(f"Initialized {feature_count} default features")
    return feature_count

def validate_feature_consistency(df: DataFrameType, feature_ids: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validates that a DataFrame contains features with consistent types and ranges.
    
    Args:
        df: DataFrame containing features to validate
        feature_ids: Optional list of feature IDs to validate (if None, uses all columns)
        
    Returns:
        Tuple[bool, List[str]]: Validation result and list of inconsistent features
    """
    # If feature_ids not provided, use all columns
    if feature_ids is None:
        feature_ids = df.columns.tolist()
    
    inconsistent_features = []
    
    for feature_id in feature_ids:
        # Skip if feature doesn't exist in DataFrame
        if feature_id not in df.columns:
            logger.warning(f"Feature {feature_id} not in DataFrame, skipping validation")
            inconsistent_features.append(f"{feature_id} (missing)")
            continue
        
        # Skip if feature doesn't exist in registry
        metadata = get_feature(feature_id)
        if metadata is None:
            logger.warning(f"Feature {feature_id} not in registry, skipping validation")
            continue
        
        # Validate data type
        expected_type = metadata.get('data_type')
        if expected_type:
            valid_type = True
            
            if expected_type == 'int':
                valid_type = pd.api.types.is_integer_dtype(df[feature_id]) or (
                    pd.api.types.is_float_dtype(df[feature_id]) and 
                    df[feature_id].dropna().apply(lambda x: x.is_integer()).all()
                )
            elif expected_type == 'float':
                valid_type = pd.api.types.is_float_dtype(df[feature_id])
            elif expected_type == 'bool':
                valid_type = pd.api.types.is_bool_dtype(df[feature_id])
            elif expected_type == 'categorical':
                valid_type = pd.api.types.is_categorical_dtype(df[feature_id])
            
            if not valid_type:
                logger.warning(f"Feature {feature_id} has incorrect data type. Expected {expected_type}")
                inconsistent_features.append(f"{feature_id} (type)")
        
        # Validate value range
        valid_range = metadata.get('valid_range')
        if valid_range and len(valid_range) == 2:
            min_val, max_val = valid_range
            if not df[feature_id].dropna().between(min_val, max_val).all():
                logger.warning(f"Feature {feature_id} has values outside valid range [{min_val}, {max_val}]")
                inconsistent_features.append(f"{feature_id} (range)")
        
        # Validate categorical values
        valid_categories = metadata.get('categories')
        if valid_categories and not df[feature_id].dropna().isin(valid_categories).all():
            invalid_values = df[feature_id].dropna()[~df[feature_id].dropna().isin(valid_categories)].unique()
            logger.warning(f"Feature {feature_id} has invalid categorical values: {invalid_values}")
            inconsistent_features.append(f"{feature_id} (categories)")
    
    is_valid = len(inconsistent_features) == 0
    
    if is_valid:
        logger.info(f"All {len(feature_ids)} features validated successfully")
    else:
        logger.warning(f"Found {len(inconsistent_features)} inconsistent features: {', '.join(inconsistent_features)}")
    
    return is_valid, inconsistent_features

def get_feature_dependencies(feature_id: str) -> List[str]:
    """
    Retrieves the dependencies for a feature.
    
    Args:
        feature_id: Unique identifier for the feature
        
    Returns:
        List[str]: List of feature IDs that this feature depends on
    """
    metadata = get_feature(feature_id)
    
    if metadata and 'dependencies' in metadata:
        return metadata['dependencies']
    
    return []

def get_dependent_features(feature_id: str) -> List[str]:
    """
    Retrieves features that depend on a given feature.
    
    Args:
        feature_id: Unique identifier for the feature
        
    Returns:
        List[str]: List of feature IDs that depend on this feature
    """
    dependent_features = []
    
    for fid, metadata in _feature_registry.items():
        if 'dependencies' in metadata and feature_id in metadata['dependencies']:
            dependent_features.append(fid)
    
    return dependent_features

class FeatureRegistry:
    """
    Class-based interface for the feature registry.
    """
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize a new FeatureRegistry instance.
        
        Args:
            file_path: Optional path to the registry file
        """
        self._registry = {}
        self._file_path = file_path or REGISTRY_FILE_PATH
        
        # Try to load existing registry
        self.load()
    
    def register_feature(self, feature_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Register a new feature in the registry.
        
        Args:
            feature_id: Unique identifier for the feature
            metadata: Dictionary containing feature metadata
            
        Returns:
            bool: True if registration was successful
        """
        # Validate feature_id
        if not feature_id or not isinstance(feature_id, str):
            logger.error(f"Invalid feature_id: {feature_id}")
            return False
        
        # Validate required metadata fields
        required_fields = ['name', 'data_type', 'group']
        if not all(field in metadata for field in required_fields):
            logger.error(f"Missing required metadata fields for feature {feature_id}. Required: {required_fields}")
            return False
        
        # Validate feature group
        if metadata['group'] not in FEATURE_GROUPS:
            logger.error(f"Invalid feature group '{metadata['group']}' for feature {feature_id}. Valid groups: {FEATURE_GROUPS}")
            return False
        
        # Add timestamp to metadata
        metadata['last_updated'] = datetime.now().isoformat()
        if 'created_at' not in metadata:
            metadata['created_at'] = metadata['last_updated']
        
        # Register the feature
        self._registry[feature_id] = metadata
        logger.info(f"Registered feature {feature_id} in group {metadata['group']}")
        
        # Save the registry
        self.save()
        
        return True
    
    def get_feature(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a feature's metadata by ID.
        
        Args:
            feature_id: Unique identifier for the feature
            
        Returns:
            Optional[Dict[str, Any]]: Feature metadata or None
        """
        if feature_id in self._registry:
            return self._registry[feature_id]
        
        logger.warning(f"Feature {feature_id} not found in registry")
        return None
    
    def get_features_by_group(self, group: FeatureGroupType) -> Dict[str, Dict[str, Any]]:
        """
        Get all features in a specific group.
        
        Args:
            group: Feature group category
            
        Returns:
            Dict[str, Dict[str, Any]]: Features in the group
        """
        # Validate group
        if group not in FEATURE_GROUPS:
            logger.error(f"Invalid feature group '{group}'. Valid groups: {FEATURE_GROUPS}")
            return {}
        
        # Filter features by group
        features = {
            feature_id: metadata
            for feature_id, metadata in self._registry.items()
            if metadata.get('group') == group
        }
        
        return features
    
    def get_all_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all features in the registry.
        
        Returns:
            Dict[str, Dict[str, Any]]: All features
        """
        return dict(self._registry)  # Return a copy
    
    def delete_feature(self, feature_id: str) -> bool:
        """
        Remove a feature from the registry.
        
        Args:
            feature_id: Unique identifier for the feature
            
        Returns:
            bool: True if deletion was successful
        """
        if feature_id not in self._registry:
            logger.warning(f"Feature {feature_id} not found in registry, cannot delete")
            return False
        
        # Remove the feature
        del self._registry[feature_id]
        logger.info(f"Deleted feature {feature_id} from registry")
        
        # Save the registry
        self.save()
        
        return True
    
    def update_feature_metadata(self, feature_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing feature.
        
        Args:
            feature_id: Unique identifier for the feature
            metadata_updates: Dictionary containing metadata fields to update
            
        Returns:
            bool: True if update was successful
        """
        if feature_id not in self._registry:
            logger.warning(f"Feature {feature_id} not found in registry, cannot update metadata")
            return False
        
        # Get existing metadata
        metadata = self._registry[feature_id]
        
        # Update fields
        for key, value in metadata_updates.items():
            metadata[key] = value
        
        # Update timestamp
        metadata['last_updated'] = datetime.now().isoformat()
        
        # Save the registry
        self.save()
        
        logger.info(f"Updated metadata for feature {feature_id}")
        return True
    
    def update_feature_importance(self, feature_id: str, importance_score: float, model_version: Optional[str] = None) -> bool:
        """
        Update importance score for a feature.
        
        Args:
            feature_id: Unique identifier for the feature
            importance_score: New importance score value
            model_version: Optional model version that generated this importance score
            
        Returns:
            bool: True if update was successful
        """
        if feature_id not in self._registry:
            logger.warning(f"Feature {feature_id} not found in registry, cannot update importance")
            return False
        
        # Get existing metadata
        metadata = self._registry[feature_id]
        
        # Initialize importance_history if it doesn't exist
        if 'importance_history' not in metadata:
            metadata['importance_history'] = []
        
        # Add new importance score to history
        importance_entry = {
            'score': importance_score,
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version
        }
        metadata['importance_history'].append(importance_entry)
        
        # Update current importance score
        metadata['importance_score'] = importance_score
        metadata['last_updated'] = datetime.now().isoformat()
        
        # Save the registry
        self.save()
        
        logger.info(f"Updated importance for feature {feature_id}: {importance_score}")
        return True
    
    def save(self, file_path: Optional[str] = None) -> bool:
        """
        Save registry to disk.
        
        Args:
            file_path: Optional path to save the registry file
            
        Returns:
            bool: True if save was successful
        """
        # Use instance file_path if none provided
        if file_path is None:
            file_path = self._file_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            # Convert to JSON
            registry_json = json.dumps(self._registry, indent=2)
            
            # Write to file
            with open(file_path, 'w') as f:
                f.write(registry_json)
            
            logger.info(f"Saved feature registry to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save feature registry: {str(e)}")
            return False
    
    def load(self, file_path: Optional[str] = None) -> bool:
        """
        Load registry from disk.
        
        Args:
            file_path: Optional path to load the registry file from
            
        Returns:
            bool: True if load was successful
        """
        # Use instance file_path if none provided
        if file_path is None:
            file_path = self._file_path
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Feature registry file not found at {file_path}")
            return False
        
        try:
            # Read from file
            with open(file_path, 'r') as f:
                registry_json = f.read()
            
            # Parse JSON
            self._registry = json.loads(registry_json)
            
            logger.info(f"Loaded feature registry from {file_path} with {len(self._registry)} features")
            return True
        except Exception as e:
            logger.error(f"Failed to load feature registry: {str(e)}")
            return False
    
    def initialize_default_features(self) -> int:
        """
        Initialize registry with default features.
        
        Returns:
            int: Number of features initialized
        """
        # Use the function-based implementation
        # Reset registry first to ensure clean state
        self._registry = {}
        self.save()
        
        # Register default features using the global function
        count = initialize_default_features()
        
        # Load the registry back
        self.load()
        
        return count
    
    def validate_feature_consistency(self, df: DataFrameType, feature_ids: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate feature consistency in a DataFrame.
        
        Args:
            df: DataFrame containing features to validate
            feature_ids: Optional list of feature IDs to validate (if None, uses all columns)
            
        Returns:
            Tuple[bool, List[str]]: Validation result and inconsistent features
        """
        # If feature_ids not provided, use all columns
        if feature_ids is None:
            feature_ids = df.columns.tolist()
        
        inconsistent_features = []
        
        for feature_id in feature_ids:
            # Skip if feature doesn't exist in DataFrame
            if feature_id not in df.columns:
                logger.warning(f"Feature {feature_id} not in DataFrame, skipping validation")
                inconsistent_features.append(f"{feature_id} (missing)")
                continue
            
            # Skip if feature doesn't exist in registry
            metadata = self.get_feature(feature_id)
            if metadata is None:
                logger.warning(f"Feature {feature_id} not in registry, skipping validation")
                continue
            
            # Validate data type
            expected_type = metadata.get('data_type')
            if expected_type:
                valid_type = True
                
                if expected_type == 'int':
                    valid_type = pd.api.types.is_integer_dtype(df[feature_id]) or (
                        pd.api.types.is_float_dtype(df[feature_id]) and 
                        df[feature_id].dropna().apply(lambda x: x.is_integer()).all()
                    )
                elif expected_type == 'float':
                    valid_type = pd.api.types.is_float_dtype(df[feature_id])
                elif expected_type == 'bool':
                    valid_type = pd.api.types.is_bool_dtype(df[feature_id])
                elif expected_type == 'categorical':
                    valid_type = pd.api.types.is_categorical_dtype(df[feature_id])
                
                if not valid_type:
                    logger.warning(f"Feature {feature_id} has incorrect data type. Expected {expected_type}")
                    inconsistent_features.append(f"{feature_id} (type)")
            
            # Validate value range
            valid_range = metadata.get('valid_range')
            if valid_range and len(valid_range) == 2:
                min_val, max_val = valid_range
                if not df[feature_id].dropna().between(min_val, max_val).all():
                    logger.warning(f"Feature {feature_id} has values outside valid range [{min_val}, {max_val}]")
                    inconsistent_features.append(f"{feature_id} (range)")
            
            # Validate categorical values
            valid_categories = metadata.get('categories')
            if valid_categories and not df[feature_id].dropna().isin(valid_categories).all():
                invalid_values = df[feature_id].dropna()[~df[feature_id].dropna().isin(valid_categories)].unique()
                logger.warning(f"Feature {feature_id} has invalid categorical values: {invalid_values}")
                inconsistent_features.append(f"{feature_id} (categories)")
        
        is_valid = len(inconsistent_features) == 0
        
        if is_valid:
            logger.info(f"All {len(feature_ids)} features validated successfully")
        else:
            logger.warning(f"Found {len(inconsistent_features)} inconsistent features: {', '.join(inconsistent_features)}")
        
        return is_valid, inconsistent_features