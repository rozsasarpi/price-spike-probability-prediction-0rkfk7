"""
Implements a model registry for the ERCOT RTLMP spike prediction system that manages
the storage, retrieval, versioning, and metadata tracking of machine learning models.

This module provides a centralized repository for model artifacts with consistent
versioning and metadata management to ensure reproducibility and traceability of model
training and inference.
"""

import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

import joblib  # version 1.2+
import shutil

from ../../utils.logging import get_logger, log_execution_time, log_function_call
from ../../utils.error_handling import retry_with_backoff, handle_errors, ModelError, ModelLoadError
from ../../utils.type_definitions import ModelConfigDict, PathType
from ../../models.versioning import (
    VersionManager, increment_version, get_latest_version
)

# Configure logger
logger = get_logger(__name__)

# Global constants
DEFAULT_REGISTRY_PATH = Path('models')
MODEL_FILE_NAME = 'model.joblib'
METADATA_FILE_NAME = 'metadata.json'
PERFORMANCE_FILE_NAME = 'performance.json'


@retry_with_backoff(exceptions=(OSError, IOError), max_retries=3, initial_delay=1.0)
@log_execution_time(logger, 'INFO')
def save_model(model_obj: Any, file_path: PathType) -> bool:
    """
    Saves a model object to the specified path with error handling.
    
    Args:
        model_obj: The model object to save
        file_path: Path where the model should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save the model
        joblib.dump(model_obj, file_path)
        logger.info(f"Successfully saved model to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save model to {file_path}: {e}")
        return False


@retry_with_backoff(exceptions=(OSError, IOError), max_retries=3, initial_delay=1.0)
@log_execution_time(logger, 'INFO')
def load_model(file_path: PathType) -> Any:
    """
    Loads a model object from the specified path with error handling.
    
    Args:
        file_path: Path from which to load the model
        
    Returns:
        The loaded model object
        
    Raises:
        ModelLoadError: If the model cannot be loaded
    """
    if not os.path.exists(file_path):
        raise ModelLoadError(f"Model file not found: {file_path}")
    
    try:
        model = joblib.load(file_path)
        logger.info(f"Successfully loaded model from {file_path}")
        return model
    
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {file_path}: {e}")


@retry_with_backoff(exceptions=(OSError, IOError), max_retries=3, initial_delay=1.0)
def save_metadata(metadata: ModelConfigDict, file_path: PathType) -> bool:
    """
    Saves model metadata to a JSON file.
    
    Args:
        metadata: Dictionary containing model metadata
        file_path: Path where metadata should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        serializable_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, datetime):
                serializable_metadata[key] = value.isoformat()
            else:
                serializable_metadata[key] = value
        
        # Write the metadata to the file
        with open(file_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        logger.info(f"Successfully saved metadata to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save metadata to {file_path}: {e}")
        return False


@retry_with_backoff(exceptions=(OSError, IOError), max_retries=3, initial_delay=1.0)
def load_metadata(file_path: PathType) -> ModelConfigDict:
    """
    Loads model metadata from a JSON file.
    
    Args:
        file_path: Path from which to load metadata
        
    Returns:
        Dictionary containing model metadata
        
    Raises:
        ModelError: If the metadata cannot be loaded
    """
    if not os.path.exists(file_path):
        raise ModelError(f"Metadata file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        
        # Convert ISO format strings back to datetime objects
        for key, value in metadata.items():
            if isinstance(value, str) and 'T' in value and value.count('-') >= 2:
                try:
                    metadata[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Not a valid datetime string, keep as is
        
        logger.info(f"Successfully loaded metadata from {file_path}")
        return metadata
    
    except Exception as e:
        raise ModelError(f"Failed to load metadata from {file_path}: {e}")


@log_execution_time(logger, 'INFO')
def list_models(registry_path: Optional[PathType] = None, model_type: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Lists all models in the registry, optionally filtered by model type.
    
    Args:
        registry_path: Path to the model registry
        model_type: Optional filter for model type
        
    Returns:
        Dictionary of model types, IDs, and versions
    """
    # If no registry path is provided, use the default
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH
    
    # Check if registry path exists
    if not os.path.exists(registry_path):
        logger.warning(f"Registry path not found: {registry_path}")
        return {}
    
    result = {}
    
    # If model_type is provided, only scan that directory
    if model_type:
        model_type_path = Path(registry_path) / model_type
        if os.path.exists(model_type_path) and os.path.isdir(model_type_path):
            result[model_type] = {}
            
            # List all model ID directories
            for model_id_dir in os.listdir(model_type_path):
                model_id_path = model_type_path / model_id_dir
                if os.path.isdir(model_id_path):
                    # List all version directories
                    versions = []
                    for version_dir in os.listdir(model_id_path):
                        version_path = model_id_path / version_dir
                        if os.path.isdir(version_path):
                            versions.append(version_dir)
                    
                    result[model_type][model_id_dir] = versions
    else:
        # Scan all model type directories
        for model_type_dir in os.listdir(registry_path):
            model_type_path = Path(registry_path) / model_type_dir
            if os.path.isdir(model_type_path):
                result[model_type_dir] = {}
                
                # List all model ID directories
                for model_id_dir in os.listdir(model_type_path):
                    model_id_path = model_type_path / model_id_dir
                    if os.path.isdir(model_id_path):
                        # List all version directories
                        versions = []
                        for version_dir in os.listdir(model_id_path):
                            version_path = model_id_path / version_dir
                            if os.path.isdir(version_path):
                                versions.append(version_dir)
                        
                        result[model_type_dir][model_id_dir] = versions
    
    return result


@handle_errors(exceptions=Exception, error_message='Failed to get latest model version', default_return=None)
def get_latest_model_version(model_type: str, model_id: str, registry_path: Optional[PathType] = None) -> Optional[str]:
    """
    Gets the latest version of a specific model.
    
    Args:
        model_type: Type of the model
        model_id: Identifier for the model
        registry_path: Path to the model registry
        
    Returns:
        Latest version string or None if no versions exist
    """
    # If no registry path is provided, use the default
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH
    
    # Construct the path to the model's directory
    model_path = Path(registry_path) / model_type / model_id
    
    # Check if the directory exists
    if not os.path.exists(model_path):
        logger.warning(f"Model directory not found: {model_path}")
        return None
    
    # List all version directories
    versions = []
    for item in os.listdir(model_path):
        item_path = model_path / item
        if os.path.isdir(item_path):
            versions.append(item)
    
    # Get the latest version
    return get_latest_version(versions)


@handle_errors(exceptions=ValueError, error_message='Failed to increment version', default_return=None)
def increment_version(current_version: str, increment_type: str) -> str:
    """
    Increments a version string according to semantic versioning rules.
    
    Args:
        current_version: Current version string
        increment_type: Type of increment ('major', 'minor', or 'patch')
        
    Returns:
        Incremented version string
    """
    # Delegate to the increment_version function from the versioning module
    return increment_version(current_version, increment_type)


class ModelRegistry:
    """
    Class that manages the storage, retrieval, and versioning of machine learning models.
    """
    
    def __init__(self, registry_path: Optional[PathType] = None):
        """
        Initialize the ModelRegistry with a base path.
        
        Args:
            registry_path: Base directory path for model storage
        """
        # Set the registry path
        self.registry_path = registry_path if registry_path is not None else DEFAULT_REGISTRY_PATH
        
        # Ensure the registry directory exists
        os.makedirs(self.registry_path, exist_ok=True)
        
        # Initialize the version manager
        self.version_manager = VersionManager(self.registry_path)
    
    @log_execution_time(logger, 'INFO')
    @log_function_call(logger, 'INFO', log_args=True, log_result=True)
    def register_model(
        self,
        model_obj: Any,
        metadata: ModelConfigDict,
        version: Optional[str] = None,
        increment_type: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Registers a model in the registry with metadata and versioning.
        
        Args:
            model_obj: The model object to register
            metadata: Dictionary containing model metadata
            version: Optional specific version to use
            increment_type: If version not provided, type of increment to apply
            
        Returns:
            Tuple of (model_type, model_id, version)
        """
        # Extract model type and id from metadata
        model_type = metadata['model_type']
        model_id = metadata['model_id']
        
        # Determine the version
        if version is None:
            # If we have a latest version and increment_type, increment it
            latest_version = self.version_manager.get_latest_version(model_type, model_id)
            
            if latest_version:
                # Increment the version
                increment_type = increment_type or 'patch'
                version = increment_version(latest_version, increment_type)
            else:
                # No existing version, use initial version
                version = '0.1.0'
        
        # Construct path to model version directory
        model_dir = self.ensure_model_directory(model_type, model_id, version)
        
        # Save the model
        model_path = model_dir / MODEL_FILE_NAME
        save_model(model_obj, model_path)
        
        # Save the metadata
        metadata_path = model_dir / METADATA_FILE_NAME
        save_metadata(metadata, metadata_path)
        
        logger.info(f"Successfully registered model {model_type}/{model_id}/{version}")
        return model_type, model_id, version
    
    @log_execution_time(logger, 'INFO')
    def get_model(self, model_type: str, model_id: str, version: Optional[str] = None) -> Tuple[Any, ModelConfigDict]:
        """
        Retrieves a model from the registry by type, ID, and optional version.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Optional specific version to retrieve
            
        Returns:
            Tuple of (model_object, metadata)
        """
        # If no version is provided, get the latest
        if version is None:
            version = self.version_manager.get_latest_version(model_type, model_id)
            
            if version is None:
                raise ModelError(f"No versions found for model {model_type}/{model_id}")
        
        # Construct path to model version directory
        model_dir = self.get_model_path(model_type, model_id, version)
        
        # Load the model
        model_path = model_dir / MODEL_FILE_NAME
        model_obj = load_model(model_path)
        
        # Load the metadata
        metadata_path = model_dir / METADATA_FILE_NAME
        metadata = load_metadata(metadata_path)
        
        return model_obj, metadata
    
    @log_execution_time(logger, 'INFO')
    def get_latest_model(self, model_type: str, model_id: Optional[str] = None) -> Tuple[Any, ModelConfigDict]:
        """
        Gets the latest version of a model by type and ID.
        
        Args:
            model_type: Type of the model
            model_id: Optional identifier for the model
            
        Returns:
            Tuple of (model_object, metadata)
        """
        # If model_id is not provided, find the model with the best performance metrics
        if model_id is None:
            # Get all models of this type
            models = list_models(self.registry_path, model_type)
            
            if not models or model_type not in models or not models[model_type]:
                raise ModelError(f"No models found of type {model_type}")
            
            # Find the model with the best performance
            best_model_id = None
            best_performance = -float('inf')
            
            for id, versions in models[model_type].items():
                if not versions:
                    continue
                
                # Get the latest version for this model ID
                latest_version = get_latest_version(versions)
                if latest_version:
                    # Get the performance metrics
                    performance = self.get_model_performance(model_type, id, latest_version)
                    
                    # Check primary metric (AUC by default)
                    primary_metric = performance.get('auc', 0)
                    
                    if primary_metric > best_performance:
                        best_performance = primary_metric
                        best_model_id = id
            
            if best_model_id is None:
                raise ModelError(f"No models with performance metrics found for type {model_type}")
            
            model_id = best_model_id
        
        # Get the latest version for this model
        latest_version = self.version_manager.get_latest_version(model_type, model_id)
        
        if latest_version is None:
            raise ModelError(f"No versions found for model {model_type}/{model_id}")
        
        # Get the model
        return self.get_model(model_type, model_id, latest_version)
    
    @log_execution_time(logger, 'INFO')
    @handle_errors(exceptions=Exception, error_message='Failed to delete model', default_return=False)
    def delete_model(self, model_type: str, model_id: str, version: Optional[str] = None) -> bool:
        """
        Deletes a model from the registry.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Optional specific version to delete
            
        Returns:
            True if successful, False otherwise
        """
        if version:
            # Delete specific version
            version_path = Path(self.registry_path) / model_type / model_id / version
            
            if not os.path.exists(version_path):
                logger.warning(f"Version path not found: {version_path}")
                return False
            
            shutil.rmtree(version_path)
            logger.info(f"Deleted model version {model_type}/{model_id}/{version}")
        else:
            # Delete all versions (the entire model)
            model_path = Path(self.registry_path) / model_type / model_id
            
            if not os.path.exists(model_path):
                logger.warning(f"Model path not found: {model_path}")
                return False
            
            shutil.rmtree(model_path)
            logger.info(f"Deleted model {model_type}/{model_id}")
        
        return True
    
    @log_execution_time(logger, 'INFO')
    @handle_errors(exceptions=Exception, error_message='Failed to update model metadata', default_return=None)
    def update_model_metadata(self, model_type: str, model_id: str, version: str, metadata_updates: Dict[str, Any]) -> ModelConfigDict:
        """
        Updates the metadata for an existing model.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Version of the model
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            Updated metadata dictionary
        """
        # Get the path to the metadata file
        model_dir = self.get_model_path(model_type, model_id, version)
        metadata_path = model_dir / METADATA_FILE_NAME
        
        # Load existing metadata
        metadata = load_metadata(metadata_path)
        
        # Update metadata
        metadata.update(metadata_updates)
        
        # Save updated metadata
        save_metadata(metadata, metadata_path)
        
        return metadata
    
    @log_execution_time(logger, 'INFO')
    @handle_errors(exceptions=Exception, error_message='Failed to create model version', default_return=None)
    def create_model_version(
        self,
        model_type: str,
        model_id: str,
        base_version: str,
        increment_type: str,
        new_model_obj: Optional[Any] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Creates a new version of an existing model.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            base_version: Base version to increment from
            increment_type: Type of increment ('major', 'minor', or 'patch')
            new_model_obj: Optional new model object
            metadata_updates: Optional metadata updates
            
        Returns:
            New version string
        """
        # Get the base model and metadata
        base_model, base_metadata = self.get_model(model_type, model_id, base_version)
        
        # Create a new version
        new_version = increment_version(base_version, increment_type)
        
        # Update metadata if needed
        if metadata_updates:
            base_metadata.update(metadata_updates)
        
        # Update the version in metadata
        base_metadata['version'] = new_version
        
        # Register the new model version
        self.register_model(
            model_obj=new_model_obj if new_model_obj is not None else base_model,
            metadata=base_metadata,
            version=new_version
        )
        
        return new_version
    
    @handle_errors(exceptions=Exception, error_message='Failed to get model performance', default_return={})
    def get_model_performance(self, model_type: str, model_id: str, version: Optional[str] = None) -> Dict[str, float]:
        """
        Gets performance metrics for a specific model version.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Optional specific version
            
        Returns:
            Dictionary of performance metrics
        """
        # If no version is provided, get the latest
        if version is None:
            version = self.version_manager.get_latest_version(model_type, model_id)
            
            if version is None:
                logger.warning(f"No versions found for model {model_type}/{model_id}")
                return {}
        
        # Get the metadata
        model_dir = self.get_model_path(model_type, model_id, version)
        metadata_path = model_dir / METADATA_FILE_NAME
        
        try:
            metadata = load_metadata(metadata_path)
            return metadata.get('performance_metrics', {})
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
            return {}
    
    @log_execution_time(logger, 'INFO')
    @handle_errors(exceptions=Exception, error_message='Failed to compare models', default_return={})
    def compare_models(self, model_type: str, model_id_1: str, version_1: str, model_id_2: str, version_2: str) -> Dict[str, Any]:
        """
        Compares two model versions based on their metadata and performance.
        
        Args:
            model_type: Type of the models
            model_id_1: Identifier for the first model
            version_1: Version of the first model
            model_id_2: Identifier for the second model
            version_2: Version of the second model
            
        Returns:
            Comparison results
        """
        # Get metadata for both models
        _, metadata_1 = self.get_model(model_type, model_id_1, version_1)
        _, metadata_2 = self.get_model(model_type, model_id_2, version_2)
        
        # Compare feature sets
        features_1 = set(metadata_1.get('feature_names', []))
        features_2 = set(metadata_2.get('feature_names', []))
        
        common_features = features_1.intersection(features_2)
        unique_features_1 = features_1 - features_2
        unique_features_2 = features_2 - features_1
        
        # Compare hyperparameters
        hyperparams_1 = metadata_1.get('hyperparameters', {})
        hyperparams_2 = metadata_2.get('hyperparameters', {})
        
        all_hyperparams = set(hyperparams_1.keys()).union(hyperparams_2.keys())
        hyperparam_diff = {}
        
        for param in all_hyperparams:
            if param not in hyperparams_1:
                hyperparam_diff[param] = {'status': 'added', 'value': hyperparams_2[param]}
            elif param not in hyperparams_2:
                hyperparam_diff[param] = {'status': 'removed', 'value': hyperparams_1[param]}
            elif hyperparams_1[param] != hyperparams_2[param]:
                hyperparam_diff[param] = {
                    'status': 'changed',
                    'old_value': hyperparams_1[param],
                    'new_value': hyperparams_2[param]
                }
        
        # Compare performance metrics
        perf_1 = metadata_1.get('performance_metrics', {})
        perf_2 = metadata_2.get('performance_metrics', {})
        
        all_metrics = set(perf_1.keys()).union(perf_2.keys())
        perf_diff = {}
        
        for metric in all_metrics:
            if metric not in perf_1:
                perf_diff[metric] = {'status': 'added', 'value': perf_2[metric]}
            elif metric not in perf_2:
                perf_diff[metric] = {'status': 'removed', 'value': perf_1[metric]}
            else:
                perf_diff[metric] = {
                    'model_1': perf_1[metric],
                    'model_2': perf_2[metric],
                    'diff': perf_2[metric] - perf_1[metric]
                }
        
        # Create comparison result
        comparison = {
            'model_1': {
                'model_id': model_id_1,
                'version': version_1,
                'training_date': metadata_1.get('training_date')
            },
            'model_2': {
                'model_id': model_id_2,
                'version': version_2,
                'training_date': metadata_2.get('training_date')
            },
            'features': {
                'common': list(common_features),
                'unique_to_model_1': list(unique_features_1),
                'unique_to_model_2': list(unique_features_2)
            },
            'hyperparameters': hyperparam_diff,
            'performance': perf_diff
        }
        
        return comparison
    
    @log_execution_time(logger, 'INFO')
    def list_models(self, model_type: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Lists all models in the registry, optionally filtered by model type.
        
        Args:
            model_type: Optional filter for model type
            
        Returns:
            Dictionary of model types, IDs, and versions
        """
        return list_models(self.registry_path, model_type)
    
    def get_model_path(self, model_type: str, model_id: str, version: Optional[str] = None) -> PathType:
        """
        Gets the file system path for a specific model version.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Optional specific version
            
        Returns:
            Path to the model version directory
        """
        # If no version is provided, get the latest
        if version is None:
            version = self.version_manager.get_latest_version(model_type, model_id)
            
            if version is None:
                raise ModelError(f"No versions found for model {model_type}/{model_id}")
        
        # Construct and return the path
        return Path(self.registry_path) / model_type / model_id / version
    
    @handle_errors(exceptions=OSError, error_message='Failed to create model directory', default_return=None)
    def ensure_model_directory(self, model_type: str, model_id: str, version: str) -> PathType:
        """
        Ensures that the directory structure for a model exists.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Version of the model
            
        Returns:
            Path to the model version directory
        """
        # Construct the path
        model_dir = Path(self.registry_path) / model_type / model_id / version
        
        # Create the directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        return model_dir