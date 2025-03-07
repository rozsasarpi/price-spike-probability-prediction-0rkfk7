"""
Model persistence functionality for the ERCOT RTLMP spike prediction system.

This module provides utilities for saving, loading, and managing trained machine learning
models and their associated metadata to ensure reproducibility and proper model lifecycle
management.
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, Union, Type, cast
from pathlib import Path

import joblib  # version 1.2+

from ..utils.type_definitions import PathType, ModelConfigDict
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import ModelError, ModelLoadError, handle_errors

# Set up logger
logger = get_logger(__name__)

# Constants
MODEL_FILE_NAME = "model.joblib"
METADATA_FILE_NAME = "metadata.json"
DEFAULT_MODEL_DIR = "models"


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=Exception, error_message='Failed to save model', default_return=None)
def save_model_to_path(model: Any, path: PathType) -> PathType:
    """
    Saves a model object to a specified file path using joblib.
    
    Args:
        model: The model object to save
        path: Path where the model will be saved
        
    Returns:
        Path to the saved model file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    
    # Determine the full path to the model file
    model_path = Path(path) / MODEL_FILE_NAME
    
    # Save the model using joblib for efficiency
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    return model_path


@log_execution_time(logger, 'INFO')
@handle_errors(exceptions=Exception, error_message='Failed to load model', default_return=None)
def load_model_from_path(path: PathType) -> Any:
    """
    Loads a model object from a specified file path using joblib.
    
    Args:
        path: Path to the model file
        
    Returns:
        Loaded model object
    """
    model_path = Path(path) / MODEL_FILE_NAME
    
    # Check if the model file exists
    if not model_path.exists() or not model_path.is_file():
        error_msg = f"Model file not found at {model_path}"
        logger.error(error_msg)
        raise ModelLoadError(error_msg)
    
    # Load the model
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    return model


@handle_errors(exceptions=Exception, error_message='Failed to save metadata', default_return=None)
def save_metadata(metadata: ModelConfigDict, path: PathType) -> PathType:
    """
    Saves model metadata to a JSON file.
    
    Args:
        metadata: Dictionary containing model metadata
        path: Path where the metadata will be saved
        
    Returns:
        Path to the saved metadata file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    
    # Determine the full path to the metadata file
    metadata_path = Path(path) / METADATA_FILE_NAME
    
    # Create a copy of the metadata to avoid modifying the original
    metadata_copy = dict(metadata)
    
    # Convert datetime objects to ISO format strings for JSON serialization
    for key, value in metadata_copy.items():
        if isinstance(value, datetime):
            metadata_copy[key] = value.isoformat()
            
    # Save the metadata as JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata_copy, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    
    return metadata_path


@handle_errors(exceptions=Exception, error_message='Failed to load metadata', default_return=None)
def load_metadata(path: PathType) -> ModelConfigDict:
    """
    Loads model metadata from a JSON file.
    
    Args:
        path: Path to the metadata file
        
    Returns:
        Dictionary containing model metadata
    """
    metadata_path = Path(path) / METADATA_FILE_NAME
    
    # Check if the metadata file exists
    if not metadata_path.exists() or not metadata_path.is_file():
        error_msg = f"Metadata file not found at {metadata_path}"
        logger.error(error_msg)
        raise ModelLoadError(error_msg)
    
    # Load the metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Convert ISO format date strings back to datetime objects
    for key, value in metadata.items():
        if isinstance(value, str) and key == 'training_date':
            try:
                metadata[key] = datetime.fromisoformat(value)
            except ValueError:
                logger.warning(f"Could not parse date string '{value}' for key '{key}'")
    
    logger.info(f"Loaded metadata from {metadata_path}")
    
    return cast(ModelConfigDict, metadata)


def get_model_path(
    model_id: str, 
    model_type: str, 
    version: Optional[str] = None, 
    base_path: Optional[PathType] = None
) -> PathType:
    """
    Constructs a standard path for a model based on its ID and version.
    
    Args:
        model_id: Unique identifier for the model
        model_type: Type of model (e.g., 'xgboost', 'lightgbm')
        version: Model version (e.g., '1.0.0')
        base_path: Base directory for models
        
    Returns:
        Path to the model directory
    """
    # If base_path is not provided, use the default model directory
    if base_path is None:
        base_path = DEFAULT_MODEL_DIR
    
    # If version is not provided, use 'latest'
    version_str = version if version is not None else 'latest'
    
    # Construct the model path
    model_path = Path(base_path) / model_type / model_id / version_str
    
    return model_path


@log_execution_time(logger, 'INFO')
def list_models(
    model_type: Optional[str] = None, 
    base_path: Optional[PathType] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Lists all available models in the model directory.
    
    Args:
        model_type: Filter by model type
        base_path: Base directory for models
        
    Returns:
        Dictionary of available models with their metadata
    """
    # If base_path is not provided, use the default model directory
    if base_path is None:
        base_path = DEFAULT_MODEL_DIR
    
    base_path = Path(base_path)
    results = {}
    
    # If model_type is provided, only check that directory
    if model_type:
        model_type_dirs = [base_path / model_type]
    else:
        # Otherwise, check all model type directories
        try:
            model_type_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        except FileNotFoundError:
            logger.warning(f"Model directory {base_path} not found")
            return {}
    
    # Iterate through model type directories
    for model_type_dir in model_type_dirs:
        if not model_type_dir.exists() or not model_type_dir.is_dir():
            continue
        
        model_type_name = model_type_dir.name
        results[model_type_name] = {}
        
        # Iterate through model ID directories
        try:
            model_id_dirs = [d for d in model_type_dir.iterdir() if d.is_dir()]
        except FileNotFoundError:
            continue
        
        for model_id_dir in model_id_dirs:
            model_id = model_id_dir.name
            results[model_type_name][model_id] = {}
            
            # Iterate through version directories
            try:
                version_dirs = [d for d in model_id_dir.iterdir() if d.is_dir()]
            except FileNotFoundError:
                continue
            
            for version_dir in version_dirs:
                version = version_dir.name
                
                # Load metadata if available
                try:
                    metadata = load_metadata(version_dir)
                    results[model_type_name][model_id][version] = metadata
                except Exception as e:
                    logger.warning(f"Failed to load metadata for model {model_id} version {version}: {e}")
                    results[model_type_name][model_id][version] = {"error": str(e)}
    
    return results


@handle_errors(exceptions=Exception, error_message='Failed to delete model', default_return=False)
def delete_model(
    model_id: str, 
    model_type: str, 
    version: Optional[str] = None, 
    base_path: Optional[PathType] = None
) -> bool:
    """
    Deletes a model and its metadata from storage.
    
    Args:
        model_id: Unique identifier for the model
        model_type: Type of model
        version: Specific version to delete, or None to delete all versions
        base_path: Base directory for models
        
    Returns:
        True if deletion was successful, False otherwise
    """
    # Get the model path
    model_path = get_model_path(model_id, model_type, version, base_path)
    
    # Check if the path exists
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        return False
    
    # If a specific version is provided, delete only that version
    if version is not None:
        # Delete the model file and metadata file
        model_file = model_path / MODEL_FILE_NAME
        metadata_file = model_path / METADATA_FILE_NAME
        
        if model_file.exists():
            model_file.unlink()
        
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Remove the version directory if it's empty
        try:
            model_path.rmdir()
            logger.info(f"Deleted model {model_id} version {version}")
        except OSError:
            logger.warning(f"Could not remove directory {model_path}, it may not be empty")
    else:
        # Delete the entire model directory (all versions)
        import shutil
        shutil.rmtree(model_path.parent, ignore_errors=True)
        logger.info(f"Deleted all versions of model {model_id}")
    
    return True


@handle_errors(exceptions=Exception, error_message='Failed to copy model', default_return=False)
def copy_model(
    source_model_id: str,
    target_model_id: str,
    model_type: str,
    source_version: Optional[str] = None,
    target_version: Optional[str] = None,
    base_path: Optional[PathType] = None
) -> bool:
    """
    Copies a model and its metadata to a new location.
    
    Args:
        source_model_id: ID of the source model
        target_model_id: ID for the target model
        model_type: Type of model
        source_version: Version of the source model, or None for latest
        target_version: Version for the target model, or None to use source version
        base_path: Base directory for models
        
    Returns:
        True if copy was successful, False otherwise
    """
    # Get the source model path
    source_path = get_model_path(source_model_id, model_type, source_version, base_path)
    
    # Determine target version
    if target_version is None:
        target_version = source_version
    
    # Get the target model path
    target_path = get_model_path(target_model_id, model_type, target_version, base_path)
    
    # Check if the source path exists
    if not source_path.exists():
        logger.error(f"Source model not found at {source_path}")
        return False
    
    # Check if the target path already exists
    if target_path.exists():
        logger.warning(f"Target path {target_path} already exists, will not overwrite")
        return False
    
    # Ensure the target directory exists
    os.makedirs(target_path, exist_ok=True)
    
    try:
        # Load the model and metadata
        model = load_model_from_path(source_path)
        metadata = load_metadata(source_path)
        
        # Update the metadata with the new model_id and version
        metadata['model_id'] = target_model_id
        if target_version is not None:
            metadata['version'] = target_version
        
        # Save the model and metadata to the target path
        save_model_to_path(model, target_path)
        save_metadata(metadata, target_path)
        
        logger.info(f"Copied model from {source_path} to {target_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying model: {e}")
        # Clean up any partial copy
        if target_path.exists():
            import shutil
            shutil.rmtree(target_path, ignore_errors=True)
        return False


class ModelPersistence:
    """
    Class that handles model persistence operations with a consistent interface.
    """
    
    def __init__(self, base_path: Optional[PathType] = None):
        """
        Initialize the ModelPersistence with a base path.
        
        Args:
            base_path: Base directory for models
        """
        self.base_path = base_path if base_path is not None else DEFAULT_MODEL_DIR
    
    @log_execution_time(logger, 'INFO')
    def save_model(
        self, 
        model: Any, 
        metadata: ModelConfigDict, 
        custom_path: Optional[PathType] = None
    ) -> PathType:
        """
        Saves a model and its metadata to storage.
        
        Args:
            model: Model object to save
            metadata: Dictionary containing model metadata
            custom_path: Optional custom path for saving the model
            
        Returns:
            Path to the saved model directory
        """
        # Extract model_id, model_type, and version from metadata
        model_id = metadata.get('model_id')
        model_type = metadata.get('model_type')
        version = metadata.get('version')
        
        if not all([model_id, model_type]):
            raise ValueError("Metadata must contain 'model_id' and 'model_type'")
        
        # Determine the save path
        if custom_path:
            save_path = Path(custom_path)
        else:
            save_path = get_model_path(model_id, model_type, version, self.base_path)
        
        # Save the model and metadata
        save_model_to_path(model, save_path)
        save_metadata(metadata, save_path)
        
        return save_path
    
    @log_execution_time(logger, 'INFO')
    def load_model(
        self, 
        model_id: str, 
        model_type: str, 
        version: Optional[str] = None, 
        custom_path: Optional[PathType] = None
    ) -> tuple[Any, ModelConfigDict]:
        """
        Loads a model and its metadata from storage.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model
            version: Model version, or None for latest
            custom_path: Optional custom path for loading the model
            
        Returns:
            Tuple of (model, metadata)
        """
        # Determine the load path
        if custom_path:
            load_path = Path(custom_path)
        else:
            load_path = get_model_path(model_id, model_type, version, self.base_path)
        
        # Load the model and metadata
        model = load_model_from_path(load_path)
        metadata = load_metadata(load_path)
        
        return model, metadata
    
    def delete_model(
        self, 
        model_id: str, 
        model_type: str, 
        version: Optional[str] = None
    ) -> bool:
        """
        Deletes a model and its metadata from storage.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model
            version: Specific version to delete, or None to delete all versions
            
        Returns:
            True if deletion was successful, False otherwise
        """
        return delete_model(model_id, model_type, version, self.base_path)
    
    def list_models(
        self, 
        model_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Lists all available models in the model directory.
        
        Args:
            model_type: Filter by model type
            
        Returns:
            Dictionary of available models with their metadata
        """
        return list_models(model_type, self.base_path)
    
    def copy_model(
        self,
        source_model_id: str,
        target_model_id: str,
        model_type: str,
        source_version: Optional[str] = None,
        target_version: Optional[str] = None
    ) -> bool:
        """
        Copies a model and its metadata to a new location.
        
        Args:
            source_model_id: ID of the source model
            target_model_id: ID for the target model
            model_type: Type of model
            source_version: Version of the source model, or None for latest
            target_version: Version for the target model, or None to use source version
            
        Returns:
            True if copy was successful, False otherwise
        """
        return copy_model(
            source_model_id,
            target_model_id,
            model_type,
            source_version,
            target_version,
            self.base_path
        )
    
    def get_model_path(
        self,
        model_id: str,
        model_type: str,
        version: Optional[str] = None
    ) -> PathType:
        """
        Gets the path for a model based on its ID and version.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model
            version: Model version, or None for latest
            
        Returns:
            Path to the model directory
        """
        return get_model_path(model_id, model_type, version, self.base_path)