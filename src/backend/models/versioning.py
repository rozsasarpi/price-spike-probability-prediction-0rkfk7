"""
Semantic versioning utilities for machine learning models in the ERCOT RTLMP spike prediction system.

This module implements semantic versioning functionality for machine learning models,
providing utilities for version management, comparison, and tracking to ensure
reproducibility and proper model lifecycle management.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
import semver  # version 2.13+
from typing import Dict, List, Optional, Tuple, Union, Any, Literal

from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import handle_errors, ModelError
from ..utils.type_definitions import ModelConfigDict, PathType

# Set up logger
logger = get_logger(__name__)

# Global constants
VERSION_PATTERN = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')
INITIAL_VERSION = '0.1.0'
VERSION_METADATA_FILE = 'version_metadata.json'
IncrementType = Literal['major', 'minor', 'patch']


@handle_errors(exceptions=ValueError, error_message='Invalid version format', default_return=None)
def parse_version(version: str) -> Tuple[int, int, int]:
    """
    Parses a version string into its major, minor, and patch components.
    
    Args:
        version: Version string in format 'major.minor.patch'
        
    Returns:
        Tuple of (major, minor, patch) version components
        
    Raises:
        ValueError: If the version string does not match the expected format
    """
    match = VERSION_PATTERN.match(version)
    if not match:
        raise ValueError(f"Invalid version format: {version}. Expected 'major.minor.patch'")
    
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


@handle_errors(exceptions=ValueError, error_message='Failed to increment version', default_return=None)
def increment_version(current_version: str, increment_type: IncrementType) -> str:
    """
    Increments a version string according to semantic versioning rules.
    
    Args:
        current_version: Current version string in format 'major.minor.patch'
        increment_type: Type of increment ('major', 'minor', or 'patch')
        
    Returns:
        Incremented version string
        
    Raises:
        ValueError: If the version string is invalid or increment_type is invalid
    """
    major, minor, patch = parse_version(current_version)
    
    if increment_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif increment_type == 'minor':
        minor += 1
        patch = 0
    elif increment_type == 'patch':
        patch += 1
    else:
        raise ValueError(f"Invalid increment type: {increment_type}. Expected 'major', 'minor', or 'patch'")
    
    return f"{major}.{minor}.{patch}"


@handle_errors(exceptions=ValueError, error_message='Failed to compare versions', default_return=None)
def compare_versions(version1: str, version2: str) -> int:
    """
    Compares two version strings according to semantic versioning rules.
    
    Args:
        version1: First version string
        version2: Second version string
        
    Returns:
        1 if version1 > version2, -1 if version1 < version2, 0 if equal
        
    Raises:
        ValueError: If either version string is invalid
    """
    # Parse both versions
    v1_major, v1_minor, v1_patch = parse_version(version1)
    v2_major, v2_minor, v2_patch = parse_version(version2)
    
    # Compare major versions
    if v1_major > v2_major:
        return 1
    elif v1_major < v2_major:
        return -1
    
    # If major versions are equal, compare minor versions
    if v1_minor > v2_minor:
        return 1
    elif v1_minor < v2_minor:
        return -1
    
    # If minor versions are equal, compare patch versions
    if v1_patch > v2_patch:
        return 1
    elif v1_patch < v2_patch:
        return -1
    
    # All components are equal
    return 0


@handle_errors(exceptions=ValueError, error_message='Failed to determine latest version', default_return=None)
def get_latest_version(versions: List[str]) -> Optional[str]:
    """
    Determines the latest version from a list of version strings.
    
    Args:
        versions: List of version strings
        
    Returns:
        Latest version string or None if the list is empty
    """
    if not versions:
        return None
    
    latest_version = versions[0]
    for version in versions[1:]:
        if compare_versions(version, latest_version) > 0:
            latest_version = version
    
    return latest_version


def is_valid_version(version: str) -> bool:
    """
    Checks if a string is a valid semantic version.
    
    Args:
        version: Version string to validate
        
    Returns:
        True if the version is valid, False otherwise
    """
    try:
        parse_version(version)
        return True
    except ValueError:
        return False


@handle_errors(exceptions=Exception, error_message='Failed to save version metadata', default_return=False)
def save_version_metadata(metadata: Dict[str, Any], path: PathType) -> bool:
    """
    Saves version metadata to a JSON file.
    
    Args:
        metadata: Metadata dictionary to save
        path: Path to save the metadata file
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Convert datetime objects to strings for JSON serialization
    serializable_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, datetime):
            serializable_metadata[key] = value.isoformat()
        else:
            serializable_metadata[key] = value
    
    # Write the metadata to the file
    with open(path, 'w') as f:
        json.dump(serializable_metadata, f, indent=2)
    
    logger.info(f"Saved version metadata to {path}")
    return True


@handle_errors(exceptions=Exception, error_message='Failed to load version metadata', default_return=None)
def load_version_metadata(path: PathType) -> Dict[str, Any]:
    """
    Loads version metadata from a JSON file.
    
    Args:
        path: Path to the metadata file
        
    Returns:
        Loaded metadata dictionary
    """
    if not os.path.exists(path):
        logger.warning(f"Version metadata file not found: {path}")
        return {}
    
    with open(path, 'r') as f:
        metadata = json.load(f)
    
    # Convert ISO format date strings back to datetime objects
    for key, value in metadata.items():
        if isinstance(value, str) and 'T' in value and value.count('-') >= 2:
            try:
                metadata[key] = datetime.fromisoformat(value)
            except ValueError:
                # If it's not a valid datetime string, keep the original value
                pass
    
    return metadata


class VersionManager:
    """
    Class that manages version information for models.
    """
    
    def __init__(self, base_path: Optional[PathType] = None):
        """
        Initialize the VersionManager with a base path.
        
        Args:
            base_path: Base directory path for model storage
        """
        self.base_path = base_path if base_path is not None else Path('models')
        
        # Ensure the base directory exists
        os.makedirs(self.base_path, exist_ok=True)
    
    @log_execution_time(logger, 'INFO')
    def get_latest_version(self, model_type: str, model_id: str) -> Optional[str]:
        """
        Gets the latest version for a specific model.
        
        Args:
            model_type: Type of the model (e.g., 'xgboost', 'lightgbm')
            model_id: Identifier for the model
            
        Returns:
            Latest version string or None if no versions exist
        """
        # Construct the path to the model's directory
        model_path = Path(self.base_path) / model_type / model_id
        
        # Check if the directory exists
        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_path}")
            return None
        
        # List all subdirectories that match valid version format
        versions = []
        for item in model_path.iterdir():
            if item.is_dir() and is_valid_version(item.name):
                versions.append(item.name)
        
        # Get the latest version
        return get_latest_version(versions)
    
    @log_execution_time(logger, 'INFO')
    def create_new_version(
        self, 
        model_type: str, 
        model_id: str, 
        base_version: Optional[str] = None, 
        increment_type: IncrementType = 'patch'
    ) -> str:
        """
        Creates a new version for a model based on an existing version.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            base_version: Optional base version to increment from
            increment_type: Type of increment ('major', 'minor', or 'patch')
            
        Returns:
            New version string
        """
        # If base_version is not provided, get the latest version
        if base_version is None:
            base_version = self.get_latest_version(model_type, model_id)
            
            # If no existing versions, use initial version
            if base_version is None:
                logger.info(f"No existing versions found for {model_type}/{model_id}, using initial version")
                return INITIAL_VERSION
        elif not is_valid_version(base_version):
            raise ValueError(f"Invalid base version: {base_version}")
        
        # Create new version by incrementing the base version
        new_version = increment_version(base_version, increment_type)
        
        # Create version directory
        version_path = Path(self.base_path) / model_type / model_id / new_version
        os.makedirs(version_path, exist_ok=True)
        
        # Create metadata
        metadata = {
            'created': datetime.now(),
            'base_version': base_version,
            'increment_type': increment_type
        }
        
        # Save metadata
        metadata_path = version_path / VERSION_METADATA_FILE
        save_version_metadata(metadata, metadata_path)
        
        logger.info(f"Created new version {new_version} for {model_type}/{model_id}")
        return new_version
    
    @handle_errors(exceptions=Exception, error_message='Failed to get version metadata', default_return={})
    def get_version_metadata(self, model_type: str, model_id: str, version: str) -> Dict[str, Any]:
        """
        Gets metadata for a specific model version.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Version string
            
        Returns:
            Version metadata dictionary
        """
        # Construct the path to the version metadata file
        metadata_path = Path(self.base_path) / model_type / model_id / version / VERSION_METADATA_FILE
        
        # Load and return the metadata
        return load_version_metadata(metadata_path)
    
    @handle_errors(exceptions=Exception, error_message='Failed to list versions', default_return=[])
    def list_versions(self, model_type: str, model_id: str) -> List[str]:
        """
        Lists all versions for a specific model.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            
        Returns:
            List of version strings
        """
        # Construct the path to the model's directory
        model_path = Path(self.base_path) / model_type / model_id
        
        # Check if the directory exists
        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_path}")
            return []
        
        # List all subdirectories that match valid version format
        versions = []
        for item in model_path.iterdir():
            if item.is_dir() and is_valid_version(item.name):
                versions.append(item.name)
        
        # Sort the versions using compare_versions function
        versions.sort(key=lambda v: semver.VersionInfo.parse(v))
        
        return versions
    
    @handle_errors(exceptions=Exception, error_message='Failed to delete version', default_return=False)
    def delete_version(self, model_type: str, model_id: str, version: str) -> bool:
        """
        Deletes a specific model version.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Version to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Validate the version format
        if not is_valid_version(version):
            raise ValueError(f"Invalid version format: {version}")
        
        # Construct the path to the version directory
        version_path = Path(self.base_path) / model_type / model_id / version
        
        # Check if the directory exists
        if not version_path.exists():
            logger.warning(f"Version directory not found: {version_path}")
            return False
        
        # Remove the directory and all its contents
        import shutil
        shutil.rmtree(version_path)
        
        logger.info(f"Deleted version {version} of {model_type}/{model_id}")
        return True
    
    @log_execution_time(logger, 'INFO')
    def compare_model_versions(
        self, 
        model_type: str, 
        model_id: str, 
        version1: str, 
        version2: str
    ) -> Dict[str, Any]:
        """
        Compares two model versions based on their metadata.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
        """
        # Get metadata for both versions
        metadata1 = self.get_version_metadata(model_type, model_id, version1)
        metadata2 = self.get_version_metadata(model_type, model_id, version2)
        
        # Compare version numbers
        version_comparison = compare_versions(version1, version2)
        
        # Initialize comparison results
        comparison = {
            'version_comparison': version_comparison,
            'version1': version1,
            'version2': version2,
            'metadata1': metadata1,
            'metadata2': metadata2,
            'differences': {}
        }
        
        # Compare creation timestamps
        if 'created' in metadata1 and 'created' in metadata2:
            time_diff = metadata2['created'] - metadata1['created']
            comparison['time_difference'] = time_diff.total_seconds()
        
        # Compare other metadata fields
        all_keys = set(metadata1.keys()) | set(metadata2.keys())
        for key in all_keys:
            # Skip 'created' field as it was already compared
            if key == 'created':
                continue
                
            value1 = metadata1.get(key)
            value2 = metadata2.get(key)
            
            if key not in metadata1:
                comparison['differences'][key] = {'status': 'added', 'value': value2}
            elif key not in metadata2:
                comparison['differences'][key] = {'status': 'removed', 'value': value1}
            elif value1 != value2:
                comparison['differences'][key] = {
                    'status': 'changed',
                    'old_value': value1,
                    'new_value': value2
                }
        
        return comparison
    
    @log_execution_time(logger, 'INFO')
    def get_version_history(self, model_type: str, model_id: str) -> List[Dict[str, Any]]:
        """
        Gets the version history for a model.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            
        Returns:
            List of version metadata dictionaries
        """
        # Get list of all versions
        versions = self.list_versions(model_type, model_id)
        
        # For each version, get its metadata
        history = []
        for version in versions:
            metadata = self.get_version_metadata(model_type, model_id, version)
            
            # Add version to metadata
            metadata['version'] = version
            
            history.append(metadata)
        
        # Sort the list by version number
        history.sort(key=lambda x: semver.VersionInfo.parse(x['version']))
        
        return history
    
    def get_version_path(self, model_type: str, model_id: str, version: str) -> PathType:
        """
        Gets the file system path for a specific model version.
        
        Args:
            model_type: Type of the model
            model_id: Identifier for the model
            version: Version string
            
        Returns:
            Path to the model version directory
        """
        # Validate the version format
        if not is_valid_version(version):
            raise ValueError(f"Invalid version format: {version}")
        
        # Construct and return the path to the model version directory
        return Path(self.base_path) / model_type / model_id / version