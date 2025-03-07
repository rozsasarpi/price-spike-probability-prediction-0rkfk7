"""
Implements a repository for storing and retrieving forecast data in the ERCOT RTLMP spike prediction system.
This module provides functionality to manage forecast data with time-based partitioning, versioning,
and efficient retrieval operations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, cast
import datetime
import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+

from ...utils.logging import get_logger
from ...utils.type_definitions import DataFrameType, PathType, ForecastResultDict, ThresholdValue, NodeID
from ...utils.error_handling import retry_with_backoff, handle_errors, DataError, DataFormatError
from ...data.validators.pandera_schemas import validate_with_forecast_schema
from .parquet_store import write_dataframe_to_parquet, read_dataframe_from_parquet, find_parquet_files, read_dataframes_from_partitions

# Set up logger
logger = get_logger(__name__)

# Global constants
DEFAULT_FORECAST_ROOT = Path(os.environ.get('FORECAST_ROOT', './forecasts'))
DEFAULT_COMPRESSION = 'snappy'
FORECAST_FILE_PREFIX = 'forecast_'
METADATA_FILE_NAME = 'metadata.json'


def get_forecast_partition_path(base_dir: PathType, forecast_timestamp: datetime.datetime) -> PathType:
    """
    Generates a partition path for forecast data based on forecast timestamp.
    
    Args:
        base_dir: Base directory for forecast storage
        forecast_timestamp: Timestamp when the forecast was generated
        
    Returns:
        Path to the forecast partition directory
    """
    year = forecast_timestamp.year
    month = forecast_timestamp.month
    day = forecast_timestamp.day
    
    return Path(base_dir) / str(year) / f"{month:02d}" / f"{day:02d}"


def generate_forecast_file_name(forecast_timestamp: datetime.datetime, suffix: str = "") -> str:
    """
    Generates a file name for a forecast file based on timestamp and optional suffix.
    
    Args:
        forecast_timestamp: Timestamp when the forecast was generated
        suffix: Optional suffix to add to the filename
        
    Returns:
        Forecast file name
    """
    timestamp_str = forecast_timestamp.strftime("%Y%m%d_%H%M%S")
    file_name = f"{FORECAST_FILE_PREFIX}{timestamp_str}"
    
    if suffix:
        file_name = f"{file_name}_{suffix}"
    
    file_name = f"{file_name}.parquet"
    return file_name


@retry_with_backoff(exceptions=(OSError, IOError), max_retries=3, initial_delay=1.0)
@handle_errors(exceptions=Exception, error_message='Failed to save forecast metadata', default_return=False)
def save_forecast_metadata(metadata: dict, file_path: PathType) -> bool:
    """
    Saves forecast metadata to a JSON file.
    
    Args:
        metadata: Dictionary containing forecast metadata
        file_path: Path to save the metadata file
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure the parent directory exists
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert datetime objects to ISO format strings
    metadata_copy = metadata.copy()
    for key, value in metadata_copy.items():
        if isinstance(value, datetime.datetime):
            metadata_copy[key] = value.isoformat()
    
    # Write to file with pretty formatting
    with open(file_path, 'w') as f:
        json.dump(metadata_copy, f, indent=2)
    
    logger.debug(f"Saved forecast metadata to {file_path}")
    return True


@retry_with_backoff(exceptions=(OSError, IOError), max_retries=3, initial_delay=1.0)
@handle_errors(exceptions=Exception, error_message='Failed to load forecast metadata', default_return=None)
def load_forecast_metadata(file_path: PathType) -> dict:
    """
    Loads forecast metadata from a JSON file.
    
    Args:
        file_path: Path to the metadata file
        
    Returns:
        Dictionary containing forecast metadata
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        logger.warning(f"Metadata file not found: {file_path}")
        return {}
    
    # Read and parse JSON
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    
    # Convert ISO format strings back to datetime objects
    for key, value in metadata.items():
        if isinstance(value, str) and 'T' in value:
            try:
                metadata[key] = datetime.datetime.fromisoformat(value)
            except ValueError:
                pass  # Not a valid ISO format, keep as string
    
    logger.debug(f"Loaded forecast metadata from {file_path}")
    return metadata


@handle_errors(exceptions=Exception, error_message='Failed to find latest forecast directory', default_return=None)
def find_latest_forecast_directory(base_dir: PathType) -> Optional[PathType]:
    """
    Finds the most recent forecast directory.
    
    Args:
        base_dir: Base directory for forecast storage
        
    Returns:
        Path to the most recent forecast directory, or None if not found
    """
    base_dir = Path(base_dir)
    
    # Check if base directory exists
    if not base_dir.exists():
        logger.warning(f"Base directory does not exist: {base_dir}")
        return None
    
    # Find all year directories
    year_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not year_dirs:
        logger.warning(f"No year directories found in {base_dir}")
        return None
    
    # Sort by name in descending order (most recent year first)
    year_dirs.sort(key=lambda d: d.name, reverse=True)
    
    for year_dir in year_dirs:
        # Find all month directories
        month_dirs = [d for d in year_dir.iterdir() if d.is_dir()]
        if not month_dirs:
            continue
        
        # Sort by name in descending order (most recent month first)
        month_dirs.sort(key=lambda d: d.name, reverse=True)
        
        for month_dir in month_dirs:
            # Find all day directories
            day_dirs = [d for d in month_dir.iterdir() if d.is_dir()]
            if not day_dirs:
                continue
            
            # Sort by name in descending order (most recent day first)
            day_dirs.sort(key=lambda d: d.name, reverse=True)
            
            # Return the first (most recent) day directory
            return day_dirs[0]
    
    # No directories found
    return None


class ForecastRepository:
    """
    Class that provides a high-level interface for storing and retrieving forecast data.
    Implements time-based partitioning, versioning, and efficient retrieval operations.
    """
    
    def __init__(self, forecast_root: Optional[PathType] = None, compression: Optional[str] = None):
        """
        Initialize the ForecastRepository with storage configuration.
        
        Args:
            forecast_root: Root directory for forecast storage
            compression: Compression algorithm to use for Parquet files
        """
        self._forecast_root = Path(forecast_root) if forecast_root else DEFAULT_FORECAST_ROOT
        self._compression = compression or DEFAULT_COMPRESSION
        
        # Ensure forecast root directory exists
        self._forecast_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ForecastRepository with root at {self._forecast_root}")
    
    def store_forecast(self, forecast_df: DataFrameType, metadata: Dict[str, Any], validate: bool = True) -> Tuple[PathType, PathType]:
        """
        Stores forecast data in Parquet format with time-based partitioning.
        
        Args:
            forecast_df: DataFrame containing forecast data
            metadata: Dictionary containing forecast metadata
            validate: Whether to validate the data against the forecast schema
            
        Returns:
            Tuple of (forecast_path, metadata_path)
            
        Raises:
            DataFormatError: If validation fails or required columns are missing
        """
        # Validate forecast data if requested
        if validate:
            forecast_df = validate_with_forecast_schema(forecast_df)
        
        # Ensure required columns are present
        required_columns = ['forecast_timestamp', 'target_timestamp']
        missing_columns = [col for col in required_columns if col not in forecast_df.columns]
        if missing_columns:
            raise DataFormatError(f"Missing required columns in forecast data: {missing_columns}")
        
        # Extract forecast timestamp from the first row
        forecast_timestamp = forecast_df['forecast_timestamp'].iloc[0]
        
        # Generate partition path
        partition_path = get_forecast_partition_path(self._forecast_root, forecast_timestamp)
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Generate file name
        file_name = generate_forecast_file_name(forecast_timestamp)
        forecast_path = partition_path / file_name
        
        # Write DataFrame to Parquet file
        write_dataframe_to_parquet(
            forecast_df,
            forecast_path,
            compression=self._compression,
            create_dir=True
        )
        
        # Create metadata path
        metadata_path = forecast_path.with_suffix('.json')
        
        # Update metadata with additional information
        metadata_to_save = metadata.copy()
        metadata_to_save.update({
            'forecast_timestamp': forecast_timestamp,
            'created_at': datetime.datetime.now(),
            'file_path': str(forecast_path),
            'row_count': len(forecast_df),
            'columns': list(forecast_df.columns),
        })
        
        # Save metadata
        save_forecast_metadata(metadata_to_save, metadata_path)
        
        logger.info(f"Stored forecast with {len(forecast_df)} rows at {forecast_path}")
        return forecast_path, metadata_path
    
    def retrieve_forecast(self, file_path: PathType) -> Tuple[DataFrameType, Dict[str, Any]]:
        """
        Retrieves a specific forecast by file path.
        
        Args:
            file_path: Path to the forecast file
            
        Returns:
            Tuple of (forecast_df, metadata)
            
        Raises:
            FileNotFoundError: If the forecast file does not exist
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"Forecast file not found: {file_path}")
            raise FileNotFoundError(f"Forecast file not found: {file_path}")
        
        # Read the forecast file
        forecast_df = read_dataframe_from_parquet(file_path)
        
        # Get the metadata file path
        metadata_path = file_path.with_suffix('.json')
        
        # Load metadata
        metadata = load_forecast_metadata(metadata_path)
        
        logger.debug(f"Retrieved forecast with {len(forecast_df)} rows from {file_path}")
        return forecast_df, metadata
    
    def retrieve_forecasts_by_date_range(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        thresholds: Optional[List[ThresholdValue]] = None,
        nodes: Optional[List[NodeID]] = None
    ) -> DataFrameType:
        """
        Retrieves forecasts within a specified date range.
        
        Args:
            start_date: Start date for the range
            end_date: End date for the range
            thresholds: Optional list of threshold values to filter by
            nodes: Optional list of node IDs to filter by
            
        Returns:
            DataFrame containing the combined forecast data
        """
        # Calculate the list of year/month/day partitions to check
        partitions = []
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day
            partitions.append((year, month, day))
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        # Read data from each partition
        dfs = []
        for year, month, day in partitions:
            partition_path = self._forecast_root / str(year) / f"{month:02d}" / f"{day:02d}"
            if not partition_path.exists():
                logger.debug(f"Partition path does not exist: {partition_path}")
                continue
            
            # Find all forecast files in this partition
            forecast_files = find_parquet_files(partition_path, pattern=f"{FORECAST_FILE_PREFIX}*.parquet")
            if not forecast_files:
                continue
            
            # Read each file and filter as needed
            for file_path in forecast_files:
                try:
                    df = read_dataframe_from_parquet(file_path)
                    
                    # Filter by threshold if specified
                    if thresholds is not None and 'threshold_value' in df.columns:
                        df = df[df['threshold_value'].isin(thresholds)]
                    
                    # Filter by node if specified
                    if nodes is not None and 'node_id' in df.columns:
                        df = df[df['node_id'].isin(nodes)]
                    
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error reading forecast file {file_path}: {str(e)}")
                    continue
        
        if not dfs:
            logger.warning(f"No forecasts found in date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        # Combine all DataFrames
        result = pd.concat(dfs, ignore_index=True)
        
        # Sort by forecast_timestamp and target_timestamp
        if 'forecast_timestamp' in result.columns and 'target_timestamp' in result.columns:
            result = result.sort_values(['forecast_timestamp', 'target_timestamp'])
        
        logger.info(f"Retrieved {len(result)} forecast rows for date range {start_date} to {end_date}")
        return result
    
    def get_latest_forecast(
        self, 
        thresholds: Optional[List[ThresholdValue]] = None,
        nodes: Optional[List[NodeID]] = None
    ) -> Tuple[Optional[DataFrameType], Optional[Dict[str, Any]], Optional[datetime.datetime]]:
        """
        Retrieves the most recent forecast data.
        
        Args:
            thresholds: Optional list of threshold values to filter by
            nodes: Optional list of node IDs to filter by
            
        Returns:
            Tuple of (forecast_df, metadata, forecast_timestamp) or (None, None, None) if not found
        """
        # Find the most recent forecast directory
        latest_dir = find_latest_forecast_directory(self._forecast_root)
        if latest_dir is None:
            logger.warning("No forecast directories found")
            return None, None, None
        
        # Find all forecast files in this directory
        forecast_files = find_parquet_files(latest_dir, pattern=f"{FORECAST_FILE_PREFIX}*.parquet")
        if not forecast_files:
            logger.warning(f"No forecast files found in {latest_dir}")
            return None, None, None
        
        # Sort by name in descending order (most recent first)
        forecast_files.sort(key=lambda f: f.name, reverse=True)
        
        # Try each file until we find one that matches our criteria
        for file_path in forecast_files:
            try:
                df = read_dataframe_from_parquet(file_path)
                
                # Filter by threshold if specified
                if thresholds is not None and 'threshold_value' in df.columns:
                    df = df[df['threshold_value'].isin(thresholds)]
                
                # Filter by node if specified
                if nodes is not None and 'node_id' in df.columns:
                    df = df[df['node_id'].isin(nodes)]
                
                if not df.empty:
                    # Load metadata
                    metadata_path = file_path.with_suffix('.json')
                    metadata = load_forecast_metadata(metadata_path)
                    
                    # Extract forecast timestamp
                    forecast_timestamp = df['forecast_timestamp'].iloc[0]
                    
                    logger.info(f"Found latest forecast from {forecast_timestamp}")
                    return df, metadata, forecast_timestamp
            except Exception as e:
                logger.warning(f"Error reading forecast file {file_path}: {str(e)}")
                continue
        
        logger.warning("No matching forecast found")
        return None, None, None
    
    def list_available_forecasts(self, limit: int = 100, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Lists all available forecasts with their metadata.
        
        Args:
            limit: Maximum number of forecasts to list
            include_metadata: Whether to include full metadata for each forecast
            
        Returns:
            List of dictionaries with forecast information
        """
        result = []
        
        # Get all year directories
        year_dirs = [d for d in self._forecast_root.iterdir() if d.is_dir()]
        year_dirs.sort(key=lambda d: d.name, reverse=True)  # Most recent first
        
        for year_dir in year_dirs:
            if len(result) >= limit:
                break
                
            # Get all month directories
            month_dirs = [d for d in year_dir.iterdir() if d.is_dir()]
            month_dirs.sort(key=lambda d: d.name, reverse=True)  # Most recent first
            
            for month_dir in month_dirs:
                if len(result) >= limit:
                    break
                    
                # Get all day directories
                day_dirs = [d for d in month_dir.iterdir() if d.is_dir()]
                day_dirs.sort(key=lambda d: d.name, reverse=True)  # Most recent first
                
                for day_dir in day_dirs:
                    if len(result) >= limit:
                        break
                        
                    # Find all forecast files
                    forecast_files = find_parquet_files(day_dir, pattern=f"{FORECAST_FILE_PREFIX}*.parquet")
                    forecast_files.sort(key=lambda f: f.name, reverse=True)  # Most recent first
                    
                    for file_path in forecast_files:
                        if len(result) >= limit:
                            break
                            
                        metadata_path = file_path.with_suffix('.json')
                        
                        # Extract forecast timestamp from filename
                        forecast_timestamp = None
                        file_name = file_path.name
                        
                        # Try to parse timestamp from filename
                        if file_name.startswith(FORECAST_FILE_PREFIX):
                            timestamp_str = file_name[len(FORECAST_FILE_PREFIX):].split('_')[0]
                            try:
                                forecast_timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d")
                            except ValueError:
                                pass
                        
                        # Basic info
                        forecast_info = {
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'created': file_path.stat().st_mtime,
                            'size': file_path.stat().st_size,
                        }
                        
                        # Add metadata if requested
                        if include_metadata and metadata_path.exists():
                            try:
                                metadata = load_forecast_metadata(metadata_path)
                                forecast_info['metadata'] = metadata
                                
                                # Use metadata timestamp if available
                                if 'forecast_timestamp' in metadata:
                                    forecast_timestamp = metadata['forecast_timestamp']
                            except Exception as e:
                                logger.warning(f"Error loading metadata for {file_path}: {str(e)}")
                        
                        # Add timestamp if found
                        if forecast_timestamp:
                            forecast_info['forecast_timestamp'] = forecast_timestamp
                            
                        result.append(forecast_info)
        
        logger.debug(f"Listed {len(result)} available forecasts")
        return result
    
    def delete_forecast(self, file_path: PathType) -> bool:
        """
        Deletes a specific forecast and its metadata.
        
        Args:
            file_path: Path to the forecast file
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Forecast file not found: {file_path}")
            return False
        
        # Get the metadata file path
        metadata_path = file_path.with_suffix('.json')
        
        # Delete forecast file
        try:
            file_path.unlink()
            logger.debug(f"Deleted forecast file: {file_path}")
            
            # Delete metadata file if it exists
            if metadata_path.exists():
                metadata_path.unlink()
                logger.debug(f"Deleted metadata file: {metadata_path}")
            
            logger.info(f"Successfully deleted forecast: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting forecast {file_path}: {str(e)}")
            return False
    
    def delete_forecasts_by_date_range(self, start_date: datetime.datetime, end_date: datetime.datetime) -> int:
        """
        Deletes all forecasts within a specified date range.
        
        Args:
            start_date: Start date for the range
            end_date: End date for the range
            
        Returns:
            Number of forecasts deleted
        """
        # Calculate the list of year/month/day partitions to check
        partitions = []
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day
            partitions.append((year, month, day))
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        # Process each partition
        deleted_count = 0
        for year, month, day in partitions:
            partition_path = self._forecast_root / str(year) / f"{month:02d}" / f"{day:02d}"
            if not partition_path.exists():
                continue
            
            # Find all forecast files in this partition
            forecast_files = find_parquet_files(partition_path, pattern=f"{FORECAST_FILE_PREFIX}*.parquet")
            
            # Delete each file
            for file_path in forecast_files:
                if self.delete_forecast(file_path):
                    deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} forecasts in date range {start_date} to {end_date}")
        return deleted_count
    
    def get_forecast_statistics(self) -> Dict[str, Any]:
        """
        Calculates statistics about stored forecasts.
        
        Returns:
            Dictionary with forecast statistics
        """
        stats = {
            'total_forecasts': 0,
            'total_size_bytes': 0,
            'oldest_forecast': None,
            'newest_forecast': None,
            'forecasts_by_threshold': {},
            'forecasts_by_node': {},
        }
        
        # Get all forecast files
        forecast_files = []
        try:
            # Walk through all directories
            for year_dir in [d for d in self._forecast_root.iterdir() if d.is_dir()]:
                for month_dir in [d for d in year_dir.iterdir() if d.is_dir()]:
                    for day_dir in [d for d in month_dir.iterdir() if d.is_dir()]:
                        forecast_files.extend(find_parquet_files(day_dir, pattern=f"{FORECAST_FILE_PREFIX}*.parquet"))
        except Exception as e:
            logger.warning(f"Error finding forecast files: {str(e)}")
        
        stats['total_forecasts'] = len(forecast_files)
        
        if not forecast_files:
            return stats
        
        # Calculate total size
        stats['total_size_bytes'] = sum(f.stat().st_size for f in forecast_files)
        
        # Find oldest and newest forecasts
        forecast_files.sort(key=lambda f: f.stat().st_mtime)
        stats['oldest_forecast'] = datetime.datetime.fromtimestamp(forecast_files[0].stat().st_mtime).isoformat()
        stats['newest_forecast'] = datetime.datetime.fromtimestamp(forecast_files[-1].stat().st_mtime).isoformat()
        
        # Sample a few forecasts to get threshold and node statistics
        sample_size = min(100, len(forecast_files))
        sample_files = forecast_files[-sample_size:]  # Use most recent files
        
        # Process each sample file
        for file_path in sample_files:
            try:
                df = read_dataframe_from_parquet(file_path)
                
                # Count by threshold
                if 'threshold_value' in df.columns:
                    thresholds = df['threshold_value'].unique()
                    for threshold in thresholds:
                        threshold_str = str(threshold)
                        if threshold_str not in stats['forecasts_by_threshold']:
                            stats['forecasts_by_threshold'][threshold_str] = 0
                        stats['forecasts_by_threshold'][threshold_str] += 1
                
                # Count by node
                if 'node_id' in df.columns:
                    nodes = df['node_id'].unique()
                    for node in nodes:
                        if node not in stats['forecasts_by_node']:
                            stats['forecasts_by_node'][node] = 0
                        stats['forecasts_by_node'][node] += 1
            except Exception as e:
                logger.warning(f"Error reading forecast file {file_path}: {str(e)}")
        
        return stats