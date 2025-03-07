"""
Parquet storage implementation for the ERCOT RTLMP spike prediction system.

This module provides efficient columnar storage for time series data with partitioning
by date and data type, supporting the system's batch processing requirements.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import datetime

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
import pyarrow  # version 11.0+

from ...utils.logging import get_logger
from ...utils.type_definitions import DataFrameType, PathType
from ...utils.error_handling import (
    retry_with_backoff, 
    handle_errors, 
    DataError, 
    DataFormatError
)
from ...utils.validation import (
    validate_dataframe_schema, 
    validate_temporal_consistency
)
from ..validators.pandera_schemas import (
    validate_with_rtlmp_schema,
    validate_with_weather_schema,
    validate_with_grid_condition_schema,
    validate_with_feature_schema,
    validate_with_forecast_schema
)

# Set up logger
logger = get_logger(__name__)

# Global constants
DEFAULT_STORAGE_ROOT = Path(os.environ.get('STORAGE_ROOT', './data'))
DEFAULT_COMPRESSION = 'snappy'
DEFAULT_ROW_GROUP_SIZE = 100000


@retry_with_backoff(exceptions=(IOError, OSError), max_retries=3)
@handle_errors(exceptions=(IOError, OSError), error_message='Failed to write DataFrame to Parquet file')
def write_dataframe_to_parquet(
    df: DataFrameType,
    file_path: PathType,
    compression: str = DEFAULT_COMPRESSION,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    create_dir: bool = True
) -> PathType:
    """
    Writes a pandas DataFrame to a Parquet file with optimized settings.

    Args:
        df: DataFrame to write
        file_path: Path to write the Parquet file to
        compression: Compression algorithm to use (default: snappy)
        row_group_size: Number of rows per group in Parquet file
        create_dir: Whether to create parent directories if they don't exist

    Returns:
        Path to the written Parquet file
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    
    # Convert to Path object if it's a string
    file_path = Path(file_path)
    
    # Create parent directories if needed
    if create_dir:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set defaults if not provided
    compression = compression or DEFAULT_COMPRESSION
    row_group_size = row_group_size or DEFAULT_ROW_GROUP_SIZE
    
    # Write DataFrame to Parquet file with optimized settings
    df.to_parquet(
        file_path,
        engine='pyarrow',
        compression=compression,
        row_group_size=row_group_size,
        index=False  # Don't include index unless it contains meaningful data
    )
    
    logger.debug(f"Successfully wrote DataFrame with {len(df)} rows to {file_path}")
    return file_path


@retry_with_backoff(exceptions=(IOError, OSError), max_retries=3)
@handle_errors(exceptions=(IOError, OSError), error_message='Failed to read DataFrame from Parquet file')
def read_dataframe_from_parquet(
    file_path: PathType,
    columns: Optional[List[str]] = None,
    filters: Optional[List[Tuple]] = None
) -> DataFrameType:
    """
    Reads a pandas DataFrame from a Parquet file with optimized settings.

    Args:
        file_path: Path to the Parquet file to read
        columns: Optional list of columns to read
        filters: Optional filters to apply (pyarrow filter syntax)

    Returns:
        DataFrame read from the Parquet file
    """
    # Convert to Path object if it's a string
    file_path = Path(file_path)
    
    # Verify file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(
        file_path,
        engine='pyarrow',
        columns=columns,
        filters=filters
    )
    
    logger.debug(f"Successfully read DataFrame with {len(df)} rows from {file_path}")
    return df


@handle_errors(exceptions=(IOError, OSError), error_message='Failed to find Parquet files')
def find_parquet_files(
    directory: PathType,
    pattern: str = "*.parquet",
    recursive: bool = False
) -> List[PathType]:
    """
    Finds Parquet files matching specified criteria in a directory.

    Args:
        directory: Directory to search for Parquet files
        pattern: Glob pattern to match files
        recursive: Whether to search recursively in subdirectories

    Returns:
        List of paths to matching Parquet files
    """
    # Convert to Path object if it's a string
    directory = Path(directory)
    
    # Verify directory exists
    if not directory.exists():
        raise NotADirectoryError(f"Directory not found: {directory}")
    elif not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    # Find matching files
    if recursive:
        # Use glob.glob for recursive search
        matching_files = [Path(p) for p in glob.glob(
            str(directory / "**" / pattern), 
            recursive=True
        )]
    else:
        # Use Path.glob for non-recursive search
        matching_files = list(directory.glob(pattern))
    
    # Sort files for consistent ordering
    matching_files.sort()
    
    logger.debug(f"Found {len(matching_files)} Parquet files in {directory}")
    return matching_files


@handle_errors(exceptions=(IOError, OSError), error_message='Failed to read DataFrames from partitions')
def read_dataframes_from_partitions(
    directory: PathType,
    columns: Optional[List[str]] = None,
    filters: Optional[List[Tuple]] = None,
    recursive: bool = False
) -> DataFrameType:
    """
    Reads and concatenates DataFrames from multiple Parquet partitions.

    Args:
        directory: Directory containing Parquet partitions
        columns: Optional list of columns to read
        filters: Optional filters to apply
        recursive: Whether to search recursively in subdirectories

    Returns:
        Concatenated DataFrame from all partitions
    """
    # Find all Parquet files in the directory
    parquet_files = find_parquet_files(directory, recursive=recursive)
    
    if not parquet_files:
        logger.warning(f"No Parquet files found in {directory}")
        return pd.DataFrame()
    
    # Read each Parquet file and collect the DataFrames
    dfs = []
    for file_path in parquet_files:
        try:
            df = read_dataframe_from_parquet(file_path, columns=columns, filters=filters)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {str(e)}")
    
    if not dfs:
        logger.warning("No valid DataFrames found")
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    result = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp if present
    if 'timestamp' in result.columns:
        result = result.sort_values('timestamp')
    
    logger.debug(f"Concatenated {len(dfs)} DataFrames with total {len(result)} rows")
    return result


def get_partition_path(
    base_dir: PathType,
    data_type: str,
    timestamp: datetime.datetime
) -> PathType:
    """
    Generates a partition path based on data type and timestamp.

    Args:
        base_dir: Base directory for data storage
        data_type: Type of data (e.g., 'rtlmp', 'weather')
        timestamp: Timestamp for partitioning

    Returns:
        Path to the partition directory
    """
    year = timestamp.year
    month = timestamp.month
    return Path(base_dir) / data_type / str(year) / f"{month:02d}"


class ParquetStore:
    """
    Class that provides a high-level interface for storing and retrieving data in Parquet format.
    This class implements time-based partitioning and optimized storage for different data types.
    """
    
    def __init__(
        self,
        storage_root: Optional[PathType] = None,
        compression: str = DEFAULT_COMPRESSION,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE
    ):
        """
        Initialize the ParquetStore with storage configuration.

        Args:
            storage_root: Root directory for data storage
            compression: Compression algorithm to use for Parquet files
            row_group_size: Number of rows per group in Parquet files
        """
        self._storage_root = Path(storage_root) if storage_root else DEFAULT_STORAGE_ROOT
        self._compression = compression
        self._row_group_size = row_group_size
        
        # Ensure storage root directory exists
        self._storage_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self._logger = get_logger(f"{__name__}.ParquetStore")
        self._logger.info(f"Initialized ParquetStore with root at {self._storage_root}")
    
    def store_rtlmp_data(self, df: DataFrameType, validate: bool = True) -> PathType:
        """
        Stores RTLMP data in Parquet format with time-based partitioning.

        Args:
            df: DataFrame with RTLMP data
            validate: Whether to validate the data schema

        Returns:
            Path to the stored data
        """
        if validate:
            df = validate_with_rtlmp_schema(df)
        
        if 'timestamp' not in df.columns:
            raise DataFormatError("RTLMP data must have a 'timestamp' column")
        
        # Ensure timestamp is datetime type
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group data by year and month for partitioning
        base_path = self._storage_root / 'rtlmp'
        grouped = df.groupby([df['timestamp'].dt.year, df['timestamp'].dt.month])
        
        for (year, month), group_df in grouped:
            # Create partition directory
            partition_dir = base_path / str(year) / f"{month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file name based on node_id if available, otherwise use a timestamp
            if 'node_id' in group_df.columns and group_df['node_id'].nunique() == 1:
                node_id = group_df['node_id'].iloc[0]
                file_name = f"rtlmp_{node_id}_{year}-{month:02d}.parquet"
            else:
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"rtlmp_{year}-{month:02d}_{current_time}.parquet"
            
            file_path = partition_dir / file_name
            
            # Write DataFrame to Parquet file
            write_dataframe_to_parquet(
                group_df,
                file_path,
                compression=self._compression,
                row_group_size=self._row_group_size
            )
        
        self._logger.info(f"Stored RTLMP data with {len(df)} rows in {base_path}")
        return base_path
    
    def store_weather_data(self, df: DataFrameType, validate: bool = True) -> PathType:
        """
        Stores weather data in Parquet format with time-based partitioning.

        Args:
            df: DataFrame with weather data
            validate: Whether to validate the data schema

        Returns:
            Path to the stored data
        """
        if validate:
            df = validate_with_weather_schema(df)
        
        if 'timestamp' not in df.columns:
            raise DataFormatError("Weather data must have a 'timestamp' column")
        
        # Ensure timestamp is datetime type
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group data by year and month for partitioning
        base_path = self._storage_root / 'weather'
        grouped = df.groupby([df['timestamp'].dt.year, df['timestamp'].dt.month])
        
        for (year, month), group_df in grouped:
            # Create partition directory
            partition_dir = base_path / str(year) / f"{month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file name based on location_id if available, otherwise use a timestamp
            if 'location_id' in group_df.columns and group_df['location_id'].nunique() == 1:
                location_id = group_df['location_id'].iloc[0]
                file_name = f"weather_{location_id}_{year}-{month:02d}.parquet"
            else:
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"weather_{year}-{month:02d}_{current_time}.parquet"
            
            file_path = partition_dir / file_name
            
            # Write DataFrame to Parquet file
            write_dataframe_to_parquet(
                group_df,
                file_path,
                compression=self._compression,
                row_group_size=self._row_group_size
            )
        
        self._logger.info(f"Stored weather data with {len(df)} rows in {base_path}")
        return base_path
    
    def store_grid_condition_data(self, df: DataFrameType, validate: bool = True) -> PathType:
        """
        Stores grid condition data in Parquet format with time-based partitioning.

        Args:
            df: DataFrame with grid condition data
            validate: Whether to validate the data schema

        Returns:
            Path to the stored data
        """
        if validate:
            df = validate_with_grid_condition_schema(df)
        
        if 'timestamp' not in df.columns:
            raise DataFormatError("Grid condition data must have a 'timestamp' column")
        
        # Ensure timestamp is datetime type
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group data by year and month for partitioning
        base_path = self._storage_root / 'grid_condition'
        grouped = df.groupby([df['timestamp'].dt.year, df['timestamp'].dt.month])
        
        for (year, month), group_df in grouped:
            # Create partition directory
            partition_dir = base_path / str(year) / f"{month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file name with timestamp
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"grid_condition_{year}-{month:02d}_{current_time}.parquet"
            
            file_path = partition_dir / file_name
            
            # Write DataFrame to Parquet file
            write_dataframe_to_parquet(
                group_df,
                file_path,
                compression=self._compression,
                row_group_size=self._row_group_size
            )
        
        self._logger.info(f"Stored grid condition data with {len(df)} rows in {base_path}")
        return base_path
    
    def store_feature_data(
        self, 
        df: DataFrameType, 
        feature_group: str,
        validate: bool = True
    ) -> PathType:
        """
        Stores engineered feature data in Parquet format with time-based partitioning.

        Args:
            df: DataFrame with feature data
            feature_group: Group name for the features (e.g., 'time', 'statistical')
            validate: Whether to validate the data schema

        Returns:
            Path to the stored data
        """
        if validate:
            df = validate_with_feature_schema(df)
        
        if 'timestamp' not in df.columns:
            raise DataFormatError("Feature data must have a 'timestamp' column")
        
        # Ensure timestamp is datetime type
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group data by year and month for partitioning
        base_path = self._storage_root / 'features' / feature_group
        grouped = df.groupby([df['timestamp'].dt.year, df['timestamp'].dt.month])
        
        for (year, month), group_df in grouped:
            # Create partition directory
            partition_dir = base_path / str(year) / f"{month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file name with timestamp
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"features_{feature_group}_{year}-{month:02d}_{current_time}.parquet"
            
            file_path = partition_dir / file_name
            
            # Write DataFrame to Parquet file
            write_dataframe_to_parquet(
                group_df,
                file_path,
                compression=self._compression,
                row_group_size=self._row_group_size
            )
        
        self._logger.info(f"Stored {feature_group} feature data with {len(df)} rows in {base_path}")
        return base_path
    
    def store_forecast_data(self, df: DataFrameType, validate: bool = True) -> PathType:
        """
        Stores forecast data in Parquet format with time-based partitioning.

        Args:
            df: DataFrame with forecast data
            validate: Whether to validate the data schema

        Returns:
            Path to the stored data
        """
        if validate:
            df = validate_with_forecast_schema(df)
        
        required_cols = ['forecast_timestamp', 'target_timestamp']
        for col in required_cols:
            if col not in df.columns:
                raise DataFormatError(f"Forecast data must have a '{col}' column")
        
        # Ensure timestamps are datetime type
        df['forecast_timestamp'] = pd.to_datetime(df['forecast_timestamp'])
        df['target_timestamp'] = pd.to_datetime(df['target_timestamp'])
        
        # Group data by forecast date for partitioning
        base_path = self._storage_root / 'forecasts'
        grouped = df.groupby(df['forecast_timestamp'].dt.date)
        
        for forecast_date, group_df in grouped:
            # Create partition directory
            year = forecast_date.year
            month = forecast_date.month
            day = forecast_date.day
            partition_dir = base_path / str(year) / f"{month:02d}" / f"{day:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file name with threshold and timestamp
            threshold_str = "multi"
            if 'threshold_value' in group_df.columns and group_df['threshold_value'].nunique() == 1:
                threshold = group_df['threshold_value'].iloc[0]
                threshold_str = f"{threshold:.1f}"
            
            forecast_timestamp = group_df['forecast_timestamp'].iloc[0]
            timestamp_str = forecast_timestamp.strftime("%Y%m%d_%H%M%S")
            
            file_name = f"forecast_{threshold_str}_{timestamp_str}.parquet"
            file_path = partition_dir / file_name
            
            # Write DataFrame to Parquet file
            write_dataframe_to_parquet(
                group_df,
                file_path,
                compression=self._compression,
                row_group_size=self._row_group_size
            )
        
        self._logger.info(f"Stored forecast data with {len(df)} rows in {base_path}")
        return base_path
    
    def retrieve_rtlmp_data(
        self, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        nodes: Optional[List[str]] = None,
        columns: Optional[List[str]] = None
    ) -> DataFrameType:
        """
        Retrieves RTLMP data for a specified date range and nodes.

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            nodes: Optional list of node IDs to filter by
            columns: Optional list of columns to retrieve

        Returns:
            DataFrame with retrieved RTLMP data
        """
        # Calculate the list of year/month partitions to read
        partitions = []
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            partitions.append((year, month))
            
            # Move to next month
            if month == 12:
                current_date = datetime.datetime(year + 1, 1, 1)
            else:
                current_date = datetime.datetime(year, month + 1, 1)
        
        # Create filters for nodes if provided
        filters = None
        if nodes:
            filters = [('node_id', 'in', nodes)]
        
        # Read data from each partition
        dfs = []
        for year, month in partitions:
            partition_path = self._storage_root / 'rtlmp' / str(year) / f"{month:02d}"
            if partition_path.exists():
                try:
                    partition_df = read_dataframes_from_partitions(
                        partition_path,
                        columns=columns,
                        filters=filters
                    )
                    dfs.append(partition_df)
                except Exception as e:
                    self._logger.warning(f"Error reading RTLMP partition {year}-{month:02d}: {str(e)}")
        
        if not dfs:
            self._logger.warning(f"No RTLMP data found for date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        # Concatenate all partitions
        result = pd.concat(dfs, ignore_index=True)
        
        # Filter to exact date range
        if 'timestamp' in result.columns:
            result = result[
                (result['timestamp'] >= start_date) & 
                (result['timestamp'] <= end_date)
            ]
        
        self._logger.info(f"Retrieved {len(result)} rows of RTLMP data")
        return result
    
    def retrieve_weather_data(
        self, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        locations: Optional[List[str]] = None,
        columns: Optional[List[str]] = None
    ) -> DataFrameType:
        """
        Retrieves weather data for a specified date range and locations.

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            locations: Optional list of location IDs to filter by
            columns: Optional list of columns to retrieve

        Returns:
            DataFrame with retrieved weather data
        """
        # Calculate the list of year/month partitions to read
        partitions = []
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            partitions.append((year, month))
            
            # Move to next month
            if month == 12:
                current_date = datetime.datetime(year + 1, 1, 1)
            else:
                current_date = datetime.datetime(year, month + 1, 1)
        
        # Create filters for locations if provided
        filters = None
        if locations:
            filters = [('location_id', 'in', locations)]
        
        # Read data from each partition
        dfs = []
        for year, month in partitions:
            partition_path = self._storage_root / 'weather' / str(year) / f"{month:02d}"
            if partition_path.exists():
                try:
                    partition_df = read_dataframes_from_partitions(
                        partition_path,
                        columns=columns,
                        filters=filters
                    )
                    dfs.append(partition_df)
                except Exception as e:
                    self._logger.warning(f"Error reading weather partition {year}-{month:02d}: {str(e)}")
        
        if not dfs:
            self._logger.warning(f"No weather data found for date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        # Concatenate all partitions
        result = pd.concat(dfs, ignore_index=True)
        
        # Filter to exact date range
        if 'timestamp' in result.columns:
            result = result[
                (result['timestamp'] >= start_date) & 
                (result['timestamp'] <= end_date)
            ]
        
        self._logger.info(f"Retrieved {len(result)} rows of weather data")
        return result
    
    def retrieve_grid_condition_data(
        self, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        columns: Optional[List[str]] = None
    ) -> DataFrameType:
        """
        Retrieves grid condition data for a specified date range.

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            columns: Optional list of columns to retrieve

        Returns:
            DataFrame with retrieved grid condition data
        """
        # Calculate the list of year/month partitions to read
        partitions = []
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            partitions.append((year, month))
            
            # Move to next month
            if month == 12:
                current_date = datetime.datetime(year + 1, 1, 1)
            else:
                current_date = datetime.datetime(year, month + 1, 1)
        
        # Read data from each partition
        dfs = []
        for year, month in partitions:
            partition_path = self._storage_root / 'grid_condition' / str(year) / f"{month:02d}"
            if partition_path.exists():
                try:
                    partition_df = read_dataframes_from_partitions(
                        partition_path,
                        columns=columns
                    )
                    dfs.append(partition_df)
                except Exception as e:
                    self._logger.warning(f"Error reading grid_condition partition {year}-{month:02d}: {str(e)}")
        
        if not dfs:
            self._logger.warning(f"No grid condition data found for date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        # Concatenate all partitions
        result = pd.concat(dfs, ignore_index=True)
        
        # Filter to exact date range
        if 'timestamp' in result.columns:
            result = result[
                (result['timestamp'] >= start_date) & 
                (result['timestamp'] <= end_date)
            ]
        
        self._logger.info(f"Retrieved {len(result)} rows of grid condition data")
        return result
    
    def retrieve_feature_data(
        self, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        feature_group: str,
        feature_names: Optional[List[str]] = None
    ) -> DataFrameType:
        """
        Retrieves engineered feature data for a specified date range and feature group.

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            feature_group: Feature group to retrieve
            feature_names: Optional list of feature names to filter by

        Returns:
            DataFrame with retrieved feature data
        """
        # Calculate the list of year/month partitions to read
        partitions = []
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            partitions.append((year, month))
            
            # Move to next month
            if month == 12:
                current_date = datetime.datetime(year + 1, 1, 1)
            else:
                current_date = datetime.datetime(year, month + 1, 1)
        
        # Create filters for feature_names if provided
        filters = None
        if feature_names and 'feature_name' in feature_names:
            filters = [('feature_name', 'in', feature_names)]
        
        # Read data from each partition
        dfs = []
        for year, month in partitions:
            partition_path = self._storage_root / 'features' / feature_group / str(year) / f"{month:02d}"
            if partition_path.exists():
                try:
                    partition_df = read_dataframes_from_partitions(
                        partition_path,
                        filters=filters
                    )
                    dfs.append(partition_df)
                except Exception as e:
                    self._logger.warning(
                        f"Error reading {feature_group} feature partition {year}-{month:02d}: {str(e)}"
                    )
        
        if not dfs:
            self._logger.warning(
                f"No {feature_group} feature data found for date range {start_date} to {end_date}"
            )
            return pd.DataFrame()
        
        # Concatenate all partitions
        result = pd.concat(dfs, ignore_index=True)
        
        # Filter to exact date range
        if 'timestamp' in result.columns:
            result = result[
                (result['timestamp'] >= start_date) & 
                (result['timestamp'] <= end_date)
            ]
        
        self._logger.info(f"Retrieved {len(result)} rows of {feature_group} feature data")
        return result
    
    def retrieve_forecast_data(
        self, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        threshold_value: Optional[float] = None,
        nodes: Optional[List[str]] = None
    ) -> DataFrameType:
        """
        Retrieves forecast data for a specified date range and threshold.

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            threshold_value: Optional threshold value to filter by
            nodes: Optional list of node IDs to filter by

        Returns:
            DataFrame with retrieved forecast data
        """
        # Get all dates in the range
        date_range = []
        current_date = start_date.date()
        while current_date <= end_date.date():
            date_range.append(current_date)
            current_date += datetime.timedelta(days=1)
        
        # Create filters for threshold and nodes
        filters = []
        if threshold_value is not None:
            filters.append(('threshold_value', '==', threshold_value))
        if nodes:
            filters.append(('node_id', 'in', nodes))
        
        # Read data from each date partition
        dfs = []
        for date in date_range:
            year = date.year
            month = date.month
            day = date.day
            partition_path = self._storage_root / 'forecasts' / str(year) / f"{month:02d}" / f"{day:02d}"
            
            if partition_path.exists():
                try:
                    partition_df = read_dataframes_from_partitions(
                        partition_path,
                        filters=filters if filters else None
                    )
                    dfs.append(partition_df)
                except Exception as e:
                    self._logger.warning(f"Error reading forecast partition {date}: {str(e)}")
        
        if not dfs:
            threshold_str = f" for threshold {threshold_value}" if threshold_value else ""
            self._logger.warning(f"No forecast data found{threshold_str} in date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        # Concatenate all partitions
        result = pd.concat(dfs, ignore_index=True)
        
        # Filter to exact date range for forecast_timestamp
        result = result[
            (result['forecast_timestamp'] >= start_date) & 
            (result['forecast_timestamp'] <= end_date)
        ]
        
        self._logger.info(f"Retrieved {len(result)} rows of forecast data")
        return result
    
    def get_latest_forecast(
        self,
        threshold_value: float,
        node_id: str
    ) -> DataFrameType:
        """
        Retrieves the most recent forecast data for a specified threshold and node.

        Args:
            threshold_value: Price threshold for spike definition
            node_id: Node ID to retrieve forecast for

        Returns:
            DataFrame with the latest forecast data
        """
        # Find the most recent forecast directory
        forecast_dir = self._storage_root / 'forecasts'
        if not forecast_dir.exists():
            self._logger.warning("No forecast directory found")
            return pd.DataFrame()
        
        # Get all year directories
        year_dirs = sorted([d for d in forecast_dir.iterdir() if d.is_dir()], reverse=True)
        if not year_dirs:
            self._logger.warning("No year directories found in forecast directory")
            return pd.DataFrame()
        
        # Find the most recent forecast by traversing directories
        latest_forecast_df = None
        
        for year_dir in year_dirs:
            # Get month directories in reverse order (most recent first)
            month_dirs = sorted([d for d in year_dir.iterdir() if d.is_dir()], reverse=True)
            if not month_dirs:
                continue
            
            for month_dir in month_dirs:
                # Get day directories in reverse order
                day_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()], reverse=True)
                if not day_dirs:
                    continue
                
                for day_dir in day_dirs:
                    # Get all parquet files and find those matching our criteria
                    parquet_files = find_parquet_files(day_dir)
                    if not parquet_files:
                        continue
                    
                    # Try to find files that match the threshold in the filename
                    threshold_files = [f for f in parquet_files if f"_{threshold_value:.1f}_" in f.name]
                    if not threshold_files:
                        # If no exact match, try to read all files and filter
                        threshold_files = parquet_files
                    
                    # Read the files, starting with the most recently modified
                    threshold_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                    
                    for file_path in threshold_files:
                        try:
                            df = read_dataframe_from_parquet(file_path)
                            
                            # Filter by threshold and node_id
                            filtered_df = df[
                                (df['threshold_value'] == threshold_value) &
                                (df['node_id'] == node_id)
                            ]
                            
                            if not filtered_df.empty:
                                # Found matching forecast
                                if 'forecast_timestamp' in filtered_df.columns:
                                    # Sort by forecast timestamp and get the latest
                                    filtered_df = filtered_df.sort_values(
                                        'forecast_timestamp', ascending=False
                                    )
                                    
                                latest_forecast_df = filtered_df
                                forecast_time = filtered_df['forecast_timestamp'].iloc[0]
                                self._logger.info(
                                    f"Found latest forecast from {forecast_time} for threshold {threshold_value} "
                                    f"and node {node_id}"
                                )
                                return latest_forecast_df
                        except Exception as e:
                            self._logger.warning(f"Error reading forecast file {file_path}: {str(e)}")
        
        if latest_forecast_df is None:
            self._logger.warning(f"No forecast found for threshold {threshold_value} and node {node_id}")
            return pd.DataFrame()
        
        return latest_forecast_df