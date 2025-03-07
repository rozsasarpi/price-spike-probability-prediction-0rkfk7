"""
Provides a unified API for retrieving data from various sources for the ERCOT RTLMP spike prediction system.

This module acts as a facade over the different data fetchers, simplifying data access for other components
and providing consistent error handling and data validation.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd  # version 2.0+

from ..data.fetchers.ercot_api import ERCOTDataFetcher, DEFAULT_NODES
from ..data.fetchers.weather_api import WeatherAPIFetcher
from ..data.storage.parquet_store import ParquetStore
from ..utils.type_definitions import DataFrameType, NodeID
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import handle_errors, ConnectionError, DataFormatError, MissingDataError

# Set up logger
logger = get_logger(__name__)

# Global constants
DEFAULT_STORAGE_PATH = os.environ.get('STORAGE_PATH', './data')
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30


@log_execution_time(logger)
@handle_errors(exceptions=(ConnectionError, DataFormatError, MissingDataError), 
              error_message='Failed to retrieve historical RTLMP data')
def get_historical_rtlmp_data(
    start_date: datetime, 
    end_date: datetime, 
    nodes: Optional[List[str]] = None,
    use_cache: bool = True,
    store_result: bool = True
) -> DataFrameType:
    """
    Retrieves historical RTLMP data for a specified date range and nodes.
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        nodes: List of node IDs to fetch data for (defaults to DEFAULT_NODES)
        use_cache: Whether to check cache before fetching from API
        store_result: Whether to store the fetched data in ParquetStore
    
    Returns:
        DataFrame containing historical RTLMP data
    """
    # Initialize ERCOT data fetcher
    ercot_fetcher = ERCOTDataFetcher(
        timeout=DEFAULT_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES,
        cache_ttl=DEFAULT_CACHE_TTL
    )
    
    # Use DEFAULT_NODES if nodes is None
    if nodes is None:
        nodes = DEFAULT_NODES
    
    # Initialize storage
    storage = ParquetStore(storage_root=DEFAULT_STORAGE_PATH)
    
    # Try to retrieve from cache if use_cache is True
    if use_cache:
        try:
            cached_data = storage.retrieve_rtlmp_data(
                start_date=start_date,
                end_date=end_date,
                nodes=nodes
            )
            if not cached_data.empty:
                logger.info(f"Retrieved {len(cached_data)} rows of RTLMP data from cache")
                return cached_data
        except Exception as e:
            logger.warning(f"Failed to retrieve RTLMP data from cache: {str(e)}")
    
    # Fetch from ERCOT API
    logger.info(f"Fetching RTLMP data from API for {start_date} to {end_date}")
    rtlmp_data = ercot_fetcher.fetch_historical_data(
        start_date=start_date,
        end_date=end_date,
        identifiers=nodes
    )
    
    # Store the result if requested
    if store_result and not rtlmp_data.empty:
        try:
            storage.store_rtlmp_data(rtlmp_data)
            logger.info(f"Stored {len(rtlmp_data)} rows of RTLMP data")
        except Exception as e:
            logger.warning(f"Failed to store RTLMP data: {str(e)}")
    
    return rtlmp_data


@log_execution_time(logger)
@handle_errors(exceptions=(ConnectionError, DataFormatError, MissingDataError), 
              error_message='Failed to retrieve historical weather data')
def get_historical_weather_data(
    start_date: datetime, 
    end_date: datetime, 
    locations: Optional[List[str]] = None,
    use_cache: bool = True,
    store_result: bool = True
) -> DataFrameType:
    """
    Retrieves historical weather data for a specified date range and locations.
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        locations: List of location IDs (defaults to WeatherAPIFetcher default locations)
        use_cache: Whether to check cache before fetching from API
        store_result: Whether to store the fetched data in ParquetStore
    
    Returns:
        DataFrame containing historical weather data
    """
    # Initialize Weather API fetcher
    weather_fetcher = WeatherAPIFetcher(
        timeout=DEFAULT_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES,
        cache_ttl=DEFAULT_CACHE_TTL
    )
    
    # Initialize storage
    storage = ParquetStore(storage_root=DEFAULT_STORAGE_PATH)
    
    # Try to retrieve from cache if use_cache is True
    if use_cache:
        try:
            cached_data = storage.retrieve_weather_data(
                start_date=start_date,
                end_date=end_date,
                locations=locations
            )
            if not cached_data.empty:
                logger.info(f"Retrieved {len(cached_data)} rows of weather data from cache")
                return cached_data
        except Exception as e:
            logger.warning(f"Failed to retrieve weather data from cache: {str(e)}")
    
    # Fetch from Weather API
    logger.info(f"Fetching weather data from API for {start_date} to {end_date}")
    weather_data = weather_fetcher.fetch_historical_data(
        start_date=start_date,
        end_date=end_date,
        identifiers=locations if locations else []
    )
    
    # Store the result if requested
    if store_result and not weather_data.empty:
        try:
            storage.store_weather_data(weather_data)
            logger.info(f"Stored {len(weather_data)} rows of weather data")
        except Exception as e:
            logger.warning(f"Failed to store weather data: {str(e)}")
    
    return weather_data


@log_execution_time(logger)
@handle_errors(exceptions=(ConnectionError, DataFormatError, MissingDataError), 
              error_message='Failed to retrieve weather forecast data')
def get_weather_forecast(
    forecast_date: datetime,
    horizon: int,
    locations: Optional[List[str]] = None
) -> DataFrameType:
    """
    Retrieves weather forecast data for a specified date and horizon.
    
    Args:
        forecast_date: Date for which to retrieve forecasts
        horizon: Forecast horizon in hours
        locations: List of location IDs (defaults to WeatherAPIFetcher default locations)
    
    Returns:
        DataFrame containing weather forecast data
    """
    # Initialize Weather API fetcher
    weather_fetcher = WeatherAPIFetcher(
        timeout=DEFAULT_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES,
        cache_ttl=DEFAULT_CACHE_TTL
    )
    
    # Fetch forecast data
    logger.info(f"Fetching weather forecast data for {forecast_date} with {horizon} hour horizon")
    forecast_data = weather_fetcher.fetch_forecast_data(
        forecast_date=forecast_date,
        horizon=horizon,
        identifiers=locations if locations else []
    )
    
    # Validate the data
    if forecast_data.empty:
        logger.warning(f"No weather forecast data retrieved for {forecast_date}")
    else:
        logger.info(f"Retrieved {len(forecast_data)} rows of weather forecast data")
    
    return forecast_data


@log_execution_time(logger)
@handle_errors(exceptions=(ConnectionError, DataFormatError, MissingDataError), 
              error_message='Failed to retrieve historical grid condition data')
def get_historical_grid_conditions(
    start_date: datetime, 
    end_date: datetime, 
    use_cache: bool = True,
    store_result: bool = True
) -> DataFrameType:
    """
    Retrieves historical grid condition data for a specified date range.
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        use_cache: Whether to check cache before fetching from API
        store_result: Whether to store the fetched data in ParquetStore
    
    Returns:
        DataFrame containing historical grid condition data
    """
    # Initialize ERCOT data fetcher
    ercot_fetcher = ERCOTDataFetcher(
        timeout=DEFAULT_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES,
        cache_ttl=DEFAULT_CACHE_TTL
    )
    
    # Initialize storage
    storage = ParquetStore(storage_root=DEFAULT_STORAGE_PATH)
    
    # Try to retrieve from cache if use_cache is True
    if use_cache:
        try:
            cached_data = storage.retrieve_grid_condition_data(
                start_date=start_date,
                end_date=end_date
            )
            if not cached_data.empty:
                logger.info(f"Retrieved {len(cached_data)} rows of grid condition data from cache")
                return cached_data
        except Exception as e:
            logger.warning(f"Failed to retrieve grid condition data from cache: {str(e)}")
    
    # Fetch from ERCOT API
    logger.info(f"Fetching grid condition data from API for {start_date} to {end_date}")
    grid_data = ercot_fetcher.fetch_grid_conditions(
        start_date=start_date,
        end_date=end_date
    )
    
    # Store the result if requested
    if store_result and not grid_data.empty:
        try:
            storage.store_grid_condition_data(grid_data)
            logger.info(f"Stored {len(grid_data)} rows of grid condition data")
        except Exception as e:
            logger.warning(f"Failed to store grid condition data: {str(e)}")
    
    return grid_data


@log_execution_time(logger)
@handle_errors(exceptions=(ConnectionError, DataFormatError, MissingDataError), 
              error_message='Failed to retrieve combined historical data')
def get_combined_historical_data(
    start_date: datetime, 
    end_date: datetime, 
    nodes: Optional[List[str]] = None,
    use_cache: bool = True
) -> DataFrameType:
    """
    Retrieves and combines historical RTLMP, weather, and grid condition data.
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        nodes: List of node IDs (defaults to DEFAULT_NODES)
        use_cache: Whether to check cache before fetching from API
    
    Returns:
        DataFrame containing combined historical data
    """
    logger.info(f"Retrieving combined historical data from {start_date} to {end_date}")
    
    # Get individual data components
    rtlmp_data = get_historical_rtlmp_data(
        start_date=start_date,
        end_date=end_date,
        nodes=nodes,
        use_cache=use_cache
    )
    
    weather_data = get_historical_weather_data(
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )
    
    grid_data = get_historical_grid_conditions(
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )
    
    # Check if we have data to combine
    if rtlmp_data.empty or weather_data.empty or grid_data.empty:
        logger.warning("One or more data sources returned empty results")
        # Return whatever data we have, with preference to RTLMP data
        return rtlmp_data if not rtlmp_data.empty else (
            weather_data if not weather_data.empty else grid_data
        )
    
    # Combine the data
    # First, ensure all DataFrames have 'timestamp' column
    if 'timestamp' not in rtlmp_data.columns or 'timestamp' not in weather_data.columns or 'timestamp' not in grid_data.columns:
        raise DataFormatError("One or more data sources missing 'timestamp' column")
    
    # Merge RTLMP and weather data
    # For each node, match to the nearest weather location
    # This is a simplified approach - in a real implementation, we would have a mapping
    merged_data = pd.merge_asof(
        rtlmp_data.sort_values('timestamp'),
        weather_data.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )
    
    # Merge with grid condition data
    merged_data = pd.merge_asof(
        merged_data.sort_values('timestamp'),
        grid_data.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )
    
    logger.info(f"Combined data has {len(merged_data)} rows and {len(merged_data.columns)} columns")
    return merged_data


class DataAPI:
    """
    Class that provides a unified interface for retrieving data from various sources.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize the DataAPI with data fetchers and storage.
        
        Args:
            storage_path: Path to store data
            cache_ttl: Time-to-live for cached data in seconds
            max_retries: Maximum number of retry attempts for API calls
            timeout: Timeout for API calls in seconds
        """
        self._storage_path = storage_path or DEFAULT_STORAGE_PATH
        self._cache_ttl = cache_ttl or DEFAULT_CACHE_TTL
        self._max_retries = max_retries or DEFAULT_MAX_RETRIES
        self._timeout = timeout or DEFAULT_TIMEOUT
        
        # Initialize storage
        self._storage = ParquetStore(storage_root=self._storage_path)
        
        # Initialize data fetchers
        self._ercot_fetcher = ERCOTDataFetcher(
            timeout=self._timeout,
            max_retries=self._max_retries,
            cache_ttl=self._cache_ttl
        )
        
        self._weather_fetcher = WeatherAPIFetcher(
            timeout=self._timeout,
            max_retries=self._max_retries,
            cache_ttl=self._cache_ttl
        )
        
        logger.info(f"Initialized DataAPI with storage_path={self._storage_path}")
    
    @log_execution_time(logger)
    def get_historical_rtlmp(
        self,
        start_date: datetime, 
        end_date: datetime, 
        nodes: Optional[List[str]] = None,
        use_cache: bool = True,
        store_result: bool = True
    ) -> DataFrameType:
        """
        Retrieves historical RTLMP data for a specified date range and nodes.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            nodes: List of node IDs to fetch data for (defaults to DEFAULT_NODES)
            use_cache: Whether to check cache before fetching from API
            store_result: Whether to store the fetched data
        
        Returns:
            DataFrame containing historical RTLMP data
        """
        # Use DEFAULT_NODES if nodes is None
        if nodes is None:
            nodes = DEFAULT_NODES
        
        # Try to retrieve from cache if use_cache is True
        if use_cache:
            try:
                cached_data = self._storage.retrieve_rtlmp_data(
                    start_date=start_date,
                    end_date=end_date,
                    nodes=nodes
                )
                if not cached_data.empty:
                    logger.info(f"Retrieved {len(cached_data)} rows of RTLMP data from cache")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to retrieve RTLMP data from cache: {str(e)}")
        
        # Fetch from ERCOT API
        logger.info(f"Fetching RTLMP data from API for {start_date} to {end_date}")
        rtlmp_data = self._ercot_fetcher.fetch_historical_data(
            start_date=start_date,
            end_date=end_date,
            identifiers=nodes
        )
        
        # Store the result if requested
        if store_result and not rtlmp_data.empty:
            try:
                self._storage.store_rtlmp_data(rtlmp_data)
                logger.info(f"Stored {len(rtlmp_data)} rows of RTLMP data")
            except Exception as e:
                logger.warning(f"Failed to store RTLMP data: {str(e)}")
        
        return rtlmp_data
    
    @log_execution_time(logger)
    def get_historical_weather(
        self,
        start_date: datetime, 
        end_date: datetime, 
        locations: Optional[List[str]] = None,
        use_cache: bool = True,
        store_result: bool = True
    ) -> DataFrameType:
        """
        Retrieves historical weather data for a specified date range and locations.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            locations: List of location IDs
            use_cache: Whether to check cache before fetching from API
            store_result: Whether to store the fetched data
        
        Returns:
            DataFrame containing historical weather data
        """
        # Try to retrieve from cache if use_cache is True
        if use_cache:
            try:
                cached_data = self._storage.retrieve_weather_data(
                    start_date=start_date,
                    end_date=end_date,
                    locations=locations
                )
                if not cached_data.empty:
                    logger.info(f"Retrieved {len(cached_data)} rows of weather data from cache")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to retrieve weather data from cache: {str(e)}")
        
        # Fetch from Weather API
        logger.info(f"Fetching weather data from API for {start_date} to {end_date}")
        weather_data = self._weather_fetcher.fetch_historical_data(
            start_date=start_date,
            end_date=end_date,
            identifiers=locations if locations else []
        )
        
        # Store the result if requested
        if store_result and not weather_data.empty:
            try:
                self._storage.store_weather_data(weather_data)
                logger.info(f"Stored {len(weather_data)} rows of weather data")
            except Exception as e:
                logger.warning(f"Failed to store weather data: {str(e)}")
        
        return weather_data
    
    @log_execution_time(logger)
    def get_weather_forecast(
        self,
        forecast_date: datetime,
        horizon: int,
        locations: Optional[List[str]] = None
    ) -> DataFrameType:
        """
        Retrieves weather forecast data for a specified date and horizon.
        
        Args:
            forecast_date: Date for which to retrieve forecasts
            horizon: Forecast horizon in hours
            locations: List of location IDs
        
        Returns:
            DataFrame containing weather forecast data
        """
        # Fetch forecast data
        logger.info(f"Fetching weather forecast data for {forecast_date} with {horizon} hour horizon")
        forecast_data = self._weather_fetcher.fetch_forecast_data(
            forecast_date=forecast_date,
            horizon=horizon,
            identifiers=locations if locations else []
        )
        
        # Validate the data
        if forecast_data.empty:
            logger.warning(f"No weather forecast data retrieved for {forecast_date}")
        else:
            logger.info(f"Retrieved {len(forecast_data)} rows of weather forecast data")
        
        return forecast_data
    
    @log_execution_time(logger)
    def get_historical_grid_conditions(
        self,
        start_date: datetime, 
        end_date: datetime, 
        use_cache: bool = True,
        store_result: bool = True
    ) -> DataFrameType:
        """
        Retrieves historical grid condition data for a specified date range.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            use_cache: Whether to check cache before fetching from API
            store_result: Whether to store the fetched data
        
        Returns:
            DataFrame containing historical grid condition data
        """
        # Try to retrieve from cache if use_cache is True
        if use_cache:
            try:
                cached_data = self._storage.retrieve_grid_condition_data(
                    start_date=start_date,
                    end_date=end_date
                )
                if not cached_data.empty:
                    logger.info(f"Retrieved {len(cached_data)} rows of grid condition data from cache")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to retrieve grid condition data from cache: {str(e)}")
        
        # Fetch from ERCOT API
        logger.info(f"Fetching grid condition data from API for {start_date} to {end_date}")
        grid_data = self._ercot_fetcher.fetch_grid_conditions(
            start_date=start_date,
            end_date=end_date
        )
        
        # Store the result if requested
        if store_result and not grid_data.empty:
            try:
                self._storage.store_grid_condition_data(grid_data)
                logger.info(f"Stored {len(grid_data)} rows of grid condition data")
            except Exception as e:
                logger.warning(f"Failed to store grid condition data: {str(e)}")
        
        return grid_data
    
    @log_execution_time(logger)
    def get_combined_historical_data(
        self,
        start_date: datetime, 
        end_date: datetime, 
        nodes: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> DataFrameType:
        """
        Retrieves and combines historical RTLMP, weather, and grid condition data.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            nodes: List of node IDs (defaults to DEFAULT_NODES)
            use_cache: Whether to check cache before fetching from API
        
        Returns:
            DataFrame containing combined historical data
        """
        logger.info(f"Retrieving combined historical data from {start_date} to {end_date}")
        
        # Get individual data components
        rtlmp_data = self.get_historical_rtlmp(
            start_date=start_date,
            end_date=end_date,
            nodes=nodes,
            use_cache=use_cache
        )
        
        weather_data = self.get_historical_weather(
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        grid_data = self.get_historical_grid_conditions(
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        # Check if we have data to combine
        if rtlmp_data.empty or weather_data.empty or grid_data.empty:
            logger.warning("One or more data sources returned empty results")
            # Return whatever data we have, with preference to RTLMP data
            return rtlmp_data if not rtlmp_data.empty else (
                weather_data if not weather_data.empty else grid_data
            )
        
        # Combine the data
        # First, ensure all DataFrames have 'timestamp' column
        if 'timestamp' not in rtlmp_data.columns or 'timestamp' not in weather_data.columns or 'timestamp' not in grid_data.columns:
            raise DataFormatError("One or more data sources missing 'timestamp' column")
        
        # Merge RTLMP and weather data
        # For each node, match to the nearest weather location
        # This is a simplified approach - in a real implementation, we would have a mapping
        merged_data = pd.merge_asof(
            rtlmp_data.sort_values('timestamp'),
            weather_data.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Merge with grid condition data
        merged_data = pd.merge_asof(
            merged_data.sort_values('timestamp'),
            grid_data.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        logger.info(f"Combined data has {len(merged_data)} rows and {len(merged_data.columns)} columns")
        return merged_data
    
    def refresh_fetchers(self) -> None:
        """
        Refreshes the data fetchers with new authentication credentials.
        """
        # Re-initialize data fetchers
        self._ercot_fetcher = ERCOTDataFetcher(
            timeout=self._timeout,
            max_retries=self._max_retries,
            cache_ttl=self._cache_ttl
        )
        
        self._weather_fetcher = WeatherAPIFetcher(
            timeout=self._timeout,
            max_retries=self._max_retries,
            cache_ttl=self._cache_ttl
        )
        
        logger.info("Refreshed data fetchers with new credentials")