"""
Abstract base class for data fetchers in the ERCOT RTLMP spike prediction system.

This module defines a standardized interface for retrieving data from various sources
including ERCOT market data and weather forecasts. It implements common functionality
like caching, request handling, and error management.
"""

import os
import time
import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Type, TypeVar, Generic, cast

import pandas as pd  # version 2.0+
import requests  # version 2.28+

from ...utils.type_definitions import DataFrameType, DataFetcherProtocol
from ...utils.error_handling import (
    retry_with_backoff, handle_errors, 
    ConnectionError, RateLimitError, DataFormatError, MissingDataError
)
from ...utils.logging import get_logger, log_execution_time
from ...utils.validation import validate_dataframe_schema

# Set up logger
logger = get_logger(__name__)

# Default settings
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../../data/cache')


def generate_cache_key(params: Dict[str, Any]) -> str:
    """
    Generates a unique cache key based on request parameters.
    
    Args:
        params: Dictionary of parameters used for the request
        
    Returns:
        A unique hash string to use as cache key
    """
    # Convert the params dictionary to a sorted, normalized JSON string
    param_str = json.dumps(params, sort_keys=True)
    
    # Generate a SHA-256 hash of the JSON string
    hash_obj = hashlib.sha256(param_str.encode())
    
    return hash_obj.hexdigest()


class BaseDataFetcher(ABC):
    """
    Abstract base class that implements common functionality for all data fetchers.
    
    This class provides standard methods for fetching, validating, and caching data
    from various sources. It is designed to be extended by specific data fetcher
    implementations for ERCOT, weather data, etc.
    """
    
    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        cache_dir: str = DEFAULT_CACHE_DIR
    ):
        """
        Initialize the base data fetcher with common configuration.
        
        Args:
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            cache_ttl: Time-to-live for cached data in seconds
            cache_dir: Directory to store cached data files
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl
        self.cache_dir = cache_dir
        self._cache = {}  # In-memory cache
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} with timeout={timeout}, "
                   f"max_retries={max_retries}, cache_ttl={cache_ttl}")
    
    @abstractmethod
    def fetch_data(self, params: Dict[str, Any]) -> DataFrameType:
        """
        Generic method to fetch data based on parameters.
        
        This method should be implemented by subclasses to handle the specific
        data source and return a standardized DataFrame.
        
        Args:
            params: Dictionary of parameters for the data fetch
            
        Returns:
            DataFrame with the fetched data
        """
        pass
    
    @abstractmethod
    def fetch_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        identifiers: List[str]
    ) -> DataFrameType:
        """
        Fetch historical data for a specific date range.
        
        This method should be implemented by subclasses to retrieve historical data
        for the specified date range and identifiers.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            identifiers: List of identifiers (e.g., node IDs, location IDs)
            
        Returns:
            DataFrame with historical data
        """
        pass
    
    @abstractmethod
    def fetch_forecast_data(
        self,
        forecast_date: datetime,
        horizon: int,
        identifiers: List[str]
    ) -> DataFrameType:
        """
        Fetch forecast data for a specific date and horizon.
        
        This method should be implemented by subclasses to retrieve forecast data
        for the specified forecast date, horizon, and identifiers.
        
        Args:
            forecast_date: Date for which to retrieve forecasts
            horizon: Forecast horizon in hours
            identifiers: List of identifiers (e.g., node IDs, location IDs)
            
        Returns:
            DataFrame with forecast data
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: DataFrameType) -> bool:
        """
        Validate the structure and content of fetched data.
        
        This method should be implemented by subclasses to validate that the
        fetched data has the expected structure and content.
        
        Args:
            data: DataFrame with data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    @retry_with_backoff(exceptions=(ConnectionError, RateLimitError), max_retries=None)
    def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None
    ) -> requests.Response:
        """
        Make an HTTP request with retry and error handling.
        
        This method handles common HTTP request operations with retry logic for
        transient failures and appropriate error handling.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters for the request
            headers: HTTP headers for the request
            data: Request body data
            
        Returns:
            Response object from the HTTP request
            
        Raises:
            ConnectionError: If the request fails due to connection issues
            RateLimitError: If the request is rate limited
            DataFormatError: If the response has an unexpected format
        """
        logger.debug(f"Making {method} request to {url} with params={params}")
        
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                data=data,
                timeout=self.timeout
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                logger.warning(f"Rate limited: {url}, retry after {retry_after} seconds")
                raise RateLimitError(
                    f"Rate limit exceeded for {url}", 
                    context={"url": url, "retry_after": retry_after}
                )
            
            logger.debug(f"Request to {url} succeeded with status {response.status_code}")
            return response
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {str(e)}")
            raise ConnectionError(f"Failed to connect to {url}: {str(e)}")
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout for {url}: {str(e)}")
            raise ConnectionError(f"Request to {url} timed out: {str(e)}")
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else None
            
            if status_code == 429:
                retry_after = e.response.headers.get("Retry-After", "60")
                logger.warning(f"Rate limited: {url}, retry after {retry_after} seconds")
                raise RateLimitError(
                    f"Rate limit exceeded for {url}", 
                    context={"url": url, "retry_after": retry_after}
                )
            elif status_code and 500 <= status_code < 600:
                logger.error(f"Server error for {url}: {str(e)}")
                raise ConnectionError(f"Server error for {url}: {str(e)}")
            else:
                logger.error(f"HTTP error for {url}: {str(e)}")
                raise ConnectionError(f"HTTP error for {url}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise ConnectionError(f"Error making request to {url}: {str(e)}")
    
    @handle_errors(exceptions=Exception, default_return=None)
    def _get_from_cache(self, params: Dict[str, Any]) -> Optional[DataFrameType]:
        """
        Retrieve data from cache if available and not expired.
        
        Args:
            params: Dictionary of parameters used to generate the cache key
            
        Returns:
            Cached data if available and not expired, None otherwise
        """
        cache_key = generate_cache_key(params)
        
        # Check in-memory cache first
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            expiration_time = cache_entry.get("timestamp", 0) + self.cache_ttl
            
            if time.time() < expiration_time:
                logger.debug(f"Cache hit (memory) for key: {cache_key}")
                return cache_entry.get("data")
        
        # Check file cache if not in memory or expired
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        
        if os.path.exists(cache_file):
            # Check file modification time for expiration
            modification_time = os.path.getmtime(cache_file)
            
            if time.time() < modification_time + self.cache_ttl:
                logger.debug(f"Cache hit (file) for key: {cache_key}")
                try:
                    data = pd.read_parquet(cache_file)
                    
                    # Update memory cache
                    self._cache[cache_key] = {
                        "data": data,
                        "timestamp": modification_time
                    }
                    
                    return data
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_file}: {str(e)}")
        
        logger.debug(f"Cache miss for key: {cache_key}")
        return None
    
    @handle_errors(exceptions=Exception, default_return=False)
    def _store_in_cache(self, params: Dict[str, Any], data: DataFrameType) -> bool:
        """
        Store data in cache with expiration time.
        
        Args:
            params: Dictionary of parameters used to generate the cache key
            data: DataFrame to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        cache_key = generate_cache_key(params)
        timestamp = time.time()
        
        # Store in memory cache
        self._cache[cache_key] = {
            "data": data,
            "timestamp": timestamp
        }
        
        # Store in file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        
        try:
            data.to_parquet(cache_file, index=True)
            logger.debug(f"Stored data in cache: {cache_key}")
            return True
        except Exception as e:
            logger.warning(f"Error storing data in cache file {cache_file}: {str(e)}")
            return False
    
    def _clear_cache(self, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clear all cached data or specific entries.
        
        Args:
            params: Optional parameters to clear specific cache entries.
                   If None, all cache entries are cleared.
                   
        Returns:
            True if successfully cleared, False otherwise
        """
        try:
            if params is not None:
                # Clear specific cache entry
                cache_key = generate_cache_key(params)
                
                # Clear from memory
                if cache_key in self._cache:
                    del self._cache[cache_key]
                
                # Clear from file system
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                
                logger.debug(f"Cleared cache entry: {cache_key}")
            else:
                # Clear all cache entries
                self._cache = {}
                
                # Clear all cache files
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith(".parquet"):
                        os.remove(os.path.join(self.cache_dir, filename))
                
                logger.debug("Cleared all cache entries")
            
            return True
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")
            return False
    
    def _validate_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        max_days: Optional[int] = None
    ) -> bool:
        """
        Validate that a date range is valid for data fetching.
        
        Args:
            start_date: Start date of the range
            end_date: End date of the range
            max_days: Optional maximum number of days in the range
            
        Returns:
            True if date range is valid, False otherwise
        """
        # Check that start_date is before end_date
        if start_date >= end_date:
            logger.warning(f"Invalid date range: start_date {start_date} is not before end_date {end_date}")
            return False
        
        # Check that the range doesn't exceed max_days if provided
        if max_days is not None:
            days_diff = (end_date - start_date).days
            if days_diff > max_days:
                logger.warning(f"Date range exceeds maximum of {max_days} days: {days_diff} days requested")
                return False
        
        # Check that end_date is not in the future
        now = datetime.now()
        if end_date > now:
            logger.warning(f"Invalid date range: end_date {end_date} is in the future")
            return False
        
        logger.debug(f"Valid date range: {start_date} to {end_date}")
        return True