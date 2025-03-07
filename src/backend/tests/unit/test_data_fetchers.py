"""
Unit tests for the data fetcher components of the ERCOT RTLMP spike prediction system.

This module tests the functionality of BaseDataFetcher, ERCOTDataFetcher, 
WeatherAPIFetcher, and MockDataFetcher classes, ensuring they correctly retrieve,
validate, and format data from various sources.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import json

# Import the components to test
from ../../data/fetchers/base import BaseDataFetcher, generate_cache_key
from ../../data/fetchers/ercot_api import ERCOTDataFetcher, parse_rtlmp_response, parse_grid_conditions_response
from ../../data/fetchers/weather_api import WeatherAPIFetcher, parse_weather_response
from ../../data/fetchers/mock import MockDataFetcher

# Import test fixtures
from ../fixtures/sample_data import (
    get_sample_rtlmp_data,
    get_sample_weather_data,
    get_sample_grid_condition_data,
    SAMPLE_NODES,
    SAMPLE_START_DATE,
    SAMPLE_END_DATE,
    DEFAULT_LOCATIONS
)
from ../fixtures/mock_responses import (
    get_mock_rtlmp_response,
    get_mock_grid_conditions_response,
    get_mock_weather_response
)

# Import error handling classes
from ../../utils/error_handling import ConnectionError, RateLimitError, DataFormatError, MissingDataError


class TestBaseDataFetcher:
    """Tests for the BaseDataFetcher abstract base class."""
    
    def setup_method(self, method):
        """Set up test environment before each test."""
        # Create a temporary directory for cache testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a concrete implementation of BaseDataFetcher for testing
        class TestFetcher(BaseDataFetcher):
            def fetch_data(self, params):
                return pd.DataFrame({'test': [1, 2, 3]})
            
            def fetch_historical_data(self, start_date, end_date, identifiers):
                return pd.DataFrame({'test': [1, 2, 3]})
            
            def fetch_forecast_data(self, forecast_date, horizon, identifiers):
                return pd.DataFrame({'test': [1, 2, 3]})
            
            def validate_data(self, data):
                return True
        
        self.test_fetcher = TestFetcher(cache_dir=self.temp_dir)
    
    def teardown_method(self, method):
        """Clean up test environment after each test."""
        # Remove the temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    def test_generate_cache_key(self):
        """Test that cache keys are generated consistently."""
        # Define test parameters
        params1 = {'start_date': '2022-01-01', 'end_date': '2022-01-07', 'nodes': ['HB_NORTH']}
        params2 = {'start_date': '2022-01-01', 'end_date': '2022-01-07', 'nodes': ['HB_NORTH']}
        params3 = {'start_date': '2022-01-01', 'end_date': '2022-01-10', 'nodes': ['HB_SOUTH']}
        
        # Generate cache keys
        key1 = generate_cache_key(params1)
        key2 = generate_cache_key(params2)
        key3 = generate_cache_key(params3)
        
        # Assert that identical params produce identical keys
        assert key1 == key2
        # Assert that different params produce different keys
        assert key1 != key3
    
    def test_cache_operations(self):
        """Test cache storage, retrieval, and clearing operations."""
        # Create test parameters and sample data
        params = {'start_date': '2022-01-01', 'end_date': '2022-01-07', 'nodes': ['HB_NORTH']}
        data = pd.DataFrame({'test': [1, 2, 3]})
        
        # Store data in cache
        self.test_fetcher._store_in_cache(params, data)
        
        # Retrieve data from cache
        cached_data = self.test_fetcher._get_from_cache(params)
        
        # Assert that retrieved data matches original data
        pd.testing.assert_frame_equal(data, cached_data)
        
        # Clear specific cache entry
        self.test_fetcher._clear_cache(params)
        
        # Attempt to retrieve cleared data
        assert self.test_fetcher._get_from_cache(params) is None
        
        # Store multiple data entries in cache
        params1 = {'start_date': '2022-01-01', 'end_date': '2022-01-07', 'nodes': ['HB_NORTH']}
        params2 = {'start_date': '2022-01-01', 'end_date': '2022-01-07', 'nodes': ['HB_SOUTH']}
        self.test_fetcher._store_in_cache(params1, data)
        self.test_fetcher._store_in_cache(params2, data)
        
        # Clear all cache
        self.test_fetcher._clear_cache()
        
        # Assert that all cache entries are cleared
        assert self.test_fetcher._get_from_cache(params1) is None
        assert self.test_fetcher._get_from_cache(params2) is None
    
    def test_cache_expiration(self):
        """Test that cached data expires after TTL."""
        # Create test parameters and sample data
        params = {'start_date': '2022-01-01', 'end_date': '2022-01-07', 'nodes': ['HB_NORTH']}
        data = pd.DataFrame({'test': [1, 2, 3]})
        
        # Initialize fetcher with very short cache_ttl
        test_fetcher = BaseDataFetcher.__subclasses__()[0](cache_ttl=1, cache_dir=self.temp_dir)
        
        # Store data in cache
        test_fetcher._store_in_cache(params, data)
        
        # Retrieve data immediately and verify it's available
        cached_data = test_fetcher._get_from_cache(params)
        assert cached_data is not None
        
        # Mock time to advance beyond TTL
        import time
        time.sleep(2)
        
        # Attempt to retrieve data after expiration
        expired_data = test_fetcher._get_from_cache(params)
        
        # Assert that data is no longer returned from cache
        assert expired_data is None


class TestERCOTDataFetcher:
    """Tests for the ERCOTDataFetcher implementation."""
    
    def setup_method(self, method):
        """Set up test environment before each test."""
        # Create a temporary directory for cache testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize mock for requests.Response
        self.mock_response = Mock()
        self.mock_response.status_code = 200
        
        # Set up mock for _make_request method
        self.patcher = patch('../../data/fetchers/ercot_api.ERCOTDataFetcher._make_request')
        self.mock_make_request = self.patcher.start()
        self.mock_make_request.return_value = self.mock_response
        
        # Initialize ERCOTDataFetcher with test parameters
        self.fetcher = ERCOTDataFetcher(
            timeout=1,
            max_retries=1,
            cache_ttl=3600,
            cache_dir=self.temp_dir
        )
    
    def teardown_method(self, method):
        """Clean up test environment after each test."""
        # Clean up temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(self.temp_dir)
        
        # Reset any mocks or patches
        self.patcher.stop()
    
    def test_fetch_rtlmp_data(self):
        """Test fetching RTLMP data from ERCOT API."""
        # Set up mock response with sample RTLMP data
        mock_response = get_mock_rtlmp_response()
        self.mock_response.json.return_value = mock_response
        
        # Mock validate_data to always return True
        with patch.object(self.fetcher, 'validate_data', return_value=True):
            # Call fetch_rtlmp_data with test parameters
            result = self.fetcher.fetch_rtlmp_data(
                SAMPLE_START_DATE,
                SAMPLE_END_DATE,
                SAMPLE_NODES
            )
            
            # Assert that _make_request was called with correct URL and parameters
            self.mock_make_request.assert_called_once()
            
            # Assert that returned DataFrame has expected structure and data
            assert isinstance(result, pd.DataFrame)
            assert 'timestamp' in result.columns
            assert 'node_id' in result.columns
            assert 'price' in result.columns
            
            # Assert that validate_data was called
            self.fetcher.validate_data.assert_called_once()
    
    def test_fetch_grid_conditions(self):
        """Test fetching grid condition data from ERCOT API."""
        # Set up mock response with sample grid condition data
        mock_response = get_mock_grid_conditions_response()
        self.mock_response.json.return_value = mock_response
        
        # Mock validate_data to always return True
        with patch.object(self.fetcher, 'validate_data', return_value=True):
            # Call fetch_grid_conditions with test parameters
            result = self.fetcher.fetch_grid_conditions(
                SAMPLE_START_DATE,
                SAMPLE_END_DATE
            )
            
            # Assert that _make_request was called with correct URL and parameters
            self.mock_make_request.assert_called_once()
            
            # Assert that returned DataFrame has expected structure and data
            assert isinstance(result, pd.DataFrame)
            assert 'timestamp' in result.columns
            assert 'total_load' in result.columns
            assert 'available_capacity' in result.columns
            
            # Assert that validate_data was called
            self.fetcher.validate_data.assert_called_once()
    
    def test_fetch_historical_data(self):
        """Test fetching historical data with different identifiers."""
        # Set up mock responses for different data types
        with patch.object(self.fetcher, 'fetch_rtlmp_data') as mock_fetch_rtlmp:
            with patch.object(self.fetcher, 'fetch_grid_conditions') as mock_fetch_grid:
                # Set up returns for mocks
                mock_fetch_rtlmp.return_value = pd.DataFrame({'test': [1, 2, 3]})
                mock_fetch_grid.return_value = pd.DataFrame({'test': [4, 5, 6]})
                
                # Call fetch_historical_data with node identifiers for RTLMP data
                result1 = self.fetcher.fetch_historical_data(
                    SAMPLE_START_DATE,
                    SAMPLE_END_DATE,
                    SAMPLE_NODES
                )
                mock_fetch_rtlmp.assert_called_once_with(
                    SAMPLE_START_DATE,
                    SAMPLE_END_DATE,
                    SAMPLE_NODES
                )
                
                # Call fetch_historical_data with empty identifiers for grid conditions
                result2 = self.fetcher.fetch_historical_data(
                    SAMPLE_START_DATE,
                    SAMPLE_END_DATE,
                    []
                )
                mock_fetch_grid.assert_called_once_with(
                    SAMPLE_START_DATE,
                    SAMPLE_END_DATE
                )
    
    def test_fetch_forecast_data(self):
        """Test fetching forecast data with different identifiers."""
        # Set up mock responses for different data types
        with patch.object(self.fetcher, 'fetch_rtlmp_data') as mock_fetch_rtlmp:
            with patch.object(self.fetcher, 'fetch_grid_conditions') as mock_fetch_grid:
                # Set up returns for mocks
                mock_fetch_rtlmp.return_value = pd.DataFrame({'test': [1, 2, 3]})
                mock_fetch_grid.return_value = pd.DataFrame({'test': [4, 5, 6]})
                
                # Call fetch_forecast_data with node identifiers for RTLMP data
                result1 = self.fetcher.fetch_forecast_data(
                    SAMPLE_START_DATE,
                    72,
                    SAMPLE_NODES
                )
                mock_fetch_rtlmp.assert_called_once()
                
                # Call fetch_forecast_data with empty identifiers for grid conditions
                result2 = self.fetcher.fetch_forecast_data(
                    SAMPLE_START_DATE,
                    72,
                    []
                )
                mock_fetch_grid.assert_called_once()
    
    def test_parse_rtlmp_response(self):
        """Test parsing RTLMP API responses."""
        # Create mock RTLMP API response using get_mock_rtlmp_response
        mock_response = get_mock_rtlmp_response()
        
        # Call parse_rtlmp_response with the mock response
        result = parse_rtlmp_response(mock_response)
        
        # Assert that returned data has expected structure and fields
        assert isinstance(result, list)
        assert len(result) > 0
        assert 'timestamp' in result[0]
        assert 'node_id' in result[0]
        assert 'price' in result[0]
        
        # Test with malformed response and assert that DataFormatError is raised
        malformed_response = {'status': '200'}
        with pytest.raises(DataFormatError):
            parse_rtlmp_response(malformed_response)
    
    def test_parse_grid_conditions_response(self):
        """Test parsing grid conditions API responses."""
        # Create mock grid conditions API response using get_mock_grid_conditions_response
        mock_response = get_mock_grid_conditions_response()
        
        # Call parse_grid_conditions_response with the mock response
        result = parse_grid_conditions_response(mock_response)
        
        # Assert that returned data has expected structure and fields
        assert isinstance(result, list)
        assert len(result) > 0
        assert 'timestamp' in result[0]
        assert 'total_load' in result[0]
        assert 'available_capacity' in result[0]
        
        # Test with malformed response and assert that DataFormatError is raised
        malformed_response = {'status': '200'}
        with pytest.raises(DataFormatError):
            parse_grid_conditions_response(malformed_response)
    
    def test_error_handling(self):
        """Test error handling for API requests."""
        # Set up mock to raise ConnectionError
        self.mock_make_request.side_effect = ConnectionError("Connection failed")
        with pytest.raises(ConnectionError):
            self.fetcher.fetch_rtlmp_data(SAMPLE_START_DATE, SAMPLE_END_DATE, SAMPLE_NODES)
        
        # Set up mock to raise RateLimitError
        self.mock_make_request.side_effect = RateLimitError("Rate limit exceeded")
        with pytest.raises(RateLimitError):
            self.fetcher.fetch_rtlmp_data(SAMPLE_START_DATE, SAMPLE_END_DATE, SAMPLE_NODES)
        
        # Set up mock to return malformed data
        self.mock_make_request.side_effect = None
        self.mock_response.json.return_value = {'status': '200'}  # Missing data field
        with pytest.raises(DataFormatError):
            self.fetcher.fetch_rtlmp_data(SAMPLE_START_DATE, SAMPLE_END_DATE, SAMPLE_NODES)
    
    def test_validate_data(self):
        """Test data validation for ERCOT data."""
        # Create sample RTLMP DataFrame
        rtlmp_data = get_sample_rtlmp_data()
        
        # Call validate_data with valid RTLMP data
        with patch.object(self.fetcher, 'validate_data', wraps=self.fetcher.validate_data):
            result = self.fetcher.validate_data(rtlmp_data)
            assert result
            
            # Create invalid RTLMP DataFrame (missing columns)
            invalid_rtlmp = rtlmp_data.drop(columns=['price'])
            result = self.fetcher.validate_data(invalid_rtlmp)
            assert not result
            
            # Repeat tests with grid condition data
            grid_data = get_sample_grid_condition_data()
            result = self.fetcher.validate_data(grid_data)
            assert result
            
            invalid_grid = grid_data.drop(columns=['total_load'])
            result = self.fetcher.validate_data(invalid_grid)
            assert not result


class TestWeatherAPIFetcher:
    """Tests for the WeatherAPIFetcher implementation."""
    
    def setup_method(self, method):
        """Set up test environment before each test."""
        # Create a temporary directory for cache testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize mock for requests.Response
        self.mock_response = Mock()
        self.mock_response.status_code = 200
        
        # Set up mock for _make_request method
        self.patcher = patch('../../data/fetchers/weather_api.WeatherAPIFetcher._make_request')
        self.mock_make_request = self.patcher.start()
        self.mock_make_request.return_value = self.mock_response
        
        # Initialize WeatherAPIFetcher with test parameters and API key
        self.fetcher = WeatherAPIFetcher(
            api_key="test_api_key",
            timeout=1,
            max_retries=1,
            cache_ttl=3600,
            cache_dir=self.temp_dir
        )
    
    def teardown_method(self, method):
        """Clean up test environment after each test."""
        # Clean up temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(self.temp_dir)
        
        # Reset any mocks or patches
        self.patcher.stop()
    
    def test_fetch_historical_data(self):
        """Test fetching historical weather data."""
        # Set up mock response with sample weather data
        mock_response = get_mock_weather_response()
        self.mock_response.json.return_value = mock_response
        
        # Mock validate_data to always return True
        with patch.object(self.fetcher, 'validate_data', return_value=True):
            # Call fetch_historical_data with test parameters
            result = self.fetcher.fetch_historical_data(
                SAMPLE_START_DATE,
                SAMPLE_END_DATE,
                DEFAULT_LOCATIONS
            )
            
            # Assert that _make_request was called with correct URL and parameters
            self.mock_make_request.assert_called_once()
            
            # Assert that returned DataFrame has expected structure and data
            assert isinstance(result, pd.DataFrame)
            assert 'timestamp' in result.columns
            assert 'location_id' in result.columns
            assert 'temperature' in result.columns
            
            # Assert that validate_data was called
            self.fetcher.validate_data.assert_called_once()
    
    def test_fetch_forecast_data(self):
        """Test fetching forecast weather data."""
        # Set up mock response with sample forecast data
        mock_response = get_mock_weather_response(is_forecast=True)
        self.mock_response.json.return_value = mock_response
        
        # Mock validate_data to always return True
        with patch.object(self.fetcher, 'validate_data', return_value=True):
            # Call fetch_forecast_data with test parameters
            result = self.fetcher.fetch_forecast_data(
                SAMPLE_START_DATE,
                72,
                DEFAULT_LOCATIONS
            )
            
            # Assert that _make_request was called with correct URL and parameters
            self.mock_make_request.assert_called_once()
            
            # Assert that returned DataFrame has expected structure and data
            assert isinstance(result, pd.DataFrame)
            assert 'timestamp' in result.columns
            assert 'location_id' in result.columns
            assert 'temperature' in result.columns
            
            # Assert that validate_data was called
            self.fetcher.validate_data.assert_called_once()
    
    def test_parse_weather_response(self):
        """Test parsing weather API responses."""
        # Create mock weather API response using get_mock_weather_response
        mock_response = get_mock_weather_response()
        
        # Call parse_weather_response with the mock response
        if 'data' in mock_response and isinstance(mock_response['data'], list) and len(mock_response['data']) > 0:
            result = parse_weather_response(mock_response['data'][0], DEFAULT_LOCATIONS[0])
            
            # Assert that returned data has expected structure and fields
            assert isinstance(result, list)
            assert len(result) > 0
            assert 'timestamp' in result[0]
            assert 'location_id' in result[0]
            assert 'temperature' in result[0]
        
        # Test with malformed response and assert that DataFormatError is raised
        malformed_response = {'status': '200'}
        with pytest.raises(Exception):
            parse_weather_response(malformed_response, DEFAULT_LOCATIONS[0])
    
    def test_error_handling(self):
        """Test error handling for weather API requests."""
        # Set up mock to raise ConnectionError
        self.mock_make_request.side_effect = ConnectionError("Connection failed")
        with pytest.raises(ConnectionError):
            self.fetcher.fetch_data({
                'start_date': SAMPLE_START_DATE,
                'end_date': SAMPLE_END_DATE,
                'identifiers': DEFAULT_LOCATIONS
            })
        
        # Set up mock to raise RateLimitError
        self.mock_make_request.side_effect = RateLimitError("Rate limit exceeded")
        with pytest.raises(RateLimitError):
            self.fetcher.fetch_data({
                'start_date': SAMPLE_START_DATE,
                'end_date': SAMPLE_END_DATE,
                'identifiers': DEFAULT_LOCATIONS
            })
        
        # Set up mock to return malformed data
        self.mock_make_request.side_effect = None
        self.mock_response.json.return_value = {'status': '200'}  # Missing data field
        with pytest.raises(DataFormatError):
            self.fetcher.fetch_data({
                'start_date': SAMPLE_START_DATE,
                'end_date': SAMPLE_END_DATE,
                'identifiers': DEFAULT_LOCATIONS
            })
    
    def test_validate_data(self):
        """Test data validation for weather data."""
        # Create sample weather DataFrame
        weather_data = get_sample_weather_data()
        
        # Call validate_data with valid weather data
        with patch.object(self.fetcher, 'validate_data', wraps=self.fetcher.validate_data):
            result = self.fetcher.validate_data(weather_data)
            assert result
            
            # Create invalid weather DataFrame (missing columns)
            invalid_weather = weather_data.drop(columns=['temperature'])
            result = self.fetcher.validate_data(invalid_weather)
            assert not result


class TestMockDataFetcher:
    """Tests for the MockDataFetcher implementation."""
    
    def setup_method(self, method):
        """Set up test environment before each test."""
        # Create a temporary directory for cache testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize MockDataFetcher with test parameters
        self.fetcher = MockDataFetcher(
            add_noise=True,
            spike_probability=0.05
        )
    
    def teardown_method(self, method):
        """Clean up test environment after each test."""
        # Clean up temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(self.temp_dir)
        
        # Reset any mocks or patches
        pass
    
    def test_fetch_historical_rtlmp_data(self):
        """Test fetching historical RTLMP data from mock fetcher."""
        # Call fetch_historical_data with node identifiers
        result = self.fetcher.fetch_historical_data(
            SAMPLE_START_DATE,
            SAMPLE_END_DATE,
            SAMPLE_NODES
        )
        
        # Assert that returned DataFrame has expected structure for RTLMP data
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert 'node_id' in result.columns
        assert 'price' in result.columns
        
        # Assert that data contains expected columns and value ranges
        assert result['price'].min() >= 0
        assert 'congestion_price' in result.columns
        assert 'loss_price' in result.columns
        assert 'energy_price' in result.columns
        
        # Assert that timestamps are within the requested range
        assert result['timestamp'].min() >= SAMPLE_START_DATE
        assert result['timestamp'].max() <= SAMPLE_END_DATE
    
    def test_fetch_historical_weather_data(self):
        """Test fetching historical weather data from mock fetcher."""
        # Call fetch_historical_data with location identifiers
        result = self.fetcher.fetch_historical_data(
            SAMPLE_START_DATE,
            SAMPLE_END_DATE,
            DEFAULT_LOCATIONS
        )
        
        # Assert that returned DataFrame has expected structure for weather data
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert 'location_id' in result.columns
        assert 'temperature' in result.columns
        
        # Assert that data contains expected columns and value ranges
        assert 'wind_speed' in result.columns
        assert 'solar_irradiance' in result.columns
        assert 'humidity' in result.columns
        assert result['temperature'].min() >= -20
        assert result['temperature'].max() <= 50
        
        # Assert that timestamps are within the requested range
        assert result['timestamp'].min() >= SAMPLE_START_DATE
        assert result['timestamp'].max() <= SAMPLE_END_DATE
    
    def test_fetch_grid_conditions(self):
        """Test fetching grid condition data from mock fetcher."""
        # Call fetch_grid_conditions with date range
        result = self.fetcher.fetch_grid_conditions(
            SAMPLE_START_DATE,
            SAMPLE_END_DATE
        )
        
        # Assert that returned DataFrame has expected structure for grid conditions
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert 'total_load' in result.columns
        assert 'available_capacity' in result.columns
        
        # Assert that data contains expected columns and value ranges
        assert 'wind_generation' in result.columns
        assert 'solar_generation' in result.columns
        assert result['total_load'].min() >= 0
        assert result['available_capacity'].min() >= 0
        
        # Assert that timestamps are within the requested range
        assert result['timestamp'].min() >= SAMPLE_START_DATE
        assert result['timestamp'].max() <= SAMPLE_END_DATE
    
    def test_fetch_forecast_data(self):
        """Test fetching forecast data from mock fetcher."""
        # Call fetch_forecast_data with forecast date and horizon
        forecast_date = SAMPLE_START_DATE
        horizon = 72
        result = self.fetcher.fetch_forecast_data(
            forecast_date,
            horizon,
            SAMPLE_NODES
        )
        
        # Assert that returned DataFrame has expected structure
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        
        # Assert that data contains expected columns
        if 'node_id' in result.columns:
            assert 'price' in result.columns
        elif 'location_id' in result.columns:
            assert 'temperature' in result.columns
        
        # Assert that timestamps start at forecast date and extend for the specified horizon
        end_date = forecast_date + timedelta(hours=horizon)
        assert result['timestamp'].min() >= forecast_date
        assert result['timestamp'].max() <= end_date
    
    def test_config_modification(self):
        """Test modifying mock data generation configuration."""
        # Get default configuration using get_config
        original_config = self.fetcher.get_config()
        
        # Create modified configuration with different parameters
        new_config = {
            'base_price': 50.0,
            'price_volatility': 20.0,
            'spike_magnitude': 200.0
        }
        
        # Set new configuration using set_config
        self.fetcher.set_config(new_config)
        
        # Generate data with new configuration
        result = self.fetcher.fetch_historical_data(
            SAMPLE_START_DATE,
            SAMPLE_END_DATE,
            SAMPLE_NODES
        )
        
        # Assert that data reflects the modified configuration parameters
        updated_config = self.fetcher.get_config()
        assert updated_config['base_price'] == new_config['base_price']
        assert updated_config['price_volatility'] == new_config['price_volatility']
        assert updated_config['spike_magnitude'] == new_config['spike_magnitude']
    
    def test_validate_data(self):
        """Test data validation for mock data."""
        # Generate sample mock data for different types
        rtlmp_data = self.fetcher.fetch_historical_data(
            SAMPLE_START_DATE,
            SAMPLE_END_DATE,
            SAMPLE_NODES
        )
        weather_data = self.fetcher.fetch_historical_data(
            SAMPLE_START_DATE,
            SAMPLE_END_DATE,
            DEFAULT_LOCATIONS
        )
        grid_data = self.fetcher.fetch_grid_conditions(
            SAMPLE_START_DATE,
            SAMPLE_END_DATE
        )
        
        # Call validate_data with each data type
        assert self.fetcher.validate_data(rtlmp_data)
        assert self.fetcher.validate_data(weather_data)
        assert self.fetcher.validate_data(grid_data)
        
        # Create invalid data by removing columns
        invalid_rtlmp = rtlmp_data.drop(columns=['price'])
        invalid_weather = weather_data.drop(columns=['temperature'])
        invalid_grid = grid_data.drop(columns=['total_load'])
        
        # Assert that validation returns False for invalid data
        assert not self.fetcher.validate_data(invalid_rtlmp)
        assert not self.fetcher.validate_data(invalid_weather)
        assert not self.fetcher.validate_data(invalid_grid)