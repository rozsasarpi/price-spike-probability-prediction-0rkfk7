"""
Provides synthetic test data generators for the ERCOT RTLMP spike prediction system.

This module contains functions to generate realistic sample data for RTLMP prices,
weather conditions, grid conditions, and feature sets that can be used in unit tests
and integration tests without requiring external data sources.
"""

import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from datetime import datetime, timedelta
import random
from typing import List, Optional, Dict, Union, Series

from ...utils.type_definitions import RTLMPDataDict, WeatherDataDict, GridConditionDict
from ...utils.date_utils import ERCOT_TIMEZONE

# Define sample constants
SAMPLE_NODES = ['HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_HOUSTON']
SAMPLE_START_DATE = datetime(2022, 1, 1, tzinfo=ERCOT_TIMEZONE)
SAMPLE_END_DATE = datetime(2022, 1, 7, tzinfo=ERCOT_TIMEZONE)
PRICE_SPIKE_THRESHOLD = 100.0
DEFAULT_LOCATIONS = ['NORTH_TX', 'SOUTH_TX', 'WEST_TX', 'HOUSTON_TX']

def get_sample_rtlmp_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    nodes: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    include_spikes: bool = True
) -> pd.DataFrame:
    """
    Generates synthetic RTLMP data for testing.
    
    Args:
        start_date: Optional start date for the data range
        end_date: Optional end date for the data range
        nodes: Optional list of node IDs
        random_seed: Optional seed for reproducible random generation
        include_spikes: Whether to include price spikes in the data
    
    Returns:
        DataFrame containing synthetic RTLMP data
    """
    # Set default values if not provided
    start_date = start_date or SAMPLE_START_DATE
    end_date = end_date or SAMPLE_END_DATE
    nodes = nodes or SAMPLE_NODES
    
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Calculate time delta and generate timestamps at 5-minute intervals
    delta = end_date - start_date
    total_minutes = int(delta.total_seconds() / 60)
    timestamps = [start_date + timedelta(minutes=i*5) for i in range(total_minutes // 5 + 1)]
    
    # Initialize empty lists for each data column
    all_timestamps = []
    all_node_ids = []
    all_prices = []
    all_congestion_prices = []
    all_loss_prices = []
    all_energy_prices = []
    
    # For each timestamp and node combination
    for timestamp in timestamps:
        for node in nodes:
            # Generate base price with daily and hourly patterns
            base_price = 30.0  # Base price in $/MWh
            price = create_synthetic_price_pattern(
                timestamp, base_price, daily_amplitude=10.0, random_scale=5.0
            )
            
            # If include_spikes is True, randomly add price spikes
            if include_spikes and random.random() < 0.01:  # 1% chance of spike
                price += random.uniform(70, 200)
            
            # Calculate congestion_price, loss_price, and energy_price components
            energy_price = 25.0 + random.uniform(-2, 2)  # Mostly stable energy price
            congestion_price = price - energy_price - random.uniform(0, 3)  # Congestion determines most variation
            loss_price = random.uniform(0, 3)  # Small loss component
            
            # Append values to respective lists
            all_timestamps.append(timestamp)
            all_node_ids.append(node)
            all_prices.append(price)
            all_congestion_prices.append(congestion_price)
            all_loss_prices.append(loss_price)
            all_energy_prices.append(energy_price)
    
    # Create DataFrame from the lists
    data = {
        'timestamp': all_timestamps,
        'node_id': all_node_ids,
        'price': all_prices,
        'congestion_price': all_congestion_prices,
        'loss_price': all_loss_prices,
        'energy_price': all_energy_prices
    }
    
    df = pd.DataFrame(data)
    return df

def get_sample_weather_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    locations: Optional[List[str]] = None,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generates synthetic weather data for testing.
    
    Args:
        start_date: Optional start date for the data range
        end_date: Optional end date for the data range
        locations: Optional list of location IDs
        random_seed: Optional seed for reproducible random generation
    
    Returns:
        DataFrame containing synthetic weather data
    """
    # Set default values if not provided
    start_date = start_date or SAMPLE_START_DATE
    end_date = end_date or SAMPLE_END_DATE
    locations = locations or DEFAULT_LOCATIONS
    
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Calculate time delta and generate timestamps at hourly intervals
    delta = end_date - start_date
    total_hours = int(delta.total_seconds() / 3600)
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours + 1)]
    
    # Initialize empty lists for each data column
    all_timestamps = []
    all_location_ids = []
    all_temperatures = []
    all_wind_speeds = []
    all_solar_irradiances = []
    all_humidities = []
    
    # For each timestamp and location combination
    for timestamp in timestamps:
        for location in locations:
            # Generate temperature with daily pattern and seasonal component
            hour = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday
            
            # Temperature follows a daily cycle and a yearly seasonal component
            # Base pattern: lower at night, higher during day
            daily_pattern = np.sin(np.pi * hour / 12) * 10  # -10 to 10 degree variation
            seasonal_component = np.sin(np.pi * day_of_year / 182.5) * 15  # +/- 15 degrees seasonal variation
            base_temp = 20 + seasonal_component  # Base around 20C with seasonal variation
            temp = base_temp + daily_pattern + random.uniform(-2, 2)
            
            # Generate wind_speed with some randomness
            wind_speed = 5 + 2 * np.sin(np.pi * hour / 12) + random.uniform(0, 5)
            
            # Generate solar_irradiance based on hour of day (zero at night)
            if 6 <= hour < 20:  # Daylight hours
                solar_pattern = np.sin(np.pi * (hour - 6) / 14)  # 0 to 1 to 0 over daylight
                solar_irradiance = 1000 * solar_pattern * (0.7 + random.uniform(0, 0.3))
            else:
                solar_irradiance = 0
            
            # Generate humidity with inverse relationship to temperature
            humidity = 70 - daily_pattern * 1.5 + random.uniform(-10, 10)
            humidity = max(10, min(100, humidity))  # Clamp between 10% and 100%
            
            # Append values to respective lists
            all_timestamps.append(timestamp)
            all_location_ids.append(location)
            all_temperatures.append(temp)
            all_wind_speeds.append(wind_speed)
            all_solar_irradiances.append(solar_irradiance)
            all_humidities.append(humidity)
    
    # Create DataFrame from the lists
    data = {
        'timestamp': all_timestamps,
        'location_id': all_location_ids,
        'temperature': all_temperatures,
        'wind_speed': all_wind_speeds,
        'solar_irradiance': all_solar_irradiances,
        'humidity': all_humidities
    }
    
    df = pd.DataFrame(data)
    return df

def get_sample_grid_condition_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generates synthetic grid condition data for testing.
    
    Args:
        start_date: Optional start date for the data range
        end_date: Optional end date for the data range
        random_seed: Optional seed for reproducible random generation
    
    Returns:
        DataFrame containing synthetic grid condition data
    """
    # Set default values if not provided
    start_date = start_date or SAMPLE_START_DATE
    end_date = end_date or SAMPLE_END_DATE
    
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Calculate time delta and generate timestamps at hourly intervals
    delta = end_date - start_date
    total_hours = int(delta.total_seconds() / 3600)
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours + 1)]
    
    # Initialize empty lists for each data column
    all_timestamps = []
    all_total_loads = []
    all_available_capacities = []
    all_wind_generations = []
    all_solar_generations = []
    
    # For each timestamp
    for timestamp in timestamps:
        # Generate total_load with daily pattern (higher during day, lower at night)
        total_load = create_synthetic_load_pattern(
            timestamp, base_load=40000, daily_amplitude=15000, random_scale=2000
        )
        
        # Generate available_capacity with some randomness
        available_capacity = total_load * (1.2 + random.uniform(0.1, 0.3))
        
        # Generate wind_generation based on time of day and randomness
        hour = timestamp.hour
        day_factor = 0.8 + random.uniform(0, 0.4)  # Day-to-day variation
        time_factor = 1.2 if hour < 8 or hour >= 20 else 0.9  # Higher at night
        wind_generation = 5000 * day_factor * time_factor + random.uniform(-1000, 1000)
        
        # Generate solar_generation based on hour of day (zero at night)
        if 6 <= hour < 20:  # Daylight hours
            solar_pattern = np.sin(np.pi * (hour - 6) / 14)  # 0 to 1 to 0 over daylight
            solar_generation = 10000 * solar_pattern * (0.7 + random.uniform(0, 0.3))
        else:
            solar_generation = 0
        
        # Append values to respective lists
        all_timestamps.append(timestamp)
        all_total_loads.append(total_load)
        all_available_capacities.append(available_capacity)
        all_wind_generations.append(wind_generation)
        all_solar_generations.append(solar_generation)
    
    # Create DataFrame from the lists
    data = {
        'timestamp': all_timestamps,
        'total_load': all_total_loads,
        'available_capacity': all_available_capacities,
        'wind_generation': all_wind_generations,
        'solar_generation': all_solar_generations
    }
    
    df = pd.DataFrame(data)
    return df

def get_sample_feature_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    nodes: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    include_target: bool = True
) -> pd.DataFrame:
    """
    Generates a combined feature dataset for model testing.
    
    Args:
        start_date: Optional start date for the data range
        end_date: Optional end date for the data range
        nodes: Optional list of node IDs
        random_seed: Optional seed for reproducible random generation
        include_target: Whether to include target labels in the dataset
    
    Returns:
        DataFrame containing combined features for model testing
    """
    # Set default values if not provided
    start_date = start_date or SAMPLE_START_DATE
    end_date = end_date or SAMPLE_END_DATE
    nodes = nodes or SAMPLE_NODES
    
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Generate RTLMP data using get_sample_rtlmp_data
    rtlmp_data = get_sample_rtlmp_data(start_date, end_date, nodes, random_seed)
    # Generate weather data using get_sample_weather_data
    weather_data = get_sample_weather_data(start_date, end_date, random_seed=random_seed)
    # Generate grid condition data using get_sample_grid_condition_data
    grid_data = get_sample_grid_condition_data(start_date, end_date, random_seed)
    
    # Hourly resampling of RTLMP data for feature engineering
    rtlmp_hourly = rtlmp_data.set_index('timestamp').groupby(['node_id', pd.Grouper(freq='H')]).agg({
        'price': ['mean', 'max', 'min', 'std'],
        'congestion_price': 'mean',
        'loss_price': 'mean',
        'energy_price': 'mean'
    })
    rtlmp_hourly.columns = ['_'.join(col).strip() for col in rtlmp_hourly.columns.values]
    rtlmp_hourly = rtlmp_hourly.reset_index()
    
    # Create time-based features (hour_of_day, day_of_week, is_weekend)
    time_features = pd.DataFrame({'timestamp': rtlmp_hourly['timestamp'].unique()})
    time_features['hour_of_day'] = time_features['timestamp'].dt.hour
    time_features['day_of_week'] = time_features['timestamp'].dt.dayofweek
    time_features['is_weekend'] = time_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    time_features['month'] = time_features['timestamp'].dt.month
    
    # Create statistical features (rolling means, volatility)
    node_dfs = []
    for node in nodes:
        node_data = rtlmp_hourly[rtlmp_hourly['node_id'] == node].sort_values('timestamp')
        
        # Calculate rolling statistics
        node_data['rolling_mean_24h'] = node_data['price_mean'].rolling(window=24, min_periods=1).mean()
        node_data['rolling_max_24h'] = node_data['price_max'].rolling(window=24, min_periods=1).max()
        node_data['rolling_std_24h'] = node_data['price_std'].rolling(window=24, min_periods=1).mean()
        node_data['price_delta_1h'] = node_data['price_mean'].diff()
        node_data['congestion_ratio'] = node_data['congestion_price_mean'] / (node_data['price_mean'] + 0.1)  # Avoid div by zero
        
        node_dfs.append(node_data)
    
    # Combine node data
    feature_data = pd.concat(node_dfs)
    
    # Merge all features on timestamp and node_id
    feature_data = pd.merge(feature_data, time_features, on='timestamp')
    
    # Create weather features (temperature trends, wind patterns)
    weather_hourly = weather_data.groupby('timestamp').agg({
        'temperature': 'mean',
        'wind_speed': 'mean',
        'solar_irradiance': 'mean',
        'humidity': 'mean'
    }).reset_index()
    
    feature_data = pd.merge(feature_data, weather_hourly, on='timestamp')
    
    # Create market features (congestion ratios, reserve margins)
    feature_data = pd.merge(feature_data, grid_data, on='timestamp')
    
    # Calculate additional derived features
    feature_data['reserve_margin'] = (feature_data['available_capacity'] - feature_data['total_load']) / feature_data['total_load']
    feature_data['renewable_ratio'] = (feature_data['wind_generation'] + feature_data['solar_generation']) / feature_data['total_load']
    feature_data['load_factor'] = feature_data['total_load'] / feature_data['available_capacity']
    
    # If include_target is True, add spike_occurred target column
    if include_target:
        spike_labels = generate_hourly_spike_labels(rtlmp_data)
        feature_data = pd.merge(feature_data, spike_labels, on=['timestamp', 'node_id'])
    
    # Clean up NaN values
    feature_data = feature_data.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return feature_data

def generate_spike_labels(rtlmp_data: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
    """
    Generates binary labels for price spikes based on a threshold.
    
    Args:
        rtlmp_data: DataFrame containing RTLMP price data
        threshold: Price threshold for spike definition (default: PRICE_SPIKE_THRESHOLD)
    
    Returns:
        Binary series indicating price spikes
    """
    # Set default threshold if not provided
    threshold = threshold or PRICE_SPIKE_THRESHOLD
    # Create binary labels where price > threshold
    return (rtlmp_data['price'] > threshold).astype(int)

def generate_hourly_spike_labels(rtlmp_data: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Generates hourly binary labels indicating if any 5-min interval had a spike.
    
    Args:
        rtlmp_data: DataFrame containing RTLMP price data
        threshold: Price threshold for spike definition (default: PRICE_SPIKE_THRESHOLD)
    
    Returns:
        DataFrame with hourly spike indicators
    """
    # Set default threshold if not provided
    threshold = threshold or PRICE_SPIKE_THRESHOLD
    
    # Generate 5-minute spike labels using generate_spike_labels
    rtlmp_data = rtlmp_data.copy()
    rtlmp_data['spike_occurred'] = generate_spike_labels(rtlmp_data, threshold)
    
    # Resample to hourly frequency
    hourly_spikes = rtlmp_data.set_index('timestamp').groupby(['node_id', pd.Grouper(freq='H')]).agg({
        'spike_occurred': 'max'  # 1 if any 5-min period had a spike, 0 otherwise
    }).reset_index()
    
    # Create hourly DataFrame with binary spike indicators
    return hourly_spikes

def create_synthetic_price_pattern(
    timestamp: datetime,
    base_price: float,
    daily_amplitude: float,
    random_scale: float
) -> float:
    """
    Creates a synthetic price pattern with daily cycles and random components.
    
    Args:
        timestamp: Datetime for the price point
        base_price: Base price value
        daily_amplitude: Amplitude of daily price variation
        random_scale: Scale factor for random component
    
    Returns:
        Synthetic price value
    """
    # Extract hour of day from timestamp
    hour = timestamp.hour
    
    # Calculate daily component based on hour (higher during peak hours)
    morning_factor = np.exp(-0.5 * ((hour - 8) / 2) ** 2)  # Gaussian around 8 AM
    evening_factor = np.exp(-0.5 * ((hour - 19) / 2) ** 2)  # Gaussian around 7 PM
    daily_component = daily_amplitude * max(morning_factor, evening_factor)
    
    # Add random component scaled by random_scale
    random_component = random.uniform(-random_scale, random_scale)
    
    return base_price + daily_component + random_component

def create_synthetic_load_pattern(
    timestamp: datetime,
    base_load: float,
    daily_amplitude: float,
    random_scale: float
) -> float:
    """
    Creates a synthetic load pattern with daily cycles and random components.
    
    Args:
        timestamp: Datetime for the load point
        base_load: Base load value
        daily_amplitude: Amplitude of daily load variation
        random_scale: Scale factor for random component
    
    Returns:
        Synthetic load value
    """
    # Extract hour of day and day of week from timestamp
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0-6, Monday is 0
    
    # Calculate daily component based on hour (higher during business hours)
    if 0 <= hour < 5:  # Overnight
        daily_factor = 0.6
    elif 5 <= hour < 9:  # Morning ramp
        daily_factor = 0.6 + 0.3 * (hour - 5) / 4
    elif 9 <= hour < 17:  # Daytime
        daily_factor = 0.9
    elif 17 <= hour < 21:  # Evening peak
        daily_factor = 1.0
    else:  # Late evening
        daily_factor = 1.0 - 0.4 * (hour - 21) / 3
    
    daily_component = daily_amplitude * daily_factor
    
    # Calculate weekly component (lower on weekends)
    weekend_factor = 0.85 if day_of_week >= 5 else 1.0
    
    # Add random component scaled by random_scale
    random_component = random.uniform(-random_scale, random_scale)
    
    return (base_load + daily_component) * weekend_factor + random_component