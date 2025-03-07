"""
Centralized type definitions for the ERCOT RTLMP spike prediction system.

This file defines common types, protocols, and type aliases used throughout the 
system to ensure type consistency and enable static type checking.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Literal, Optional, Protocol, 
    Tuple, TypeAlias, TypedDict, TypeVar, Union, cast
)

import numpy as np  # version 1.24+
import pandas as pd  # version 2.0+

# Type aliases for common data structures
DataFrameType: TypeAlias = pd.DataFrame
SeriesType: TypeAlias = pd.Series
ArrayType: TypeAlias = np.ndarray
PathType: TypeAlias = Path
ThresholdValue: TypeAlias = float
NodeID: TypeAlias = str

# Generic type variable for model objects
ModelType = TypeVar('ModelType')

# Literal type for feature group categories
FeatureGroupType = Literal['time', 'statistical', 'weather', 'market']

# TypedDict definitions for data structures
class RTLMPDataDict(TypedDict):
    """Type definition for RTLMP data structure."""
    timestamp: datetime
    node_id: str
    price: float
    congestion_price: float
    loss_price: float
    energy_price: float

class WeatherDataDict(TypedDict):
    """Type definition for weather data structure."""
    timestamp: datetime
    location_id: str
    temperature: float
    wind_speed: float
    solar_irradiance: float
    humidity: float

class GridConditionDict(TypedDict):
    """Type definition for grid condition data structure."""
    timestamp: datetime
    total_load: float
    available_capacity: float
    wind_generation: float
    solar_generation: float

class ModelConfigDict(TypedDict):
    """Type definition for model configuration."""
    model_id: str
    model_type: str
    version: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]]
    training_date: Optional[datetime]
    feature_names: Optional[List[str]]

class ForecastResultDict(TypedDict):
    """Type definition for forecast result structure."""
    forecast_timestamp: datetime
    target_timestamp: datetime
    threshold_value: float
    spike_probability: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_version: str

# Protocol definitions for component interfaces
class DataFetcherProtocol(Protocol):
    """Protocol defining the interface for data fetchers."""
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> DataFrameType:
        """Fetch data from source."""
        ...
    
    @abstractmethod
    def fetch_historical_data(self, start_date: datetime, end_date: datetime, nodes: List[NodeID], **kwargs) -> DataFrameType:
        """Fetch historical data for a specific date range and nodes."""
        ...
    
    @abstractmethod
    def fetch_forecast_data(self, forecast_date: datetime, horizon: int, nodes: List[NodeID], **kwargs) -> DataFrameType:
        """Fetch forecast data for a specific date and horizon."""
        ...
    
    @abstractmethod
    def validate_data(self, data: DataFrameType) -> bool:
        """Validate the structure and content of fetched data."""
        ...

class ModelProtocol(Protocol[ModelType]):
    """Protocol defining the interface for prediction models."""
    
    @abstractmethod
    def train(self, X: DataFrameType, y: SeriesType, **kwargs) -> ModelType:
        """Train the model with features X and targets y."""
        ...
    
    @abstractmethod
    def predict(self, X: DataFrameType) -> SeriesType:
        """Generate predictions for features X."""
        ...
    
    @abstractmethod
    def predict_proba(self, X: DataFrameType) -> ArrayType:
        """Generate probability predictions for features X."""
        ...
    
    @abstractmethod
    def save(self, path: PathType) -> bool:
        """Save the model to the specified path."""
        ...
    
    @abstractmethod
    def load(self, path: PathType) -> ModelType:
        """Load a model from the specified path."""
        ...
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance values."""
        ...
    
    @abstractmethod
    def get_model_config(self) -> ModelConfigDict:
        """Get model configuration."""
        ...

class FeatureEngineerProtocol(Protocol):
    """Protocol defining the interface for feature engineering."""
    
    @abstractmethod
    def create_features(self, data: DataFrameType) -> DataFrameType:
        """Transform raw data into model-ready features."""
        ...
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get the names of all features created by this engineer."""
        ...
    
    @abstractmethod
    def validate_feature_set(self, features: DataFrameType) -> bool:
        """Validate that a feature set contains all required features."""
        ...

class InferenceEngineProtocol(Protocol):
    """Protocol defining the interface for inference engine."""
    
    @abstractmethod
    def load_model(self, model_path: PathType) -> ModelType:
        """Load a model for inference."""
        ...
    
    @abstractmethod
    def generate_forecast(self, features: DataFrameType, model: ModelType, thresholds: List[ThresholdValue]) -> DataFrameType:
        """Generate forecast using features and model for given thresholds."""
        ...
    
    @abstractmethod
    def store_forecast(self, forecast: DataFrameType, output_path: PathType) -> bool:
        """Store forecast results to the specified path."""
        ...
    
    @abstractmethod
    def run_inference(self, date: datetime, thresholds: List[ThresholdValue], nodes: List[NodeID], **kwargs) -> DataFrameType:
        """Run the complete inference pipeline."""
        ...

class BacktestingProtocol(Protocol):
    """Protocol defining the interface for backtesting framework."""
    
    @abstractmethod
    def run_backtest(self, start_date: datetime, end_date: datetime, model: ModelType, thresholds: List[ThresholdValue], nodes: List[NodeID], **kwargs) -> DataFrameType:
        """Run backtest for a specific period using the given model."""
        ...
    
    @abstractmethod
    def calculate_metrics(self, predictions: DataFrameType, actuals: DataFrameType) -> Dict[str, float]:
        """Calculate performance metrics from predictions and actual values."""
        ...
    
    @abstractmethod
    def generate_report(self, backtest_results: DataFrameType, output_path: Optional[PathType] = None) -> str:
        """Generate a report of backtest results."""
        ...

class StorageProtocol(Protocol):
    """Protocol defining the interface for storage components."""
    
    @abstractmethod
    def store(self, data: Any, path: PathType, **kwargs) -> bool:
        """Store data to the specified path."""
        ...
    
    @abstractmethod
    def retrieve(self, path: PathType, **kwargs) -> Any:
        """Retrieve data from the specified path."""
        ...
    
    @abstractmethod
    def delete(self, path: PathType, **kwargs) -> bool:
        """Delete data at the specified path."""
        ...
    
    @abstractmethod
    def list(self, path: PathType, **kwargs) -> List[PathType]:
        """List available items at the specified path."""
        ...