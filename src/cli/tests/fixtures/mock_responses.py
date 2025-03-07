"""
Provides mock response data for testing CLI commands of the ERCOT RTLMP spike prediction system.

This module contains predefined response structures that mimic the outputs from backend API calls,
allowing tests to run without actual backend dependencies. It includes mock data for RTLMP data,
weather data, model training results, forecasts, and error responses.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from datetime import datetime, timedelta

# Import type definitions
from ../../../backend/utils/type_definitions import DataFrameType, ThresholdValue, NodeID
from ../../cli_types import CommandType, DataType

# Mock RTLMP data response
MOCK_RTLMP_DATA = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "data_type": "rtlmp",
    "nodes": ["HB_NORTH"],
    "start_date": (datetime.now() - timedelta(days=7)).date().isoformat(),
    "end_date": datetime.now().date().isoformat(),
    "row_count": 2016,  # 7 days * 24 hours * 12 intervals (5-minute data)
    "data": {}  # Will be populated by create_mock_dataframe when needed
}

# Mock weather data response
MOCK_WEATHER_DATA = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "data_type": "weather",
    "locations": ["NORTH_CENTRAL"],
    "start_date": (datetime.now() - timedelta(days=7)).date().isoformat(),
    "end_date": datetime.now().date().isoformat(),
    "row_count": 168,  # 7 days * 24 hours
    "data": {}  # Will be populated by create_mock_dataframe when needed
}

# Mock grid conditions data response
MOCK_GRID_CONDITIONS_DATA = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "data_type": "grid_conditions",
    "start_date": (datetime.now() - timedelta(days=7)).date().isoformat(),
    "end_date": datetime.now().date().isoformat(),
    "row_count": 168,  # 7 days * 24 hours
    "data": {}  # Will be populated by create_mock_dataframe when needed
}

# Mock combined data response
MOCK_COMBINED_DATA = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "data_types": ["rtlmp", "weather", "grid_conditions"],
    "nodes": ["HB_NORTH"],
    "locations": ["NORTH_CENTRAL"],
    "start_date": (datetime.now() - timedelta(days=7)).date().isoformat(),
    "end_date": datetime.now().date().isoformat(),
    "rtlmp_row_count": 2016,
    "weather_row_count": 168,
    "grid_conditions_row_count": 168,
    "data": {
        "rtlmp": {},
        "weather": {},
        "grid_conditions": {}
    }  # Will be populated by create_mock_dataframe when needed
}

# Mock model training result
MOCK_MODEL_TRAINING_RESULT = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "model_id": "model_001",
    "model_type": "xgboost",
    "version": "1.0.0",
    "training_duration": 1345.67,  # seconds
    "training_data": {
        "start_date": (datetime.now() - timedelta(days=365)).date().isoformat(),
        "end_date": datetime.now().date().isoformat(),
        "nodes": ["HB_NORTH"],
        "thresholds": [100.0],
        "row_count": 8760  # 365 days * 24 hours
    },
    "cross_validation": {
        "folds": 5,
        "scores": {
            "auc": [0.81, 0.79, 0.83, 0.82, 0.80],
            "brier_score": [0.12, 0.13, 0.11, 0.12, 0.14],
            "precision": [0.75, 0.73, 0.78, 0.76, 0.74],
            "recall": [0.70, 0.68, 0.72, 0.71, 0.69]
        },
        "mean_scores": {
            "auc": 0.81,
            "brier_score": 0.12,
            "precision": 0.75,
            "recall": 0.70
        }
    },
    "hyperparameters": {
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200
    },
    "feature_importance": {
        "rolling_price_max_24h": 0.32,
        "hour_of_day": 0.24,
        "load_forecast": 0.21,
        "day_of_week": 0.13,
        "temperature_forecast": 0.08,
        "wind_forecast": 0.02
    },
    "model_path": "/models/xgboost/model_001_v1.0.0.joblib"
}

# Mock model list response
MOCK_MODEL_LIST = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "models": {
        "xgboost": [
            "model_001_v1.0.0",
            "model_001_v0.9.0",
            "model_002_v1.0.0"
        ],
        "lightgbm": [
            "model_003_v1.0.0"
        ],
        "random_forest": [
            "model_004_v1.0.0"
        ]
    },
    "default_model": "model_001_v1.0.0"
}

# Mock model comparison result
MOCK_MODEL_COMPARISON = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "models_compared": [
        {
            "model_id": "model_001",
            "version": "1.0.0",
            "model_type": "xgboost"
        },
        {
            "model_id": "model_001",
            "version": "0.9.0",
            "model_type": "xgboost"
        }
    ],
    "evaluation_period": {
        "start_date": (datetime.now() - timedelta(days=30)).date().isoformat(),
        "end_date": datetime.now().date().isoformat()
    },
    "metrics": {
        "model_001_v1.0.0": {
            "auc": 0.82,
            "brier_score": 0.11,
            "precision": 0.76,
            "recall": 0.71,
            "f1_score": 0.73
        },
        "model_001_v0.9.0": {
            "auc": 0.79,
            "brier_score": 0.13,
            "precision": 0.74,
            "recall": 0.68,
            "f1_score": 0.71
        }
    },
    "improvement": {
        "auc": 0.03,
        "brier_score": -0.02,  # Lower is better for Brier score
        "precision": 0.02,
        "recall": 0.03,
        "f1_score": 0.02
    }
}

# Mock forecast result
MOCK_FORECAST_RESULT = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "forecast_id": "forecast_20230715_1200",
    "model_id": "model_001",
    "model_version": "1.0.0",
    "model_type": "xgboost",
    "forecast_horizon": 72,  # hours
    "thresholds": [100.0],
    "nodes": ["HB_NORTH"],
    "execution_time": 4.56,  # seconds
    "forecast_data": {}  # Will be populated by create_mock_forecast when needed
}

# Mock forecast data
MOCK_FORECAST_DATA = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "forecast_id": "forecast_20230715_1200",
    "forecast_timestamp": datetime.now().isoformat(),
    "horizon_start": (datetime.now() + timedelta(hours=1)).isoformat(),
    "horizon_end": (datetime.now() + timedelta(hours=72)).isoformat(),
    "thresholds": [100.0],
    "nodes": ["HB_NORTH"],
    "forecast_data": {}  # Will be populated by create_mock_forecast when needed
}

# Mock backtesting result
MOCK_BACKTEST_RESULT = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "backtest_id": "backtest_20230715_001",
    "model_id": "model_001",
    "model_version": "1.0.0",
    "model_type": "xgboost",
    "backtest_period": {
        "start_date": (datetime.now() - timedelta(days=30)).date().isoformat(),
        "end_date": datetime.now().date().isoformat()
    },
    "thresholds": [100.0],
    "nodes": ["HB_NORTH"],
    "execution_time": 45.67,  # seconds
    "metrics": {
        "100.0": {
            "HB_NORTH": {
                "auc": 0.82,
                "brier_score": 0.11,
                "precision": 0.76,
                "recall": 0.71,
                "f1_score": 0.73,
                "true_positives": 45,
                "false_positives": 14,
                "true_negatives": 652,
                "false_negatives": 9
            }
        }
    },
    "backtest_data": {}  # Will be populated when needed
}

# Mock evaluation metrics
MOCK_EVALUATION_METRICS = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "model_id": "model_001",
    "model_version": "1.0.0",
    "evaluation_period": {
        "start_date": (datetime.now() - timedelta(days=30)).date().isoformat(),
        "end_date": datetime.now().date().isoformat()
    },
    "thresholds": [100.0],
    "nodes": ["HB_NORTH"],
    "metrics": {
        "100.0": {
            "HB_NORTH": {
                "auc": 0.82,
                "brier_score": 0.11,
                "precision": 0.76,
                "recall": 0.71,
                "f1_score": 0.73,
                "log_loss": 0.25,
                "calibration_error": 0.03
            }
        }
    }
}

# Mock visualization data
MOCK_VISUALIZATION_DATA = {
    "status": "success",
    "timestamp": datetime.now().isoformat(),
    "visualization_type": "forecast",
    "forecast_id": "forecast_20230715_1200",
    "model_id": "model_001",
    "model_version": "1.0.0",
    "thresholds": [100.0],
    "nodes": ["HB_NORTH"],
    "data": {}  # Will be populated when needed
}

# Mock error responses
MOCK_ERROR_RESPONSES = {
    "validation": {
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "error_type": "validation",
        "message": "Invalid parameters: start_date must be before end_date"
    },
    "data_fetch": {
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "error_type": "data_fetch",
        "message": "Failed to retrieve data from ERCOT API: Connection timeout"
    },
    "model": {
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "error_type": "model",
        "message": "Model not found: model_001_v2.0.0"
    },
    "inference": {
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "error_type": "inference",
        "message": "Failed to generate forecast: Insufficient data for feature engineering"
    },
    "system": {
        "status": "error",
        "timestamp": datetime.now().isoformat(),
        "error_type": "system",
        "message": "System error: Out of memory"
    }
}


def get_mock_response(command: CommandType, response_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns a mock response for a specific command and response type.
    
    Args:
        command: The command type (fetch-data, train, predict, etc.)
        response_type: The type of response (success, error, etc.)
        params: Optional parameters to customize the mock response
    
    Returns:
        A dictionary containing the mock response
    """
    # Default return success response
    result = {"status": "success", "timestamp": datetime.now().isoformat()}
    
    # Handle error responses
    if response_type == "error":
        error_type = params.get("error_type", "system") if params else "system"
        message = params.get("message", None) if params else None
        return create_mock_error(error_type, message)
    
    # Handle different commands
    if command == "fetch-data":
        data_type = params.get("data_type", "rtlmp") if params else "rtlmp"
        if data_type == "rtlmp":
            result = MOCK_RTLMP_DATA.copy()
            if params:
                result["nodes"] = params.get("nodes", ["HB_NORTH"])
                result["start_date"] = params.get("start_date", result["start_date"])
                result["end_date"] = params.get("end_date", result["end_date"])
                rows = params.get("rows", 168)
                columns = params.get("columns", None)
                start_date = datetime.fromisoformat(result["start_date"]) if isinstance(result["start_date"], str) else result["start_date"]
                result["data"] = create_mock_dataframe("rtlmp", rows, columns, start_date)
        elif data_type == "weather":
            result = MOCK_WEATHER_DATA.copy()
            if params:
                result["locations"] = params.get("locations", ["NORTH_CENTRAL"])
                result["start_date"] = params.get("start_date", result["start_date"])
                result["end_date"] = params.get("end_date", result["end_date"])
                rows = params.get("rows", 168)
                columns = params.get("columns", None)
                start_date = datetime.fromisoformat(result["start_date"]) if isinstance(result["start_date"], str) else result["start_date"]
                result["data"] = create_mock_dataframe("weather", rows, columns, start_date)
        elif data_type == "grid_conditions":
            result = MOCK_GRID_CONDITIONS_DATA.copy()
            if params:
                result["start_date"] = params.get("start_date", result["start_date"])
                result["end_date"] = params.get("end_date", result["end_date"])
                rows = params.get("rows", 168)
                columns = params.get("columns", None)
                start_date = datetime.fromisoformat(result["start_date"]) if isinstance(result["start_date"], str) else result["start_date"]
                result["data"] = create_mock_dataframe("grid_conditions", rows, columns, start_date)
        elif data_type == "all":
            result = MOCK_COMBINED_DATA.copy()
            if params:
                result["nodes"] = params.get("nodes", ["HB_NORTH"])
                result["locations"] = params.get("locations", ["NORTH_CENTRAL"])
                result["start_date"] = params.get("start_date", result["start_date"])
                result["end_date"] = params.get("end_date", result["end_date"])
                rows = params.get("rows", 168)
                start_date = datetime.fromisoformat(result["start_date"]) if isinstance(result["start_date"], str) else result["start_date"]
                result["data"]["rtlmp"] = create_mock_dataframe("rtlmp", rows * 12, None, start_date)
                result["data"]["weather"] = create_mock_dataframe("weather", rows, None, start_date)
                result["data"]["grid_conditions"] = create_mock_dataframe("grid_conditions", rows, None, start_date)
                
    elif command == "train":
        if response_type == "training_result":
            result = MOCK_MODEL_TRAINING_RESULT.copy()
            if params:
                result["model_id"] = params.get("model_id", result["model_id"])
                result["model_type"] = params.get("model_type", result["model_type"])
                result["version"] = params.get("version", result["version"])
                if "hyperparameters" in params:
                    result["hyperparameters"].update(params["hyperparameters"])
                result["training_data"]["nodes"] = params.get("nodes", result["training_data"]["nodes"])
                result["training_data"]["thresholds"] = params.get("thresholds", result["training_data"]["thresholds"])
        elif response_type == "model_list":
            result = MOCK_MODEL_LIST.copy()
        elif response_type == "model_info":
            model_id = params.get("model_id", "model_001") if params else "model_001"
            model_type = params.get("model_type", "xgboost") if params else "xgboost"
            version = params.get("version", "1.0.0") if params else "1.0.0"
            result = create_mock_model_info(model_id, model_type, version)
    
    elif command == "predict":
        if response_type == "forecast_result":
            result = MOCK_FORECAST_RESULT.copy()
            if params:
                result["thresholds"] = params.get("thresholds", [100.0])
                result["nodes"] = params.get("nodes", ["HB_NORTH"])
                result["model_id"] = params.get("model_id", result["model_id"])
                result["model_version"] = params.get("model_version", result["model_version"])
                horizon = params.get("horizon", 72)
                thresholds = params.get("thresholds", [100.0])
                nodes = params.get("nodes", ["HB_NORTH"])
                result["forecast_data"] = create_mock_forecast(thresholds, nodes, horizon)
        elif response_type == "forecast_data":
            result = MOCK_FORECAST_DATA.copy()
            if params:
                result["thresholds"] = params.get("thresholds", [100.0])
                result["nodes"] = params.get("nodes", ["HB_NORTH"])
                horizon = params.get("horizon", 72)
                thresholds = params.get("thresholds", [100.0])
                nodes = params.get("nodes", ["HB_NORTH"])
                result["forecast_data"] = create_mock_forecast(thresholds, nodes, horizon)
    
    elif command == "backtest":
        result = MOCK_BACKTEST_RESULT.copy()
        if params:
            result["thresholds"] = params.get("thresholds", [100.0])
            result["nodes"] = params.get("nodes", ["HB_NORTH"])
            result["model_id"] = params.get("model_id", result["model_id"])
            result["model_version"] = params.get("model_version", result["model_version"])
            if "backtest_period" in params:
                result["backtest_period"].update(params["backtest_period"])
            
    elif command == "evaluate":
        result = MOCK_EVALUATION_METRICS.copy()
        if params:
            result["thresholds"] = params.get("thresholds", [100.0])
            result["nodes"] = params.get("nodes", ["HB_NORTH"])
            result["model_id"] = params.get("model_id", result["model_id"])
            result["model_version"] = params.get("model_version", result["model_version"])
            if "evaluation_period" in params:
                result["evaluation_period"].update(params["evaluation_period"])
            
    elif command == "visualize":
        result = MOCK_VISUALIZATION_DATA.copy()
        if params:
            result["visualization_type"] = params.get("visualization_type", result["visualization_type"])
            result["thresholds"] = params.get("thresholds", [100.0])
            result["nodes"] = params.get("nodes", ["HB_NORTH"])
            
    return result


def create_mock_dataframe(data_type: str, rows: Optional[int] = None, 
                        columns: Optional[List[str]] = None, 
                        start_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Creates a mock pandas DataFrame with specified structure.
    
    Args:
        data_type: The type of data to create (rtlmp, weather, grid_conditions, forecast)
        rows: The number of rows in the DataFrame (default: 100)
        columns: Specific columns to include (default: all columns for the data_type)
        start_date: The start date for time series data (default: current date)
    
    Returns:
        A pandas DataFrame with appropriate structure
    """
    rows = 100 if rows is None else rows
    start_date = datetime.now() if start_date is None else start_date
    
    if data_type == "rtlmp":
        # RTLMP data has 5-minute intervals
        date_range = pd.date_range(start=start_date, periods=rows, freq="5T")
        
        # Default columns for RTLMP data
        if columns is None:
            columns = ["node_id", "price", "congestion_price", "loss_price", "energy_price"]
        
        # Generate random data
        data = {
            "node_id": ["HB_NORTH"] * rows
        }
        
        if "price" in columns:
            # Generate somewhat realistic price data with occasional spikes
            base_price = np.random.normal(30, 10, rows)
            spikes = np.random.random(rows) > 0.95  # 5% chance of spike
            spike_multiplier = np.random.uniform(3, 10, rows)
            data["price"] = base_price * (1 + spikes * spike_multiplier)
        
        if "congestion_price" in columns:
            data["congestion_price"] = np.random.normal(5, 2, rows)
        
        if "loss_price" in columns:
            data["loss_price"] = np.random.normal(1, 0.5, rows)
        
        if "energy_price" in columns:
            data["energy_price"] = np.random.normal(25, 5, rows)
        
        df = pd.DataFrame(data, index=date_range)
        
    elif data_type == "weather":
        # Weather data has hourly intervals
        date_range = pd.date_range(start=start_date, periods=rows, freq="H")
        
        # Default columns for weather data
        if columns is None:
            columns = ["location_id", "temperature", "wind_speed", "solar_irradiance", "humidity"]
        
        # Generate random data
        data = {
            "location_id": ["NORTH_CENTRAL"] * rows
        }
        
        if "temperature" in columns:
            # Generate somewhat realistic temperature data with daily cycle
            hour_of_day = date_range.hour
            base_temp = 75 + 15 * np.sin((hour_of_day - 12) * np.pi / 12)
            random_variation = np.random.normal(0, 3, rows)
            data["temperature"] = base_temp + random_variation
        
        if "wind_speed" in columns:
            data["wind_speed"] = np.random.gamma(2, 3, rows)
        
        if "solar_irradiance" in columns:
            # Solar irradiance follows daily cycle and is zero at night
            hour_of_day = date_range.hour
            is_day = (hour_of_day >= 6) & (hour_of_day <= 18)
            day_intensity = np.sin((hour_of_day[is_day] - 6) * np.pi / 12)
            solar = np.zeros(rows)
            solar[is_day] = day_intensity * 1000
            random_variation = np.random.normal(0, 50, rows)
            solar = np.maximum(0, solar + random_variation)
            data["solar_irradiance"] = solar
        
        if "humidity" in columns:
            data["humidity"] = np.random.uniform(30, 90, rows)
        
        df = pd.DataFrame(data, index=date_range)
        
    elif data_type == "grid_conditions":
        # Grid conditions data has hourly intervals
        date_range = pd.date_range(start=start_date, periods=rows, freq="H")
        
        # Default columns for grid conditions data
        if columns is None:
            columns = ["total_load", "available_capacity", "wind_generation", "solar_generation"]
        
        # Generate random data
        data = {}
        
        if "total_load" in columns:
            # Generate somewhat realistic load data with daily cycle
            hour_of_day = date_range.hour
            base_load = 40000 + 15000 * np.sin((hour_of_day - 12) * np.pi / 12)
            random_variation = np.random.normal(0, 2000, rows)
            data["total_load"] = base_load + random_variation
        
        if "available_capacity" in columns:
            data["available_capacity"] = np.random.uniform(70000, 80000, rows)
        
        if "wind_generation" in columns:
            data["wind_generation"] = np.random.gamma(3, 2000, rows)
        
        if "solar_generation" in columns:
            # Solar generation follows daily cycle and is zero at night
            hour_of_day = date_range.hour
            is_day = (hour_of_day >= 6) & (hour_of_day <= 18)
            day_intensity = np.sin((hour_of_day[is_day] - 6) * np.pi / 12)
            solar = np.zeros(rows)
            solar[is_day] = day_intensity * 10000
            random_variation = np.random.normal(0, 500, rows)
            solar = np.maximum(0, solar + random_variation)
            data["solar_generation"] = solar
        
        df = pd.DataFrame(data, index=date_range)
        
    elif data_type == "forecast":
        # Forecast data has hourly intervals
        date_range = pd.date_range(start=start_date, periods=rows, freq="H")
        
        # Default columns for forecast data
        if columns is None:
            columns = ["threshold_value", "node_id", "spike_probability", 
                      "confidence_interval_lower", "confidence_interval_upper"]
        
        # Generate random data
        threshold_values = [100.0] * rows
        node_ids = ["HB_NORTH"] * rows
        
        # Generate somewhat realistic probability data
        hour_of_day = date_range.hour
        day_of_week = date_range.dayofweek
        
        # Higher probabilities in afternoon and evening hours
        time_factor = 0.3 + 0.7 * np.sin((hour_of_day - 6) * np.pi / 18)
        # Higher probabilities on weekdays
        day_factor = 0.7 + 0.3 * (day_of_week < 5)
        
        base_probability = time_factor * day_factor
        random_variation = np.random.normal(0, 0.1, rows)
        probabilities = np.clip(base_probability + random_variation, 0.01, 0.99)
        
        # Generate confidence intervals
        interval_width = 0.1 + 0.2 * (1 - probabilities)  # wider intervals for lower probabilities
        lower_bound = np.clip(probabilities - interval_width/2, 0.01, 0.99)
        upper_bound = np.clip(probabilities + interval_width/2, 0.01, 0.99)
        
        data = {
            "threshold_value": threshold_values,
            "node_id": node_ids,
            "spike_probability": probabilities,
            "confidence_interval_lower": lower_bound,
            "confidence_interval_upper": upper_bound
        }
        
        df = pd.DataFrame(data, index=date_range)
    
    else:
        # Default empty DataFrame for unknown data types
        df = pd.DataFrame(index=pd.date_range(start=start_date, periods=rows, freq="H"))
    
    return df


def create_mock_model_info(model_id: Optional[str] = None, 
                         model_type: Optional[str] = None,
                         version: Optional[str] = None) -> Dict[str, Any]:
    """
    Creates mock model information dictionary.
    
    Args:
        model_id: The model ID (default: 'model_001')
        model_type: The model type (default: 'xgboost')
        version: The model version (default: '1.0.0')
    
    Returns:
        A dictionary containing mock model information
    """
    model_id = model_id or "model_001"
    model_type = model_type or "xgboost"
    version = version or "1.0.0"
    
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "model_type": model_type,
        "version": version,
        "creation_date": (datetime.now() - timedelta(days=7)).isoformat(),
        "performance_metrics": {
            "auc": 0.82,
            "brier_score": 0.11,
            "precision": 0.76,
            "recall": 0.71,
            "f1_score": 0.73
        },
        "training_data": {
            "start_date": (datetime.now() - timedelta(days=365)).date().isoformat(),
            "end_date": (datetime.now() - timedelta(days=7)).date().isoformat(),
            "nodes": ["HB_NORTH"],
            "thresholds": [100.0],
            "row_count": 8760  # 365 days * 24 hours
        },
        "feature_names": [
            "rolling_price_max_24h", 
            "hour_of_day", 
            "load_forecast", 
            "day_of_week", 
            "temperature_forecast", 
            "wind_forecast"
        ],
        "feature_importance": {
            "rolling_price_max_24h": 0.32,
            "hour_of_day": 0.24,
            "load_forecast": 0.21,
            "day_of_week": 0.13,
            "temperature_forecast": 0.08,
            "wind_forecast": 0.02
        },
        "model_path": f"/models/{model_type}/{model_id}_v{version}.joblib"
    }
    
    # Add model-specific hyperparameters
    if model_type == "xgboost":
        result["hyperparameters"] = {
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 200
        }
    elif model_type == "lightgbm":
        result["hyperparameters"] = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "n_estimators": 200
        }
    elif model_type == "random_forest":
        result["hyperparameters"] = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt"
        }
    
    return result


def create_mock_forecast(thresholds: Optional[List[ThresholdValue]] = None,
                       nodes: Optional[List[NodeID]] = None,
                       horizon: Optional[int] = None) -> Dict[str, Any]:
    """
    Creates a mock forecast result dictionary.
    
    Args:
        thresholds: List of price thresholds (default: [100.0])
        nodes: List of node IDs (default: ['HB_NORTH'])
        horizon: Forecast horizon in hours (default: 72)
    
    Returns:
        A dictionary containing mock forecast results
    """
    thresholds = thresholds or [100.0]
    nodes = nodes or ["HB_NORTH"]
    horizon = horizon or 72
    
    forecast_timestamp = datetime.now()
    start_date = forecast_timestamp + timedelta(hours=1)
    
    result = {
        "forecast_timestamp": forecast_timestamp.isoformat(),
        "horizon_start": start_date.isoformat(),
        "horizon_end": (start_date + timedelta(hours=horizon-1)).isoformat(),
        "model_id": "model_001",
        "model_version": "1.0.0",
        "thresholds": thresholds,
        "nodes": nodes,
        "forecasts": {}
    }
    
    # Create forecasts for each threshold and node
    for threshold in thresholds:
        result["forecasts"][str(threshold)] = {}
        for node in nodes:
            # Create DataFrame with hourly forecasts
            df = create_mock_dataframe("forecast", horizon, None, start_date)
            # Filter to relevant columns and convert to dict
            forecast_data = df.loc[:, ["spike_probability", "confidence_interval_lower", "confidence_interval_upper"]]
            # Convert to records format for easy JSON serialization
            result["forecasts"][str(threshold)][node] = forecast_data.reset_index().to_dict(orient="records")
    
    return result


def create_mock_error(error_type: str, message: Optional[str] = None) -> Dict[str, str]:
    """
    Creates a mock error response.
    
    Args:
        error_type: The type of error (validation, data_fetch, model, inference, system)
        message: Custom error message (default: None)
    
    Returns:
        A dictionary containing the mock error response
    """
    if error_type in MOCK_ERROR_RESPONSES:
        error = MOCK_ERROR_RESPONSES[error_type].copy()
        if message:
            error["message"] = message
        return error
    else:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": message or f"Unknown error of type: {error_type}"
        }