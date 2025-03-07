"""
Initialization module for the models package in the ERCOT RTLMP spike prediction system.
Exports key model classes, training functions, and evaluation utilities to provide a clean interface for model creation, training, and evaluation.
"""

from typing import Dict, Any  # Standard

from .base_model import BaseModel  # ./base_model
from .xgboost_model import XGBoostModel  # ./xgboost_model
from .lightgbm_model import LightGBMModel  # ./lightgbm_model
from .ensemble import EnsembleModel, EnsembleMethod  # ./ensemble
from .xgboost_model import DEFAULT_XGBOOST_PARAMS  # ./xgboost_model
from .lightgbm_model import DEFAULT_LIGHTGBM_PARAMS  # ./lightgbm_model
from .ensemble import DEFAULT_ENSEMBLE_PARAMS  # ./ensemble
from .training import create_model, train_model, train_and_evaluate, optimize_and_train, load_model, get_latest_model, compare_models, select_best_model, retrain_model, schedule_retraining, ModelTrainer  # ./training
from .evaluation import evaluate_model_performance, ModelEvaluator, ThresholdOptimizer  # ./evaluation

__version__ = "0.1.0"

# Expose key classes and functions for external use
__all__ = [
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    "EnsembleModel",
    "EnsembleMethod",
    "create_model",
    "train_model",
    "train_and_evaluate",
    "optimize_and_train",
    "load_model",
    "get_latest_model",
    "compare_models",
    "select_best_model",
    "retrain_model",
    "schedule_retraining",
    "ModelTrainer",
    "evaluate_model_performance",
    "ModelEvaluator",
    "ThresholdOptimizer",
    "DEFAULT_XGBOOST_PARAMS",
    "DEFAULT_LIGHTGBM_PARAMS",
    "DEFAULT_ENSEMBLE_PARAMS"
]