"""
Entry point for the backtesting module of the ERCOT RTLMP spike prediction system.
This file exposes the key components and functionality for simulating historical forecasts,
evaluating model performance, and analyzing prediction accuracy across various time periods and price thresholds.
"""

from typing import Dict, List, Optional, Union, Tuple, Any  # Standard

# Internal imports
from .framework import BacktestingFramework, BacktestingResult  # src/backend/backtesting/framework.py
from .scenario_definitions import ScenarioConfig, ModelConfig, MetricsConfig  # src/backend/backtesting/scenario_definitions.py
from .performance_metrics import BacktestingMetricsCalculator  # src/backend/backtesting/performance_metrics.py
from .historical_simulation import HistoricalSimulator, SimulationResult  # src/backend/backtesting/historical_simulation.py
from .scenarios import ScenarioLibrary, StandardScenarios, STANDARD_THRESHOLDS, STANDARD_NODES  # src/backend/backtesting/scenarios.py
from .scenario_definitions import DEFAULT_METRICS  # src/backend/backtesting/scenario_definitions.py

__version__ = "0.1.0"

__all__ = [
    "BacktestingFramework",
    "BacktestingResult",
    "ScenarioConfig",
    "ModelConfig",
    "MetricsConfig",
    "BacktestingMetricsCalculator",
    "HistoricalSimulator",
    "SimulationResult",
    "ScenarioLibrary",
    "StandardScenarios",
    "STANDARD_THRESHOLDS",
    "STANDARD_NODES",
    "DEFAULT_METRICS"
]