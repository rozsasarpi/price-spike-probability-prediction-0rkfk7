import pytest  # version 7.3+
import pandas as pd  # version 2.0+
import numpy as np  # version 1.24+
from datetime import datetime, timedelta  # Standard
import tempfile  # Standard
import pathlib  # Standard

from src.backend.backtesting.framework import BacktestingFramework, BacktestingResult, execute_backtesting_scenario, execute_backtesting_scenarios, compare_backtesting_results, save_backtesting_results, load_backtesting_results, visualize_backtesting_results  # src/backend/backtesting/framework.py
from src.backend.backtesting.historical_simulation import HistoricalSimulator, SimulationResult, run_historical_simulation, run_historical_simulations  # src/backend/backtesting/historical_simulation.py
from src.backend.backtesting.scenario_definitions import ScenarioConfig, ModelConfig, MetricsConfig  # src/backend/backtesting/scenario_definitions.py
from src.backend.backtesting.performance_metrics import BacktestingMetricsCalculator, calculate_backtesting_metrics  # src/backend/backtesting/performance_metrics.py
from src.backend.data.fetchers.mock import MockDataFetcher  # src/backend/data/fetchers/mock.py
from src.backend.tests.fixtures.sample_data import get_sample_rtlmp_data, get_sample_feature_data, generate_spike_labels, SAMPLE_NODES, SAMPLE_START_DATE, SAMPLE_END_DATE, PRICE_SPIKE_THRESHOLD  # src/backend/tests/fixtures/sample_data.py


@pytest.fixture
def mock_data_fetcher():
    """Pytest fixture that provides a configured MockDataFetcher instance"""
    data_fetcher = MockDataFetcher(add_noise=True, spike_probability=0.1)
    return data_fetcher


@pytest.fixture
def sample_scenario_config():
    """Pytest fixture that provides a sample ScenarioConfig for testing"""
    scenario = ScenarioConfig(name='test_scenario',
                              start_date=SAMPLE_START_DATE,
                              end_date=SAMPLE_END_DATE,
                              thresholds=[PRICE_SPIKE_THRESHOLD],
                              nodes=SAMPLE_NODES,
                              window_size=timedelta(days=1),
                              window_stride=timedelta(days=1),
                              forecast_horizon=24)
    return scenario


@pytest.fixture
def sample_model_config():
    """Pytest fixture that provides a sample ModelConfig for testing"""
    model_config = ModelConfig(model_type='xgboost',
                               retrain_per_window=True,
                               hyperparameters={'learning_rate': 0.1, 'max_depth': 5})
    return model_config


@pytest.fixture
def sample_metrics_config():
    """Pytest fixture that provides a sample MetricsConfig for testing"""
    metrics_config = MetricsConfig(metrics=['accuracy', 'precision', 'recall', 'f1', 'auc', 'brier_score'],
                                   calibration_curve=True,
                                   confusion_matrix=True,
                                   threshold_performance=True)
    return metrics_config


@pytest.fixture
def temp_output_dir():
    """Pytest fixture that provides a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = pathlib.Path(tmpdir)
        yield output_path


def test_scenario_config_validation(sample_scenario_config):
    """Tests the validation functionality of ScenarioConfig"""
    assert sample_scenario_config.validate() is True

    invalid_scenario = ScenarioConfig(name='invalid',
                                      start_date=SAMPLE_END_DATE,
                                      end_date=SAMPLE_START_DATE,
                                      thresholds=[PRICE_SPIKE_THRESHOLD],
                                      nodes=SAMPLE_NODES)
    assert invalid_scenario.validate() is False

    invalid_scenario = ScenarioConfig(name='invalid',
                                      start_date=SAMPLE_START_DATE,
                                      end_date=SAMPLE_END_DATE,
                                      thresholds=[],
                                      nodes=SAMPLE_NODES)
    assert invalid_scenario.validate() is False

    invalid_scenario = ScenarioConfig(name='invalid',
                                      start_date=SAMPLE_START_DATE,
                                      end_date=SAMPLE_END_DATE,
                                      thresholds=[PRICE_SPIKE_THRESHOLD],
                                      nodes=[])
    assert invalid_scenario.validate() is False


def test_model_config_validation(sample_model_config):
    """Tests the validation functionality of ModelConfig"""
    assert sample_model_config.validate() is True

    invalid_config = ModelConfig(model_type='')
    assert invalid_config.validate() is False

    invalid_config = ModelConfig(model_type='xgboost', retrain_per_window='abc')
    assert invalid_config.validate() is False


def test_metrics_config_validation(sample_metrics_config):
    """Tests the validation functionality of MetricsConfig"""
    assert sample_metrics_config.validate() is True

    invalid_config = MetricsConfig(metrics='abc')
    assert invalid_config.validate() is False

    invalid_config = MetricsConfig(metrics=['accuracy'], calibration_curve='abc')
    assert invalid_config.validate() is False


def test_execute_backtesting_scenario(sample_scenario_config, mock_data_fetcher, temp_output_dir):
    """Tests the execute_backtesting_scenario function"""
    result = execute_backtesting_scenario(scenario=sample_scenario_config, data_fetcher=mock_data_fetcher, output_path=temp_output_dir)
    assert isinstance(result, dict)
    assert 'scenario_name' in result
    assert 'metrics' in result
    assert 'window_results' in result
    assert PRICE_SPIKE_THRESHOLD in result['metrics']
    assert len(result['window_results']) > 0


def test_execute_backtesting_scenarios(sample_scenario_config, mock_data_fetcher, temp_output_dir):
    """Tests the execute_backtesting_scenarios function with multiple scenarios"""
    scenario2 = ScenarioConfig(name='test_scenario2',
                               start_date=SAMPLE_START_DATE + timedelta(days=1),
                               end_date=SAMPLE_END_DATE + timedelta(days=1),
                               thresholds=[PRICE_SPIKE_THRESHOLD * 2],
                               nodes=SAMPLE_NODES)
    scenarios = [sample_scenario_config, scenario2]
    result = execute_backtesting_scenarios(scenarios=scenarios, data_fetcher=mock_data_fetcher, output_path=temp_output_dir)
    assert isinstance(result, dict)
    assert sample_scenario_config.name in result
    assert scenario2.name in result
    assert 'metrics' in result[sample_scenario_config.name]
    assert 'window_results' in result[scenario2.name]

    result_parallel = execute_backtesting_scenarios(scenarios=scenarios, data_fetcher=mock_data_fetcher, output_path=temp_output_dir, parallel=True)
    assert isinstance(result_parallel, dict)
    assert sample_scenario_config.name in result_parallel
    assert scenario2.name in result_parallel
    assert 'metrics' in result_parallel[sample_scenario_config.name]
    assert 'window_results' in result_parallel[scenario2.name]


def test_backtesting_framework_class(sample_scenario_config, mock_data_fetcher, temp_output_dir):
    """Tests the BacktestingFramework class functionality"""
    framework = BacktestingFramework(data_fetcher=mock_data_fetcher, output_path=temp_output_dir)
    framework.execute_scenario(scenario=sample_scenario_config)
    assert sample_scenario_config.name in framework.get_results()

    framework.save_results()
    assert any(file.name.startswith('backtesting_results') for file in temp_output_dir.iterdir())

    framework2 = BacktestingFramework(data_fetcher=mock_data_fetcher, output_path=temp_output_dir)
    filepaths = [file for file in temp_output_dir.iterdir()]
    framework2.load_results(results_paths=filepaths)
    assert sample_scenario_config.name in framework2.get_results()

    framework.clear_results()
    assert len(framework.get_results()) == 0


def test_backtesting_result_class(sample_scenario_config, mock_data_fetcher):
    """Tests the BacktestingResult class functionality"""
    results = execute_backtesting_scenario(scenario=sample_scenario_config, data_fetcher=mock_data_fetcher)
    backtesting_result = BacktestingResult(results=results)

    assert backtesting_result.get_metric(metric='auc', threshold=PRICE_SPIKE_THRESHOLD) is not None
    assert isinstance(backtesting_result.get_metrics_dataframe(), pd.DataFrame)
    assert isinstance(backtesting_result.to_dict(), dict)

    reconstructed_result = BacktestingResult.from_dict(backtesting_result.to_dict())
    assert reconstructed_result.to_dict() == backtesting_result.to_dict()


def test_compare_backtesting_results(sample_scenario_config, mock_data_fetcher):
    """Tests the compare_backtesting_results function"""
    scenario2 = ScenarioConfig(name='test_scenario2',
                               start_date=SAMPLE_START_DATE + timedelta(days=1),
                               end_date=SAMPLE_END_DATE + timedelta(days=1),
                               thresholds=[PRICE_SPIKE_THRESHOLD * 2],
                               nodes=SAMPLE_NODES)
    results = execute_backtesting_scenarios(scenarios=[sample_scenario_config, scenario2], data_fetcher=mock_data_fetcher)
    comparison = compare_backtesting_results(results=results)
    assert isinstance(comparison, dict)
    assert any(PRICE_SPIKE_THRESHOLD in key for key in comparison.keys())
    assert any('auc' in key for key in comparison.keys())
    assert isinstance(comparison[(PRICE_SPIKE_THRESHOLD, 'auc')], pd.DataFrame)


def test_save_and_load_backtesting_results(sample_scenario_config, mock_data_fetcher, temp_output_dir):
    """Tests saving and loading backtesting results"""
    results = execute_backtesting_scenario(scenario=sample_scenario_config, data_fetcher=mock_data_fetcher)
    filepath = save_backtesting_results(results=results, output_path=temp_output_dir)
    assert filepath.exists()

    loaded_results = load_backtesting_results(results_path=filepath)
    assert isinstance(loaded_results, dict)
    # Cannot directly compare results due to non-serializable objects, but can check keys
    assert set(results.keys()) == set(loaded_results.keys())

    with pytest.raises(FileNotFoundError):
        load_backtesting_results(results_path='invalid_path')


def test_visualize_backtesting_results(sample_scenario_config, mock_data_fetcher):
    """Tests the visualization of backtesting results"""
    results = execute_backtesting_scenario(scenario=sample_scenario_config, data_fetcher=mock_data_fetcher)
    plots = visualize_backtesting_results(results=results, show_plot=False)
    assert isinstance(plots, dict)
    # TODO: Add more specific assertions about the plot types


def test_historical_simulation(sample_scenario_config, mock_data_fetcher, temp_output_dir):
    """Tests the historical simulation functionality"""
    results = run_historical_simulations(scenario=sample_scenario_config, data_fetcher=mock_data_fetcher)
    assert isinstance(results, dict)
    assert 'scenario_name' in results
    assert 'start_date' in results
    assert 'end_date' in results

    results_retrain = run_historical_simulations(scenario=sample_scenario_config, data_fetcher=mock_data_fetcher, retrain_per_window=True)
    assert isinstance(results_retrain, dict)

    results_parallel = run_historical_simulations(scenario=sample_scenario_config, data_fetcher=mock_data_fetcher, parallel=True)
    assert isinstance(results_parallel, dict)


def test_historical_simulator_class(sample_scenario_config, mock_data_fetcher, temp_output_dir):
    """Tests the HistoricalSimulator class functionality"""
    simulator = HistoricalSimulator(data_fetcher=mock_data_fetcher, output_path=temp_output_dir)
    simulator.run_simulation(scenario=sample_scenario_config)
    assert len(simulator.get_results()) > 0

    simulator.save_results()
    assert any(file.name.startswith('simulation_results') for file in temp_output_dir.iterdir())

    simulator2 = HistoricalSimulator(data_fetcher=mock_data_fetcher, output_path=temp_output_dir)
    filepaths = [file for file in temp_output_dir.iterdir()]
    simulator2.load_results(results_paths=filepaths)
    assert len(simulator2.get_results()) > 0


def test_simulation_result_class(sample_scenario_config, mock_data_fetcher):
    """Tests the SimulationResult class functionality"""
    results = run_historical_simulation(time_window=(sample_scenario_config.start_date, sample_scenario_config.end_date), scenario=sample_scenario_config, data_fetcher=mock_data_fetcher)
    simulation_result = SimulationResult(results=results)

    assert simulation_result.get_metric(metric='auc', threshold=PRICE_SPIKE_THRESHOLD) is not None
    assert isinstance(simulation_result.get_metrics_dataframe(), pd.DataFrame)
    assert isinstance(simulation_result.get_window_results(), dict)
    assert isinstance(simulation_result.to_dict(), dict)

    reconstructed_result = SimulationResult.from_dict(simulation_result.to_dict())
    assert reconstructed_result.to_dict() == simulation_result.to_dict()


def test_backtesting_metrics_calculator(sample_metrics_config):
    """Tests the BacktestingMetricsCalculator class"""
    calculator = BacktestingMetricsCalculator(metrics_config=sample_metrics_config)
    predictions = pd.DataFrame({'threshold': [0.6, 0.7, 0.8], 'actual': [0, 1, 0]})
    actuals = pd.DataFrame({'threshold': [0, 1, 0]})
    metrics = calculator.calculate_all_metrics(predictions=predictions, actuals=actuals)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    summary = calculator.get_metric_summary()
    assert isinstance(summary, pd.DataFrame)


def test_calculate_backtesting_metrics():
    """Tests the calculate_backtesting_metrics function"""
    predictions = pd.DataFrame({'threshold': [0.6, 0.7, 0.8], 'actual': [0, 1, 0]})
    actuals = pd.DataFrame({'threshold': [0, 1, 0]})
    result = calculate_backtesting_metrics(predictions=predictions, actuals=actuals)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert isinstance(result[0.6], dict)