.. _backtesting:

===========
Backtesting
===========

Module Overview
--------------

The backtesting module provides a comprehensive framework for evaluating model performance by simulating 
historical forecasts over user-specified time windows. This allows data scientists and energy scientists 
to validate model quality, compare different model configurations, and understand how models would have 
performed under historical market conditions before deployment.

In the context of ERCOT RTLMP spike prediction, backtesting is critical for:

* Validating model accuracy across different time periods and market conditions
* Evaluating the calibration of probability forecasts
* Comparing the performance of different model architectures or hyperparameters
* Understanding how model predictions would have influenced battery storage optimization decisions

The module is designed with a flexible architecture that supports various backtesting scenarios, metrics 
calculations, and result visualizations.

Related modules:
    * :ref:`model-training` - Model training components integrated with backtesting
    * :ref:`inference` - Inference engine used in backtesting simulations
    * :ref:`visualization` - Visualization tools for backtesting results

Framework Classes
----------------

BacktestingFramework
^^^^^^^^^^^^^^^^^^^

.. py:class:: BacktestingFramework

   The primary class for configuring and executing backtesting scenarios.

   .. py:method:: __init__(data_fetcher=None, feature_engineer=None, model_registry=None)

      Initialize the backtesting framework with component dependencies.

      :param data_fetcher: Component for retrieving historical data
      :type data_fetcher: DataFetcher, optional
      :param feature_engineer: Component for generating features
      :type feature_engineer: FeatureEngineer, optional
      :param model_registry: Component for accessing models
      :type model_registry: ModelRegistry, optional

   .. py:method:: run_backtest(scenario)

      Executes a backtest based on the provided scenario configuration.

      :param scenario: The scenario configuration
      :type scenario: ScenarioConfig
      :return: Results of the backtest execution
      :rtype: BacktestingResult
      
      Example::
      
          from ercot_rtlmp.backtesting import BacktestingFramework, ScenarioConfig
          
          # Initialize the framework
          framework = BacktestingFramework()
          
          # Create a scenario configuration
          scenario = ScenarioConfig(
              start_date="2022-01-01",
              end_date="2022-12-31",
              thresholds=[100.0, 200.0],
              nodes=["HB_NORTH", "HB_SOUTH"]
          )
          
          # Run the backtest
          results = framework.run_backtest(scenario)

   .. py:method:: run_multiple_backtests(scenarios)

      Executes multiple backtests based on a list of scenario configurations.

      :param scenarios: List of scenario configurations to run
      :type scenarios: list[ScenarioConfig]
      :return: List of backtest results
      :rtype: list[BacktestingResult]

   .. py:method:: compare_models(model_ids, scenario)

      Run the same scenario for multiple models and compare their performance.

      :param model_ids: List of model IDs to compare
      :type model_ids: list[str]
      :param scenario: The scenario configuration
      :type scenario: ScenarioConfig
      :return: Comparison results for the models
      :rtype: ModelComparisonResult

   .. py:method:: save_results(result, output_path)

      Save backtest results to a file.

      :param result: Backtest results to save
      :type result: BacktestingResult
      :param output_path: Path where to save the results
      :type output_path: str

   .. py:method:: load_results(input_path)

      Load backtest results from a file.

      :param input_path: Path to load results from
      :type input_path: str
      :return: Loaded backtest results
      :rtype: BacktestingResult

BacktestingResult
^^^^^^^^^^^^^^^^

.. py:class:: BacktestingResult

   Container for the results of a backtesting run.

   .. py:method:: __init__(scenario, predictions, actuals, metrics)

      Initialize a backtest result.

      :param scenario: The scenario configuration used
      :type scenario: ScenarioConfig
      :param predictions: DataFrame of predicted probabilities
      :type predictions: pandas.DataFrame
      :param actuals: DataFrame of actual outcomes
      :type actuals: pandas.DataFrame
      :param metrics: Dictionary of calculated performance metrics
      :type metrics: dict

   .. py:method:: get_metrics()

      Get the performance metrics from the backtest.

      :return: Dictionary of performance metrics
      :rtype: dict

   .. py:method:: get_predictions()

      Get the predictions DataFrame.

      :return: DataFrame of predicted probabilities
      :rtype: pandas.DataFrame

   .. py:method:: get_actuals()

      Get the actuals DataFrame.

      :return: DataFrame of actual outcomes
      :rtype: pandas.DataFrame

   .. py:method:: get_scenario()

      Get the scenario configuration used for this backtest.

      :return: The scenario configuration
      :rtype: ScenarioConfig

   .. py:method:: to_dict()

      Convert the backtest result to a dictionary representation.

      :return: Dictionary representation of the result
      :rtype: dict

   .. py:method:: from_dict(data)

      Create a BacktestingResult instance from a dictionary.

      :param data: Dictionary representation of a backtest result
      :type data: dict
      :return: BacktestingResult instance
      :rtype: BacktestingResult

   .. py:method:: generate_report(output_path=None, include_plots=True)

      Generate a comprehensive backtest report.

      :param output_path: Path to save the report
      :type output_path: str, optional
      :param include_plots: Whether to include plots in the report
      :type include_plots: bool
      :return: Report content or path to saved report
      :rtype: str

Scenario Configuration Classes
-----------------------------

ScenarioConfig
^^^^^^^^^^^^^

.. py:class:: ScenarioConfig

   Configuration for a backtesting scenario.

   .. py:method:: __init__(start_date, end_date, thresholds, nodes, model_config=None, metrics_config=None)

      Initialize a scenario configuration.

      :param start_date: Start date for the backtest period
      :type start_date: str or datetime.datetime
      :param end_date: End date for the backtest period
      :type end_date: str or datetime.datetime
      :param thresholds: Price thresholds for spike definition
      :type thresholds: list[float]
      :param nodes: List of node locations
      :type nodes: list[str]
      :param model_config: Model configuration
      :type model_config: ModelConfig, optional
      :param metrics_config: Metrics configuration
      :type metrics_config: MetricsConfig, optional
      
      Example::
      
          from ercot_rtlmp.backtesting import ScenarioConfig
          
          # Create a basic scenario configuration
          scenario = ScenarioConfig(
              start_date="2022-01-01",
              end_date="2022-12-31",
              thresholds=[100.0, 200.0],
              nodes=["HB_NORTH", "HB_SOUTH"]
          )

   .. py:method:: validate()

      Validate the scenario configuration.

      :return: True if configuration is valid
      :rtype: bool
      :raises: ValueError if configuration is invalid

   .. py:method:: to_dict()

      Convert the scenario configuration to a dictionary.

      :return: Dictionary representation
      :rtype: dict

   .. py:method:: from_dict(data)

      Create a ScenarioConfig instance from a dictionary.

      :param data: Dictionary representation
      :type data: dict
      :return: ScenarioConfig instance
      :rtype: ScenarioConfig

   .. py:method:: get_time_windows()

      Generate the time windows for the scenario.

      :return: List of (start, end) datetime tuples
      :rtype: list[tuple]

ModelConfig
^^^^^^^^^^

.. py:class:: ModelConfig

   Configuration for models used in backtesting.

   .. py:method:: __init__(model_id=None, model_version=None, model_type=None, hyperparameters=None)

      Initialize a model configuration.

      :param model_id: Identifier for the model
      :type model_id: str, optional
      :param model_version: Version of the model
      :type model_version: str, optional
      :param model_type: Type of model (e.g., "xgboost", "lightgbm")
      :type model_type: str, optional
      :param hyperparameters: Model hyperparameters
      :type hyperparameters: dict, optional

   .. py:method:: validate()

      Validate the model configuration.

      :return: True if configuration is valid
      :rtype: bool
      :raises: ValueError if configuration is invalid

   .. py:method:: to_dict()

      Convert the model configuration to a dictionary.

      :return: Dictionary representation
      :rtype: dict

   .. py:method:: from_dict(data)

      Create a ModelConfig instance from a dictionary.

      :param data: Dictionary representation
      :type data: dict
      :return: ModelConfig instance
      :rtype: ModelConfig

MetricsConfig
^^^^^^^^^^^^

.. py:class:: MetricsConfig

   Configuration for metrics used in backtesting.

   .. py:method:: __init__(metrics=None, custom_metrics=None)

      Initialize a metrics configuration.

      :param metrics: List of standard metrics to calculate
      :type metrics: list[str], optional
      :param custom_metrics: Dictionary of custom metric functions
      :type custom_metrics: dict, optional

   .. py:method:: validate()

      Validate the metrics configuration.

      :return: True if configuration is valid
      :rtype: bool
      :raises: ValueError if configuration is invalid

   .. py:method:: to_dict()

      Convert the metrics configuration to a dictionary.

      :return: Dictionary representation
      :rtype: dict

   .. py:method:: from_dict(data)

      Create a MetricsConfig instance from a dictionary.

      :param data: Dictionary representation
      :type data: dict
      :return: MetricsConfig instance
      :rtype: MetricsConfig

   .. py:method:: get_metric_functions()

      Get the metric functions to be used.

      :return: Dictionary mapping metric names to functions
      :rtype: dict

Metrics Calculation Classes
--------------------------

BacktestingMetricsCalculator
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: BacktestingMetricsCalculator

   Class for calculating performance metrics in backtesting.

   .. py:method:: __init__(metrics_config=None)

      Initialize a metrics calculator.

      :param metrics_config: Configuration for metrics calculation
      :type metrics_config: MetricsConfig, optional

   .. py:method:: calculate_metrics(predictions, actuals)

      Calculate metrics based on predictions and actual values.

      :param predictions: DataFrame of predicted probabilities
      :type predictions: pandas.DataFrame
      :param actuals: DataFrame of actual outcomes
      :type actuals: pandas.DataFrame
      :return: Dictionary of calculated metrics
      :rtype: dict
      
      Example::
      
          from ercot_rtlmp.backtesting import BacktestingMetricsCalculator
          
          # Initialize calculator with default metrics
          calculator = BacktestingMetricsCalculator()
          
          # Calculate metrics
          metrics = calculator.calculate_metrics(predictions_df, actuals_df)
          
          print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
          print(f"Brier Score: {metrics['brier_score']:.3f}")

   .. py:method:: calculate_threshold_metrics(predictions, actuals, threshold)

      Calculate metrics for a specific threshold.

      :param predictions: DataFrame of predicted probabilities
      :type predictions: pandas.DataFrame
      :param actuals: DataFrame of actual outcomes
      :type actuals: pandas.DataFrame
      :param threshold: Threshold value
      :type threshold: float
      :return: Dictionary of calculated metrics
      :rtype: dict

   .. py:method:: calculate_node_metrics(predictions, actuals, node)

      Calculate metrics for a specific node.

      :param predictions: DataFrame of predicted probabilities
      :type predictions: pandas.DataFrame
      :param actuals: DataFrame of actual outcomes
      :type actuals: pandas.DataFrame
      :param node: Node identifier
      :type node: str
      :return: Dictionary of calculated metrics
      :rtype: dict

   .. py:method:: calculate_calibration_curve(predictions, actuals, bins=10)

      Calculate calibration curve data.

      :param predictions: DataFrame of predicted probabilities
      :type predictions: pandas.DataFrame
      :param actuals: DataFrame of actual outcomes
      :type actuals: pandas.DataFrame
      :param bins: Number of bins for calibration curve
      :type bins: int
      :return: Tuple of (fraction_of_positives, mean_predicted_value)
      :rtype: tuple

   .. py:method:: calculate_roc_curve(predictions, actuals)

      Calculate ROC curve data.

      :param predictions: DataFrame of predicted probabilities
      :type predictions: pandas.DataFrame
      :param actuals: DataFrame of actual outcomes
      :type actuals: pandas.DataFrame
      :return: Tuple of (fpr, tpr, thresholds)
      :rtype: tuple

   .. py:method:: calculate_precision_recall_curve(predictions, actuals)

      Calculate precision-recall curve data.

      :param predictions: DataFrame of predicted probabilities
      :type predictions: pandas.DataFrame
      :param actuals: DataFrame of actual outcomes
      :type actuals: pandas.DataFrame
      :return: Tuple of (precision, recall, thresholds)
      :rtype: tuple

Simulation Classes
-----------------

HistoricalSimulator
^^^^^^^^^^^^^^^^^^

.. py:class:: HistoricalSimulator

   Class for simulating historical forecasts.

   .. py:method:: __init__(data_fetcher, feature_engineer, model_registry)

      Initialize a historical simulator.

      :param data_fetcher: Component for retrieving historical data
      :type data_fetcher: DataFetcher
      :param feature_engineer: Component for generating features
      :type feature_engineer: FeatureEngineer
      :param model_registry: Component for accessing models
      :type model_registry: ModelRegistry

   .. py:method:: simulate(scenario)

      Simulate forecasts for a given scenario.

      :param scenario: The scenario configuration
      :type scenario: ScenarioConfig
      :return: Simulation results
      :rtype: SimulationResult

   .. py:method:: simulate_window(start_date, end_date, model, thresholds, nodes)

      Simulate forecasts for a specific time window.

      :param start_date: Start date for the window
      :type start_date: datetime.datetime
      :param end_date: End date for the window
      :type end_date: datetime.datetime
      :param model: Model to use for prediction
      :type model: object
      :param thresholds: Price thresholds for spike definition
      :type thresholds: list[float]
      :param nodes: List of node locations
      :type nodes: list[str]
      :return: Simulation results for the window
      :rtype: SimulationResult

   .. py:method:: get_actual_spikes(start_date, end_date, thresholds, nodes)

      Get actual price spikes that occurred in a time window.

      :param start_date: Start date for the window
      :type start_date: datetime.datetime
      :param end_date: End date for the window
      :type end_date: datetime.datetime
      :param thresholds: Price thresholds for spike definition
      :type thresholds: list[float]
      :param nodes: List of node locations
      :type nodes: list[str]
      :return: DataFrame of actual spike occurrences
      :rtype: pandas.DataFrame

SimulationResult
^^^^^^^^^^^^^^^

.. py:class:: SimulationResult

   Container for results of a historical simulation.

   .. py:method:: __init__(predictions, actuals, scenario)

      Initialize a simulation result.

      :param predictions: DataFrame of predicted probabilities
      :type predictions: pandas.DataFrame
      :param actuals: DataFrame of actual outcomes
      :type actuals: pandas.DataFrame
      :param scenario: The scenario configuration used
      :type scenario: ScenarioConfig

   .. py:method:: get_predictions()

      Get the predictions DataFrame.

      :return: DataFrame of predicted probabilities
      :rtype: pandas.DataFrame

   .. py:method:: get_actuals()

      Get the actuals DataFrame.

      :return: DataFrame of actual outcomes
      :rtype: pandas.DataFrame

   .. py:method:: get_scenario()

      Get the scenario configuration used for this simulation.

      :return: The scenario configuration
      :rtype: ScenarioConfig

   .. py:method:: to_dict()

      Convert the simulation result to a dictionary representation.

      :return: Dictionary representation of the result
      :rtype: dict

   .. py:method:: from_dict(data)

      Create a SimulationResult instance from a dictionary.

      :param data: Dictionary representation of a simulation result
      :type data: dict
      :return: SimulationResult instance
      :rtype: SimulationResult

Scenario Library
---------------

ScenarioLibrary
^^^^^^^^^^^^^^^

.. py:class:: ScenarioLibrary

   Library of predefined and saved scenarios.

   .. py:method:: __init__(storage_path=None)

      Initialize a scenario library.

      :param storage_path: Path to store scenarios
      :type storage_path: str, optional

   .. py:method:: save_scenario(scenario, name)

      Save a scenario to the library.

      :param scenario: Scenario to save
      :type scenario: ScenarioConfig
      :param name: Name for the scenario
      :type name: str
      :return: Success indicator
      :rtype: bool

   .. py:method:: load_scenario(name)

      Load a scenario from the library.

      :param name: Name of the scenario to load
      :type name: str
      :return: Loaded scenario
      :rtype: ScenarioConfig

   .. py:method:: list_scenarios()

      List all available scenarios in the library.

      :return: List of scenario names
      :rtype: list[str]

   .. py:method:: delete_scenario(name)

      Delete a scenario from the library.

      :param name: Name of the scenario to delete
      :type name: str
      :return: Success indicator
      :rtype: bool

StandardScenarios
^^^^^^^^^^^^^^^^

.. py:class:: StandardScenarios

   Collection of standard backtesting scenarios.

   .. py:method:: get_standard_scenarios()

      Get all standard scenarios.

      :return: Dictionary of standard scenarios
      :rtype: dict[str, ScenarioConfig]

   .. py:method:: yearly_scenario(year, thresholds=None, nodes=None)

      Create a scenario for an entire year.

      :param year: Year to create scenario for
      :type year: int
      :param thresholds: Price thresholds (defaults to STANDARD_THRESHOLDS)
      :type thresholds: list[float], optional
      :param nodes: Nodes to include (defaults to STANDARD_NODES)
      :type nodes: list[str], optional
      :return: Scenario configuration
      :rtype: ScenarioConfig
      
      Example::
      
          from ercot_rtlmp.backtesting import StandardScenarios
          
          # Create a standard scenario for the year 2022
          standard_scenarios = StandardScenarios()
          scenario_2022 = standard_scenarios.yearly_scenario(2022)
          
          # Run backtest with the standard scenario
          framework = BacktestingFramework()
          results = framework.run_backtest(scenario_2022)

   .. py:method:: quarterly_scenario(year, quarter, thresholds=None, nodes=None)

      Create a scenario for a specific quarter.

      :param year: Year to create scenario for
      :type year: int
      :param quarter: Quarter to create scenario for (1-4)
      :type quarter: int
      :param thresholds: Price thresholds (defaults to STANDARD_THRESHOLDS)
      :type thresholds: list[float], optional
      :param nodes: Nodes to include (defaults to STANDARD_NODES)
      :type nodes: list[str], optional
      :return: Scenario configuration
      :rtype: ScenarioConfig

   .. py:method:: monthly_scenario(year, month, thresholds=None, nodes=None)

      Create a scenario for a specific month.

      :param year: Year to create scenario for
      :type year: int
      :param month: Month to create scenario for (1-12)
      :type month: int
      :param thresholds: Price thresholds (defaults to STANDARD_THRESHOLDS)
      :type thresholds: list[float], optional
      :param nodes: Nodes to include (defaults to STANDARD_NODES)
      :type nodes: list[str], optional
      :return: Scenario configuration
      :rtype: ScenarioConfig

   .. py:method:: extreme_weather_scenario(thresholds=None, nodes=None)

      Create a scenario focusing on extreme weather periods.

      :param thresholds: Price thresholds (defaults to STANDARD_THRESHOLDS)
      :type thresholds: list[float], optional
      :param nodes: Nodes to include (defaults to STANDARD_NODES)
      :type nodes: list[str], optional
      :return: Scenario configuration
      :rtype: ScenarioConfig

   .. py:method:: high_volatility_scenario(thresholds=None, nodes=None)

      Create a scenario focusing on high price volatility periods.

      :param thresholds: Price thresholds (defaults to STANDARD_THRESHOLDS)
      :type thresholds: list[float], optional
      :param nodes: Nodes to include (defaults to STANDARD_NODES)
      :type nodes: list[str], optional
      :return: Scenario configuration
      :rtype: ScenarioConfig

   .. py:method:: summer_peak_scenario(year=None, thresholds=None, nodes=None)

      Create a scenario focusing on summer peak periods.

      :param year: Year to create scenario for (defaults to latest available)
      :type year: int, optional
      :param thresholds: Price thresholds (defaults to STANDARD_THRESHOLDS)
      :type thresholds: list[float], optional
      :param nodes: Nodes to include (defaults to STANDARD_NODES)
      :type nodes: list[str], optional
      :return: Scenario configuration
      :rtype: ScenarioConfig

Utility Functions
----------------

.. py:function:: create_scenario(start_date, end_date, thresholds=None, nodes=None)

   Create a basic scenario configuration.

   :param start_date: Start date for the scenario
   :type start_date: str or datetime.datetime
   :param end_date: End date for the scenario
   :type end_date: str or datetime.datetime
   :param thresholds: Price thresholds (defaults to STANDARD_THRESHOLDS)
   :type thresholds: list[float], optional
   :param nodes: Nodes to include (defaults to STANDARD_NODES)
   :type nodes: list[str], optional
   :return: Scenario configuration
   :rtype: ScenarioConfig

   Example::
   
       from ercot_rtlmp.backtesting import create_scenario
       
       # Create a simple scenario for Q1 2022
       scenario = create_scenario(
           start_date="2022-01-01",
           end_date="2022-03-31"
       )

.. py:function:: calculate_auc_roc(predictions, actuals)

   Calculate Area Under the ROC Curve.

   :param predictions: Predicted probabilities
   :type predictions: numpy.ndarray
   :param actuals: Actual binary outcomes
   :type actuals: numpy.ndarray
   :return: AUC-ROC score
   :rtype: float

.. py:function:: calculate_brier_score(predictions, actuals)

   Calculate Brier Score for probability calibration.

   :param predictions: Predicted probabilities
   :type predictions: numpy.ndarray
   :param actuals: Actual binary outcomes
   :type actuals: numpy.ndarray
   :return: Brier score
   :rtype: float

.. py:function:: calculate_precision_recall_f1(predictions, actuals, threshold=0.5)

   Calculate precision, recall, and F1 score.

   :param predictions: Predicted probabilities
   :type predictions: numpy.ndarray
   :param actuals: Actual binary outcomes
   :type actuals: numpy.ndarray
   :param threshold: Decision threshold
   :type threshold: float
   :return: Dictionary with precision, recall, and F1 scores
   :rtype: dict

.. py:function:: visualize_backtest_results(result, output_path=None)

   Generate visualizations for backtest results.

   :param result: Backtest results
   :type result: BacktestingResult
   :param output_path: Path to save visualizations
   :type output_path: str, optional
   :return: Dictionary of figure objects or paths
   :rtype: dict

.. py:function:: compare_backtest_results(results, metric_name='auc_roc', output_path=None)

   Compare multiple backtest results.

   :param results: List of backtest results
   :type results: list[BacktestingResult]
   :param metric_name: Metric to use for comparison
   :type metric_name: str
   :param output_path: Path to save comparison
   :type output_path: str, optional
   :return: Comparison summary and visualization
   :rtype: tuple

.. py:function:: aggregate_metrics_by_threshold(result)

   Aggregate metrics by threshold value.

   :param result: Backtest results
   :type result: BacktestingResult
   :return: DataFrame of aggregated metrics
   :rtype: pandas.DataFrame

.. py:function:: aggregate_metrics_by_node(result)

   Aggregate metrics by node.

   :param result: Backtest results
   :type result: BacktestingResult
   :return: DataFrame of aggregated metrics
   :rtype: pandas.DataFrame

Constants
---------

.. py:data:: STANDARD_THRESHOLDS

   Standard price thresholds for spike definition.

   Default value: ``[50.0, 100.0, 200.0, 500.0, 1000.0]``

.. py:data:: STANDARD_NODES

   Standard nodes used in backtesting.

   Default value: ``["HB_NORTH", "HB_SOUTH", "HB_HOUSTON", "HB_WEST"]``

.. py:data:: DEFAULT_METRICS

   Default metrics used in backtesting.

   Default value: ``["auc_roc", "brier_score", "precision", "recall", "f1_score", "calibration"]``

Usage Examples
-------------

Basic Scenario
^^^^^^^^^^^^^

Example of creating and executing a basic backtesting scenario::

    from ercot_rtlmp.backtesting import BacktestingFramework, ScenarioConfig
    
    # Initialize the framework
    framework = BacktestingFramework()
    
    # Create a basic scenario
    scenario = ScenarioConfig(
        start_date="2022-01-01",
        end_date="2022-03-31",
        thresholds=[100.0, 200.0],
        nodes=["HB_NORTH", "HB_SOUTH"]
    )
    
    # Run the backtest
    result = framework.run_backtest(scenario)
    
    # Print summary metrics
    metrics = result.get_metrics()
    print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"Brier Score: {metrics['brier_score']:.3f}")
    
    # Generate and save a report
    result.generate_report(output_path="backtest_report.html")

Using Standard Scenarios
^^^^^^^^^^^^^^^^^^^^^^^

Example of using the standard scenarios library::

    from ercot_rtlmp.backtesting import BacktestingFramework, StandardScenarios
    
    # Initialize components
    framework = BacktestingFramework()
    standard_scenarios = StandardScenarios()
    
    # Create a standard scenario for summer peak in 2022
    summer_scenario = standard_scenarios.summer_peak_scenario(year=2022)
    
    # Run the backtest with the standard scenario
    result = framework.run_backtest(summer_scenario)
    
    # Print threshold-specific metrics
    metrics_by_threshold = aggregate_metrics_by_threshold(result)
    print(metrics_by_threshold)

Analyzing Results
^^^^^^^^^^^^^^^^

Example of analyzing backtesting results::

    from ercot_rtlmp.backtesting import BacktestingResult, visualize_backtest_results
    
    # Load saved results
    framework = BacktestingFramework()
    result = framework.load_results("backtesting_results.pkl")
    
    # Visualize results
    figures = visualize_backtest_results(result)
    
    # Access detailed predictions
    predictions = result.get_predictions()
    actuals = result.get_actuals()
    
    # Calculate custom metrics
    high_threshold_predictions = predictions[predictions["threshold"] == 500.0]
    high_threshold_actuals = actuals[actuals["threshold"] == 500.0]
    
    # Example of filtering by node
    north_predictions = predictions[predictions["node"] == "HB_NORTH"]
    north_actuals = actuals[actuals["node"] == "HB_NORTH"]

Comparing Multiple Models
^^^^^^^^^^^^^^^^^^^^^^^^

Example of comparing multiple models::

    from ercot_rtlmp.backtesting import BacktestingFramework, ScenarioConfig, compare_backtest_results
    
    # Initialize the framework
    framework = BacktestingFramework()
    
    # Create a scenario
    scenario = ScenarioConfig(
        start_date="2022-01-01",
        end_date="2022-12-31",
        thresholds=[100.0],
        nodes=["HB_NORTH"]
    )
    
    # Compare multiple models using the same scenario
    model_ids = ["model_v1", "model_v2", "model_v3"]
    comparison_result = framework.compare_models(model_ids, scenario)
    
    # Visualize the comparison
    summary, fig = compare_backtest_results(
        comparison_result.results, 
        metric_name="auc_roc",
        output_path="model_comparison.png"
    )
    
    print("Model Comparison Summary:")
    print(summary)