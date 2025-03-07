Visualization API
===============

The visualization module provides a comprehensive set of tools for creating static and interactive visualizations of forecasts, model performance metrics, calibration curves, and feature importance. It serves as the primary interface for visualizing the outputs of the ERCOT RTLMP spike prediction system.

Module Overview
---------------

The visualization module is organized into several submodules, each focused on a specific type of visualization:

- **Forecast Plots**: Visualizations of RTLMP spike probability forecasts
- **Performance Plots**: Visualizations of model performance metrics
- **Calibration Plots**: Visualizations of model calibration
- **Feature Importance**: Visualizations of feature importance and relationships
- **Metrics Dashboard**: Interactive dashboards for comprehensive visualization
- **Export**: Utilities for exporting visualizations to various formats

Forecast Visualization
----------------------

The forecast visualization components provide tools for visualizing RTLMP spike probability forecasts, including timeline plots, threshold comparisons, node comparisons, and heatmaps.

.. autoclass:: src.backend.visualization.forecast_plots.ForecastPlotter
   :members:
   :undoc-members:

Functions
^^^^^^^^^

Standalone functions for forecast visualization.

.. autofunction:: src.backend.visualization.forecast_plots.plot_probability_timeline
.. autofunction:: src.backend.visualization.forecast_plots.plot_threshold_comparison
.. autofunction:: src.backend.visualization.forecast_plots.plot_node_comparison
.. autofunction:: src.backend.visualization.forecast_plots.plot_heatmap
.. autofunction:: src.backend.visualization.forecast_plots.create_interactive_timeline
.. autofunction:: src.backend.visualization.forecast_plots.create_interactive_comparison
.. autofunction:: src.backend.visualization.forecast_plots.create_interactive_heatmap
.. autofunction:: src.backend.visualization.forecast_plots.create_forecast_dashboard
.. autofunction:: src.backend.visualization.forecast_plots.get_forecast_for_visualization
.. autofunction:: src.backend.visualization.forecast_plots.save_plot
.. autofunction:: src.backend.visualization.forecast_plots.save_interactive_plot

Performance Visualization
-------------------------

The performance visualization components provide tools for visualizing model performance metrics, including ROC curves, precision-recall curves, calibration curves, confusion matrices, and threshold sensitivity analysis.

.. autoclass:: src.backend.visualization.performance_plots.ModelPerformancePlotter
   :members:
   :undoc-members:

Functions
^^^^^^^^^

Standalone functions for performance visualization.

.. autofunction:: src.backend.visualization.performance_plots.plot_roc_curve
.. autofunction:: src.backend.visualization.performance_plots.plot_precision_recall_curve
.. autofunction:: src.backend.visualization.performance_plots.plot_calibration_curve
.. autofunction:: src.backend.visualization.performance_plots.plot_confusion_matrix
.. autofunction:: src.backend.visualization.performance_plots.plot_threshold_sensitivity
.. autofunction:: src.backend.visualization.performance_plots.plot_metric_comparison
.. autofunction:: src.backend.visualization.performance_plots.plot_temporal_performance
.. autofunction:: src.backend.visualization.performance_plots.create_interactive_roc_curve
.. autofunction:: src.backend.visualization.performance_plots.create_interactive_precision_recall_curve
.. autofunction:: src.backend.visualization.performance_plots.create_interactive_calibration_curve
.. autofunction:: src.backend.visualization.performance_plots.create_interactive_confusion_matrix
.. autofunction:: src.backend.visualization.performance_plots.create_interactive_threshold_sensitivity
.. autofunction:: src.backend.visualization.performance_plots.create_performance_dashboard
.. autofunction:: src.backend.visualization.performance_plots.save_performance_plot

Constants
^^^^^^^^^

Constants for performance visualization.

.. autodata:: src.backend.visualization.performance_plots.METRIC_COLORS
.. autodata:: src.backend.visualization.performance_plots.THRESHOLD_COLORS

Feature Importance Visualization
--------------------------------

The feature importance visualization components provide tools for visualizing feature importance metrics, feature relationships, and feature dependencies.

.. autoclass:: src.backend.visualization.feature_importance.FeatureImportancePlotter
   :members:
   :undoc-members:

Functions
^^^^^^^^^

Standalone functions for feature importance visualization.

.. autofunction:: src.backend.visualization.feature_importance.plot_feature_importance
.. autofunction:: src.backend.visualization.feature_importance.plot_feature_importance_history
.. autofunction:: src.backend.visualization.feature_importance.plot_feature_group_importance
.. autofunction:: src.backend.visualization.feature_importance.plot_feature_correlation_heatmap
.. autofunction:: src.backend.visualization.feature_importance.plot_feature_dependency_graph
.. autofunction:: src.backend.visualization.feature_importance.create_interactive_feature_importance
.. autofunction:: src.backend.visualization.feature_importance.create_interactive_feature_group_importance
.. autofunction:: src.backend.visualization.feature_importance.create_interactive_feature_correlation_heatmap
.. autofunction:: src.backend.visualization.feature_importance.create_interactive_feature_dependency_graph
.. autofunction:: src.backend.visualization.feature_importance.create_feature_importance_dashboard
.. autofunction:: src.backend.visualization.feature_importance.save_feature_importance_plot

Constants
^^^^^^^^^

Constants for feature importance visualization.

.. autodata:: src.backend.visualization.feature_importance.GROUP_COLORS
.. autodata:: src.backend.visualization.feature_importance.MAX_FEATURES_TO_DISPLAY

Metrics Dashboard
-----------------

The metrics dashboard components provide tools for creating interactive dashboards that combine multiple visualizations.

.. autoclass:: src.backend.visualization.metrics_dashboard.MetricsDashboard
   :members:
   :undoc-members:

Functions
^^^^^^^^^

Standalone functions for metrics dashboards.

.. autofunction:: src.backend.visualization.metrics_dashboard.create_metrics_dashboard
.. autofunction:: src.backend.visualization.metrics_dashboard.save_dashboard_to_html
.. autofunction:: src.backend.visualization.metrics_dashboard.dashboard_to_base64

Constants
^^^^^^^^^

Constants for metrics dashboards.

.. autodata:: src.backend.visualization.metrics_dashboard.DEFAULT_PORT
.. autodata:: src.backend.visualization.metrics_dashboard.DEFAULT_HOST
.. autodata:: src.backend.visualization.metrics_dashboard.DEFAULT_TITLE

Export Utilities
----------------

The export utilities provide tools for exporting visualizations to various formats, including PNG, SVG, PDF, HTML, and data formats.

.. autoclass:: src.backend.visualization.export.ExportManager
   :members:
   :undoc-members:

Functions
^^^^^^^^^

Standalone functions for exporting visualizations and data.

.. autofunction:: src.backend.visualization.export.figure_to_base64
.. autofunction:: src.backend.visualization.export.plotly_figure_to_base64
.. autofunction:: src.backend.visualization.export.dataframe_to_base64
.. autofunction:: src.backend.visualization.export.export_figure_to_file
.. autofunction:: src.backend.visualization.export.export_plotly_figure_to_file
.. autofunction:: src.backend.visualization.export.export_dataframe_to_file
.. autofunction:: src.backend.visualization.export.export_forecast_to_file
.. autofunction:: src.backend.visualization.export.export_model_performance_to_file
.. autofunction:: src.backend.visualization.export.export_backtesting_results_to_file

Constants
^^^^^^^^^

Constants for export utilities.

.. autodata:: src.backend.visualization.export.SUPPORTED_IMAGE_FORMATS
.. autodata:: src.backend.visualization.export.SUPPORTED_DATA_FORMATS

Exceptions
----------

The visualization module defines several exception classes for handling errors.

.. autoclass:: src.backend.visualization.VisualizationError
   :members:
   :undoc-members:

.. autoclass:: src.backend.visualization.export.ExportError
   :members:
   :undoc-members:

Usage Examples
--------------

Examples of how to use the visualization module.

Creating a Forecast Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rtlmp_predict.visualization import ForecastPlotter

    # Initialize the plotter
    plotter = ForecastPlotter()

    # Load the latest forecast
    plotter.load_forecast(None)  # None means latest forecast

    # Create a probability timeline plot
    fig, ax = plotter.plot_probability_timeline(threshold=100)

    # Save the plot
    plotter.save_plot(fig, 'probability_timeline.png')

Creating a Performance Dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rtlmp_predict.visualization import ModelPerformancePlotter

    # Initialize the plotter
    plotter = ModelPerformancePlotter()

    # Load model data
    plotter.load_model_data('xgboost_model', version='1.0.0')

    # Create a performance dashboard
    dashboard = plotter.create_performance_dashboard()

    # Save the dashboard
    plotter.save_plot(dashboard, 'performance_dashboard.html')

Creating a Feature Importance Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rtlmp_predict.visualization import FeatureImportancePlotter

    # Initialize the plotter
    plotter = FeatureImportancePlotter()

    # Load model data
    plotter.load_model_data('xgboost_model', version='1.0.0')

    # Create a feature importance plot
    fig, ax = plotter.plot_feature_importance(n_features=10)

    # Save the plot
    plotter.save_plot(fig, 'feature_importance.png')