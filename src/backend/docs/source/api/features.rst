Features API Reference
======================

This section provides detailed documentation for the feature engineering modules of the ERCOT RTLMP spike prediction system. These modules transform raw data into model-ready features with consistent formatting, which are critical for accurate price spike prediction.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   time_features
   statistical_features
   weather_features
   market_features
   feature_registry
   feature_selection
   feature_pipeline

Feature Engineering Overview
---------------------------

The feature engineering system is organized into several modules, each responsible for a specific category of features. The system includes time-based features, statistical features derived from price data, weather-related features, and market-specific features. These are managed through a central feature registry and can be processed through a unified feature pipeline.

Time Features
------------

The time_features module extracts temporal patterns from timestamp data, creating features such as hour of day, day of week, month, season, and holiday indicators that capture cyclical patterns in electricity prices.

.. automodule:: backend.features.time_features
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Features
------------------

The statistical_features module generates statistical measures from RTLMP time series data, including rolling statistics, volatility metrics, and price spike indicators that capture the statistical properties of price movements.

.. automodule:: backend.features.statistical_features
   :members:
   :undoc-members:
   :show-inheritance:

Weather Features
--------------

The weather_features module transforms weather data into predictive features, including temperature, wind, solar irradiance, and humidity features that capture the impact of weather conditions on electricity demand and supply.

.. automodule:: backend.features.weather_features
   :members:
   :undoc-members:
   :show-inheritance:

Market Features
-------------

The market_features module creates features based on electricity market conditions, including congestion components, grid conditions, reserve margins, and generation mix that capture market dynamics affecting price formation.

.. automodule:: backend.features.market_features
   :members:
   :undoc-members:
   :show-inheritance:

Feature Registry
--------------

The feature_registry module maintains a catalog of all available features with their properties, data types, valid ranges, and relationships, ensuring consistent feature definitions across training and inference processes.

.. automodule:: backend.features.feature_registry
   :members:
   :undoc-members:
   :show-inheritance:

Feature Selection
---------------

The feature_selection module implements various feature selection techniques including importance-based, correlation-based, and model-based selection to identify the most predictive features for RTLMP spike prediction.

.. automodule:: backend.features.feature_selection
   :members:
   :undoc-members:
   :show-inheritance:

Feature Pipeline
--------------

The feature_pipeline module orchestrates the complete feature engineering process, providing a unified interface for transforming raw data into model-ready features with consistent formatting.

.. automodule:: backend.features.feature_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Indices and tables
================

* :ref:`genindex`
* :ref:`modindex`