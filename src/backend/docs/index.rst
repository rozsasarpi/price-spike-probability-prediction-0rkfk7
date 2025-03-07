================================================
ERCOT RTLMP Spike Prediction System Documentation
================================================

Introduction
------------

Welcome to the documentation for the ERCOT RTLMP Spike Prediction System.
This system forecasts the probability of price spikes in the Real-Time Locational
Marginal Price (RTLMP) market before day-ahead market closure, enabling more
informed bidding strategies for battery storage operators.

System Overview
---------------

The ERCOT RTLMP spike prediction system is designed to predict the probability
of seeing at least one 5-minute RTLMP greater than a threshold value over a
particular hour. The system provides a 72-hour forecast horizon with hourly
probabilities and is retrained every second day to maintain accuracy.

Architecture
------------

The system follows a modular architecture with separate components for data
fetching, feature engineering, model training, and inference. This modular
design ensures maintainability, testability, and the ability to evolve
individual components independently.

Installation
------------

Instructions for installing and setting up the ERCOT RTLMP spike prediction
system, including dependencies and configuration.

Usage
-----

Guidelines for using the system, including command-line interface,
configuration options, and common workflows.

API Reference
-------------

Comprehensive documentation of all modules, classes, and functions in the system.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   source/api/data
   source/api/features
   source/api/models
   source/api/inference
   source/api/backtesting
   source/api/visualization
   source/api/utils
   source/api/orchestration

User Guides
-----------

Detailed guides for specific use cases and workflows.

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   source/guides/data_fetching
   source/guides/feature_engineering
   source/guides/model_training
   source/guides/inference
   source/guides/backtesting
   source/guides/visualization

Development
-----------

Information for developers contributing to the system.

.. toctree::
   :maxdepth: 2
   :caption: Development

   source/development/contributing
   source/development/testing
   source/development/code_style
   source/development/documentation

Indices and Tables
------------------

Useful indices and search functionality.

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`