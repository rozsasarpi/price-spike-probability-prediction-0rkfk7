=====
Data API
=====

Overview
========

The Data API module provides a comprehensive set of components for retrieving, validating, and storing data used in the ERCOT RTLMP spike prediction system. This module forms the foundation of the data pipeline, ensuring reliable access to ERCOT market data, weather forecasts, and grid condition metrics required for feature engineering and model training.

The module is designed with a focus on reliability, data quality, and efficient data management, implementing standardized interfaces for each component to ensure consistent operation throughout the system.

Data Fetchers
============

The data fetchers provide a standardized interface for retrieving data from external sources, including the ERCOT API and weather forecast services. They handle connection management, error recovery, and data formatting to ensure consistent data structures for downstream processing.

Base Data Fetcher
----------------

.. autoclass:: backend.data.fetchers.BaseDataFetcher
   :members:
   :undoc-members:
   :show-inheritance:

ERCOT Data Fetcher
-----------------

.. autoclass:: backend.data.fetchers.ERCOTDataFetcher
   :members:
   :undoc-members:
   :show-inheritance:

Weather Data Fetcher
------------------

.. autoclass:: backend.data.fetchers.WeatherAPIFetcher
   :members:
   :undoc-members:
   :show-inheritance:

Data Validators
==============

Data validators ensure the quality and consistency of data used in the prediction system. These components provide schema validation, range checking, and data integrity verification to prevent invalid data from affecting model training and inference.

JSON Schema Validators
--------------------

.. autofunction:: backend.data.validators.validate_rtlmp_json
.. autofunction:: backend.data.validators.validate_weather_json
.. autofunction:: backend.data.validators.validate_grid_condition_json

.. autoclass:: backend.data.validators.JSONSchemaValidator
   :members:
   :undoc-members:
   :show-inheritance:

Pandera Schema Validators
-----------------------

.. autoclass:: backend.data.validators.WeatherSchema
   :members:
   :undoc-members:
   :show-inheritance:

Data Storage
===========

The data storage components manage persistent storage of data in optimized formats, ensuring efficient access for model training and inference. The system uses Parquet files for time series data storage due to their columnar format and compression capabilities.

Parquet Store
-----------

.. autoclass:: backend.data.storage.ParquetStore
   :members:
   :undoc-members:
   :show-inheritance:

Storage Utilities
---------------

.. autofunction:: backend.data.storage.write_dataframe_to_parquet
.. autofunction:: backend.data.storage.read_dataframe_from_parquet
.. autofunction:: backend.data.storage.find_parquet_files

Data Types
=========

The data types module defines the structure and schema for different data sources used in the system. These definitions ensure consistent data handling throughout the pipeline.

RTLMP Data
---------

.. autodata:: backend.data.types.RTLMPDataDict
.. autodata:: backend.data.types.RTLMP_SCHEMA

Weather Data
----------

.. autodata:: backend.data.types.WeatherDataDict
.. autodata:: backend.data.types.WEATHER_SCHEMA

Grid Condition Data
-----------------

.. autodata:: backend.data.types.GridConditionDict
.. autodata:: backend.data.types.GRID_CONDITION_SCHEMA

Examples
========

Fetching RTLMP Data
------------------

.. code-block:: python

   from backend.data.fetchers import ERCOTDataFetcher
   
   fetcher = ERCOTDataFetcher()
   start_date = datetime(2023, 1, 1)
   end_date = datetime(2023, 1, 7)
   nodes = ['HB_NORTH', 'HB_SOUTH']
   
   rtlmp_data = fetcher.fetch_historical_data(start_date, end_date, nodes)

Storing Data in Parquet Format
----------------------------

.. code-block:: python

   from backend.data.storage import ParquetStore
   
   store = ParquetStore()
   store.store_rtlmp_data(rtlmp_data, validate=True)

Validating Data
-------------

.. code-block:: python

   from backend.data.validators import validate_rtlmp_json
   
   data = {...}  # RTLMP data in dictionary format
   is_valid = validate_rtlmp_json(data, raise_error=False)