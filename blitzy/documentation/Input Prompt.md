## the tech spec and the code must be concise!! no 100+ pages and unnecessarily verbose code please  
  
WHY - Vision & Purpose

### Purpose & Users

- What does your application do? ERCOT RTLMP spike prediction before day-ahead market closure, probability of seeing at least one 5-min granularity RTLMP greater than x over a particular hour, the model should predict each hour 

- Who will use it? data scientist and energy scientist, who work on battery storage optimization

- Why will they use it instead of alternatives? there is no alternative

## WHAT - Core Requirements

### Functional Requirements

\[What must your application do? Describe the specific behaviors and functions required.\]

- What action needs to happen? daily inference run, before day-ahead market closure, 72 hours forecast horizon, starting from the day right after the DAM closure

- What should the outcome be? 72 forecast probabilities per inference

## HOW - Planning & Implementation

- the code must be focused on the model alone, inference, and training, retrain cadence: every second day  
  write a general model that can take inputs, features, and has clearly defined hyperparameters to modify

- write modular code, mostly functions, no overbloated classes please

- python

- make type hints as specific as possible, e.g. dict type hinted down to keys and values, pandera dataframe shemas, etc.

- please avoid repeatedly typing out feature names and column names as text, we should have a standard way of handling this

- create a general data fetching interface that returns features in standard format (can be forecasts (even with overlapping forecast horizons, or actuals)

- add a general feature engineering module

- add a general model training module (with cross-validation)

- then code for backtesting, i.e. producitng forecast over a user-specific window

- add visualization and metric computing tools too

- include a small working example, use synthetic data for it, create an integration test