# ERCOT RTLMP Spike Prediction System
## User Guide for Energy Scientists

## Table of Contents
- [Introduction](#introduction)
  - [Purpose of this Guide](#purpose-of-this-guide)
  - [System Overview](#system-overview)
  - [Key Capabilities](#key-capabilities)
- [Getting Started](#getting-started)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Using the Command Line Interface](#using-the-command-line-interface)
  - [Basic Commands](#basic-commands)
  - [Generating Forecasts](#generating-forecasts)
  - [Running Backtests](#running-backtests)
  - [Creating Visualizations](#creating-visualizations)
  - [Exporting Results](#exporting-results)
- [Interpreting Forecast Results](#interpreting-forecast-results)
  - [Understanding Probability Values](#understanding-probability-values)
  - [Confidence Intervals](#confidence-intervals)
  - [Threshold Selection](#threshold-selection)
  - [Temporal Patterns](#temporal-patterns)
- [Visualization Dashboards](#visualization-dashboards)
  - [Forecast Visualization Dashboard](#forecast-visualization-dashboard)
  - [Model Performance Dashboard](#model-performance-dashboard)
  - [Interactive Features](#interactive-features)
  - [Custom Visualizations](#custom-visualizations)
- [Integrating with Battery Optimization](#integrating-with-battery-optimization)
  - [Using Forecasts for Bidding Strategies](#using-forecasts-for-bidding-strategies)
  - [Risk Assessment](#risk-assessment)
  - [Optimization Frameworks](#optimization-frameworks)
  - [Case Studies](#case-studies)
- [Advanced Topics](#advanced-topics)
  - [Multi-Node Analysis](#multi-node-analysis)
  - [Combining with Other Data Sources](#combining-with-other-data-sources)
  - [Custom Backtesting Scenarios](#custom-backtesting-scenarios)
  - [Performance Metrics Deep Dive](#performance-metrics-deep-dive)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Error Messages](#error-messages)
  - [Performance Optimization](#performance-optimization)
  - [Getting Help](#getting-help)
- [Appendices](#appendices)
  - [Command Reference](#command-reference)
  - [Configuration File Reference](#configuration-file-reference)
  - [Glossary](#glossary)
  - [Further Reading](#further-reading)

## Introduction

Welcome to the ERCOT RTLMP Spike Prediction System, an advanced forecasting tool designed to predict the probability of price spikes in the ERCOT Real-Time Locational Marginal Price (RTLMP) market. This system provides critical insights that can significantly enhance your battery storage optimization strategies and decision-making process.

### Purpose of this Guide

This comprehensive guide is specifically designed for energy scientists working on battery storage optimization. It provides detailed instructions on how to effectively use the RTLMP spike prediction system to maximize revenue from battery assets through improved bidding strategies and operational decisions. Whether you're new to the system or looking to advance your usage, this guide offers valuable information to help you leverage the full capabilities of the forecasting tool.

### System Overview

The ERCOT RTLMP Spike Prediction System uses machine learning models to forecast the probability of seeing at least one 5-minute RTLMP greater than a specified threshold value (e.g., $100/MWh) over a particular hour. These forecasts are generated before day-ahead market closure, providing you with actionable intelligence for your bidding strategies.

The system processes historical RTLMP data, weather forecasts, and grid condition information to generate probabilistic forecasts with a 72-hour horizon. It is retrained every second day to ensure the models adapt to changing market conditions and maintain prediction accuracy.

### Key Capabilities

The RTLMP Spike Prediction System offers several key capabilities:

- **72-Hour Forecast Horizon**: Provides hourly probability predictions for the next three days
- **Customizable Price Thresholds**: Allows setting different price thresholds for spike definitions based on your specific needs
- **Probabilistic Forecasts**: Delivers probability values rather than binary predictions, enabling risk-based decision making
- **Confidence Intervals**: Includes uncertainty estimates with each forecast
- **Visualization Tools**: Offers comprehensive dashboards for analyzing forecasts and model performance
- **Backtesting Framework**: Allows evaluation of model performance on historical data
- **Multi-Node Support**: Generates predictions for multiple ERCOT nodes

## Getting Started

### System Requirements

To run the ERCOT RTLMP Spike Prediction System, your environment should meet the following requirements:

**Hardware Requirements:**
- CPU: 4+ cores recommended (8+ for training operations)
- Memory: 16GB minimum (32GB recommended)
- Storage: 100GB minimum for historical data and model artifacts

**Software Requirements:**
- Operating System: Linux, macOS, or Windows
- Python: Version 3.10 or higher
- Required Libraries:
  - pandas 2.0+
  - numpy 1.24+
  - scikit-learn 1.2+
  - XGBoost 1.7+
  - matplotlib 3.7+
  - plotly 5.14+
  - pandera 0.15+

### Installation

Follow these steps to install the ERCOT RTLMP Spike Prediction System:

1. **Create a virtual environment** (recommended):
   ```bash
   # Using conda
   conda create -n rtlmp-predict python=3.10
   conda activate rtlmp-predict
   
   # Or using venv
   python -m venv rtlmp-env
   source rtlmp-env/bin/activate  # On Windows: rtlmp-env\Scripts\activate
   ```

2. **Install the package**:
   ```bash
   pip install ercot-rtlmp-predict
   ```

3. **Verify installation**:
   ```bash
   rtlmp_predict --version
   ```

### Configuration

Before using the system, you need to set up a configuration file to specify data sources, output directories, and other settings:

1. **Create a configuration file** named `config.yaml` with the following structure:
   ```yaml
   # Data Sources
   data_sources:
     ercot_api:
       url: "https://api.example.com/ercot"
       key: "your_api_key"  # Or use environment variable: ${ERCOT_API_KEY}
     weather_api:
       url: "https://api.example.com/weather"
       key: "your_api_key"  # Or use environment variable: ${WEATHER_API_KEY}
   
   # Storage Paths
   storage:
     data_dir: "/path/to/data"
     model_dir: "/path/to/models"
     forecast_dir: "/path/to/forecasts"
   
   # Nodes Configuration
   nodes:
     - "HB_NORTH"
     - "HB_SOUTH"
     - "HB_WEST"
     - "HB_HOUSTON"
   
   # Threshold Values ($/MWh)
   thresholds:
     - 50.0
     - 100.0
     - 200.0
   
   # Forecast Configuration
   forecast:
     horizon: 72  # hours
     confidence_interval: 0.9  # 90% confidence interval
   ```

2. **Set environment variables** for API keys (optional but recommended for security):
   ```bash
   # Linux/macOS
   export ERCOT_API_KEY="your_ercot_api_key"
   export WEATHER_API_KEY="your_weather_api_key"
   
   # Windows
   set ERCOT_API_KEY=your_ercot_api_key
   set WEATHER_API_KEY=your_weather_api_key
   ```

3. **Verify configuration**:
   ```bash
   rtlmp_predict --config path/to/config.yaml validate-config
   ```

## Using the Command Line Interface

The ERCOT RTLMP Spike Prediction System provides a comprehensive command-line interface (CLI) that allows you to generate forecasts, run backtests, create visualizations, and more.

### Basic Commands

Here are the basic CLI commands to get you started:

```bash
# Display help information
rtlmp_predict --help

# Show version information
rtlmp_predict --version

# Use a specific configuration file
rtlmp_predict --config path/to/config.yaml [COMMAND]

# Enable verbose output
rtlmp_predict --verbose [COMMAND]
```

### Generating Forecasts

To generate RTLMP spike probability forecasts:

```bash
# Generate forecasts with default settings from config file
rtlmp_predict predict

# Generate forecasts for a specific node and threshold
rtlmp_predict predict --node HB_NORTH --threshold 100

# Generate forecasts for multiple nodes
rtlmp_predict predict --node HB_NORTH --node HB_SOUTH

# Generate forecasts for multiple thresholds
rtlmp_predict predict --threshold 50 --threshold 100 --threshold 200

# Specify output format
rtlmp_predict predict --output-format csv --output-file forecast.csv

# Generate forecast with a specific start date
rtlmp_predict predict --start-date 2023-07-15
```

**Example Output:**
```
Generating 72-hour forecast for node: HB_NORTH
Threshold: $100.00/MWh
Start time: 2023-07-15 00:00:00
End time: 2023-07-17 23:00:00
Model version: 2.3.1

Forecast saved to: /path/to/forecasts/forecast_20230714_HB_NORTH_100.csv
```

### Running Backtests

To evaluate model performance on historical data:

```bash
# Run backtest with default settings
rtlmp_predict backtest

# Run backtest for a specific time period
rtlmp_predict backtest --start-date 2022-01-01 --end-date 2022-12-31

# Run backtest for specific nodes and thresholds
rtlmp_predict backtest --node HB_NORTH --threshold 100

# Export detailed results
rtlmp_predict backtest --output-file backtest_results.csv --detailed-results
```

**Example Output:**
```
Running backtest for period: 2022-01-01 to 2022-12-31
Node: HB_NORTH
Threshold: $100.00/MWh
Model version: 2.3.1

Performance Metrics:
- AUC-ROC: 0.83
- Brier Score: 0.11
- Precision: 0.78
- Recall: 0.72
- F1 Score: 0.75

Results saved to: /path/to/forecasts/backtest_20220101_20221231_HB_NORTH_100.csv
```

### Creating Visualizations

To create visualizations of forecasts and model performance:

```bash
# Create forecast visualization
rtlmp_predict visualize forecast --node HB_NORTH --threshold 100

# Create performance visualization
rtlmp_predict visualize performance --start-date 2022-01-01 --end-date 2022-12-31

# Create calibration curve
rtlmp_predict visualize calibration --node HB_NORTH --threshold 100

# Create feature importance visualization
rtlmp_predict visualize features --node HB_NORTH --threshold 100

# Save visualization to file
rtlmp_predict visualize forecast --output-file forecast_viz.png
```

### Exporting Results

To export forecast results and visualizations:

```bash
# Export forecasts to CSV
rtlmp_predict predict --output-format csv --output-file forecast.csv

# Export forecasts to JSON
rtlmp_predict predict --output-format json --output-file forecast.json

# Export visualization to PNG
rtlmp_predict visualize forecast --output-format png --output-file forecast.png

# Export visualization to HTML (interactive)
rtlmp_predict visualize forecast --output-format html --output-file forecast.html

# Export performance metrics
rtlmp_predict backtest --output-format json --output-file metrics.json
```

## Interpreting Forecast Results

Understanding how to interpret the RTLMP spike probability forecasts is essential for making informed decisions about battery storage operations.

### Understanding Probability Values

The forecast probabilities represent the likelihood of seeing at least one 5-minute RTLMP greater than the specified threshold value during a particular hour. For example:

- A probability of **0.8 (80%)** for the hour ending at 2023-07-15 14:00 with a threshold of $100/MWh means there is an 80% chance that at least one of the twelve 5-minute RTLMP values in that hour will exceed $100/MWh.

- A probability of **0.2 (20%)** indicates a 20% chance of a price spike above the threshold during that hour, or conversely, an 80% chance that all prices will remain below the threshold.

These probabilities are calibrated, meaning a forecast of 80% probability should, over time, correspond to actual price spikes occurring in approximately 80% of such forecasted instances.

### Confidence Intervals

Each probability forecast includes a confidence interval, which represents the uncertainty in the prediction:

```
Hour Ending: 2023-07-15 14:00
Threshold: $100/MWh
Spike Probability: 0.80 (80%)
Confidence Interval: [0.65, 0.90]
```

This means that while the best estimate of the spike probability is 80%, the model is 90% confident (default confidence level) that the true probability lies between 65% and 90%. Wider confidence intervals indicate greater uncertainty in the prediction.

Factors that affect confidence interval width include:
- Historical data availability for similar conditions
- Weather forecast uncertainty
- Market volatility
- Temporal distance (forecasts further in the future typically have wider intervals)

### Threshold Selection

Selecting appropriate price thresholds is crucial for effective use of the forecasts:

- **Lower thresholds** (e.g., $50/MWh) capture more frequent but less extreme price events. These may be useful for regular battery cycling strategies.

- **Medium thresholds** (e.g., $100/MWh) balance frequency and magnitude of price spikes. These are often used for standard optimization scenarios.

- **Higher thresholds** (e.g., $200/MWh) identify rarer but more extreme price events. These are valuable for capturing high-value arbitrage opportunities.

Consider these factors when selecting thresholds:
- Your battery's round-trip efficiency
- Operating costs
- Opportunity costs
- Risk tolerance
- Historical price distribution at your nodes of interest

### Temporal Patterns

Analyzing temporal patterns in the forecasts can reveal valuable insights:

- **Daily patterns**: Many nodes show higher spike probabilities during morning and evening peak load periods.

- **Weekly patterns**: Weekdays often have different probability profiles compared to weekends.

- **Seasonal patterns**: Summer and winter may show higher spike probabilities due to extreme weather conditions affecting demand and supply.

- **Special events**: Grid maintenance, fuel price fluctuations, or extreme weather events may cause temporary shifts in probability patterns.

Tips for temporal pattern analysis:
- Compare forecasts across multiple days to identify recurring patterns
- Analyze historical backtests to understand seasonal variations
- Look for correlations between spike probabilities and known market drivers

## Visualization Dashboards

The ERCOT RTLMP Spike Prediction System includes comprehensive visualization dashboards that help you analyze forecasts and model performance.

### Forecast Visualization Dashboard

The Forecast Visualization Dashboard provides a graphical representation of the RTLMP spike probability forecasts:

![Forecast Dashboard](./images/forecast_dashboard_example.png)

Key components of the dashboard include:

1. **Probability Timeline**: Shows the forecasted spike probabilities across the 72-hour horizon.
   - The x-axis represents time
   - The y-axis represents probability (0-1)
   - Shaded areas indicate confidence intervals

2. **Threshold Selector**: Allows you to switch between different price thresholds.

3. **Node Selector**: Enables viewing forecasts for different ERCOT nodes.

4. **Date Range Selector**: Controls the time period displayed.

5. **Historical Performance**: Displays model performance metrics for recent periods.

To access the dashboard:

```bash
# Launch the forecast dashboard in your browser
rtlmp_predict dashboard forecast

# Specify node and threshold
rtlmp_predict dashboard forecast --node HB_NORTH --threshold 100
```

### Model Performance Dashboard

The Model Performance Dashboard provides detailed insights into model quality and performance:

![Model Performance Dashboard](./images/model_performance_dashboard_example.png)

Key components include:

1. **Calibration Curve**: Shows how well the predicted probabilities match observed frequencies.
   - A perfect calibration would follow the diagonal line
   - Points above the line indicate underestimation of probabilities
   - Points below the line indicate overestimation

2. **ROC Curve**: Illustrates the diagnostic ability of the model.
   - The area under the curve (AUC) quantifies overall performance
   - Higher AUC values indicate better performance

3. **Feature Importance**: Shows which features most strongly influence the model's predictions.
   - Longer bars indicate greater importance
   - Helps understand the key drivers of price spike probabilities

4. **Version Comparison**: Allows comparing performance across different model versions.

To access the dashboard:

```bash
# Launch the performance dashboard in your browser
rtlmp_predict dashboard performance

# Specify model version
rtlmp_predict dashboard performance --model-version v2.3.1
```

### Interactive Features

Both dashboards include interactive features to enhance your analysis:

1. **Zooming**: Click and drag to zoom into specific time periods or regions of interest.

2. **Hovering**: Hover over data points to see detailed information in tooltips.

3. **Filtering**: Use dropdown menus and date selectors to filter the displayed data.

4. **Export Options**: Export the current view as an image or the underlying data as CSV.

5. **Theme Switching**: Toggle between light and dark themes for different viewing environments.

6. **Responsive Layout**: Dashboards adjust to different screen sizes and resolutions.

### Custom Visualizations

The system also allows you to create custom visualizations for specific analysis needs:

```bash
# Create a custom multi-threshold comparison
rtlmp_predict visualize custom --type threshold-comparison --node HB_NORTH --threshold 50 --threshold 100 --threshold 200

# Create a custom node comparison
rtlmp_predict visualize custom --type node-comparison --threshold 100 --node HB_NORTH --node HB_SOUTH

# Create a custom heatmap of daily patterns
rtlmp_predict visualize custom --type daily-heatmap --node HB_NORTH --threshold 100 --days 30

# Create a custom probability distribution
rtlmp_predict visualize custom --type probability-distribution --node HB_NORTH --threshold 100
```

## Integrating with Battery Optimization

One of the primary applications of the RTLMP spike prediction system is to enhance battery storage optimization strategies.

### Using Forecasts for Bidding Strategies

The probability forecasts can directly inform your day-ahead market bidding strategies:

1. **Basic Probability Threshold Strategy**:
   - Set a probability threshold (e.g., 0.7 or 70%)
   - For hours where the spike probability exceeds this threshold, bid to discharge during those hours
   - For hours with low spike probabilities, bid to charge

2. **Expected Value Approach**:
   - Calculate expected value: `E[Value] = P(spike) * (spike_price - threshold) + (1-P(spike)) * (normal_price - threshold)`
   - Bid to discharge during hours with the highest expected values
   - Bid to charge during hours with the lowest prices

3. **Risk-Adjusted Bidding**:
   - Incorporate confidence intervals to adjust bids based on forecast uncertainty
   - For narrower confidence intervals (more certain predictions), make larger capacity commitments
   - For wider intervals (more uncertain predictions), make more conservative commitments

Example of integrating with a bidding algorithm:

```python
# Example code snippet (not for direct execution)
def optimize_battery_bids(forecast_df, battery_capacity, efficiency, risk_tolerance):
    # Sort hours by expected value
    forecast_df['expected_value'] = forecast_df['probability'] * forecast_df['historical_avg_spike_price'] + \
                                   (1 - forecast_df['probability']) * forecast_df['historical_avg_normal_price']
    
    # Adjust based on confidence interval width and risk tolerance
    forecast_df['certainty'] = 1 - (forecast_df['ci_upper'] - forecast_df['ci_lower'])
    forecast_df['risk_adjusted_value'] = forecast_df['expected_value'] * \
                                         (risk_tolerance + (1 - risk_tolerance) * forecast_df['certainty'])
    
    # Identify best hours for discharge (highest values) and charge (lowest values)
    forecast_df_sorted = forecast_df.sort_values('risk_adjusted_value', ascending=False)
    
    discharge_hours = forecast_df_sorted.head(int(battery_capacity / efficiency))
    charge_hours = forecast_df_sorted.tail(int(battery_capacity))
    
    return {
        'discharge_bids': discharge_hours[['hour', 'risk_adjusted_value']],
        'charge_bids': charge_hours[['hour', 'risk_adjusted_value']]
    }
```

### Risk Assessment

The probability forecasts enable sophisticated risk assessment for battery operations:

1. **Volatility Exposure**:
   - Higher spike probabilities indicate greater price volatility
   - Adjust position sizes based on your risk tolerance
   - Consider the width of confidence intervals as a measure of forecast uncertainty

2. **Opportunity Cost Analysis**:
   - Compare the expected value of participating in energy arbitrage versus ancillary services
   - Use spike probabilities to quantify the potential revenue from energy price spikes

3. **Risk-Return Tradeoff**:
   - Plot expected return versus risk (using confidence interval width as a risk proxy)
   - Identify optimal operating points on the risk-return curve

Example risk assessment approach:

```
For each hour h:
1. Calculate basic expected value: EV(h) = P(spike) * (estimated_spike_price) + (1-P(spike)) * (estimated_normal_price)
2. Calculate risk measure: Risk(h) = (ci_upper - ci_lower) * P(spike) * (estimated_spike_price)
3. Calculate risk-adjusted expected value: RAEV(h) = EV(h) - (risk_aversion_factor * Risk(h))
4. Rank hours by RAEV for bidding decisions
```

### Optimization Frameworks

The RTLMP spike predictions can be integrated with various battery optimization frameworks:

1. **Mixed Integer Linear Programming (MILP)**:
   - Use spike probabilities as inputs to constrained optimization problems
   - Define objective function incorporating expected revenue based on probabilities
   - Add constraints for battery capacity, power limits, and state of charge

2. **Stochastic Optimization**:
   - Use probability distributions derived from the forecasts and confidence intervals
   - Generate scenarios based on these distributions
   - Optimize across multiple scenarios to find robust solutions

3. **Dynamic Programming**:
   - Define state transitions based on battery operations and market outcomes
   - Use probabilities to weight different potential outcomes
   - Solve recursively to find optimal policies

4. **Reinforcement Learning**:
   - Use forecasts as part of the state representation
   - Train agents to make optimal charge/discharge decisions
   - Include forecast uncertainty as part of the state space

Example integration with an optimization framework:

```python
# Example code snippet (not for direct execution)
def build_optimization_model(forecast_df, battery_params):
    import pyomo.environ as pyo
    
    # Initialize model
    model = pyo.ConcreteModel()
    
    # Define time periods
    model.T = pyo.Set(initialize=forecast_df.index.tolist())
    
    # Decision variables
    model.charge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, battery_params['max_charge_rate']))
    model.discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, battery_params['max_discharge_rate']))
    model.energy_level = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, battery_params['capacity']))
    
    # Objective: maximize expected profit
    def obj_rule(model):
        return sum((forecast_df.loc[t, 'probability'] * forecast_df.loc[t, 'estimated_spike_price'] + 
                   (1 - forecast_df.loc[t, 'probability']) * forecast_df.loc[t, 'estimated_normal_price']) * 
                   model.discharge[t] - 
                   forecast_df.loc[t, 'charging_price'] * model.charge[t] for t in model.T)
    
    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    
    # Add constraints (state of charge, etc.)
    # [...]
    
    return model
```

### Case Studies

Real-world examples demonstrate the value of RTLMP spike predictions for battery optimization:

**Case Study 1: Summer Peak Management**
- **Scenario**: A 10MW/20MWh battery system in ERCOT's Houston zone during summer 2022
- **Approach**: Used spike probability forecasts to target afternoon price spikes
- **Results**: 
  - 23% increase in revenue compared to standard day-ahead price forecast strategy
  - Successfully captured 85% of significant price spike events
  - Reduced unnecessary cycling by 18%

**Case Study 2: Winter Storm Preparation**
- **Scenario**: A 5MW/10MWh battery system during an approaching cold front
- **Approach**: Used spike probability forecasts with higher thresholds to prepare for extreme events
- **Results**:
  - Identified potential extreme price spike 48 hours in advance
  - Battery was fully charged before the event
  - Single discharge during price spike of $1,500/MWh generated equivalent of two weeks of normal operation revenue

**Case Study 3: Multi-Service Optimization**
- **Scenario**: A 20MW/40MWh battery system providing both energy arbitrage and grid services
- **Approach**: Used spike probabilities to optimize allocation between markets
- **Results**:
  - 15% revenue increase through optimal service allocation
  - Better management of battery degradation
  - Improved bidding strategy for day-ahead vs. ancillary service markets

## Advanced Topics

### Multi-Node Analysis

Analyzing multiple nodes simultaneously can reveal valuable patterns and opportunities:

1. **Node Correlation Analysis**:
   - Calculate correlations between spike probabilities at different nodes
   - Identify groups of nodes that tend to spike together
   - Find nodes with complementary patterns for portfolio diversification

2. **Geographical Spread Strategy**:
   - For operators with batteries at multiple locations, optimize discharge scheduling across the portfolio
   - Prioritize dispatch at nodes with highest spike probabilities
   - Create a diversified bidding strategy across nodes

3. **Congestion Pattern Identification**:
   - Compare nodes across congestion boundaries
   - Identify patterns in probability differences that may indicate transmission constraints
   - Use these insights to predict congestion-related price separations

Example command for multi-node analysis:

```bash
# Compare spike probabilities across multiple nodes
rtlmp_predict visualize custom --type node-correlation --threshold 100 --node HB_NORTH --node HB_SOUTH --node HB_WEST --node HB_HOUSTON

# Generate geographical heatmap of probabilities
rtlmp_predict visualize custom --type geo-heatmap --threshold 100 --date 2023-07-15T14:00
```

### Combining with Other Data Sources

Enhancing the RTLMP spike predictions with additional data sources can provide more comprehensive market insights:

1. **Weather Forecast Integration**:
   - Overlay weather forecasts with spike probabilities
   - Identify correlations between weather patterns and spike probabilities
   - Create custom visualizations combining both data types

2. **Grid Condition Analysis**:
   - Combine spike probabilities with forecasted reserve margins
   - Analyze relationship between generation mix and spike probabilities
   - Identify conditions that consistently precede price spikes

3. **Fuel Price Integration**:
   - Incorporate natural gas price forecasts alongside RTLMP spike probabilities
   - Analyze how changes in fuel prices affect spike probabilities
   - Develop predictive models for how fuel price movements might shift future spike probabilities

Example of combining data sources:

```bash
# Generate visualization combining spike probabilities and weather forecast
rtlmp_predict visualize custom --type weather-overlay --node HB_HOUSTON --threshold 100

# Analyze correlation between spike probabilities and system-wide load
rtlmp_predict visualize custom --type load-correlation --node HB_NORTH --threshold 100
```

### Custom Backtesting Scenarios

Creating custom backtesting scenarios allows you to evaluate model performance under specific conditions:

1. **Weather Event Backtest**:
   - Test model performance during extreme weather events
   - Compare forecast accuracy during normal vs. extreme conditions
   - Identify potential improvements for extreme event prediction

2. **Market Change Backtest**:
   - Evaluate performance before and after significant market rule changes
   - Assess model adaptation to new market conditions
   - Identify potential need for model retraining or feature engineering

3. **Seasonal Performance Analysis**:
   - Compare model performance across different seasons
   - Identify seasonal patterns in forecast accuracy
   - Develop season-specific strategies

Example custom backtest commands:

```bash
# Run backtest specifically for summer months
rtlmp_predict backtest --start-date 2022-06-01 --end-date 2022-08-31 --node HB_HOUSTON --threshold 100 --label "Summer 2022"

# Run backtest for specific weather events
rtlmp_predict backtest --event-list extreme_weather_events.csv --node HB_NORTH --threshold 200 --window 48

# Compare performance across years
rtlmp_predict backtest --yearly-comparison --start-year 2020 --end-year 2022 --node HB_SOUTH --threshold 100
```

### Performance Metrics Deep Dive

Understanding the various performance metrics can help you better assess model quality and make more informed decisions:

1. **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**:
   - Measures the model's ability to discriminate between hours with and without price spikes
   - Values range from 0.5 (random guessing) to 1.0 (perfect discrimination)
   - A higher AUC indicates better model performance
   - Interpretation: An AUC of 0.85 means there's an 85% chance that the model will assign a higher probability to a randomly chosen hour with a spike than to a randomly chosen hour without a spike

2. **Brier Score**:
   - Measures the accuracy of probabilistic predictions
   - Values range from 0 (perfect) to 1 (worst)
   - Lower values indicate better calibration
   - Interpretation: A Brier score of 0.1 indicates well-calibrated probabilities, while 0.3 would indicate poor calibration

3. **Precision and Recall**:
   - Precision: Of the hours predicted to have spikes (above some probability threshold), what percentage actually had spikes
   - Recall: Of the hours that actually had spikes, what percentage were correctly predicted
   - F1 Score: Harmonic mean of precision and recall
   - Interpretation: High precision with low recall means the model rarely predicts spikes but is usually correct when it does; high recall with low precision means the model catches most spikes but has many false alarms

4. **Calibration Curve (Reliability Diagram)**:
   - Shows how well the predicted probabilities match observed frequencies
   - Perfectly calibrated predictions would follow the diagonal line
   - Points above the line indicate underestimation of probabilities
   - Points below the line indicate overestimation
   - Interpretation: If the model predicts a 70% chance of a spike, approximately 70% of such instances should actually experience spikes

Example commands for detailed performance analysis:

```bash
# Generate comprehensive performance metrics report
rtlmp_predict analyze performance --node HB_NORTH --threshold 100 --start-date 2022-01-01 --end-date 2022-12-31

# Create detailed calibration curve with confidence bands
rtlmp_predict visualize calibration --node HB_HOUSTON --threshold 100 --detailed

# Generate precision-recall curve with different probability thresholds
rtlmp_predict visualize custom --type precision-recall --node HB_SOUTH --threshold 200
```

## Troubleshooting

### Common Issues

**Issue 1: Forecasts Not Generating**
- **Symptoms**: Command completes but no forecast files are produced
- **Possible Causes**:
  - Configuration file path is incorrect
  - API keys are missing or invalid
  - Output directory does not exist or is not writeable
- **Solutions**:
  - Verify configuration file path with `rtlmp_predict validate-config`
  - Check API keys with `rtlmp_predict test-connection`
  - Ensure output directories exist and have write permissions

**Issue 2: Unexpected Forecast Values**
- **Symptoms**: Forecast probabilities seem unusually high or low
- **Possible Causes**:
  - Using an outdated model
  - Data quality issues in recent inputs
  - Unusual market conditions
- **Solutions**:
  - Check model version with `rtlmp_predict status`
  - Verify input data quality with `rtlmp_predict validate-data`
  - Compare with historical patterns using `rtlmp_predict visualize historical-comparison`

**Issue 3: Missing Data in Visualizations**
- **Symptoms**: Dashboards or visualizations show gaps or missing data
- **Possible Causes**:
  - Incomplete forecast runs
  - Data not available for requested date range
  - Visualization filters excluding data
- **Solutions**:
  - Check forecast completion status with `rtlmp_predict status`
  - Verify date range has available data
  - Reset filters or expand date range

**Issue 4: Performance Problems**
- **Symptoms**: Commands run slowly or timeout
- **Possible Causes**:
  - Insufficient memory
  - Large date ranges requested
  - Resource contention
- **Solutions**:
  - Reduce date range for analysis
  - Close other memory-intensive applications
  - Use the `--optimize-memory` flag for large operations

### Error Messages

| Error Message | Meaning | Solution |
|---------------|---------|----------|
| `ConfigurationError: Invalid configuration file` | The config file is missing or has invalid format | Check file path and format with `validate-config` |
| `APIError: Could not connect to ERCOT API` | Connection to ERCOT API failed | Verify API key and internet connection |
| `DataError: Missing required data for date range` | Data is incomplete for the requested dates | Try a different date range or run `fetch-data` |
| `ModelError: Failed to load model` | The requested model could not be loaded | Check model path and version |
| `ValueError: Node 'X' not found in configuration` | The specified node is not in the config | Add the node to your config file or use a supported node |
| `MemoryError: Insufficient memory for operation` | Not enough RAM to complete the operation | Use a smaller date range or add `--optimize-memory` flag |

### Performance Optimization

To optimize system performance, especially for large-scale analyses:

1. **Memory Optimization**:
   - Use the `--optimize-memory` flag for operations on large datasets
   - Limit date ranges to necessary periods
   - Close other memory-intensive applications

2. **Processing Optimization**:
   - Use the `--parallel` flag to enable multi-core processing
   - Set `--workers N` to specify the number of parallel workers
   - Schedule resource-intensive operations during off-hours

3. **Storage Optimization**:
   - Regularly clean old forecast files with `rtlmp_predict cleanup --older-than 90d`
   - Use the `--compress` flag when exporting large datasets
   - Consider using the `--low-precision` flag for faster but slightly less accurate calculations

4. **Visualization Optimization**:
   - Use the `--simplified` flag for faster rendering of visualizations
   - Set `--max-points 1000` to limit the number of points in plots
   - Use the `--static` flag to generate non-interactive visualizations for better performance

### Getting Help

If you encounter issues not covered in this guide:

1. **Check Documentation**:
   - Review the full documentation at [https://ercot-rtlmp-predict.readthedocs.io/](https://ercot-rtlmp-predict.readthedocs.io/)
   - Look for examples and tutorials in the documentation

2. **Generate Diagnostic Information**:
   ```bash
   rtlmp_predict diagnostics --output diagnostic_report.zip
   ```

3. **Contact Support**:
   - Email: support@ercot-rtlmp-predict.example.com
   - Include your diagnostic report and a detailed description of the issue

4. **Community Resources**:
   - GitHub Issues: [https://github.com/example/ercot-rtlmp-predict/issues](https://github.com/example/ercot-rtlmp-predict/issues)
   - Community Forum: [https://community.ercot-rtlmp-predict.example.com](https://community.ercot-rtlmp-predict.example.com)

## Appendices

### Command Reference

#### General Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--help` | Show help information | `rtlmp_predict --help` |
| `--version` | Show version information | `rtlmp_predict --version` |
| `--config PATH` | Specify configuration file | `rtlmp_predict --config config.yaml predict` |
| `--verbose` | Enable verbose output | `rtlmp_predict --verbose predict` |

#### Data Management Commands

| Command | Description | Example |
|---------|-------------|---------|
| `fetch-data` | Fetch data from external sources | `rtlmp_predict fetch-data --start-date 2023-01-01 --end-date 2023-01-31` |
| `validate-data` | Check data quality and completeness | `rtlmp_predict validate-data --node HB_NORTH` |
| `export-data` | Export raw or processed data | `rtlmp_predict export-data --type rtlmp --node HB_NORTH --output rtlmp_data.csv` |

#### Forecast Commands

| Command | Description | Example |
|---------|-------------|---------|
| `predict` | Generate forecasts | `rtlmp_predict predict --node HB_NORTH --threshold 100` |
| `backtest` | Run backtesting on historical data | `rtlmp_predict backtest --start-date 2022-01-01 --end-date 2022-12-31` |
| `evaluate` | Evaluate model performance | `rtlmp_predict evaluate --model-version v2.3.1` |

#### Visualization Commands

| Command | Description | Example |
|---------|-------------|---------|
| `visualize forecast` | Create forecast visualization | `rtlmp_predict visualize forecast --node HB_NORTH` |
| `visualize performance` | Create performance visualization | `rtlmp_predict visualize performance --threshold 100` |
| `visualize calibration` | Create calibration curve | `rtlmp_predict visualize calibration --node HB_SOUTH` |
| `visualize features` | Create feature importance visualization | `rtlmp_predict visualize features --threshold 100` |
| `dashboard` | Launch interactive dashboard | `rtlmp_predict dashboard forecast` |

#### System Commands

| Command | Description | Example |
|---------|-------------|---------|
| `status` | Check system status | `rtlmp_predict status` |
| `validate-config` | Validate configuration file | `rtlmp_predict validate-config --config config.yaml` |
| `cleanup` | Remove old files | `rtlmp_predict cleanup --older-than 90d` |
| `diagnostics` | Generate diagnostic information | `rtlmp_predict diagnostics --output diagnostics.zip` |

### Configuration File Reference

The configuration file (`config.yaml`) supports the following sections and options:

#### Data Sources Section

```yaml
data_sources:
  ercot_api:
    url: "https://api.example.com/ercot"
    key: "your_api_key"  # Or ${ERCOT_API_KEY}
    timeout: 60  # seconds
    retry_attempts: 3
    
  weather_api:
    url: "https://api.example.com/weather"
    key: "your_api_key"  # Or ${WEATHER_API_KEY}
    timeout: 30  # seconds
    retry_attempts: 3
    
  cache:
    enabled: true
    expiration: 24  # hours
    location: "/path/to/cache"
```

#### Storage Section

```yaml
storage:
  data_dir: "/path/to/data"
  model_dir: "/path/to/models"
  forecast_dir: "/path/to/forecasts"
  log_dir: "/path/to/logs"
  
  retention:
    raw_data: 730  # days (2 years)
    features: 730  # days (2 years)
    forecasts: 365  # days (1 year)
    logs: 90  # days
```

#### Nodes Section

```yaml
nodes:
  - id: "HB_NORTH"
    name: "ERCOT North Hub"
    enabled: true
    
  - id: "HB_SOUTH"
    name: "ERCOT South Hub"
    enabled: true
    
  - id: "HB_WEST"
    name: "ERCOT West Hub"
    enabled: false
    
  - id: "HB_HOUSTON"
    name: "ERCOT Houston Hub"
    enabled: true
```

#### Thresholds Section

```yaml
thresholds:
  - value: 50.0
    label: "Low"
    enabled: true
    
  - value: 100.0
    label: "Medium"
    enabled: true
    
  - value: 200.0
    label: "High"
    enabled: true
```

#### Forecast Section

```yaml
forecast:
  horizon: 72  # hours
  confidence_interval: 0.9  # 90% confidence interval
  update_frequency: 24  # hours
  
  model:
    version: "latest"  # or specific version like "v2.3.1"
    fallback_version: "v2.2.0"  # version to use if latest fails
    
  execution:
    timeout: 300  # seconds
    memory_optimization: false
    parallel_workers: 4
```

#### Visualization Section

```yaml
visualization:
  default_theme: "light"  # or "dark"
  color_scheme: "default"  # or "colorblind"
  interactive_default: true
  max_points: 5000
  
  dashboards:
    port: 8050
    host: "127.0.0.1"
    auto_refresh: 3600  # seconds
```

### Glossary

| Term | Definition |
|------|------------|
| RTLMP | Real-Time Locational Marginal Price - The price of energy at a specific location in the grid in real-time (5-minute intervals in ERCOT) |
| Price Spike | A sudden, significant increase in RTLMP, typically defined as exceeding a specific threshold value |
| Day-Ahead Market (DAM) | Market where electricity is purchased for delivery the following day |
| DAM Closure | The deadline after which no more bids can be submitted to the day-ahead market |
| Forecast Horizon | The time period into the future for which predictions are made (72 hours in this system) |
| Battery Storage Optimization | The process of determining optimal charging/discharging schedules for battery assets to maximize revenue |
| Feature Engineering | The process of transforming raw data into inputs suitable for machine learning models |
| Cross-Validation | A technique to assess model performance by training and testing on different data subsets |
| Backtesting | The process of testing a predictive model on historical data to evaluate performance |
| Calibration | The alignment between predicted probabilities and observed frequencies of events |
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve - A performance metric for classification problems |
| Brier Score | A measure of the accuracy of probabilistic predictions |
| Confidence Interval | A range of values that likely contains the true value of a parameter |
| Round-Trip Efficiency | The percentage of energy retrieved from storage relative to the energy used to charge it |
| Node | A specific location in the ERCOT grid where prices are calculated |

### Further Reading

#### ERCOT Markets
- [ERCOT Market Information](http://www.ercot.com/mktinfo)
- "Understanding ERCOT's Market Design" by Potomac Economics
- "The Changing Texas Electricity Market" by Joshua D. Rhodes

#### Battery Storage Optimization
- "Battery Energy Storage Systems for Ancillary Services Provision" by Praveen Sharma
- "Optimal Bidding Strategy for Battery Storage Systems in Day-Ahead Energy Markets" by Matthew Chagnon
- "Energy Arbitrage and Battery Storage Optimization" by John B. Goodenough

#### Machine Learning for Energy Forecasting
- "Probabilistic Electricity Price Forecasting" by Rafał Weron
- "Machine Learning for Energy Markets" by Katherine Dykes
- "Time Series Forecasting for Energy Markets" by Peter J. Brockwell and Richard A. Davis

#### Risk Management in Energy Markets
- "Energy Risk Management" by Paul Ekins
- "Managing Price Risk in Volatile Energy Markets" by Vincent Kaminski
- "Financial Risk Management in Electricity Markets" by Fred Espen Benth

#### Online Resources
- [ERCOT Real-Time Market Information](http://www.ercot.com/content/cdr/html/real_time_spp.html)
- [Battery Storage Integration Guide](https://www.nrel.gov/docs/fy19osti/74426.pdf)
- [Energy Storage Association Resources](https://energystorage.org/resources/)