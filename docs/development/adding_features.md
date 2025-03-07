# Adding Features to the ERCOT RTLMP Spike Prediction System

## Introduction

This guide provides detailed instructions for developers who want to add new features to the ERCOT RTLMP spike prediction system. Features are critical inputs to the prediction models that forecast the probability of price spikes in the Real-Time Locational Marginal Price (RTLMP) market. Well-designed features can significantly improve model performance and prediction accuracy.

Feature engineering is a crucial step in the machine learning pipeline, where raw data is transformed into meaningful inputs (features) that enable models to identify patterns and make predictions. For the ERCOT RTLMP spike prediction system, effective features capture the underlying factors that influence electricity price volatility and spikes.

## Feature Engineering Architecture

The feature engineering system consists of several key components: feature modules for different categories, a feature registry for metadata management, a feature pipeline for orchestration, and feature selection mechanisms. All new features must be properly integrated with these components to ensure consistency and maintainability.

### Feature Pipeline

The feature pipeline is responsible for transforming raw data into model-ready features through a series of processing steps:

1. **Data Loading**: Retrieves raw data from various sources
2. **Data Cleaning**: Handles missing values, outliers, and inconsistencies
3. **Feature Extraction**: Applies transformations to create individual features
4. **Feature Selection**: Filters and prioritizes features based on importance criteria
5. **Feature Scaling**: Normalizes or standardizes features as needed
6. **Feature Output**: Produces a finalized feature set for model training or inference

### Component Interaction

The feature engineering components interact with other system components as follows:

- **Data Fetcher**: Provides raw data for feature generation
- **Model Trainer**: Consumes engineered features for model training
- **Inference Engine**: Uses consistent feature generation for predictions
- **Feature Registry**: Maintains metadata about all available features

## Feature Categories

The system supports four main categories of features:

1. **Time-based features**: Derived from timestamps, capturing temporal patterns like hour of day, day of week, seasonality, and holidays
2. **Statistical features**: Calculated from historical RTLMP data, including rolling statistics, volatility measures, and threshold-based indicators
3. **Weather features**: Derived from weather forecast data, including temperature, wind, solar irradiance, and their interactions
4. **Market features**: Based on ERCOT grid conditions, generation mix, load forecasts, and other market indicators

### Time-based Features

Time-based features capture temporal patterns in electricity demand and pricing. These features help the model understand:

- Daily patterns (peak vs. off-peak hours)
- Weekly patterns (weekdays vs. weekends)
- Monthly and seasonal patterns
- Holiday effects
- Daylight saving time transitions

Examples include:
- `hour_of_day`: Hour from the timestamp (0-23)
- `day_of_week`: Day of the week (0-6, where 0 is Monday)
- `is_weekend`: Boolean indicator for weekends
- `month`: Month of the year (1-12)
- `season`: Categorical variable for season

### Statistical Features

Statistical features derive insights from historical RTLMP data patterns. These features help capture:

- Recent price trends
- Price volatility
- Historical spike patterns
- Threshold crossings
- Autoregressive effects

Examples include:
- `rolling_mean_24h`: Mean RTLMP over the previous 24 hours
- `rolling_max_7d`: Maximum RTLMP over the previous 7 days
- `price_volatility_24h`: Standard deviation of RTLMP over the previous 24 hours
- `spike_count_7d`: Number of price spikes in the previous 7 days

### Weather Features

Weather features incorporate the impact of weather conditions on electricity demand and renewable generation. These features help capture:

- Temperature effects on demand
- Wind generation potential
- Solar generation potential
- Extreme weather events

Examples include:
- `temperature_forecast`: Forecasted temperature for the target hour
- `wind_speed_forecast`: Forecasted wind speed for the target hour
- `solar_irradiance_forecast`: Forecasted solar irradiance for the target hour
- `is_extreme_weather`: Boolean indicator for extreme weather conditions

### Market Features

Market features reflect the current and expected state of the ERCOT grid. These features help capture:

- Supply-demand balance
- Generation mix
- Transmission constraints
- Reserve margins
- Market participant behavior

Examples include:
- `load_forecast`: Forecasted load for the target hour
- `reserve_margin`: Forecasted reserve margin for the target hour
- `wind_generation_forecast`: Forecasted wind generation for the target hour
- `solar_generation_forecast`: Forecasted solar generation for the target hour
- `congestion_indicator`: Indicator of transmission congestion

## Feature Registry

The feature registry maintains metadata about all features, including:

- Feature name and ID
- Data type and valid range
- Description and purpose
- Dependencies on other features
- Historical importance scores

All new features must be registered to ensure proper documentation and validation.

### Registry Structure

The feature registry is implemented as a structured collection of feature metadata, with the following key components:

```python
{
    "feature_id": {
        "name": "human_readable_name",
        "category": "time|statistical|weather|market",
        "description": "Detailed description of the feature",
        "data_type": "float|int|bool|categorical",
        "valid_range": [min_value, max_value],  # Optional
        "dependencies": ["other_feature_id"],   # Optional
        "importance_history": [
            {"date": "YYYY-MM-DD", "score": 0.XX, "model_version": "vX.Y.Z"}
        ],
        "creation_date": "YYYY-MM-DD",
        "created_by": "developer_name",
        "modification_history": [
            {"date": "YYYY-MM-DD", "description": "Change description"}
        ]
    }
}
```

### Registry Operations

The feature registry supports the following operations:

- **Register**: Add a new feature to the registry
- **Update**: Modify existing feature metadata
- **Query**: Retrieve feature information by ID, category, or other criteria
- **Validate**: Check feature properties against registry specifications
- **Import/Export**: Serialize and deserialize the registry

## Adding a New Feature

Follow these steps to add a new feature to the system:

1. Identify the appropriate feature category
2. Implement the feature extraction function
3. Define feature metadata
4. Register the feature in the registry
5. Update the feature module's creation function
6. Test the feature for correctness and predictive power
7. Document the feature's purpose and implementation

### Step 1: Identify the Feature Category

Determine which of the four categories your feature belongs to:
- Time-based
- Statistical
- Weather
- Market

This will determine which module and namespace the feature should be implemented in.

### Step 2: Implement the Feature Extraction Function

Create a function that transforms raw data into your feature. Follow these principles:
- Make the function pure (no side effects)
- Handle missing data gracefully
- Use vectorized operations for efficiency
- Document parameters and return values

Example:

```python
def calculate_temperature_deviation(
    weather_data: pd.DataFrame,
    baseline_column: str = "normal_temperature",
    forecast_column: str = "temperature_forecast"
) -> pd.Series:
    """
    Calculate the deviation of forecasted temperature from historical normal.
    
    Parameters:
    -----------
    weather_data : pd.DataFrame
        DataFrame containing weather data
    baseline_column : str
        Column name for the historical normal temperature
    forecast_column : str
        Column name for the forecasted temperature
        
    Returns:
    --------
    pd.Series
        Series containing temperature deviation values
    """
    return weather_data[forecast_column] - weather_data[baseline_column]
```

### Step 3: Define Feature Metadata

Create a dictionary containing the metadata for your feature:

```python
temperature_deviation_metadata = {
    "name": "Temperature Deviation from Normal",
    "id": "temperature_deviation",
    "category": "weather",
    "description": "Difference between forecasted temperature and historical normal temperature",
    "data_type": "float",
    "valid_range": [-50.0, 50.0],  # in degrees Celsius or Fahrenheit
    "dependencies": ["temperature_forecast", "normal_temperature"],
    "creation_date": "2023-07-20",
    "created_by": "John Doe"
}
```

### Step 4: Register the Feature

Use the feature registry to register your new feature:

```python
from feature_engineering.registry import FeatureRegistry

registry = FeatureRegistry()
registry.register_feature("temperature_deviation", temperature_deviation_metadata)
```

### Step 5: Update Feature Creation Function

Add your feature to the appropriate feature creation function in the relevant module:

```python
# In weather_features.py

def create_weather_features(weather_data: pd.DataFrame) -> pd.DataFrame:
    """Create all weather-related features."""
    features = pd.DataFrame()
    
    # Existing features
    features["temperature_forecast"] = weather_data["temperature"]
    features["wind_speed_forecast"] = weather_data["wind_speed"]
    
    # Add new feature
    features["temperature_deviation"] = calculate_temperature_deviation(
        weather_data, 
        baseline_column="normal_temperature",
        forecast_column="temperature"
    )
    
    return features
```

### Step 6: Test the Feature

Write unit tests to verify your feature's correctness:

```python
# In test_weather_features.py

def test_temperature_deviation():
    """Test the temperature deviation feature calculation."""
    # Create test data
    test_data = pd.DataFrame({
        "temperature": [25.0, 30.0, 15.0],
        "normal_temperature": [20.0, 20.0, 20.0]
    })
    
    # Calculate feature
    result = calculate_temperature_deviation(
        test_data, 
        baseline_column="normal_temperature",
        forecast_column="temperature"
    )
    
    # Verify results
    expected = pd.Series([5.0, 10.0, -5.0])
    pd.testing.assert_series_equal(result, expected)
```

### Step 7: Document the Feature

Update the documentation to include your new feature:

```python
"""
Temperature Deviation Feature
-----------------------------

This feature captures the difference between forecasted temperature and 
historical normal temperature, which can indicate unusual weather patterns 
that may impact electricity demand and prices.

Calculation:
    temperature_deviation = temperature_forecast - normal_temperature

Typical range: -50.0 to 50.0 degrees

Related features:
    - temperature_forecast
    - normal_temperature
"""
```

## Time-based Features

When adding time-based features, follow these guidelines:

### Design Considerations

- Use consistent timezone handling (UTC recommended)
- Account for daylight saving time transitions
- Consider both cyclical and categorical representations for periodic features
- Be aware of holiday calendars for different regions

### Implementation Patterns

For cyclical features (like hour of day, day of week), consider using sine/cosine transformations to preserve the cyclical nature:

```python
def cyclical_encoding(value, period):
    """
    Create cyclical encoding using sine and cosine transformations.
    
    Parameters:
    -----------
    value : numeric
        The value to encode (e.g., hour of day)
    period : int
        The period of the cycle (e.g., 24 for hours)
        
    Returns:
    --------
    tuple
        (sin_component, cos_component)
    """
    sin_component = np.sin(2 * np.pi * value / period)
    cos_component = np.cos(2 * np.pi * value / period)
    return sin_component, cos_component
```

Example usage:

```python
def create_time_features(timestamps: pd.Series) -> pd.DataFrame:
    """Create time-based features from timestamps."""
    features = pd.DataFrame(index=timestamps.index)
    
    # Extract basic components
    hour = timestamps.dt.hour
    day_of_week = timestamps.dt.dayofweek  # 0=Monday, 6=Sunday
    month = timestamps.dt.month
    
    # Basic categorical features
    features["hour_of_day"] = hour
    features["day_of_week"] = day_of_week
    features["month"] = month
    features["is_weekend"] = (day_of_week >= 5).astype(int)
    
    # Cyclical encodings
    hour_sin, hour_cos = zip(*[cyclical_encoding(h, 24) for h in hour])
    features["hour_sin"] = hour_sin
    features["hour_cos"] = hour_cos
    
    dow_sin, dow_cos = zip(*[cyclical_encoding(d, 7) for d in day_of_week])
    features["day_of_week_sin"] = dow_sin
    features["day_of_week_cos"] = dow_cos
    
    month_sin, month_cos = zip(*[cyclical_encoding(m, 12) for m in month])
    features["month_sin"] = month_sin
    features["month_cos"] = month_cos
    
    return features
```

### Common Time-based Features

- Basic time components (hour, day, month, year)
- Day type (weekday/weekend/holiday)
- Season
- Fiscal periods (quarter, fiscal year)
- Special events (market events, regulatory events)

## Statistical Features

When adding statistical features, follow these guidelines:

### Design Considerations

- Define appropriate lookback windows (e.g., 24 hours, 7 days)
- Consider different aggregation functions (mean, max, min, std, quantiles)
- Handle missing data in historical windows
- Balance information content against computational cost

### Implementation Patterns

Use pandas' rolling windows for efficient calculation of rolling statistics:

```python
def create_rolling_statistics(
    price_data: pd.Series,
    windows: list = [24, 168]  # 24 hours, 7 days (in hours)
) -> pd.DataFrame:
    """
    Create rolling statistical features from price data.
    
    Parameters:
    -----------
    price_data : pd.Series
        Series containing price values with datetime index
    windows : list
        List of window sizes (in hours) to use for rolling statistics
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing rolling statistics features
    """
    features = pd.DataFrame(index=price_data.index)
    
    # Ensure data is sorted by time
    price_data = price_data.sort_index()
    
    for window in windows:
        # Calculate rolling statistics
        window_str = f"{window}h"
        rolling = price_data.rolling(window_str)
        
        features[f"rolling_mean_{window}h"] = rolling.mean()
        features[f"rolling_max_{window}h"] = rolling.max()
        features[f"rolling_min_{window}h"] = rolling.min()
        features[f"rolling_std_{window}h"] = rolling.std()
        
        # Relative features
        features[f"price_rel_to_mean_{window}h"] = price_data / rolling.mean()
        
        # Volatility features
        features[f"price_volatility_{window}h"] = rolling.std() / rolling.mean()
        
    return features
```

### Common Statistical Features

- Rolling means, maximums, minimums
- Rolling standard deviations and variances
- Rolling quantiles (median, 95th percentile)
- Volatility measures
- Trend indicators
- Threshold crossings (e.g., counts of prices exceeding thresholds)
- Autoregressive features (lagged values)

## Weather Features

When adding weather features, follow these guidelines:

### Design Considerations

- Align weather data temporally and spatially with price data
- Consider forecast horizons and uncertainty
- Incorporate domain knowledge about weather impacts on electricity
- Account for seasonal baseline differences

### Implementation Patterns

Consider creating derived features that capture energy-relevant weather impacts:

```python
def create_heating_cooling_degree_features(
    temperature_data: pd.Series,
    heating_threshold: float = 65.0,  # Fahrenheit
    cooling_threshold: float = 75.0   # Fahrenheit
) -> pd.DataFrame:
    """
    Create heating and cooling degree features from temperature data.
    
    Parameters:
    -----------
    temperature_data : pd.Series
        Series containing temperature values (Fahrenheit)
    heating_threshold : float
        Base temperature for heating degree calculation
    cooling_threshold : float
        Base temperature for cooling degree calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing heating and cooling degree features
    """
    features = pd.DataFrame(index=temperature_data.index)
    
    # Heating degrees (how much below heating threshold)
    features["heating_degrees"] = heating_threshold - temperature_data
    features["heating_degrees"] = features["heating_degrees"].clip(lower=0)
    
    # Cooling degrees (how much above cooling threshold)
    features["cooling_degrees"] = temperature_data - cooling_threshold
    features["cooling_degrees"] = features["cooling_degrees"].clip(lower=0)
    
    return features
```

### Common Weather Features

- Basic weather variables (temperature, wind speed, solar irradiance)
- Derived energy-relevant metrics (heating/cooling degrees)
- Extreme weather indicators
- Weather deviations from normal
- Temporal weather changes (temperature gradients)
- Spatial weather variations
- Combined weather impacts (e.g., wind chill)

## Market Features

When adding market features, follow these guidelines:

### Design Considerations

- Incorporate ERCOT-specific market knowledge
- Consider the timing of market information availability
- Account for changes in market rules and conditions
- Balance complexity against interpretability

### Implementation Patterns

Create features that capture market conditions relevant to price formation:

```python
def create_supply_margin_features(
    load_forecast: pd.Series,
    available_capacity: pd.Series
) -> pd.DataFrame:
    """
    Create supply margin features from load and capacity data.
    
    Parameters:
    -----------
    load_forecast : pd.Series
        Series containing load forecast values (MW)
    available_capacity : pd.Series
        Series containing available generation capacity (MW)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing supply margin features
    """
    features = pd.DataFrame(index=load_forecast.index)
    
    # Absolute margin (MW)
    features["supply_margin_mw"] = available_capacity - load_forecast
    
    # Relative margin (%)
    features["supply_margin_pct"] = (
        (available_capacity - load_forecast) / load_forecast * 100
    )
    
    # Tight supply indicator (boolean)
    features["tight_supply"] = (
        features["supply_margin_pct"] < 10
    ).astype(int)
    
    return features
```

### Common Market Features

- Load forecasts and actuals
- Generation capacity and mix
- Reserve margins
- Transmission congestion indicators
- Fuel prices
- Historical price patterns
- Market participant behavior indicators
- Regulatory and operational events

## Feature Selection

New features should be evaluated for their predictive power using the feature selection mechanisms. The system supports importance-based, correlation-based, and model-based selection methods. Features with low importance or high correlation with existing features may be filtered out during the selection process.

### Feature Selection Workflow

1. **Candidate Generation**: Add new features to the candidate pool
2. **Initial Screening**: Filter features based on data quality and basic criteria
3. **Correlation Analysis**: Identify and handle highly correlated features
4. **Importance Evaluation**: Assess feature importance using tree-based models
5. **Stability Analysis**: Evaluate feature stability across different time periods
6. **Final Selection**: Create the optimal feature subset

### Feature Selection Methods

The system supports multiple feature selection methods:

#### Correlation-based Selection

Identify and remove highly correlated features to reduce redundancy:

```python
def select_uncorrelated_features(
    features: pd.DataFrame,
    correlation_threshold: float = 0.95
) -> List[str]:
    """
    Select a subset of features with correlation below the threshold.
    
    Parameters:
    -----------
    features : pd.DataFrame
        DataFrame containing feature values
    correlation_threshold : float
        Maximum allowed correlation between features
        
    Returns:
    --------
    List[str]
        List of selected feature names
    """
    corr_matrix = features.corr().abs()
    
    # Create a mask for the upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation greater than threshold
    to_drop = [
        column for column in upper.columns 
        if any(upper[column] > correlation_threshold)
    ]
    
    # Return list of features to keep
    return [col for col in features.columns if col not in to_drop]
```

#### Importance-based Selection

Use trained models to evaluate feature importance:

```python
def select_important_features(
    features: pd.DataFrame,
    target: pd.Series,
    importance_threshold: float = 0.01,
    n_iterations: int = 5
) -> List[str]:
    """
    Select features with importance above threshold.
    
    Parameters:
    -----------
    features : pd.DataFrame
        DataFrame containing feature values
    target : pd.Series
        Series containing target values
    importance_threshold : float
        Minimum required feature importance
    n_iterations : int
        Number of model training iterations
        
    Returns:
    --------
    List[str]
        List of selected feature names
    """
    importances = pd.DataFrame(index=features.columns)
    
    # Train multiple models to get stable importance
    for i in range(n_iterations):
        model = XGBClassifier(n_estimators=100, random_state=i)
        model.fit(features, target)
        
        # Get feature importances
        iter_imp = pd.Series(
            model.feature_importances_,
            index=features.columns
        )
        importances[f"iter_{i}"] = iter_imp
    
    # Calculate mean importance
    importances["mean_importance"] = importances.mean(axis=1)
    
    # Select features above threshold
    selected = importances[
        importances["mean_importance"] > importance_threshold
    ].index.tolist()
    
    return selected
```

## Testing Features

All new features should be tested for:

- Correct data types and ranges
- Handling of edge cases and missing values
- Consistency across training and inference
- Predictive power through backtesting
- Performance impact on the feature pipeline

### Unit Testing

Write unit tests for each feature function to verify correctness:

```python
def test_heating_cooling_degree_features():
    """Test heating and cooling degree feature calculations."""
    # Create test data
    temperatures = pd.Series(
        [50.0, 60.0, 70.0, 80.0, 90.0],
        index=pd.date_range("2023-01-01", periods=5, freq="H")
    )
    
    # Set thresholds
    heating_threshold = 65.0
    cooling_threshold = 75.0
    
    # Calculate features
    result = create_heating_cooling_degree_features(
        temperatures,
        heating_threshold=heating_threshold,
        cooling_threshold=cooling_threshold
    )
    
    # Check heating degrees
    expected_heating = pd.Series([15.0, 5.0, 0.0, 0.0, 0.0], index=result.index)
    pd.testing.assert_series_equal(result["heating_degrees"], expected_heating)
    
    # Check cooling degrees
    expected_cooling = pd.Series([0.0, 0.0, 0.0, 5.0, 15.0], index=result.index)
    pd.testing.assert_series_equal(result["cooling_degrees"], expected_cooling)
```

### Integration Testing

Verify that the feature works within the full pipeline:

```python
def test_feature_in_pipeline():
    """Test that the feature integrates correctly with the pipeline."""
    # Setup test data
    test_data = load_test_data()
    
    # Run the feature pipeline
    feature_set = run_feature_pipeline(test_data)
    
    # Check that the feature exists and has expected properties
    assert "heating_degrees" in feature_set.columns
    assert not feature_set["heating_degrees"].isna().any()
    assert (feature_set["heating_degrees"] >= 0).all()
```

### Predictive Power Testing

Evaluate the feature's contribution to model performance:

```python
def test_feature_predictive_power():
    """Test the predictive power of the feature."""
    # Setup train and test data
    X_train, X_test, y_train, y_test = load_train_test_data()
    
    # Train model without the feature
    X_train_without = X_train.drop("heating_degrees", axis=1)
    X_test_without = X_test.drop("heating_degrees", axis=1)
    
    model_without = XGBClassifier()
    model_without.fit(X_train_without, y_train)
    
    y_pred_without = model_without.predict_proba(X_test_without)[:, 1]
    auc_without = roc_auc_score(y_test, y_pred_without)
    
    # Train model with the feature
    model_with = XGBClassifier()
    model_with.fit(X_train, y_train)
    
    y_pred_with = model_with.predict_proba(X_test)[:, 1]
    auc_with = roc_auc_score(y_test, y_pred_with)
    
    # Check for improvement
    assert auc_with > auc_without, "Feature does not improve model performance"
```

## Best Practices

- Keep features simple and interpretable when possible
- Document the rationale behind each feature
- Consider the computational cost of feature extraction
- Validate features against domain knowledge
- Maintain backward compatibility
- Follow naming conventions for consistency

### Naming Conventions

Follow these naming conventions for feature consistency:

- Use lowercase with underscores
- Include the feature category prefix
- For rolling windows, include the window size
- For boolean features, use "is_" prefix

Examples:
- `time_hour_of_day`
- `time_is_weekend`
- `stat_rolling_mean_24h`
- `weather_temperature_forecast`
- `market_reserve_margin_pct`

### Documentation Standards

Document each feature with:

- Description of what the feature represents
- Formula or calculation method
- Expected range of values
- Domain justification (why this feature should be predictive)
- Dependencies on other features
- Edge cases and handling

### Performance Considerations

- Use vectorized operations (avoid loops)
- Cache intermediate results when appropriate
- Consider memory usage for large datasets
- Profile feature extraction time
- Create grouped features to reduce computation

## Example: Adding a New Time-based Feature

Let's walk through adding a new time-based feature: `time_is_peak_hour`, which identifies peak electricity demand hours.

### Step 1: Identify the Feature Category

This is a time-based feature since it depends on the hour of the day.

### Step 2: Implement the Feature Extraction Function

```python
def is_peak_hour(timestamps: pd.Series) -> pd.Series:
    """
    Identify peak electricity demand hours (typically 3pm-8pm).
    
    Parameters:
    -----------
    timestamps : pd.Series
        Series of timestamps
        
    Returns:
    --------
    pd.Series
        Boolean series indicating peak hours (True/False)
    """
    hours = timestamps.dt.hour
    
    # Define peak hours as 3pm-8pm (15-20)
    is_peak = (hours >= 15) & (hours <= 20)
    
    return is_peak.astype(int)  # Convert boolean to 0/1
```

### Step 3: Define Feature Metadata

```python
peak_hour_metadata = {
    "name": "Is Peak Hour",
    "id": "time_is_peak_hour",
    "category": "time",
    "description": "Indicates whether the hour is during peak electricity demand (3pm-8pm)",
    "data_type": "int",
    "valid_range": [0, 1],
    "dependencies": [],
    "creation_date": "2023-07-20",
    "created_by": "John Doe"
}
```

### Step 4: Register the Feature

```python
from feature_engineering.registry import FeatureRegistry

registry = FeatureRegistry()
registry.register_feature("time_is_peak_hour", peak_hour_metadata)
```

### Step 5: Update Feature Creation Function

```python
# In time_features.py

def create_time_features(timestamps: pd.Series) -> pd.DataFrame:
    """Create all time-based features."""
    features = pd.DataFrame(index=timestamps.index)
    
    # Existing features
    features["time_hour_of_day"] = timestamps.dt.hour
    features["time_day_of_week"] = timestamps.dt.dayofweek
    features["time_is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    
    # Add new feature
    features["time_is_peak_hour"] = is_peak_hour(timestamps)
    
    return features
```

### Step 6: Test the Feature

```python
# In test_time_features.py

def test_is_peak_hour():
    """Test the is_peak_hour feature calculation."""
    # Create test data
    test_timestamps = pd.Series(
        pd.date_range("2023-01-01 00:00", "2023-01-01 23:00", freq="H")
    )
    
    # Calculate feature
    result = is_peak_hour(test_timestamps)
    
    # Expected values: 0 for hours 0-14, 1 for hours 15-20, 0 for hours 21-23
    expected = pd.Series(
        [0] * 15 + [1] * 6 + [0] * 3,
        index=test_timestamps.index
    )
    
    # Verify results
    pd.testing.assert_series_equal(result, expected)
```

### Step 7: Document the Feature

```python
"""
Is Peak Hour Feature
-------------------

This feature identifies hours during peak electricity demand (3pm-8pm),
which often correspond to higher prices and greater likelihood of price spikes.

Calculation:
    time_is_peak_hour = 1 if hour in [15, 16, 17, 18, 19, 20] else 0

Typical range: 0 or 1 (binary)

Related features:
    - time_hour_of_day
"""
```

## Example: Adding a New Statistical Feature

Let's walk through adding a new statistical feature: `stat_price_momentum_24h`, which measures the rate of change in RTLMP over the past 24 hours.

### Step 1: Identify the Feature Category

This is a statistical feature since it's derived from historical price data.

### Step 2: Implement the Feature Extraction Function

```python
def calculate_price_momentum(
    price_data: pd.Series,
    window: int = 24  # Hours
) -> pd.Series:
    """
    Calculate price momentum (rate of change) over specified window.
    
    Parameters:
    -----------
    price_data : pd.Series
        Series containing price values with datetime index
    window : int
        Window size in hours
        
    Returns:
    --------
    pd.Series
        Series containing momentum values
    """
    # Ensure data is sorted by time
    price_data = price_data.sort_index()
    
    # Calculate current price / price from 'window' hours ago
    momentum = price_data / price_data.shift(window) - 1
    
    return momentum
```

### Step 3: Define Feature Metadata

```python
price_momentum_metadata = {
    "name": "Price Momentum (24h)",
    "id": "stat_price_momentum_24h",
    "category": "statistical",
    "description": "Rate of change in RTLMP over the past 24 hours",
    "data_type": "float",
    "valid_range": [-1.0, float('inf')],  # Can't drop below -100%, but can rise indefinitely
    "dependencies": [],
    "creation_date": "2023-07-20",
    "created_by": "Jane Smith"
}
```

### Step 4: Register the Feature

```python
from feature_engineering.registry import FeatureRegistry

registry = FeatureRegistry()
registry.register_feature("stat_price_momentum_24h", price_momentum_metadata)
```

### Step 5: Update Feature Creation Function

```python
# In statistical_features.py

def create_statistical_features(price_data: pd.Series) -> pd.DataFrame:
    """Create all statistical features from price data."""
    features = pd.DataFrame(index=price_data.index)
    
    # Existing features
    features["stat_rolling_mean_24h"] = price_data.rolling('24h').mean()
    features["stat_rolling_max_24h"] = price_data.rolling('24h').max()
    
    # Add new feature
    features["stat_price_momentum_24h"] = calculate_price_momentum(price_data, window=24)
    
    return features
```

### Step 6: Test the Feature

```python
# In test_statistical_features.py

def test_price_momentum():
    """Test the price momentum feature calculation."""
    # Create test data
    dates = pd.date_range("2023-01-01", periods=48, freq="H")
    
    # Create price series with known pattern: flat then doubling
    prices = [10.0] * 24 + [20.0] * 24
    price_data = pd.Series(prices, index=dates)
    
    # Calculate feature
    result = calculate_price_momentum(price_data, window=24)
    
    # Expected values: NaN for first 24 hours, then 1.0 (100% increase)
    expected = pd.Series([np.nan] * 24 + [1.0] * 24, index=dates)
    
    # Verify results
    pd.testing.assert_series_equal(result, expected)
```

## Example: Adding a New Weather Feature

Let's walk through adding a new weather feature: `weather_wind_capacity_factor`, which estimates the percentage of wind generation capacity that can be utilized based on wind speed.

### Step 1: Identify the Feature Category

This is a weather feature since it's derived from weather forecast data.

### Step 2: Implement the Feature Extraction Function

```python
def calculate_wind_capacity_factor(
    wind_speed: pd.Series,
    cut_in_speed: float = 3.0,  # m/s
    rated_speed: float = 12.0,  # m/s
    cut_out_speed: float = 25.0  # m/s
) -> pd.Series:
    """
    Calculate wind turbine capacity factor based on wind speed.
    
    Parameters:
    -----------
    wind_speed : pd.Series
        Series containing wind speed values (m/s)
    cut_in_speed : float
        Minimum wind speed for generation
    rated_speed : float
        Wind speed at which rated power is reached
    cut_out_speed : float
        Maximum wind speed (turbines shut down above this)
        
    Returns:
    --------
    pd.Series
        Series containing capacity factor values (0-1)
    """
    # Initialize with zeros
    capacity_factor = pd.Series(0.0, index=wind_speed.index)
    
    # Below cut-in speed: capacity factor = 0
    # Already initialized to 0
    
    # Between cut-in and rated speed: cubic relationship
    mask_ramp = (wind_speed >= cut_in_speed) & (wind_speed < rated_speed)
    capacity_factor.loc[mask_ramp] = (
        (wind_speed.loc[mask_ramp] ** 3 - cut_in_speed ** 3) / 
        (rated_speed ** 3 - cut_in_speed ** 3)
    )
    
    # Between rated and cut-out speed: capacity factor = 1
    mask_rated = (wind_speed >= rated_speed) & (wind_speed <= cut_out_speed)
    capacity_factor.loc[mask_rated] = 1.0
    
    # Above cut-out speed: capacity factor = 0
    # Already initialized to 0
    
    return capacity_factor
```

### Step 3: Define Feature Metadata

```python
wind_capacity_factor_metadata = {
    "name": "Wind Capacity Factor",
    "id": "weather_wind_capacity_factor",
    "category": "weather",
    "description": "Estimated percentage of wind generation capacity that can be utilized based on wind speed",
    "data_type": "float",
    "valid_range": [0.0, 1.0],
    "dependencies": ["weather_wind_speed_forecast"],
    "creation_date": "2023-07-20",
    "created_by": "Alex Johnson"
}
```

### Step 4: Register the Feature

```python
from feature_engineering.registry import FeatureRegistry

registry = FeatureRegistry()
registry.register_feature("weather_wind_capacity_factor", wind_capacity_factor_metadata)
```

### Step 5: Update Feature Creation Function

```python
# In weather_features.py

def create_weather_features(weather_data: pd.DataFrame) -> pd.DataFrame:
    """Create all weather-related features."""
    features = pd.DataFrame(index=weather_data.index)
    
    # Existing features
    features["weather_temperature_forecast"] = weather_data["temperature"]
    features["weather_wind_speed_forecast"] = weather_data["wind_speed"]
    
    # Add new feature
    features["weather_wind_capacity_factor"] = calculate_wind_capacity_factor(
        weather_data["wind_speed"]
    )
    
    return features
```

### Step 6: Test the Feature

```python
# In test_weather_features.py

def test_wind_capacity_factor():
    """Test the wind capacity factor feature calculation."""
    # Create test data with wind speeds in different ranges
    wind_speeds = pd.Series(
        [0.0, 2.0, 3.0, 6.0, 12.0, 15.0, 25.0, 30.0],
        index=range(8)
    )
    
    # Calculate feature
    result = calculate_wind_capacity_factor(wind_speeds)
    
    # Expected values
    # 0.0 and 2.0 m/s: below cut-in -> 0
    # 3.0 m/s: at cut-in -> 0
    # 6.0 m/s: between cut-in and rated -> ~0.125
    # 12.0 m/s: at rated -> 1.0
    # 15.0 m/s: between rated and cut-out -> 1.0
    # 25.0 m/s: at cut-out -> 1.0
    # 30.0 m/s: above cut-out -> 0.0
    
    expected = pd.Series(
        [0.0, 0.0, 0.0, 0.125, 1.0, 1.0, 1.0, 0.0],
        index=range(8)
    )
    
    # Verify results (with tolerance for floating-point precision)
    pd.testing.assert_series_equal(
        result, expected, 
        check_exact=False, 
        rtol=1e-2
    )
```

## Example: Adding a New Market Feature

Let's walk through adding a new market feature: `market_ramp_requirement`, which calculates the expected ramping requirement based on load forecast changes.

### Step 1: Identify the Feature Category

This is a market feature since it's derived from grid operational data.

### Step 2: Implement the Feature Extraction Function

```python
def calculate_ramp_requirement(
    load_forecast: pd.Series,
    window: int = 1  # Hours
) -> pd.Series:
    """
    Calculate ramping requirement based on load forecast changes.
    
    Parameters:
    -----------
    load_forecast : pd.Series
        Series containing load forecast values (MW)
    window : int
        Window size in hours for calculating ramp
        
    Returns:
    --------
    pd.Series
        Series containing ramping requirement values (MW/hour)
    """
    # Ensure data is sorted by time
    load_forecast = load_forecast.sort_index()
    
    # Calculate change in load over the window
    ramp_requirement = load_forecast.diff(window)
    
    return ramp_requirement
```

### Step 3: Define Feature Metadata

```python
ramp_requirement_metadata = {
    "name": "Ramp Requirement",
    "id": "market_ramp_requirement",
    "category": "market",
    "description": "Expected ramping requirement based on load forecast changes",
    "data_type": "float",
    "valid_range": [-10000.0, 10000.0],  # MW/hour
    "dependencies": ["market_load_forecast"],
    "creation_date": "2023-07-20",
    "created_by": "Sam Wilson"
}
```

### Step 4: Register the Feature

```python
from feature_engineering.registry import FeatureRegistry

registry = FeatureRegistry()
registry.register_feature("market_ramp_requirement", ramp_requirement_metadata)
```

### Step 5: Update Feature Creation Function

```python
# In market_features.py

def create_market_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """Create all market-related features."""
    features = pd.DataFrame(index=market_data.index)
    
    # Existing features
    features["market_load_forecast"] = market_data["load_forecast"]
    features["market_available_capacity"] = market_data["available_capacity"]
    
    # Add new feature
    features["market_ramp_requirement"] = calculate_ramp_requirement(
        market_data["load_forecast"]
    )
    
    return features
```

### Step 6: Test the Feature

```python
# In test_market_features.py

def test_ramp_requirement():
    """Test the ramp requirement feature calculation."""
    # Create test data
    dates = pd.date_range("2023-01-01", periods=5, freq="H")
    
    # Create load forecast with known pattern
    load_values = [30000, 32000, 35000, 33000, 31000]  # MW
    load_forecast = pd.Series(load_values, index=dates)
    
    # Calculate feature
    result = calculate_ramp_requirement(load_forecast)
    
    # Expected values: NaN for first hour, then differences
    expected = pd.Series(
        [np.nan, 2000, 3000, -2000, -2000],
        index=dates
    )
    
    # Verify results
    pd.testing.assert_series_equal(result, expected)
```

## Troubleshooting

### Common Issues and Solutions

#### Missing Data Handling

**Issue**: Feature calculation fails due to missing data in inputs.

**Solution**: Implement proper missing data handling in your feature function:

```python
def robust_feature_calculation(data: pd.Series) -> pd.Series:
    """Calculate feature with robust missing data handling."""
    # Check for missing data
    if data.isna().any():
        # Option 1: Fill missing values
        data_filled = data.fillna(method='ffill')
        
        # Option 2: Set missing results explicitly
        result = some_calculation(data_filled)
        result[data.isna()] = np.nan
        
        return result
    else:
        return some_calculation(data)
```

#### Feature Availability at Inference Time

**Issue**: Feature uses data that isn't available during inference.

**Solution**: Ensure feature only relies on data that will be available at prediction time:

```python
def inference_compatible_feature(data: pd.DataFrame, is_training: bool = False) -> pd.Series:
    """Calculate feature that works for both training and inference."""
    if is_training:
        # Use all available data for training
        return full_calculation(data)
    else:
        # Use only forecast data for inference
        return forecast_based_calculation(data)
```

#### Computational Performance

**Issue**: Feature calculation is too slow for production.

**Solution**: Optimize using vectorized operations and avoid loops:

```python
# Slow implementation
def slow_feature(data: pd.Series) -> pd.Series:
    result = pd.Series(index=data.index)
    for i in range(len(data)):
        result.iloc[i] = complex_calculation(data.iloc[i])
    return result

# Fast implementation
def fast_feature(data: pd.Series) -> pd.Series:
    return data.apply(lambda x: simplified_calculation(x))
    # Or even better, use vectorized operations
    # return vectorized_calculation(data)
```

#### Feature Value Range Issues

**Issue**: Feature produces unexpected values outside valid range.

**Solution**: Add value validation and clipping:

```python
def bounded_feature(data: pd.Series, min_val: float = 0.0, max_val: float = 1.0) -> pd.Series:
    """Calculate feature and ensure values are within bounds."""
    result = raw_calculation(data)
    
    # Clip to valid range
    result = result.clip(lower=min_val, upper=max_val)
    
    # Validate result
    assert not result.isna().all(), "Feature produced all NaN values"
    assert (result >= min_val).all() and (result <= max_val).all(), "Values outside valid range"
    
    return result
```

### Debugging Feature Engineering Pipeline

If your feature is causing issues in the pipeline:

1. **Isolate the feature**: Test it independently with controlled inputs
2. **Check intermediate results**: Add logging of intermediate calculations
3. **Validate inputs**: Verify that input data matches your expectations
4. **Inspect edge cases**: Test with extreme values, missing data, etc.
5. **Performance profiling**: Use timing decorators to identify bottlenecks

Example debugging decorator:

```python
import time
from functools import wraps

def debug_feature(func):
    """Decorator to debug feature functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {len(args)} args, {len(kwargs)} kwargs")
        
        # Check for NaN values in inputs
        for i, arg in enumerate(args):
            if hasattr(arg, 'isna'):
                print(f"Arg {i} has {arg.isna().sum()} NaN values")
        
        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Check result
        if hasattr(result, 'isna'):
            print(f"Result has {result.isna().sum()} NaN values")
            
        print(f"{func.__name__} execution time: {end_time - start_time:.4f} seconds")
        
        return result
    
    return wrapper

# Usage
@debug_feature
def my_feature_function(data):
    # Implementation
    return result
```

## References

### Internal Documentation

- [Feature Engineering Module Design](/docs/design/feature_engineering.md)
- [Feature Registry Documentation](/docs/api/feature_registry.md)
- [Model Training Guide](/docs/development/model_training.md)
- [Data Schema Reference](/docs/data/schemas.md)

### External Resources

- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [ERCOT Market Information](http://www.ercot.com/mktinfo)
- [Pandas Documentation: Time Series / Date Functionality](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)