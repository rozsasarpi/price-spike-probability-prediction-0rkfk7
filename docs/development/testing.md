# Testing Guide for ERCOT RTLMP Spike Prediction System

This document provides comprehensive guidance on testing practices for the ERCOT RTLMP spike prediction system. It covers unit testing, integration testing, data quality testing, model testing, and test automation.

## 1. Testing Philosophy

Testing is a critical component of the ERCOT RTLMP spike prediction system development process. As a system that supports critical business operations in battery storage optimization, we prioritize thorough testing throughout the development lifecycle.

### Core Testing Principles

- **Reliability First**: The system must consistently deliver accurate forecasts before day-ahead market closure.
- **Balance Coverage and Development Speed**: Focus testing efforts on critical components that affect forecast quality.
- **Data-Centric Testing**: Prioritize robust testing of data fetching, transformation, and feature engineering.
- **Model Quality Validation**: Ensure models meet minimum performance thresholds.
- **Comprehensive Automation**: Tests should run automatically to catch regressions early.

The testing strategy is designed to maintain high confidence in system reliability while allowing for ongoing development and improvement.

## 2. Testing Structure

Our testing structure follows a clear organization to maintain consistency and clarity:

```
tests/
├── unit/                   # Unit tests for isolated components
│   ├── test_data_fetcher.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── test_inference.py
├── integration/            # Tests for component interactions
│   ├── test_data_to_features.py
│   ├── test_features_to_model.py
│   └── test_end_to_end.py
├── fixtures/               # Shared test fixtures and data
│   ├── sample_data.py
│   ├── mock_services.py
│   └── test_helpers.py
└── conftest.py             # pytest configuration and shared fixtures
```

### Naming Conventions

- Test files are prefixed with `test_`
- Test functions are named `test_<functionality_being_tested>`
- Test classes are named `Test<ComponentName>`
- Fixtures are named descriptively based on what they provide

### Organization Principles

1. Group tests by component and functionality
2. Use fixtures for common setup and test data
3. Separate unit and integration tests
4. Use markers to categorize tests (e.g., slow, data, model)

## 3. Unit Testing

Unit tests verify the correctness of individual functions and classes in isolation. These tests should be fast, reliable, and focused on specific functionality.

### Test Isolation

Each unit test should focus on testing a single function or method. External dependencies should be mocked or replaced with test doubles:

```python
def test_fetch_rtlmp_data():
    # Arrange
    fetcher = MockDataFetcher()
    start_date = datetime(2022, 1, 1, tzinfo=ERCOT_TIMEZONE)
    end_date = datetime(2022, 1, 2, tzinfo=ERCOT_TIMEZONE)
    nodes = ['HB_NORTH']
    
    # Act
    result = fetcher.fetch_rtlmp_data(start_date, end_date, nodes)
    
    # Assert
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'timestamp' in result.columns
    assert 'node_id' in result.columns
    assert 'price' in result.columns
    assert all(node in nodes for node in result['node_id'].unique())
```

### Structure: Arrange-Act-Assert

Unit tests should follow the Arrange-Act-Assert pattern:
1. **Arrange**: Set up the test conditions
2. **Act**: Call the function or method being tested
3. **Assert**: Verify the results

### Mocking External Dependencies

Use `pytest-mock` or `unittest.mock` to isolate components from external systems:

```python
def test_ercot_api_client(mocker):
    # Arrange
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"data": [...]}
    mock_response.status_code = 200
    
    mock_session = mocker.Mock()
    mock_session.get.return_value = mock_response
    
    client = ERCOTAPIClient(session=mock_session)
    
    # Act
    result = client.fetch_rtlmp_data('HB_NORTH', '2023-01-01', '2023-01-02')
    
    # Assert
    mock_session.get.assert_called_once()
    assert isinstance(result, dict)
    assert "data" in result
```

### Testing Edge Cases

Always test edge cases and boundary conditions:
- Empty input data
- Missing values
- Extreme values
- Time boundary cases (e.g., daylight saving changes)
- Error handling

### Component-Specific Guidelines

#### Data Fetcher Testing

- Test API error handling
- Test data validation logic
- Test retry mechanisms
- Verify output data format

#### Feature Engineering Testing

- Test feature calculation correctness
- Test handling of missing values
- Verify time-based feature consistency
- Test feature scaling and normalization

#### Model Training Testing

- Test hyperparameter validation
- Test cross-validation splitting
- Verify model persistence
- Test performance metric calculations

#### Inference Testing

- Test prediction format and ranges
- Test handling of different thresholds
- Verify confidence interval calculations
- Test forecast completion before deadlines

## 4. Integration Testing

Integration tests verify that components work correctly together. These tests focus on the data flow between components and end-to-end workflows.

### Component Interaction Testing

Test data transformations across component boundaries:

```python
@pytest.mark.integration
def test_data_fetcher_to_feature_pipeline():
    # Arrange
    fetcher = MockDataFetcher()
    rtlmp_data = fetcher.fetch_rtlmp_data(
        SAMPLE_START_DATE, SAMPLE_END_DATE, SAMPLE_NODES
    )
    weather_data = fetcher.fetch_weather_data(
        SAMPLE_START_DATE, SAMPLE_END_DATE
    )
    grid_data = fetcher.fetch_grid_conditions(
        SAMPLE_START_DATE, SAMPLE_END_DATE
    )
    
    # Act
    pipeline = FeaturePipeline()
    pipeline.add_data_source('rtlmp', rtlmp_data)
    pipeline.add_data_source('weather', weather_data)
    pipeline.add_data_source('grid', grid_data)
    features = pipeline.create_features()
    
    # Assert
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    # Check for expected feature columns
    assert 'hour_of_day' in features.columns
    assert 'rolling_price_mean_24h' in features.columns
    assert 'temperature' in features.columns
    assert 'total_load' in features.columns
```

### End-to-End Workflow Testing

Test complete workflows from data fetching to forecast generation:

1. Fetch data from mock services
2. Process features
3. Train a simple model (with reduced hyperparameters)
4. Generate forecasts
5. Verify forecast format and completeness

### Data Flow Testing

Verify that data is properly transformed as it flows through the system:
- Raw data → Cleaned data
- Cleaned data → Features
- Features → Model input
- Model output → Forecast format

### Testing with Realistic Data

Use representative (but reduced) datasets to test system behavior under realistic conditions:
- Historical RTLMP data samples
- Weather forecast samples
- Grid condition samples

## 5. Data Quality Testing

Data quality testing verifies that data meets expected standards throughout the processing pipeline.

### Schema Validation

Use pandera to define and validate data schemas:

```python
import pandera as pa

# Define schema for RTLMP data
rtlmp_schema = pa.DataFrameSchema({
    "timestamp": pa.Column(pa.DateTime),
    "node_id": pa.Column(pa.String),
    "price": pa.Column(pa.Float, pa.Check.greater_than_or_equal_to(0)),
    "congestion_price": pa.Column(pa.Float),
    "loss_price": pa.Column(pa.Float),
    "energy_price": pa.Column(pa.Float)
})

def test_rtlmp_data_quality():
    # Arrange
    data = get_sample_rtlmp_data()
    
    # Act & Assert
    # Check for required columns
    assert all(col in data.columns for col in [
        'timestamp', 'node_id', 'price', 'congestion_price', 'loss_price', 'energy_price'
    ])
    
    # Check data types
    assert pd.api.types.is_datetime64_dtype(data['timestamp'])
    assert pd.api.types.is_string_dtype(data['node_id'])
    assert pd.api.types.is_float_dtype(data['price'])
    
    # Check value ranges
    assert data['price'].min() >= 0, "Negative prices found"
    assert not data.isna().any().any(), "Missing values found"
    
    # Check time frequency
    timestamps = data['timestamp'].sort_values().unique()
    time_diffs = pd.Series(timestamps).diff().dropna()
    assert (time_diffs == pd.Timedelta(minutes=5)).all(), "Inconsistent time frequency"
```

### Completeness Checks

Verify that data does not have unexpected gaps:
- Test for missing timestamps
- Check for missing nodes in RTLMP data
- Verify complete weather data for forecast periods

### Statistical Validation

Use hypothesis for property-based testing of data distributions:

```python
from hypothesis import given, strategies as st
import pandas as pd

@given(st.data())
def test_statistical_properties(data):
    # Generate test data with hypothesis
    df = data.draw(
        st.data_frames([
            st.column('price', elements=st.floats(min_value=0, max_value=10000)),
            st.column('timestamp', elements=st.datetimes())
        ])
    )
    
    # Calculate statistics
    result = calculate_rolling_statistics(df, window='24h')
    
    # Verify statistical properties
    # Mean should be within the range of the data
    assert result['rolling_mean'].min() >= df['price'].min()
    assert result['rolling_mean'].max() <= df['price'].max()
    
    # Standard deviation should be non-negative
    assert (result['rolling_std'] >= 0).all()
```

### Data Consistency Tests

Verify relationships between data fields:
- Energy price + congestion price + loss price ≈ total price
- Grid load is consistent with generation
- Weather data is consistent across nearby locations

## 6. Model Testing

Model testing verifies the quality and reliability of prediction models.

### Performance Testing

Test that models meet minimum performance thresholds:

```python
def test_model_performance_metrics():
    # Arrange
    X_train, X_test, y_train, y_test = get_test_train_data()
    model = create_model(model_type='xgboost')
    
    # Act
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    metrics = calculate_performance_metrics(y_test, predictions)
    
    # Assert
    assert metrics['auc'] >= 0.75, f"AUC score {metrics['auc']} below threshold"
    assert metrics['brier_score'] <= 0.15, f"Brier score {metrics['brier_score']} above threshold"
    assert metrics['precision'] >= 0.7, f"Precision {metrics['precision']} below threshold"
    assert metrics['recall'] >= 0.6, f"Recall {metrics['recall']} below threshold"
```

### Stability Testing

Verify that model performance is consistent:
- Train with different random seeds
- Test with different data subsets
- Verify performance across different time periods

### Sensitivity Analysis

Test how models respond to input variations:
- Feature perturbation tests
- Feature importance consistency
- Response to extreme input values

### Backtesting

Simulate historical forecasts to evaluate performance:
- Train on data before time T
- Predict for time T+n
- Compare predictions with actual outcomes
- Calculate performance metrics

### Calibration Testing

Verify that predicted probabilities match observed frequencies:
- Generate reliability diagrams
- Calculate calibration metrics
- Test across different prediction thresholds

## 7. Test Fixtures and Utilities

Test fixtures provide reusable test components and data to simplify test development.

### Common Fixtures

```python
@pytest.fixture
def mock_data_fetcher():
    """Provides a configured MockDataFetcher for testing."""
    fetcher = MockDataFetcher(
        add_noise=True,
        spike_probability=0.1
    )
    return fetcher

@pytest.fixture
def temp_output_dir():
    """Provides a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
```

### Sample Data Generators

Create fixtures that generate test data:
- Synthetic RTLMP data with controlled spikes
- Weather forecast data with seasonal patterns
- Grid condition data with realistic load profiles

### Time Utilities

Fixtures for handling time-based testing:
- Time freezing utilities
- Timezone handling
- Mock scheduling functions

### Model Utilities

Fixtures for model testing:
- Pre-trained simple models
- Mock model objects
- Standardized evaluation datasets

## 8. Test Automation

Our CI/CD pipeline automates test execution to maintain code quality.

### GitHub Actions Configuration

Tests are automatically executed on push and pull requests:

```yaml
# .github/workflows/test.yml
name: Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install
    
    - name: Run linters
      run: |
        poetry run black --check .
        poetry run isort --check .
        poetry run mypy .
    
    - name: Run tests
      run: |
        poetry run pytest tests/unit
        poetry run pytest tests/integration
    
    - name: Generate coverage report
      run: |
        poetry run pytest --cov=rtlmp_predict --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

### Test Execution Strategy

Tests are categorized and executed strategically:

1. **Fast tests** (unit tests) run on every commit
2. **Integration tests** run on pull requests
3. **Full test suite** runs nightly
4. **Model quality tests** run after retraining

### Quality Gates

Pull requests must pass these quality gates:
- All unit and integration tests pass
- Code coverage meets minimum threshold (≥85%)
- No critical linting errors
- Type checking passes

### Test Reporting

Test results are reported in multiple formats:
- JUnit XML for CI integration
- HTML reports for human review
- Coverage reports to track test completeness

## 9. Code Coverage

Code coverage measures how much of the codebase is exercised by tests.

### Coverage Measurement

We use pytest-cov to measure code coverage:

```bash
# Run tests with coverage
pytest --cov=rtlmp_predict --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Coverage Requirements

- Overall code coverage: ≥85%
- Critical modules coverage: ≥90%
  - Data fetching
  - Feature engineering
  - Model training
  - Inference

### Improving Coverage

To improve coverage:
1. Identify uncovered code with coverage reports
2. Prioritize high-impact, uncovered components
3. Add tests for edge cases and error handling
4. Use property-based testing for complex functions

### Coverage Reports

Coverage reports are:
- Generated on every CI run
- Tracked over time to prevent regressions
- Used to identify testing gaps

## 10. Best Practices

Follow these best practices for testing the RTLMP spike prediction system:

### Test Organization

- Group related tests together
- Use clear, descriptive test names
- Follow the Arrange-Act-Assert pattern
- Keep tests independent and isolated

### Naming Conventions

- Test modules: `test_<component>.py`
- Test functions: `test_<functionality>_<condition>`
- Test classes: `Test<ComponentName>`
- Fixtures: descriptive names based on what they provide

### Documentation

- Document test fixtures
- Explain complex test setup
- Document test data sources
- Include the purpose of each test class

### Test Performance

- Keep unit tests fast (<100ms per test)
- Use markers for slow tests
- Minimize file I/O in tests
- Use test parametrization to reduce duplication

### Mocking Guidelines

- Only mock what you need to
- Prefer dependency injection for easier mocking
- Verify mock interactions when relevant
- Reset mocks between tests

## 11. Troubleshooting

Common testing issues and their solutions:

### Flaky Tests

Tests that sometimes pass and sometimes fail.

**Solutions:**
- Identify and eliminate time dependencies
- Add retries for network operations
- Use deterministic random seeds
- Ensure test isolation

### Slow Tests

Tests that take too long to run.

**Solutions:**
- Use smaller datasets
- Mock expensive operations
- Add `@pytest.mark.slow` marker
- Run slow tests less frequently

### Environment-Specific Issues

Tests that pass locally but fail in CI.

**Solutions:**
- Use Docker for consistent environments
- Check for timezone issues
- Verify dependency versions
- Add more logging for CI failures

### Data Inconsistencies

Tests that fail due to data issues.

**Solutions:**
- Use fixed test datasets
- Add data validation
- Implement data cleaning in tests
- Check for correct data types

## Appendix A: Example Tests

### Unit Test Examples

```python
def test_feature_engineering_time_features():
    # Arrange
    input_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=48, freq='H')
    })
    
    # Act
    result = engineer_time_features(input_data)
    
    # Assert
    assert 'hour_of_day' in result.columns
    assert 'day_of_week' in result.columns
    assert 'is_weekend' in result.columns
    assert result['hour_of_day'].min() >= 0
    assert result['hour_of_day'].max() <= 23
    assert result['day_of_week'].min() >= 0
    assert result['day_of_week'].max() <= 6
```

### Integration Test Examples

```python
@pytest.mark.integration
def test_end_to_end_training_workflow(mock_data_fetcher, temp_output_dir):
    # Arrange
    config = TrainingConfig(
        start_date='2023-01-01',
        end_date='2023-01-31',
        threshold=100.0,
        model_type='xgboost',
        output_dir=temp_output_dir
    )
    
    # Act
    data = fetch_training_data(mock_data_fetcher, config)
    features = engineer_features(data, config)
    model, metrics = train_model(features, config)
    
    # Assert
    assert model is not None
    assert Path(temp_output_dir / 'model.joblib').exists()
    assert metrics['auc'] > 0.5
    assert 0 <= metrics['brier_score'] <= 1.0
```

### Data Quality Test Examples

```python
def test_weather_data_completeness():
    # Arrange
    data = get_sample_weather_data()
    expected_locations = ['NORTH', 'SOUTH', 'WEST', 'HOUSTON']
    
    # Act & Assert
    # Check all locations present
    actual_locations = data['location'].unique()
    assert set(expected_locations).issubset(set(actual_locations))
    
    # Check no missing timestamps
    for location in expected_locations:
        location_data = data[data['location'] == location]
        timestamps = location_data['timestamp']
        
        # Check hourly frequency
        assert timestamps.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        
        # Check full date range
        assert timestamps.min() <= pd.Timestamp('2023-01-01')
        assert timestamps.max() >= pd.Timestamp('2023-01-31')
```

## Appendix B: Testing Cheatsheet

### Common Commands

```bash
# Run all tests
pytest

# Run tests in a specific file
pytest tests/unit/test_data_fetcher.py

# Run tests matching a pattern
pytest -k "feature_engineering"

# Run tests with a specific marker
pytest -m integration

# Run tests with coverage
pytest --cov=rtlmp_predict

# Run tests and generate HTML coverage report
pytest --cov=rtlmp_predict --cov-report=html

# Run tests verbosely
pytest -v

# Run tests and stop on first failure
pytest -xvs
```

### Useful Pytest Plugins

- **pytest-cov**: Code coverage measurement
- **pytest-mock**: Mocking utilities
- **pytest-xdist**: Parallel test execution
- **pytest-timeout**: Test timeout enforcement
- **pytest-benchmark**: Performance benchmarking

### Common Test Markers

```python
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.data
@pytest.mark.model
@pytest.mark.parametrize('input,expected', [(1, 2), (2, 4)])
```

### Assertion Examples

```python
# Value assertions
assert result == expected
assert result != unexpected
assert result >= minimum

# Type assertions
assert isinstance(result, pd.DataFrame)
assert callable(function)

# Collection assertions
assert 'key' in dictionary
assert item in list
assert all(value > 0 for value in values)
assert any(condition(item) for item in items)

# DataFrame assertions
assert 'column' in df.columns
assert df['column'].dtype == np.float64
assert not df.empty
assert (df >= 0).all().all()

# Exception assertions
with pytest.raises(ValueError):
    function_that_raises()
```