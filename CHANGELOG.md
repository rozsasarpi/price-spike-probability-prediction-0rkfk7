# Changelog

All notable changes to the ERCOT RTLMP spike prediction system will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Integration with additional weather data providers
- Support for custom threshold configuration via config file
- Enhanced visualization options for probability curves

### Changed
- Improved feature selection algorithm for better model performance
- Optimized data fetching process for faster execution

### Fixed
- Known issues in current development cycle will be documented here

## [1.0.0] - 2023-08-01

Initial release of the ERCOT RTLMP spike prediction system.

### Added
- Data fetching interface for ERCOT market data with robust error handling
- Feature engineering module with time-based and statistical features
- Weather data integration for improved forecast accuracy
- Model training module with time-based cross-validation
- Inference engine for generating 72-hour probability forecasts
- Backtesting framework for historical simulation and model evaluation
- Visualization and metrics tools including calibration curves and ROC analysis
- Comprehensive logging and monitoring system
- Automated model retraining on bi-daily schedule
- Command-line interface for system operations

### Performance
- Optimized feature calculation for large historical datasets
- Efficient model storage and loading mechanisms
- Parallel processing for backtesting operations

### Documentation
- User guide with example usage scenarios
- Developer documentation with API specifications
- Installation and configuration instructions
- Model interpretation guidelines

[Unreleased]: https://github.com/yourusername/ercot-rtlmp-prediction/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/ercot-rtlmp-prediction/releases/tag/v1.0.0