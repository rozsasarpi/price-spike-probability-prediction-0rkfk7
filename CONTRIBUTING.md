# Contributing to the ERCOT RTLMP Spike Prediction System

Thank you for your interest in contributing to the ERCOT RTLMP spike prediction system! This document provides guidelines for contributing to the project, helping us maintain high-quality code and consistent development practices.

The system predicts the probability of price spikes in the ERCOT Real-Time Locational Marginal Price (RTLMP) market, helping battery storage operators optimize their strategies. Your contributions help improve the accuracy and reliability of these predictions.

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and considerate of others
- Use inclusive language and be mindful of different perspectives
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

Unacceptable behavior will not be tolerated. If you witness or experience unacceptable behavior, please report it to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/ercot-rtlmp-prediction.git
   cd ercot-rtlmp-prediction
   ```
3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/original-owner/ercot-rtlmp-prediction.git
   ```
4. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. Install development dependencies:
   ```bash
   # For backend development
   pip install -e "src/backend[dev]"
   
   # For CLI development
   pip install -e "src/backend"
   pip install -e "src/cli[dev]"
   ```
6. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

For more detailed setup instructions, see the [local setup guide](docs/deployment/local_setup.md).

## Development Workflow

### Branching Strategy

We use a simplified Git flow with the following branches:

- `main`: Production-ready code
- `develop`: Integration branch for features and bug fixes
- Feature branches: Named as `feature/short-description` or `bugfix/issue-number`

### Creating a New Feature or Bug Fix

1. Ensure your fork is up to date:
   ```bash
   git checkout develop
   git pull upstream develop
   ```
2. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   or for bug fixes:
   ```bash
   git checkout -b bugfix/issue-number
   ```
3. Make your changes, following the coding standards
4. Commit your changes using conventional commit messages:
   ```bash
   git commit -m "feat(component): add new feature"
   ```
   or
   ```bash
   git commit -m "fix(component): resolve issue #123"
   ```
5. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a pull request against the `develop` branch

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types include:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code changes that neither fix bugs nor add features
- `test`: Adding or updating tests
- `chore`: Changes to the build process or auxiliary tools

Example:
```
feat(feature-engineering): add support for multiple time windows

Implements the ability to specify multiple time windows for rolling statistics
in feature engineering. This allows for capturing patterns at different time
scales (hourly, daily, weekly).

Closes #123
```

## Coding Standards

### Code Style

We use the following tools to maintain code quality:

- **Black**: For code formatting (line length: 100 characters)
- **isort**: For organizing imports
- **mypy**: For static type checking
- **flake8**: For linting

These tools are configured in the pre-commit hooks and will run automatically when you commit changes.

### Type Annotations

All code must use type annotations. This improves code readability, enables better IDE support, and catches type-related errors early:

```python
from typing import Dict, List, Optional, Union
import pandas as pd

def engineer_features(
    data: pd.DataFrame,
    feature_names: List[str],
    window_size: int = 24,
    include_statistics: bool = True
) -> pd.DataFrame:
    """Engineer time-based features from raw data."""
    # Implementation
    return result
```

### Docstrings

All modules, classes, and functions should have docstrings following the Google style:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of the function.
    
    Longer description explaining the function's purpose and behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When and why this exception is raised
    """
```

### Error Handling

- Use specific exception types rather than generic exceptions
- Provide informative error messages
- Document exceptions in function docstrings
- Handle errors at the appropriate level of abstraction

### Logging

Use the standard logging module with appropriate log levels:

- DEBUG: Detailed information for troubleshooting
- INFO: Confirmation that things are working as expected
- WARNING: Indication that something unexpected happened
- ERROR: Due to a more serious problem, the software couldn't perform a function
- CRITICAL: A serious error indicating the program may be unable to continue running

## Testing Requirements

### Test Coverage

All code must have test coverage of at least 85%. This is enforced by the CI pipeline.

### Writing Tests

We use pytest for testing. Tests should be organized to mirror the package structure:

```
src/backend/tests/
├── unit/                 # Unit tests
│   ├── test_data_fetchers.py
│   ├── test_feature_engineering.py
│   └── ...
├── integration/          # Integration tests
│   ├── test_data_flow.py
│   └── ...
└── fixtures/             # Test fixtures
    ├── sample_data.py
    └── ...
```

Test files should be named with the `test_` prefix, and test functions should also start with `test_`.

### Test Example

```python
import pytest
import pandas as pd
from datetime import datetime
from backend.features.time_features import extract_hour_of_day

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=48, freq='H')
    data = pd.DataFrame({
        'price': [50.0] * 48
    }, index=dates)
    return data

def test_extract_hour_of_day(sample_data):
    """Test extraction of hour of day feature."""
    # Act
    result = extract_hour_of_day(sample_data)
    
    # Assert
    assert 'hour_of_day' in result.columns
    assert result['hour_of_day'].min() >= 0
    assert result['hour_of_day'].max() <= 23
    assert len(result) == len(sample_data)
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src --cov-report=term

# Run specific test file
pytest src/backend/tests/unit/test_feature_engineering.py

# Run tests matching a pattern
pytest -k "feature or model"
```

For more detailed testing guidelines, see the [testing documentation](docs/development/testing.md).

## Pull Request Process

### Before Submitting a Pull Request

1. Ensure all tests pass locally
2. Ensure code coverage meets the minimum threshold (85%)
3. Verify that all linting and type checking passes
4. Update documentation if necessary
5. Add or update tests for your changes

### Creating a Pull Request

1. Push your branch to your fork
2. Go to the original repository and create a pull request
3. Fill out the pull request template with all required information
4. Link any related issues using keywords like "Closes #123" or "Fixes #456"
5. Request a review from appropriate team members

### Pull Request Template

The pull request template will guide you through providing necessary information about your changes. It includes sections for:

- Description of changes
- Related issues
- Type of change
- Checklist of completed items
- Performance impact
- Model quality impact

### Code Review Process

1. At least one approval is required before merging
2. All CI checks must pass
3. Address all reviewer comments and suggestions
4. Once approved, the PR can be merged by a maintainer

### After Merging

1. Delete your feature branch
2. Update your local repository:
   ```bash
   git checkout develop
   git pull upstream develop
   ```

## Documentation

### Code Documentation

All code should be well-documented with:

- Module-level docstrings explaining the purpose of the module
- Class and function docstrings following the Google style
- Inline comments for complex or non-obvious code

### Project Documentation

The project documentation is organized as follows:

- `README.md`: Project overview and quick start guide
- `docs/`: Detailed documentation
  - `architecture/`: System architecture documentation
  - `deployment/`: Deployment and setup guides
  - `user_guides/`: Guides for different user roles
  - `development/`: Development guides and standards

### Updating Documentation

When making changes that affect user-facing functionality or APIs, update the relevant documentation:

1. Update docstrings for modified code
2. Update README.md if necessary
3. Update or add documentation in the docs/ directory
4. Add examples for new features

For significant changes to the documentation structure, discuss with the team first.

## Feature Development

### Adding New Features

When adding new features to the system, follow these steps:

1. Discuss the feature with the team first, preferably by creating a feature request issue
2. Design the feature with consideration for:
   - Integration with existing components
   - Performance impact
   - Testing strategy
   - Documentation needs
3. Implement the feature following the coding standards
4. Add comprehensive tests for the feature
5. Update documentation to include the new feature
6. Submit a pull request with the implementation

### Feature Engineering

When adding new features for the prediction model:

1. Follow the guidelines in [Adding Features](docs/development/adding_features.md)
2. Register new features in the feature registry
3. Implement feature extraction functions
4. Add tests verifying feature behavior
5. Document the feature's purpose and expected impact on model performance

### Model Development

When working on model improvements:

1. Establish a baseline using existing models
2. Document your approach and expected improvements
3. Implement model changes with appropriate abstractions
4. Evaluate model performance using standard metrics
5. Document performance results and comparison to baseline
6. Ensure model serialization and loading works correctly

## Bug Reporting and Fixing

### Reporting Bugs

If you find a bug in the system:

1. Check if the bug has already been reported
2. Create a new issue using the bug report template
3. Provide detailed steps to reproduce the bug
4. Include information about your environment
5. Describe the expected behavior and actual behavior
6. Add any relevant logs or screenshots

### Fixing Bugs

When fixing bugs:

1. Create a branch named `bugfix/issue-number`
2. Write a test that reproduces the bug
3. Fix the bug and ensure the test passes
4. Ensure no regression in existing functionality
5. Update documentation if necessary
6. Submit a pull request with the fix
7. Reference the issue in the pull request using "Fixes #123"

## Release Process

### Version Numbering

We follow semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible new functionality
- PATCH: Backwards-compatible bug fixes

### Release Preparation

1. Update the CHANGELOG.md with all notable changes
2. Update version numbers in relevant files
3. Create a release branch from develop named `release/vX.Y.Z`
4. Run final tests and verification on the release branch
5. Submit a pull request from the release branch to main

### Release Finalization

1. Merge the release PR into main
2. Tag the release in Git with the version number
3. Merge main back into develop
4. Create a GitHub release with release notes
5. Publish the package to the appropriate repositories

## Additional Resources

### Project Documentation

- [System Architecture](docs/architecture/system_overview.md)
- [Data Flow](docs/architecture/data_flow.md)
- [Component Interaction](docs/architecture/component_interaction.md)
- [Local Setup Guide](docs/deployment/local_setup.md)
- [Scheduled Execution](docs/deployment/scheduled_execution.md)
- [Data Scientist Guide](docs/user_guides/data_scientists.md)
- [Energy Scientist Guide](docs/user_guides/energy_scientists.md)
- [Testing Guide](docs/development/testing.md)
- [Adding Features Guide](docs/development/adding_features.md)

### External Resources

- [Python Documentation](https://docs.python.org/3/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [ERCOT Market Information](http://www.ercot.com/)

## Getting Help

If you need help with your contribution:

1. Check the documentation in the docs/ directory
2. Look for similar issues in the issue tracker
3. Ask questions in the relevant issue or pull request
4. Contact the project maintainers

We appreciate your contributions and are happy to help you get started!