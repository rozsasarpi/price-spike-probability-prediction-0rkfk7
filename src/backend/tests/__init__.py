"""
Package initialization file for the test suite of the ERCOT RTLMP spike prediction system.

This file marks the directory as a Python package, enables proper test discovery,
and provides common imports and utilities for both unit and integration tests.
"""

import pytest  # version 7.3+

# Import test fixtures for use across all test modules
from . import fixtures

# Register the fixtures module as a pytest plugin
pytest_plugins = ['src.backend.tests.fixtures']