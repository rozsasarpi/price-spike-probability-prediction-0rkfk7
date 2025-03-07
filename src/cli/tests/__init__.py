"""
Initialization file for the CLI tests package.

This module makes the test modules discoverable by pytest and provides
common imports and utilities for all test modules.
"""

import pytest  # pytest 7.0+
from pathlib import Path  # standard library

# Define the tests directory path for use in test modules
TEST_DIR = Path(__file__).parent

# Export TEST_DIR for use in other test modules
__all__ = ["TEST_DIR"]