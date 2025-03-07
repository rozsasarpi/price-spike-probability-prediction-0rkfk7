"""
Integration tests package for the ERCOT RTLMP spike prediction system.

This package contains tests that verify interactions between system components,
ensuring they work together correctly as integrated units. Integration tests focus
on validating data flows across component boundaries and end-to-end workflows.
"""

import pytest  # version 7.3+

# Register the fixtures module so it's available to all integration tests
pytest_plugins = ['src.backend.tests.fixtures']