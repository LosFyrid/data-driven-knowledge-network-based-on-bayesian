"""
Pytest configuration and shared fixtures for CI/CD tests.

This file provides common fixtures and utilities for fast automated tests.
These fixtures are available to ALL tests in tests/cicd/.
"""

import pytest


@pytest.fixture
def sampling_params_fast():
    """
    Fast MCMC sampling parameters for CI/CD tests.

    These parameters prioritize speed over accuracy to keep tests under 1 minute.
    Use these for smoke tests and interface contract tests.

    Returns:
        Dictionary of PyMC sampling parameters
    """
    return {
        'draws': 50,
        'tune': 50,
        'chains': 1,
        'random_seed': 42
    }

