"""
Pytest fixtures specific to rules module tests.

These fixtures are only available to tests in the rules/ subdirectory.
"""

import pytest
import numpy as np


@pytest.fixture
def synthetic_linear_data():
    """
    Generate simple synthetic linear data for testing.

    Returns:
        X, y arrays with known linear relationship: y = 2.0 * x + 1.0 + noise
    """
    np.random.seed(42)
    X = np.linspace(0, 10, 20)
    y = 2.0 * X + 1.0 + np.random.normal(0, 0.1, 20)
    return X, y


@pytest.fixture
def true_linear_params():
    """
    True parameters for synthetic linear data.

    Returns:
        Dictionary with true parameter values
    """
    return {
        'a': 2.0,
        'b': 1.0,
        'sigma': 0.1
    }
