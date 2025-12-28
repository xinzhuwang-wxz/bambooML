"""Pytest configuration and fixtures."""
import pytest


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    import numpy as np
    return {
        "X": np.random.randn(100, 10),
        "y": np.random.randint(0, 3, 100),
    }

