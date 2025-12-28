"""Tests for metrics module."""
import numpy as np
import pytest

from bambooml.tasks.metrics import accuracy, f1, mse, precision, recall


def test_accuracy():
    """Test accuracy calculation."""
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1])
    assert accuracy(y_true, y_pred) == 1.0

    y_pred = np.array([1, 1, 2, 0, 1])
    assert accuracy(y_true, y_pred) < 1.0


def test_accuracy_with_probabilities():
    """Test accuracy with probability predictions."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    assert accuracy(y_true, y_pred) == 1.0


def test_mse():
    """Test mean squared error calculation."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mse(y_true, y_pred) == 0.0

    y_pred = np.array([2.0, 3.0, 4.0])
    assert mse(y_true, y_pred) > 0.0


def test_precision():
    """Test precision calculation."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    prec = precision(y_true, y_pred)
    assert 0.0 <= prec <= 1.0


def test_recall():
    """Test recall calculation."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    rec = recall(y_true, y_pred)
    assert 0.0 <= rec <= 1.0


def test_f1():
    """Test F1 score calculation."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    f1_score = f1(y_true, y_pred)
    assert 0.0 <= f1_score <= 1.0

