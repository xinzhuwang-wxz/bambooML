"""Tests for config utilities."""
import tempfile
from pathlib import Path

import pytest

from bambooml.core.config import dump_json, dump_yaml, load_json, load_yaml


def test_load_dump_yaml():
    """Test YAML loading and dumping."""
    data = {"key1": "value1", "key2": 42, "nested": {"key3": [1, 2, 3]}}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name

    try:
        dump_yaml(data, temp_path)
        loaded = load_yaml(temp_path)
        assert loaded == data
    finally:
        Path(temp_path).unlink()


def test_load_dump_json():
    """Test JSON loading and dumping."""
    data = {"key1": "value1", "key2": 42, "nested": {"key3": [1, 2, 3]}}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        dump_json(data, temp_path)
        loaded = load_json(temp_path)
        assert loaded == data
    finally:
        Path(temp_path).unlink()

