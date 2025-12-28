"""Configuration utilities for loading and saving config files."""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Directories
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = Path(ROOT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model registry (can be configured via environment variable)
MODEL_REGISTRY = Path(os.environ.get("BAMBOOML_MODEL_REGISTRY", str(Path(ROOT_DIR, "models"))))
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

# MLflow tracking URI (can be configured via environment variable)
MLFLOW_TRACKING_URI = os.environ.get("BAMBOOML_MLFLOW_TRACKING_URI", f"file://{MODEL_REGISTRY.absolute()}")


def load_yaml(path: str | Path) -> dict:
    """Load YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Dictionary loaded from YAML file.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(obj: dict, path: str | Path, sort_keys: bool = False) -> None:
    """Dump dictionary to YAML file.

    Args:
        obj: Dictionary to dump.
        path: Path to save YAML file.
        sort_keys: Whether to sort keys. Defaults to False.
    """
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=sort_keys)


def load_json(path: str | Path) -> dict:
    """Load JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Dictionary loaded from JSON file.
    """
    with open(path) as f:
        return json.load(f)


def dump_json(obj: dict, path: str | Path, indent: int = 2) -> None:
    """Dump dictionary to JSON file.

    Args:
        obj: Dictionary to dump.
        path: Path to save JSON file.
        indent: Indentation level. Defaults to 2.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)
        f.write("\n")
