"""Utility functions for bambooML."""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .core.config import dump_json, load_json


def set_seeds(seed: int = 42) -> None:
    """Set seeds for reproducibility.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str | Path) -> Dict:
    """Load a dictionary from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Dictionary loaded from JSON file.
    """
    return load_json(path)


def save_dict(d: Dict, path: str | Path, sort_keys: bool = False) -> None:
    """Save a dictionary to a JSON file.

    Args:
        d: Dictionary to save.
        path: Path to save JSON file.
        sort_keys: Whether to sort keys alphabetically. Defaults to False.
    """
    dump_json(d, path, indent=2)


def dict_to_list(data: Dict, keys: list[str]) -> list[Dict[str, Any]]:
    """Convert a dictionary to a list of dictionaries.

    Args:
        data: Input dictionary.
        keys: Keys to include in the output list of dictionaries.

    Returns:
        List of dictionaries with specified keys.
    """
    list_of_dicts = []
    if not keys or not data:
        return list_of_dicts

    # Get the length from the first key
    length = len(data[keys[0]]) if keys[0] in data else 0

    for i in range(length):
        new_dict = {key: data[key][i] if key in data else None for key in keys}
        list_of_dicts.append(new_dict)
    return list_of_dicts

