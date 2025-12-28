"""Model evaluation utilities."""
import datetime
import json
from typing import Dict, Optional

import numpy as np

from ..core.config import dump_json
from ..core.logging import get_logger
from ..tasks.metrics import get_classification_metrics

logger = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[list] = None,
    task_type: str = "cls",
) -> Dict:
    """Evaluate model performance.

    Args:
        y_true: Ground truth labels or values.
        y_pred: Predicted labels or values.
        y_prob: Predicted probabilities (for classification).
        class_names: Optional list of class names.
        task_type: Task type ("cls" for classification, "reg" for regression).

    Returns:
        Dictionary containing evaluation metrics.
    """
    if task_type == "cls":
        metrics = get_classification_metrics(y_true, y_pred, class_names)
    else:
        from ..tasks.metrics import mse
        metrics = {
            "mse": mse(y_true, y_pred),
            "rmse": float(np.sqrt(mse(y_true, y_pred))),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "num_samples": len(y_true),
        }

    results = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "task_type": task_type,
        **metrics,
    }

    logger.info(json.dumps(results, indent=2))
    return results


def save_evaluation_results(results: Dict, output_path: str) -> None:
    """Save evaluation results to file.

    Args:
        results: Evaluation results dictionary.
        output_path: Path to save results.
    """
    dump_json(results, output_path)
    logger.info(f"Evaluation results saved to {output_path}")

