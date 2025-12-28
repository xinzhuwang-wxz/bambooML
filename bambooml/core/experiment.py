"""Experiment tracking utilities using MLflow."""
import os
from pathlib import Path
from typing import Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ..core.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY


def setup_mlflow(tracking_uri: Optional[str] = None) -> None:
    """Setup MLflow tracking.

    Args:
        tracking_uri: MLflow tracking URI. If None, uses default from config.
    """
    if not MLFLOW_AVAILABLE:
        return

    if tracking_uri is None:
        tracking_uri = MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)


def start_run(experiment_name: str, run_name: Optional[str] = None) -> Optional[object]:
    """Start an MLflow run.

    Args:
        experiment_name: Name of the experiment.
        run_name: Name of the run. If None, MLflow generates one.

    Returns:
        MLflow run object or None if MLflow is not available.
    """
    if not MLFLOW_AVAILABLE:
        return None

    setup_mlflow()
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict) -> None:
    """Log parameters to MLflow.

    Args:
        params: Dictionary of parameters to log.
    """
    if MLFLOW_AVAILABLE:
        mlflow.log_params(params)


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: Dictionary of metrics to log.
        step: Step number for the metrics.
    """
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """Log an artifact to MLflow.

    Args:
        local_path: Local path to the artifact.
        artifact_path: Path within the artifact directory.
    """
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(local_path, artifact_path)


def end_run() -> None:
    """End the current MLflow run."""
    if MLFLOW_AVAILABLE:
        mlflow.end_run()

