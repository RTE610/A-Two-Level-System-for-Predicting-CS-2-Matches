"""MLflow tracking for v2.0 training runs (FR-6, FR-7)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "cs2-two-stage"
PREDICTOR_NAME = "cs2-match-predictor"
FILTER_NAME = "cs2-binary-filter"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def flatten_metrics(prefix: str, values: Mapping[str, Any]) -> dict[str, float]:
    """Flatten nested numeric metrics: {"test": {"coverage": 0.4}} -> {"test.coverage": 0.4}."""
    out: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            out.update(flatten_metrics(name, value))
        elif isinstance(value, list | tuple) and len(value) == 2 and all(isinstance(v, int | float) for v in value):
            out[f"{name}_low"], out[f"{name}_high"] = float(value[0]), float(value[1])
        elif isinstance(value, int | float) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def log_run(
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    data_path: Path,
    artifacts_dir: Path | None = None,
    models: Mapping[str, Any] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    tags: Mapping[str, str] | None = None,
) -> str:
    """
    Log params, flattened metrics, data version and artifacts; register sklearn models.

    `models` maps a registry name to a fitted sklearn estimator/pipeline.
    Returns the MLflow run id.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"data.file": data_path.name, "data.sha256": file_sha256(data_path), **(tags or {})})
        mlflow.log_params({k: json.dumps(v) if isinstance(v, list | dict | tuple) else v for k, v in params.items()})
        mlflow.log_metrics(flatten_metrics("", metrics))
        if artifacts_dir is not None and artifacts_dir.exists():
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="outputs")
        for name, model in (models or {}).items():
            mlflow.sklearn.log_model(model, name=name, registered_model_name=name)
        return str(run.info.run_id)


def latest_model_version(name: str, tracking_uri: str = DEFAULT_TRACKING_URI) -> str | None:
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))
