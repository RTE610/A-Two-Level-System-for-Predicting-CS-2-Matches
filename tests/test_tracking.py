from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.tracking import file_sha256, flatten_metrics, latest_model_version, log_run


def test_flatten_metrics() -> None:
    nested = {"test": {"accuracy": 0.7, "ci95": [0.6, 0.8], "model": "LR", "n": 10, "flag": True}}
    assert flatten_metrics("", nested) == {
        "test.accuracy": 0.7,
        "test.ci95_low": 0.6,
        "test.ci95_high": 0.8,
        "test.n": 10.0,
    }


def test_file_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text("[]")
    assert file_sha256(path) == file_sha256(path)
    assert len(file_sha256(path)) == 64


def test_log_run_registers_model(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_registry_uri(uri)
    data = tmp_path / "data.json"
    data.write_text("[]")
    artifacts = tmp_path / "out"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text("{}")
    X, y = np.array([[0.0], [1.0]]), [0, 1]
    models = {"unit-model": LogisticRegression().fit(X, y), "unit-mlp": MLPClassifier(max_iter=50).fit(X, y)}

    run_id = log_run(
        run_name="unit",
        params={"split": "temporal", "fractions": [0.7, 0.15, 0.15]},
        metrics={"test": {"proposed_accuracy": 0.69}},
        data_path=data,
        artifacts_dir=artifacts,
        models=models,
        tracking_uri=uri,
    )

    run = mlflow.get_run(run_id)
    assert run.data.metrics["test.proposed_accuracy"] == 0.69
    assert run.data.tags["data.sha256"] == file_sha256(data)
    assert latest_model_version("unit-model", tracking_uri=uri) == "1"
    assert latest_model_version("unit-mlp", tracking_uri=uri) == "1"
