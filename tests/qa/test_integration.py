"""Integration tests: MLflow registry -> service (FR-6, FR-7) and UI -> API contract."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_tc06_service_loads_latest_model_from_registry(
    league: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-6: the service takes the model from the MLflow registry, not from local files."""
    import mlflow

    from services.api.app import load_model
    from src.tracking import log_run

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)
    mlflow.set_registry_uri(uri)
    out_dir = league["bundle_path"].parent
    log_run(
        "integration",
        params={"split": "temporal"},
        metrics={"test": league["result"]["test"]},
        data_path=league["data_dir"] / "matches_v2.json",
        artifacts_dir=out_dir,
        models={"cs2-match-predictor": league["bundle"].model},
        tracking_uri=uri,
    )
    monkeypatch.delenv("BUNDLE_PATH", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    loaded = load_model()

    assert loaded.version == "cs2-match-predictor/v1"
    assert loaded.model.headers == league["bundle"].headers


def test_tc06_ui_contains_no_model_code() -> None:
    """ADR-0001: the UI does not import ML libraries or train models."""
    tree = ast.parse((ROOT / "ui" / "app.py").read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    assert not imported & {"sklearn", "xgboost", "lightgbm", "catboost", "joblib", "src"}


def test_tc20_ui_renders_api_responses(client: Any) -> None:
    """UI -> API: the Streamlit app renders the schedule served by the real API app."""
    from streamlit.testing.v1 import AppTest

    def routed(method: str) -> Any:
        def call(url: str, **kwargs: Any) -> Any:
            path = url.replace("http://localhost:8000", "")
            if method == "post":
                response = client.post(path, json=kwargs.get("json"))
            else:
                response = client.get(path, params=kwargs.get("params"))
            response.raise_for_status = lambda: None
            return response

        return call

    with mock.patch("requests.get", routed("get")), mock.patch("requests.post", routed("post")):
        app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert any("CS2 Match Predictor" in t.value for t in app.title)
    assert app.dataframe, "schedule table is rendered"
    assert any("Дані станом на" in c.value for c in app.caption)
