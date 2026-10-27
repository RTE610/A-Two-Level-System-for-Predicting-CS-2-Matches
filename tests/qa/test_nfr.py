"""Non-functional requirements from Lab 2 (performance, observability, reproducibility)."""

from __future__ import annotations

import json
import logging
import statistics
import time
from typing import Any

import pytest

pytestmark = pytest.mark.nfr


def test_tc13_prediction_latency_under_2s(client: Any, league: dict[str, Any]) -> None:
    """NFR performance: response to a prediction within 2 seconds (p95)."""
    ids = list(league["strength"])
    timings = []
    for i in range(40):
        a, b = ids[i % len(ids)], ids[(i + 3) % len(ids)]
        started = time.perf_counter()
        assert client.post("/predict", json={"team1_id": a, "team2_id": b}).status_code == 200
        timings.append(time.perf_counter() - started)
    p95 = statistics.quantiles(timings, n=20)[18]
    assert p95 < 2.0


def test_tc16_prediction_is_logged(client: Any, league: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    """NFR observability: every prediction is logged with match, filter decision and model version."""
    a, b = list(league["strength"])[:2]
    with caplog.at_level(logging.INFO, logger="prediction_api"):
        client.post("/predict", json={"team1_id": a, "team2_id": b})
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == "prediction_api"]
    event = next(e for e in events if e["event"] == "prediction")
    assert {"team1", "team2", "decision", "model_version", "latency_ms"} <= event.keys()


def test_tc14_training_is_reproducible(tmp_path: Any) -> None:
    """NFR reproducibility: the same data and seed give the same metrics."""
    import run_temporal

    from src.splits import temporal_split
    from tests.fixtures import synthetic_matches

    path = tmp_path / "m.json"
    path.write_text(json.dumps(synthetic_matches(160)))
    frames = run_temporal.build_frames(path)
    first, _ = run_temporal.evaluate_two_stage(frames, temporal_split(frames.dates), "ultra", tmp_path / "a")
    second, _ = run_temporal.evaluate_two_stage(frames, temporal_split(frames.dates), "ultra", tmp_path / "b")
    assert first["test"] == second["test"]
