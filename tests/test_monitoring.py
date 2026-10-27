from __future__ import annotations

import json

from monitor_predictions import parse, summarize


def line(**event: object) -> str:
    return "2026-10-28 10:00:00,000 prediction_api " + json.dumps(event)


def test_monitor_aggregates_and_alerts() -> None:
    lines = [
        line(event="prediction", decision="accepted", incomplete_data=True, latency_ms=100.0),
        line(event="prediction", decision="rejected", incomplete_data=True, latency_ms=3000.0),
        line(event="prediction", decision="rejected", incomplete_data=False, latency_ms=2500.0),
        line(event="canary_failed", error="boom"),
        "INFO:     127.0.0.1 - GET /health 200",
    ]
    events, canary = parse(lines)
    stats, alerts = summarize(events, canary)

    assert stats["predictions"] == 3 and canary == 1
    assert round(stats["rejected_share"], 2) == 0.67
    assert "canary prediction failed" in alerts
    assert "latency p95 above 2 s" in alerts
    assert "more than half of predictions use incomplete data (risk R6)" in alerts
