from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.snapshot import (
    FeatureSetVersion,
    iter_json_array,
    quality_by_month,
    scheduled_matches,
    select_feature_set,
    split_by_cutoff,
    suggest_cutoff,
    unwrap,
)


def make_record(
    match_id: int,
    date: str,
    calc: float = 2.4,
    coef: int = 1,
    updated: str = "2025-01-01T00:00:00Z",
    game_stats: bool = True,
    winner: int | None = 1,
    reason: str = "",
) -> dict[str, Any]:
    avg = {"gameAverages": {"adr": 350.0}} if game_stats else None
    return {
        "matchId": {"$numberLong": str(match_id)},
        "matchStartingDateTime": {"$date": date},
        "powerRankingCalculatorVersion": calc,
        "powerRankingCoefficientsVersion": coef,
        "summaryStats": [{"pandaScoreTeamId": 1, "allAverages": avg}, {"pandaScoreTeamId": 2, "allAverages": avg}],
        "actualWinnerId": {"$numberLong": str(winner)} if winner else None,
        "noPredictionReason": reason,
        "updatedAt": {"$date": updated},
    }


def test_unwrap_mongo_scalars() -> None:
    assert unwrap({"$numberLong": "42"}) == 42
    assert unwrap({"$date": "2025-01-01T00:00:00Z"}) == "2025-01-01T00:00:00Z"
    assert unwrap(3.5) == 3.5
    assert unwrap({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_select_feature_set_filters_version_and_dedups() -> None:
    records = [
        make_record(1, "2025-02-01T10:00:00Z", updated="2025-02-01T00:00:00Z"),
        make_record(1, "2025-02-01T10:00:00Z", updated="2025-02-02T00:00:00Z", reason="late"),
        make_record(1, "2025-02-01T10:00:00Z", calc=3.0),
        make_record(2, "2025-01-01T10:00:00Z"),
        make_record(3, "2025-03-01T10:00:00Z", coef=2),
    ]
    selected = select_feature_set(records, FeatureSetVersion("2.4", "1"))

    assert [unwrap(r["matchId"]) for r in selected] == [2, 1]
    assert selected[1]["noPredictionReason"] == "late"


def test_quality_report_and_cutoff() -> None:
    records = [
        make_record(1, "2025-12-05T10:00:00Z"),
        make_record(2, "2026-01-05T10:00:00Z"),
        make_record(3, "2026-02-05T10:00:00Z", game_stats=False, reason="x"),
        make_record(4, "2026-02-06T10:00:00Z"),
        make_record(5, "2026-03-05T10:00:00Z", game_stats=False, winner=None),
    ]
    report = quality_by_month(records)

    feb = report.set_index("month").loc["2026-02"]
    assert feb["matches"] == 2
    assert feb["missing_game_stats_pct"] == pytest.approx(50.0)
    assert feb["rejected_pct"] == pytest.approx(50.0)
    assert suggest_cutoff(report, max_missing_pct=5.0) == "2026-01"

    before, after = split_by_cutoff(records, "2026-01")
    assert len(before) == 2 and len(after) == 3


def test_suggest_cutoff_none_when_clean() -> None:
    report = quality_by_month([make_record(1, "2025-01-01T00:00:00Z")])
    assert suggest_cutoff(report) is None


def test_scheduled_matches_window(tmp_path: Path) -> None:
    def match(mid: int, status: str, when: str) -> dict[str, Any]:
        return {
            "pandaScoreData": {
                "_id": {"$numberLong": str(mid)},
                "status": status,
                "scheduledAt": {"$date": when},
                "opponents": [{"_id": 10, "name": "A"}, {"_id": 20, "name": "B"}],
            }
        }

    path = tmp_path / "matches.json"
    path.write_text(
        json.dumps(
            [
                match(1, "not_started", "2024-05-01T10:00:00Z"),
                match(2, "not_started", "2026-09-29T10:00:00Z"),
                match(3, "finished", "2026-09-29T11:00:00Z"),
                match(4, "not_started", "2026-11-01T10:00:00Z"),
            ]
        )
    )
    schedule = scheduled_matches(path, datetime(2026, 9, 28, tzinfo=UTC), horizon_days=14)

    assert [m["matchId"] for m in schedule] == [2]
    assert schedule[0]["team1"]["name"] == "A"


def test_iter_json_array_streams(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"x": 1.5}, {"x": 2}]))
    assert [r["x"] for r in iter_json_array(path)] == [1.5, 2]
