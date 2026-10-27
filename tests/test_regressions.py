"""Regression tests for defects found in Lab 4 testing (one test per bug report)."""

from __future__ import annotations

import copy
import itertools
from typing import Any

from src.data_source import SnapshotDataSource


def test_bug19_predict_with_one_team_without_game_stats(league: dict[str, Any]) -> None:
    """#19: /predict returned 500 when only one team had game statistics."""
    ds = SnapshotDataSource(league["data_dir"])
    a, b = list(league["strength"])[:2]
    record = ds.build_record(a, b)
    assert record is not None
    record = copy.deepcopy(record)
    record["summaryStats"][0]["allAverages"] = {"gameAverages": {"adr": 360.0, "kills": 71.0}}
    record["summaryStats"][1]["allAverages"] = None

    prediction = league["bundle"].predict([record])[0]

    assert 0.0 <= prediction.team1_win_probability <= 1.0
    assert prediction.incomplete_data


def test_bug20_schedule_keeps_matches_without_data(client: Any) -> None:
    """#20: matches with a team without statistics disappeared from the schedule."""
    schedule = client.get("/schedule", params={"days": 14}).json()
    ghost = next(m for m in schedule if m["match_id"] == 9200)

    assert len(schedule) == 6
    assert ghost["decision"] == "no_data"
    assert ghost["winner"] is None
    assert "Ghost Team" in ghost["reason"]


def test_bug21_prediction_does_not_depend_on_team_order(league: dict[str, Any]) -> None:
    """#21: the winner flipped when the teams were swapped."""
    ds = SnapshotDataSource(league["data_dir"])
    for a, b in itertools.combinations(list(league["strength"])[:8], 2):
        ab = league["bundle"].predict([ds.build_record(a, b)])[0]
        ba = league["bundle"].predict([ds.build_record(b, a)])[0]
        assert abs(ab.team1_win_probability - (1.0 - ba.team1_win_probability)) < 1e-9
        assert ab.accepted == ba.accepted


def test_bug22_team_name_falls_back_to_schedule(client: Any) -> None:
    """#22: a team missing from teams.json was shown as 'Team <id>'."""
    schedule = client.get("/schedule", params={"days": 14}).json()
    ghost = next(m for m in schedule if m["match_id"] == 9200)
    assert ghost["team2"]["name"] == "Ghost Team"


def test_bug27_schedule_is_computed_once_per_version(client: Any, league: dict[str, Any]) -> None:
    """#27: the schedule was re-predicted match by match on every request."""
    import time

    client.get("/schedule")
    started = time.perf_counter()
    for _ in range(5):
        assert client.get("/schedule", params={"days": 14}).status_code == 200
    assert (time.perf_counter() - started) / 5 < 0.5
