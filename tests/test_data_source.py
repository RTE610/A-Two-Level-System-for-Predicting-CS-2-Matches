from __future__ import annotations

from typing import Any

from src.data_source import SnapshotDataSource
from tests.fixtures import AS_OF


def test_profile_uses_latest_record_before_as_of(league: dict[str, Any]) -> None:
    ds = SnapshotDataSource(league["data_dir"])
    team_id = next(iter(league["strength"]))
    profile = ds.team_profile(team_id)

    assert profile is not None
    assert profile.last_match_at < AS_OF
    assert profile.name == "Team A"


def test_schedule_respects_horizon(league: dict[str, Any]) -> None:
    ds = SnapshotDataSource(league["data_dir"])
    assert [m.match_id for m in ds.schedule(14)] == [9000, 9001, 9200, 9002, 9003, 9004]
    assert 9100 in [m.match_id for m in ds.schedule(60)]


def test_build_record_is_oriented(league: dict[str, Any]) -> None:
    ds = SnapshotDataSource(league["data_dir"])
    a, b = list(league["strength"])[:2]
    record = ds.build_record(a, b)
    reverse = ds.build_record(b, a)

    assert record is not None and reverse is not None
    ids = [int(s["pandaScoreTeamId"]["$numberLong"]) for s in record["summaryStats"]]
    assert ids == [a, b]
    assert record["summaryStats"] == list(reversed(reverse["summaryStats"]))


def test_unknown_team_has_no_record(league: dict[str, Any]) -> None:
    ds = SnapshotDataSource(league["data_dir"])
    assert ds.build_record(1, 2) is None
    assert ds.team_profile(1) is None


def test_search_by_name(league: dict[str, Any]) -> None:
    ds = SnapshotDataSource(league["data_dir"])
    assert [p.name for p in ds.search_teams("team a")] == ["Team A"]
