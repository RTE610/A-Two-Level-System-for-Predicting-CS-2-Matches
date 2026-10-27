from __future__ import annotations

from typing import Any

from src.schema import MatchPredictionRecord, validate_records
from tests.fixtures import synthetic_league


def valid_record() -> dict[str, Any]:
    records, _ = synthetic_league(n_matches=5)
    return records[0]


def test_valid_record_passes() -> None:
    record = MatchPredictionRecord.model_validate(valid_record())
    assert record.matchId == 5000
    assert len(record.summaryStats) == 2


def test_invalid_records_are_counted_by_reason() -> None:
    ok = valid_record()
    wrong_winner = {**ok, "actualWinnerId": {"$numberLong": "1"}}
    one_team = {**ok, "summaryStats": ok["summaryStats"][:1]}
    bad_pct = {**ok, "summaryStats": [{**ok["summaryStats"][0], "winPercentage": 140.0}, ok["summaryStats"][1]]}

    valid, report = validate_records([ok, wrong_winner, one_team, bad_pct])

    assert valid == [ok]
    assert report.total == 4 and report.invalid == 3
    assert report.invalid_share == 0.75
    assert any("winner is not one of the teams" in e for e in report.errors)


def test_power_ranks_must_follow_summary_order() -> None:
    ok = valid_record()
    swapped = {**ok, "powerRanks": list(reversed(ok["powerRanks"]))}
    _, report = validate_records([swapped])
    assert report.invalid == 1
