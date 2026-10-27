"""Functional tests by acceptance criteria FR-1…FR-9 (test plan: docs/qa/test-plan.md)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pytest


@pytest.fixture(scope="module")
def schedule(client: Any) -> list[dict[str, Any]]:
    response = client.get("/schedule", params={"days": 14})
    assert response.status_code == 200
    return response.json()


def test_tc01_schedule_lists_teams_and_start_time(schedule: list[dict[str, Any]]) -> None:
    """FR-1: teams, date and start time are visible for every scheduled match."""
    assert schedule
    for match in schedule:
        assert match["team1"]["name"] and match["team2"]["name"]
        assert datetime.fromisoformat(match["scheduled_at"])
        assert match["mode"] == "scheduled"


def test_tc02_every_scheduled_match_has_a_result(schedule: list[dict[str, Any]]) -> None:
    """FR-2: a prediction (or an explicit status) for each match without user action."""
    assert all(m["decision"] in {"accepted", "rejected", "no_data"} for m in schedule)
    accepted = [m for m in schedule if m["decision"] == "accepted"]
    assert accepted, "at least one match in the synthetic schedule should be predictable"


def test_tc03_rejected_match_has_reason_and_no_winner(client: Any, league: dict[str, Any]) -> None:
    """FR-3: status 'rejected' and a reason, without a winner."""
    ids = list(league["strength"])
    results = [
        client.post("/predict", json={"team1_id": a, "team2_id": b}).json() for a in ids[:8] for b in ids[:8] if a < b
    ]
    rejected = [r for r in results if r["decision"] == "rejected"]
    assert rejected
    for r in rejected:
        assert r["reason"]
        assert r["winner"] is None and r["win_probability"] is None


def test_tc04_data_date_is_visible(client: Any, schedule: list[dict[str, Any]]) -> None:
    """FR-4: the date of the data is shown with every result and in /health."""
    health = client.get("/health").json()
    assert health["data_as_of"]
    assert {m["data_as_of"] for m in schedule} == {health["data_as_of"]}
    team = client.get(f"/teams/{schedule[0]['team1']['id']}").json()
    assert "last_match_at" in team and "profile_age_days" in team


def test_tc08_real_team_names(schedule: list[dict[str, Any]]) -> None:
    """FR-8: no placeholder names like 'Team 12345'."""
    names = [m[side]["name"] for m in schedule for side in ("team1", "team2")]
    assert not [n for n in names if re.fullmatch(r"Team \d+", n)]


def test_tc09_hypothetical_pair_is_labelled(client: Any, league: dict[str, Any]) -> None:
    """FR-9: any two teams; result marked as a hypothetical match."""
    a, b = list(league["strength"])[3:5]
    result = client.post("/predict", json={"team1_id": a, "team2_id": b}).json()
    assert result["mode"] == "hypothetical"
    assert result["match_id"] is None
    if result["decision"] == "accepted":
        assert result["winner"]["id"] in {a, b}
        assert 0.5 <= result["win_probability"] <= 1.0


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"team1_id": 200, "team2_id": 200}, 422),
        ({"team1_id": -1, "team2_id": 200}, 422),
        ({"team1_id": 200}, 422),
        ({"team1_id": 200, "team2_id": 123456}, 404),
    ],
)
def test_tc10_invalid_input(client: Any, payload: dict[str, int], status: int) -> None:
    """Negative cases: same team, invalid id, missing field, unknown team."""
    assert client.post("/predict", json=payload).status_code == status


def test_tc10_unknown_match_and_team(client: Any) -> None:
    assert client.get("/matches/1").status_code == 404
    assert client.get("/teams/1").status_code == 404


def test_tc11_incomplete_data_flag(client: Any, schedule: list[dict[str, Any]]) -> None:
    """Risk R6: predictions on profiles without game statistics are flagged."""
    flagged = [m for m in schedule if m["incomplete_data"]]
    assert flagged, "synthetic league has teams whose last profile lacks game statistics"


def test_tc12_order_invariance(client: Any, league: dict[str, Any]) -> None:
    """Consistency of FR-9: swapping teams gives the same winner and probability."""
    ids = list(league["strength"])[:6]
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            ab = client.post("/predict", json={"team1_id": a, "team2_id": b}).json()
            ba = client.post("/predict", json={"team1_id": b, "team2_id": a}).json()
            assert ab["decision"] == ba["decision"]
            if ab["decision"] == "accepted":
                assert ab["winner"]["id"] == ba["winner"]["id"]
                assert ab["win_probability"] == ba["win_probability"]


def test_tc02_match_card(client: Any, schedule: list[dict[str, Any]]) -> None:
    """FR-1/FR-4: the match card shows statistics of both teams and head-to-head count."""
    match = next(m for m in schedule if m["decision"] != "no_data")
    card = client.get(f"/matches/{match['match_id']}").json()
    assert card["team1_stats"]["id"] == match["team1"]["id"]
    assert card["head_to_head_matches"] >= 0
    assert card["prediction"]["decision"] == match["decision"]
