from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np


def synthetic_matches(n: int = 160, seed: int = 0) -> list[dict[str, Any]]:
    """Small matchPrediction-like records where the stronger team (power rank, win %) tends to win."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    records: list[dict[str, Any]] = []
    for i in range(n):
        strength = rng.normal(size=2)
        t1_wins = bool(strength[0] + rng.normal(scale=0.7) > strength[1])
        teams = []
        for k in range(2):
            teams.append(
                {
                    "pandaScoreTeamId": {"$numberLong": str(100 + 2 * i + k)},
                    "totalMatchesPlayed": int(rng.integers(3, 40)),
                    "winPercentage": float(50 + 15 * strength[k] + rng.normal(scale=5)),
                    "allAverages": {
                        "gameAverages": {"adr": float(350 + 10 * strength[k]), "kills": float(70 + 3 * strength[k])}
                    },
                }
            )
        ids = [t["pandaScoreTeamId"] for t in teams]
        records.append(
            {
                "matchId": {"$numberLong": str(1000 + i)},
                "matchStartingDateTime": {"$date": (start + timedelta(hours=6 * i)).isoformat()},
                "powerRankingCalculatorVersion": 2.4,
                "powerRankingCoefficientsVersion": 1,
                "summaryStats": teams,
                "headToHeadStats": [],
                "powerRanks": [
                    {"pandaScoreTeamId": ids[k], "powerRank": float(100 + 20 * strength[k])} for k in range(2)
                ],
                "predictedWinnerId": ids[0],
                "actualWinnerId": ids[0] if t1_wins else ids[1],
                "noPredictionReason": "" if abs(strength[0] - strength[1]) > 0.8 else "TeamsTooClose",
            }
        )
    return records


AS_OF = datetime(2025, 3, 1, tzinfo=UTC)


def synthetic_league(
    n_matches: int = 400, n_teams: int = 24, seed: int = 1
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    """Teams with a fixed strength play repeatedly, so profiles, head-to-head and schedules exist."""
    rng = np.random.default_rng(seed)
    strength = {200 + t: float(rng.normal()) for t in range(n_teams)}
    team_ids = list(strength)
    played: dict[int, int] = dict.fromkeys(team_ids, 0)
    start = AS_OF - timedelta(days=n_matches // 4)
    records: list[dict[str, Any]] = []
    for i in range(n_matches):
        a, b = (int(x) for x in rng.choice(team_ids, size=2, replace=False))
        diff = strength[a] - strength[b]
        a_wins = bool(diff + rng.normal(scale=0.7) > 0)
        stats = []
        for team in (a, b):
            played[team] += 1
            stats.append(
                {
                    "pandaScoreTeamId": {"$numberLong": str(team)},
                    "totalMatchesPlayed": played[team] + 5,
                    "winPercentage": float(np.clip(50 + 15 * strength[team] + rng.normal(scale=5), 0, 100)),
                    "allAverages": None
                    if i >= n_matches - 20
                    else {
                        "gameAverages": {
                            "adr": float(350 + 10 * strength[team]),
                            "kills": float(70 + 3 * strength[team]),
                        }
                    },
                }
            )
        ids = [s["pandaScoreTeamId"] for s in stats]
        records.append(
            {
                "matchId": {"$numberLong": str(5000 + i)},
                "matchStartingDateTime": {"$date": (start + timedelta(hours=6 * i)).isoformat()},
                "powerRankingCalculatorVersion": 2.4,
                "powerRankingCoefficientsVersion": 1,
                "summaryStats": stats,
                "headToHeadStats": [],
                "powerRanks": [
                    {"pandaScoreTeamId": ids[k], "powerRank": float(100 + 20 * strength[t])}
                    for k, t in enumerate((a, b))
                ],
                "predictedWinnerId": ids[0],
                "actualWinnerId": ids[0] if a_wins else ids[1],
                "noPredictionReason": "" if abs(diff) > 0.6 else "Teams are equal, unpredictable result",
                "updatedAt": {"$date": (start + timedelta(hours=6 * i)).isoformat()},
            }
        )
    return records, strength


def write_league_data_dir(path: Any, records: list[dict[str, Any]], strength: dict[int, float]) -> None:
    """Prepared-data layout expected by SnapshotDataSource."""
    import json

    path.mkdir(parents=True, exist_ok=True)
    past = [r for r in records if r["matchStartingDateTime"]["$date"] < AS_OF.isoformat()]
    (path / "matches_v2.json").write_text(json.dumps(past))
    (path / "team_names.json").write_text(json.dumps({str(t): f"Team {chr(65 + i)}" for i, t in enumerate(strength)}))
    ids = list(strength)
    schedule = [
        {
            "matchId": 9000 + k,
            "scheduledAt": (AS_OF + timedelta(days=1 + k)).isoformat(),
            "numberOfGames": 3,
            "team1": {"id": ids[k], "name": f"Team {chr(65 + k)}"},
            "team2": {"id": ids[k + 1], "name": f"Team {chr(66 + k)}"},
        }
        for k in range(5)
    ] + [
        {
            "matchId": 9100,
            "scheduledAt": (AS_OF + timedelta(days=40)).isoformat(),
            "numberOfGames": 1,
            "team1": {"id": ids[0], "name": "Team A"},
            "team2": {"id": ids[1], "name": "Team B"},
        }
    ]
    (path / "schedule.json").write_text(json.dumps(schedule))
    (path / "snapshot_meta.json").write_text(
        json.dumps({"feature_set": "2.4/1", "as_of": AS_OF.isoformat(), "snapshot_dir": "synthetic"})
    )
