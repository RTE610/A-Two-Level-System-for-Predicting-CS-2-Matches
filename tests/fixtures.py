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
