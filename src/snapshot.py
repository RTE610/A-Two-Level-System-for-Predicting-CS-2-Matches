"""
Loader for the CSBettor MongoDB snapshot provided by the department.

The snapshot stores several pre-match feature sets per match (one per power-ranking
calculator / coefficients version). The v1.0 model was trained on a single feature set,
so v2.0 pins one version and deduplicates by match id to avoid train/test leakage.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import ijson
import pandas as pd

MATCH_PREDICTION_FILE = "CSBettor.matchPrediction.json"
MATCHES_FILE = "CSBettor.matches.json"
TEAMS_FILE = "CSBettor.teams.json"

Record = dict[str, Any]


@dataclass(frozen=True)
class FeatureSetVersion:
    calculator: str = "2.4"
    coefficients: str = "1"

    @property
    def label(self) -> str:
        return f"{self.calculator}/{self.coefficients}"


def iter_json_array(path: Path) -> Iterator[Record]:
    """Stream items of a top-level JSON array without loading the whole file."""
    with path.open("rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def unwrap(value: Any) -> Any:
    """Unwrap Mongo extended JSON scalars ($numberLong, $date, $oid)."""
    if isinstance(value, dict) and len(value) == 1:
        key, inner = next(iter(value.items()))
        if key == "$numberLong":
            return int(inner)
        if key in {"$date", "$oid", "$numberDouble"}:
            return inner
    return value


def match_date(record: Record) -> str:
    return str(unwrap(record["matchStartingDateTime"]))


def matches_version(record: Record, version: FeatureSetVersion) -> bool:
    return (
        str(record.get("powerRankingCalculatorVersion")) == version.calculator
        and str(record.get("powerRankingCoefficientsVersion")) == version.coefficients
    )


def select_feature_set(records: Iterable[Record], version: FeatureSetVersion) -> list[Record]:
    """Keep one feature-set version and one record per match (latest `updatedAt` wins)."""
    latest: dict[int, Record] = {}
    for record in records:
        if not matches_version(record, version):
            continue
        match_id = int(unwrap(record["matchId"]))
        current = latest.get(match_id)
        if current is None or str(unwrap(record.get("updatedAt"))) >= str(unwrap(current.get("updatedAt"))):
            latest[match_id] = record
    return sorted(latest.values(), key=match_date)


def has_game_stats(record: Record) -> bool:
    stats = record.get("summaryStats") or []
    return len(stats) == 2 and all(team.get("allAverages") for team in stats)


def has_result(record: Record) -> bool:
    return unwrap(record.get("actualWinnerId")) not in (None, "", 0)


def is_rejected(record: Record) -> bool:
    return bool(str(record.get("noPredictionReason") or "").strip())


def quality_by_month(records: Iterable[Record]) -> pd.DataFrame:
    """Per-month data quality report used to choose the training cutoff."""
    rows = [
        {
            "month": match_date(r)[:7],
            "missing_game_stats": not has_game_stats(r),
            "has_result": has_result(r),
            "rejected": is_rejected(r),
        }
        for r in records
    ]
    if not rows:
        return pd.DataFrame(columns=["month", "matches", "missing_game_stats_pct", "with_result_pct", "rejected_pct"])
    df = pd.DataFrame(rows)
    report = df.groupby("month").agg(
        matches=("month", "size"),
        missing_game_stats_pct=("missing_game_stats", "mean"),
        with_result_pct=("has_result", "mean"),
        rejected_pct=("rejected", "mean"),
    )
    pct_cols = ["missing_game_stats_pct", "with_result_pct", "rejected_pct"]
    report[pct_cols] = (report[pct_cols] * 100).round(1)
    return report.reset_index()


def suggest_cutoff(report: pd.DataFrame, max_missing_pct: float = 5.0) -> str | None:
    """Last month before the first month whose missing game-stats share exceeds the threshold."""
    ordered = report.sort_values("month")
    bad = ordered[ordered["missing_game_stats_pct"] > max_missing_pct]
    if bad.empty:
        return None
    first_bad = str(bad.iloc[0]["month"])
    good = ordered[ordered["month"] < first_bad]
    return str(good.iloc[-1]["month"]) if not good.empty else None


def split_by_cutoff(records: list[Record], cutoff_month: str | None) -> tuple[list[Record], list[Record]]:
    if cutoff_month is None:
        return records, []
    before = [r for r in records if match_date(r)[:7] <= cutoff_month]
    after = [r for r in records if match_date(r)[:7] > cutoff_month]
    return before, after


def load_team_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for record in iter_json_array(path):
        data = record.get("pandaScoreData") or {}
        team_id = unwrap(data.get("_id"))
        if team_id is not None and data.get("name"):
            names[int(team_id)] = str(data["name"])
    return names


def scheduled_matches(path: Path, as_of: datetime, horizon_days: int = 14) -> list[Record]:
    """Matches with status `not_started` scheduled within [as_of, as_of + horizon]."""
    start = as_of.astimezone(UTC)
    end = start + timedelta(days=horizon_days)
    schedule: list[Record] = []
    for record in iter_json_array(path):
        data = record.get("pandaScoreData") or {}
        if data.get("status") != "not_started" or not data.get("scheduledAt"):
            continue
        scheduled_at = datetime.fromisoformat(str(unwrap(data["scheduledAt"])).replace("Z", "+00:00"))
        if not start <= scheduled_at <= end:
            continue
        opponents = data.get("opponents") or []
        if len(opponents) != 2:
            continue
        schedule.append(
            {
                "matchId": unwrap(data["_id"]),
                "scheduledAt": scheduled_at.isoformat(),
                "numberOfGames": data.get("numberOfGames"),
                "team1": {"id": unwrap(opponents[0]["_id"]), "name": opponents[0].get("name")},
                "team2": {"id": unwrap(opponents[1]["_id"]), "name": opponents[1].get("name")},
            }
        )
    return sorted(schedule, key=lambda m: m["scheduledAt"])


def write_json(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
