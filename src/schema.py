"""Schema validation for snapshot records (task 8)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict) and len(value) == 1:
        key, inner = next(iter(value.items()))
        if key in {"$numberLong", "$numberInt"}:
            return int(inner)
        if key in {"$date", "$oid", "$numberDouble"}:
            return inner
    return value


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _unwrap_mongo(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: _unwrap(v) for k, v in data.items()}
        return data


class TeamSummary(_Model):
    pandaScoreTeamId: int
    totalMatchesPlayed: int = Field(ge=0)
    winPercentage: float = Field(ge=0, le=100)


class PowerRank(_Model):
    pandaScoreTeamId: int
    powerRank: float | None = None


class MatchPredictionRecord(_Model):
    matchId: int
    matchStartingDateTime: datetime
    powerRankingCalculatorVersion: float
    powerRankingCoefficientsVersion: int
    summaryStats: list[TeamSummary] = Field(min_length=2, max_length=2)
    powerRanks: list[PowerRank] = Field(min_length=2, max_length=2)
    actualWinnerId: int | None = None
    noPredictionReason: str | None = ""

    @field_validator("actualWinnerId", mode="before")
    @classmethod
    def _empty_winner(cls, value: Any) -> Any:
        return None if value in ("", 0) else value

    @model_validator(mode="after")
    def _consistent_teams(self) -> MatchPredictionRecord:
        stats_ids = [t.pandaScoreTeamId for t in self.summaryStats]
        if stats_ids[0] == stats_ids[1]:
            raise ValueError("team plays against itself")
        if [p.pandaScoreTeamId for p in self.powerRanks] != stats_ids:
            raise ValueError("powerRanks order differs from summaryStats")
        if self.actualWinnerId is not None and self.actualWinnerId not in stats_ids:
            raise ValueError("winner is not one of the teams")
        return self


@dataclass
class ValidationReport:
    total: int = 0
    invalid: int = 0
    errors: Counter[str] = field(default_factory=Counter)

    @property
    def invalid_share(self) -> float:
        return self.invalid / self.total if self.total else 0.0


def validate_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], ValidationReport]:
    """Return valid records and a report; invalid records are counted by error type, not silently dropped."""
    valid: list[dict[str, Any]] = []
    report = ValidationReport()
    for record in records:
        report.total += 1
        try:
            MatchPredictionRecord.model_validate(record)
        except ValidationError as exc:
            report.invalid += 1
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"] if not isinstance(part, int)) or "record"
            report.errors[f"{location}: {first['msg']}"] += 1
            continue
        valid.append(record)
    return valid, report
