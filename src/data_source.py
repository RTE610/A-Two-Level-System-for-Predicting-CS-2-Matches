"""
Data-source interface (ADR-0004).

The service and UI depend only on `DataSource`; `SnapshotDataSource` reads the prepared
department snapshot. A future PandaScore implementation can replace it without other changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.snapshot import unwrap

Record = dict[str, Any]


@dataclass(frozen=True)
class TeamProfile:
    team_id: int
    name: str
    total_matches: int
    win_percentage: float
    power_rank: float | None
    last_match_at: datetime
    has_game_stats: bool

    def age_days(self, as_of: datetime) -> int:
        return (as_of - self.last_match_at).days


@dataclass(frozen=True)
class ScheduledMatch:
    match_id: int
    scheduled_at: datetime
    team1_id: int
    team1_name: str
    team2_id: int
    team2_name: str


@dataclass(frozen=True)
class DataVersion:
    snapshot: str
    feature_set: str
    as_of: datetime


class DataSource(Protocol):
    as_of: datetime

    def version(self) -> DataVersion: ...
    def team_name(self, team_id: int) -> str: ...
    def search_teams(self, query: str, limit: int) -> list[TeamProfile]: ...
    def team_profile(self, team_id: int) -> TeamProfile | None: ...
    def schedule(self, horizon_days: int) -> list[ScheduledMatch]: ...
    def build_record(self, team1_id: int, team2_id: int, match_id: int | None = None) -> Record | None: ...
    def head_to_head_count(self, team1_id: int, team2_id: int) -> int: ...


def _parse_dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(unwrap(value)).replace("Z", "+00:00"))


class SnapshotDataSource:
    """Team profiles are taken from each team's latest pre-match feature record before `as_of`."""

    def __init__(self, data_dir: Path, as_of: datetime | None = None) -> None:
        meta = json.loads((data_dir / "snapshot_meta.json").read_text(encoding="utf-8"))
        self._meta = meta
        self.as_of = as_of or datetime.fromisoformat(meta["as_of"])
        if self.as_of.tzinfo is None:
            self.as_of = self.as_of.replace(tzinfo=UTC)

        names = json.loads((data_dir / "team_names.json").read_text(encoding="utf-8"))
        self._names: dict[int, str] = {int(k): v.strip() for k, v in names.items() if v and v.strip()}
        self._schedule_raw: list[Record] = json.loads((data_dir / "schedule.json").read_text(encoding="utf-8"))
        for match in self._schedule_raw:
            for side in ("team1", "team2"):
                team, name = int(match[side]["id"]), (match[side].get("name") or "").strip()
                if name and team not in self._names:
                    self._names[team] = name

        records: list[Record] = []
        for file in ("matches_v2.json", "matches_v2_post_cutoff.json"):
            path = data_dir / file
            if path.exists():
                records.extend(json.loads(path.read_text(encoding="utf-8")))

        self._by_match_id: dict[int, Record] = {}
        self._latest_by_team: dict[int, tuple[datetime, Record, int]] = {}
        self._latest_by_pair: dict[frozenset[int], tuple[datetime, Record]] = {}
        self._h2h_count: dict[frozenset[int], int] = {}
        for record in records:
            when = _parse_dt(record["matchStartingDateTime"])
            stats = record.get("summaryStats") or []
            if len(stats) != 2:
                continue
            ids = [int(unwrap(s["pandaScoreTeamId"])) for s in stats]
            self._by_match_id[int(unwrap(record["matchId"]))] = record
            if when >= self.as_of:
                continue
            for pos, team_id in enumerate(ids):
                current = self._latest_by_team.get(team_id)
                if current is None or when > current[0]:
                    self._latest_by_team[team_id] = (when, record, pos)
            pair = frozenset(ids)
            if record.get("actualWinnerId"):
                self._h2h_count[pair] = self._h2h_count.get(pair, 0) + 1
            if pair not in self._latest_by_pair or when > self._latest_by_pair[pair][0]:
                self._latest_by_pair[pair] = (when, record)

    def version(self) -> DataVersion:
        return DataVersion(snapshot=self._meta["snapshot_dir"], feature_set=self._meta["feature_set"], as_of=self.as_of)

    def team_name(self, team_id: int) -> str:
        return self._names.get(team_id, f"Team {team_id}")

    def team_profile(self, team_id: int) -> TeamProfile | None:
        latest = self._latest_by_team.get(team_id)
        if latest is None:
            return None
        when, record, pos = latest
        stats = record["summaryStats"][pos]
        power = next(
            (p.get("powerRank") for p in record.get("powerRanks") or [] if unwrap(p["pandaScoreTeamId"]) == team_id),
            None,
        )
        return TeamProfile(
            team_id=team_id,
            name=self.team_name(team_id),
            total_matches=int(stats.get("totalMatchesPlayed") or 0),
            win_percentage=float(stats.get("winPercentage") or 0.0),
            power_rank=float(power) if isinstance(power, int | float) else None,
            last_match_at=when,
            has_game_stats=bool(stats.get("allAverages")),
        )

    def search_teams(self, query: str, limit: int = 20) -> list[TeamProfile]:
        q = query.strip().lower()
        profiles = [p for tid in self._latest_by_team if (p := self.team_profile(tid)) and q in p.name.lower()]
        return sorted(profiles, key=lambda p: (-p.total_matches, p.name))[:limit]

    def head_to_head_count(self, team1_id: int, team2_id: int) -> int:
        return self._h2h_count.get(frozenset((team1_id, team2_id)), 0)

    def schedule(self, horizon_days: int = 14) -> list[ScheduledMatch]:
        out = []
        for m in self._schedule_raw:
            when = datetime.fromisoformat(m["scheduledAt"])
            if 0 <= (when - self.as_of).total_seconds() <= horizon_days * 86400:
                out.append(
                    ScheduledMatch(
                        match_id=int(m["matchId"]),
                        scheduled_at=when,
                        team1_id=int(m["team1"]["id"]),
                        team1_name=self.team_name(int(m["team1"]["id"])),
                        team2_id=int(m["team2"]["id"]),
                        team2_name=self.team_name(int(m["team2"]["id"])),
                    )
                )
        return sorted(out, key=lambda m: m.scheduled_at)

    def build_record(self, team1_id: int, team2_id: int, match_id: int | None = None) -> Record | None:
        """Pre-match feature record for the pair, oriented as (team1, team2)."""
        if match_id is not None and match_id in self._by_match_id:
            precomputed = self._oriented(self._by_match_id[match_id], team1_id, team2_id)
            if precomputed is not None:
                return precomputed

        p1, p2 = self._latest_by_team.get(team1_id), self._latest_by_team.get(team2_id)
        if p1 is None or p2 is None:
            return None
        stats = [p1[1]["summaryStats"][p1[2]], p2[1]["summaryStats"][p2[2]]]
        ranks = [self._power_entry(p1[1], team1_id), self._power_entry(p2[1], team2_id)]

        h2h: list[Any] = []
        pair = self._latest_by_pair.get(frozenset((team1_id, team2_id)))
        if pair is not None:
            oriented = self._oriented(pair[1], team1_id, team2_id)
            h2h = (oriented or {}).get("headToHeadStats") or []

        return {
            "matchId": match_id or 0,
            "matchStartingDateTime": {"$date": self.as_of.isoformat()},
            "summaryStats": stats,
            "powerRanks": ranks,
            "headToHeadStats": h2h,
            "predictionBans": [],
            "noPredictionReason": "",
        }

    @staticmethod
    def _power_entry(record: Record, team_id: int) -> Record:
        for entry in record.get("powerRanks") or []:
            if unwrap(entry["pandaScoreTeamId"]) == team_id:
                return entry
        return {"pandaScoreTeamId": team_id, "powerRank": None}

    @staticmethod
    def _oriented(record: Record, team1_id: int, team2_id: int) -> Record | None:
        ids = [unwrap(s["pandaScoreTeamId"]) for s in record.get("summaryStats") or []]
        if ids == [team1_id, team2_id]:
            return record
        if ids == [team2_id, team1_id]:
            flipped = dict(record)
            for key in ("summaryStats", "powerRanks", "headToHeadStats"):
                if isinstance(record.get(key), list) and len(record[key]) == 2:
                    flipped[key] = list(reversed(record[key]))
            return flipped
        return None
