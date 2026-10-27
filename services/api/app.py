"""Prediction API (ADR-0001): the only component that runs the model."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from src.data_source import DataSource, ScheduledMatch, SnapshotDataSource, TeamProfile
from src.inference import Prediction, TwoStageModel

log = logging.getLogger("prediction_api")

STALE_PROFILE_DAYS = 7
SCHEDULE_HORIZON_DAYS = 60


class TeamOut(BaseModel):
    id: int
    name: str


class TeamStatsOut(TeamOut):
    total_matches: int
    win_percentage: float
    power_rank: float | None
    last_match_at: datetime
    profile_age_days: int
    stale: bool
    has_game_stats: bool


class PredictionOut(BaseModel):
    match_id: int | None
    scheduled_at: datetime | None
    team1: TeamOut
    team2: TeamOut
    mode: Literal["scheduled", "hypothetical"]
    decision: Literal["accepted", "rejected", "no_data"]
    reason: str | None
    winner: TeamOut | None
    win_probability: float | None
    incomplete_data: bool
    data_as_of: datetime
    model_version: str


class MatchCardOut(BaseModel):
    prediction: PredictionOut
    team1_stats: TeamStatsOut
    team2_stats: TeamStatsOut
    head_to_head_matches: int


class PredictRequest(BaseModel):
    team1_id: int = Field(gt=0)
    team2_id: int = Field(gt=0)

    @model_validator(mode="after")
    def _different_teams(self) -> PredictRequest:
        if self.team1_id == self.team2_id:
            raise ValueError("team1_id and team2_id must differ")
        return self


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_version: str
    model: dict[str, object]
    data_snapshot: str
    data_feature_set: str
    data_as_of: datetime


@dataclass
class LoadedModel:
    model: TwoStageModel
    version: str


def load_model() -> LoadedModel:
    """Latest registered version from MLflow (ADR-0002); BUNDLE_PATH overrides it for local runs and tests."""
    bundle_path = os.environ.get("BUNDLE_PATH")
    if bundle_path:
        return LoadedModel(joblib.load(bundle_path), version=f"file:{Path(bundle_path).name}")

    import mlflow
    from mlflow.tracking import MlflowClient

    from src.tracking import DEFAULT_TRACKING_URI, PREDICTOR_NAME, latest_model_version

    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    version = latest_model_version(PREDICTOR_NAME, tracking_uri=uri)
    if version is None:
        raise RuntimeError(f"No registered versions of {PREDICTOR_NAME} in {uri}")
    run_id = MlflowClient(tracking_uri=uri).get_model_version(PREDICTOR_NAME, version).run_id
    mlflow.set_tracking_uri(uri)
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="outputs/bundle.joblib")
    return LoadedModel(joblib.load(path), version=f"{PREDICTOR_NAME}/v{version}")


def create_app(loaded: LoadedModel | None = None, source: DataSource | None = None) -> FastAPI:
    app = FastAPI(title="CS2 Match Predictor API", version="2.0.0")
    state: dict[str, object] = {}

    def model() -> LoadedModel:
        if "model" not in state:
            state["model"] = loaded or load_model()
        return state["model"]  # type: ignore[return-value]

    def data() -> DataSource:
        if "data" not in state:
            state["data"] = source or SnapshotDataSource(Path(os.environ.get("DATA_DIR", "data")))
        return state["data"]  # type: ignore[return-value]

    def team_out(team_id: int) -> TeamOut:
        return TeamOut(id=team_id, name=data().team_name(team_id))

    def stats_out(profile: TeamProfile) -> TeamStatsOut:
        age = profile.age_days(data().as_of)
        return TeamStatsOut(
            id=profile.team_id,
            name=profile.name,
            total_matches=profile.total_matches,
            win_percentage=round(profile.win_percentage, 1),
            power_rank=round(profile.power_rank, 1) if profile.power_rank is not None else None,
            last_match_at=profile.last_match_at,
            profile_age_days=age,
            stale=age > STALE_PROFILE_DAYS,
            has_game_stats=profile.has_game_stats,
        )

    def to_out(team1_id: int, team2_id: int, match: ScheduledMatch | None, result: Prediction) -> PredictionOut:
        winner_id = (team1_id, team2_id)[result.winner_index]
        return PredictionOut(
            match_id=match.match_id if match else None,
            scheduled_at=match.scheduled_at if match else None,
            team1=team_out(team1_id),
            team2=team_out(team2_id),
            mode="scheduled" if match else "hypothetical",
            decision="accepted" if result.accepted else "rejected",
            reason=result.reason,
            winner=team_out(winner_id) if result.accepted else None,
            win_probability=round(result.win_probability, 3) if result.accepted else None,
            incomplete_data=result.incomplete_data,
            data_as_of=data().as_of,
            model_version=model().version,
        )

    def log_prediction(out: PredictionOut, latency_ms: float) -> None:
        log.info(
            json.dumps(
                {
                    "event": "prediction",
                    "match_id": out.match_id,
                    "team1": out.team1.id,
                    "team2": out.team2.id,
                    "mode": out.mode,
                    "decision": out.decision,
                    "win_probability": out.win_probability,
                    "incomplete_data": out.incomplete_data,
                    "model_version": out.model_version,
                    "latency_ms": round(latency_ms, 1),
                }
            )
        )

    def predict_pair(team1_id: int, team2_id: int, match: ScheduledMatch | None) -> PredictionOut:
        started = time.perf_counter()
        record = data().build_record(team1_id, team2_id, match.match_id if match else None)
        if record is None:
            raise HTTPException(status_code=404, detail="No statistics for one of the teams before the data date")
        out = to_out(team1_id, team2_id, match, model().model.predict([record])[0])
        log_prediction(out, (time.perf_counter() - started) * 1000)
        return out

    def scheduled_predictions() -> dict[int, PredictionOut]:
        """FR-2: predictions for the whole schedule are computed once per (data, model) version, in one batch."""
        key = (data().version(), model().version)
        cached = state.get("schedule_cache")
        if isinstance(cached, tuple) and cached[0] == key:
            return cached[1]
        started = time.perf_counter()
        matches = data().schedule(SCHEDULE_HORIZON_DAYS)
        records = {m.match_id: data().build_record(m.team1_id, m.team2_id, m.match_id) for m in matches}
        known = [m for m in matches if records[m.match_id] is not None]
        results = model().model.predict([records[m.match_id] for m in known])  # type: ignore[misc]
        per_match_ms = (time.perf_counter() - started) * 1000 / max(len(matches), 1)
        out: dict[int, PredictionOut] = {}
        for match, result in zip(known, results, strict=True):
            out[match.match_id] = to_out(match.team1_id, match.team2_id, match, result)
            log_prediction(out[match.match_id], per_match_ms)
        for match in matches:
            if match.match_id not in out:
                log.info(json.dumps({"event": "schedule_no_data", "match_id": match.match_id}))
                out[match.match_id] = no_data(match)
        state["schedule_cache"] = (key, out)
        return out

    @app.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        m, v = model(), data().version()
        return HealthOut(
            status="ok",
            model_version=m.version,
            model=m.model.metadata,
            data_snapshot=v.snapshot,
            data_feature_set=v.feature_set,
            data_as_of=v.as_of,
        )

    def no_data(match: ScheduledMatch) -> PredictionOut:
        missing = [data().team_name(t) for t in (match.team1_id, match.team2_id) if data().team_profile(t) is None]
        return PredictionOut(
            match_id=match.match_id,
            scheduled_at=match.scheduled_at,
            team1=team_out(match.team1_id),
            team2=team_out(match.team2_id),
            mode="scheduled",
            decision="no_data",
            reason=f"Немає статистики команди до дати даних: {', '.join(missing)}",
            winner=None,
            win_probability=None,
            incomplete_data=True,
            data_as_of=data().as_of,
            model_version=model().version,
        )

    @app.get("/schedule", response_model=list[PredictionOut])
    def schedule(days: int = Query(14, ge=1, le=SCHEDULE_HORIZON_DAYS)) -> list[PredictionOut]:
        predictions = scheduled_predictions()
        return [predictions[m.match_id] for m in data().schedule(days)]

    @app.get("/matches/{match_id}", response_model=MatchCardOut)
    def match_card(match_id: int) -> MatchCardOut:
        match = next((m for m in data().schedule(60) if m.match_id == match_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail="Match is not in the schedule")
        p1, p2 = data().team_profile(match.team1_id), data().team_profile(match.team2_id)
        if p1 is None or p2 is None:
            raise HTTPException(status_code=404, detail="No statistics for one of the teams")
        return MatchCardOut(
            prediction=scheduled_predictions()[match.match_id],
            team1_stats=stats_out(p1),
            team2_stats=stats_out(p2),
            head_to_head_matches=data().head_to_head_count(match.team1_id, match.team2_id),
        )

    @app.get("/teams", response_model=list[TeamStatsOut])
    def teams(query: str = "", limit: int = Query(20, ge=1, le=200)) -> list[TeamStatsOut]:
        return [stats_out(p) for p in data().search_teams(query, limit)]

    @app.get("/teams/{team_id}", response_model=TeamStatsOut)
    def team(team_id: int) -> TeamStatsOut:
        profile = data().team_profile(team_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Unknown team or no matches before the data date")
        return stats_out(profile)

    @app.post("/predict", response_model=PredictionOut)
    def predict(req: PredictRequest) -> PredictionOut:
        return predict_pair(req.team1_id, req.team2_id, None)

    return app


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(message)s")
app = create_app()
