"""Two-stage inference (Binary-Filter -> Match-Predictor) on raw match records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.binary_filter import create_filter_features
from src.data_loader import records_to_frame
from src.features import compute_team_distances_augmented, difference_distance
from src.preprocessing import preprocess

Record = dict[str, Any]


class TrainImputer:
    """Mean imputation fitted on train only; columns that are all-NaN in train are dropped."""

    def fit(self, X: pd.DataFrame) -> TrainImputer:
        self.columns_: list[str] = [c for c in X.columns if X[c].notna().any()]
        self.means_: pd.Series = X[self.columns_].mean(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.columns_).fillna(self.means_).fillna(0.0)


@dataclass(frozen=True)
class FeatureFrames:
    filter_X: pd.DataFrame
    dist_X: pd.DataFrame


def build_features(records: list[Record], require_target: bool = True) -> tuple[pd.DataFrame, FeatureFrames]:
    """Same feature path as training: flatten -> preprocess -> filter meta-features + team distances."""
    df_raw = records_to_frame(records)
    df = preprocess(df_raw, impute=False, require_target=require_target)
    distances = compute_team_distances_augmented(df, difference_distance)[2]
    dist_X = distances.drop(columns=["y"]).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frames = FeatureFrames(
        filter_X=create_filter_features(df, impute=False).reset_index(drop=True),
        dist_X=dist_X.reset_index(drop=True),
    )
    return df_raw.loc[df.index].reset_index(drop=True), frames


@dataclass(frozen=True)
class ReasonThresholds:
    """Thresholds (from accepted training matches) used to explain a rejection."""

    min_matches: float
    win_pct_diff: float
    power_rank_diff: float

    @classmethod
    def fit(cls, filter_X: pd.DataFrame, accepted: np.ndarray) -> ReasonThresholds:
        acc = filter_X[accepted.astype(bool)]
        return cls(
            min_matches=float(acc["min_matches_played"].quantile(0.10)),
            win_pct_diff=float(acc["win_pct_diff"].quantile(0.25)),
            power_rank_diff=float(acc["power_rank_diff"].quantile(0.25)),
        )

    def explain(self, row: pd.Series) -> str:
        if row["min_matches_played"] < self.min_matches:
            return f"Замало зіграних матчів в однієї з команд ({int(row['min_matches_played'])})"
        if row["win_pct_diff"] < self.win_pct_diff and row["power_rank_diff"] < self.power_rank_diff:
            return "Команди приблизно рівні за силою"
        if row["has_h2h"] == 0:
            return "Немає історії особистих зустрічей"
        return "Фільтр оцінив матч як важкопрогнозований"


@dataclass(frozen=True)
class Prediction:
    accepted: bool
    reason: str | None
    team1_win_probability: float
    incomplete_data: bool

    @property
    def winner_index(self) -> int:
        return 0 if self.team1_win_probability >= 0.5 else 1

    @property
    def win_probability(self) -> float:
        return max(self.team1_win_probability, 1.0 - self.team1_win_probability)


@dataclass
class TwoStageModel:
    filter_imputer: TrainImputer
    filter_scaler: Any
    filter_model: Any
    imputer: TrainImputer
    scaler: Any
    pca: Any
    model: Any
    reasons: ReasonThresholds
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, records: list[Record]) -> list[Prediction]:
        if not records:
            return []
        _, frames = build_features(records, require_target=False)
        fX = self.filter_imputer.transform(frames.filter_X)
        accepted = self.filter_model.predict(self.filter_scaler.transform(fX)).astype(bool)
        X = pd.DataFrame(self.scaler.transform(self.imputer.transform(frames.dist_X)), columns=self.imputer.columns_)
        proba = np.asarray(self.model.predict_proba(self.pca.transform(X)))[:, 1]
        return [
            Prediction(
                accepted=bool(accepted[i]),
                reason=None if accepted[i] else self.reasons.explain(fX.iloc[i]),
                team1_win_probability=float(proba[i]),
                incomplete_data=not _has_game_stats(records[i]),
            )
            for i in range(len(records))
        ]


def _has_game_stats(record: Record) -> bool:
    stats = record.get("summaryStats") or []
    return len(stats) == 2 and all(team.get("allAverages") for team in stats)
