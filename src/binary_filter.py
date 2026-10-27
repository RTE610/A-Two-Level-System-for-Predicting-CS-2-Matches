"""
Binary-Filter (level 1 of the two-stage system).

Target:
  noPredictionReason == "" -> 1 (accepted)
  otherwise              -> 0 (rejected)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import evaluate_model

RANDOM_STATE = 42


def _find_column_by_suffix(df: pd.DataFrame, suffix_options: list[str]) -> str | None:
    """
    Find first column that ends with one of the given suffixes.
    """
    for suffix in suffix_options:
        for col in df.columns:
            if col.endswith(suffix):
                return col
    return None


def _to_numeric_series(df: pd.DataFrame, col: str | None, default: float = np.nan) -> pd.Series:
    """
    Return numeric series for a column or a default-filled series if missing.
    """
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def create_filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build meta-features for the Binary-Filter.
    """
    t1_matches_col = _find_column_by_suffix(df, ["team1|totalMatchesPlayed"])
    t2_matches_col = _find_column_by_suffix(df, ["team2|totalMatchesPlayed"])

    t1_winpct_col = _find_column_by_suffix(df, ["team1|winPercentage"])
    t2_winpct_col = _find_column_by_suffix(df, ["team2|winPercentage"])

    t1_rank_col = _find_column_by_suffix(df, ["team1|powerRank"])
    t2_rank_col = _find_column_by_suffix(df, ["team2|powerRank"])

    h2h_col = _find_column_by_suffix(
        df,
        [
            "headToHeadStats|team1|totalMatchesPlayed",
            "team1|headToHeadStats|totalMatchesPlayed",
        ],
    )

    t1_matches = _to_numeric_series(df, t1_matches_col)
    t2_matches = _to_numeric_series(df, t2_matches_col)
    t1_winpct = _to_numeric_series(df, t1_winpct_col)
    t2_winpct = _to_numeric_series(df, t2_winpct_col)
    t1_rank = _to_numeric_series(df, t1_rank_col)
    t2_rank = _to_numeric_series(df, t2_rank_col)
    h2h_matches = _to_numeric_series(df, h2h_col, default=0.0).fillna(0.0)

    features = pd.DataFrame(index=df.index)
    features["min_matches_played"] = np.minimum(t1_matches, t2_matches)
    features["win_pct_diff"] = np.abs(t1_winpct - t2_winpct)
    features["power_rank_diff"] = np.abs(t1_rank - t2_rank)
    features["h2h_matches"] = h2h_matches
    features["has_h2h"] = (h2h_matches > 0).astype(int)
    features["max_power_rank"] = np.maximum(t1_rank, t2_rank)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean(numeric_only=True))
    if features.isna().sum().sum() > 0:
        features = features.fillna(0.0)
    return features


def create_filter_target(df_meta: pd.DataFrame) -> np.ndarray:
    """
    Build binary filter target from noPredictionReason.
    """
    if "noPredictionReason" in df_meta.columns:
        reasons = df_meta["noPredictionReason"]
    elif "match|noPredictionReason" in df_meta.columns:
        reasons = df_meta["match|noPredictionReason"]
    else:
        # If field is not present, default to accepted for all (best effort).
        reasons = pd.Series([""] * len(df_meta), index=df_meta.index)

    reasons = reasons.fillna("").astype(str).str.strip()
    return (reasons == "").astype(int).to_numpy()


def train_binary_filter(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    dataset_name: str = "",
) -> dict[str, Any]:
    """
    Train 4 Binary-Filter models and select the best by F1.
    """
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.mean(numeric_only=True)).fillna(0.0)

    unique_classes = np.unique(y)
    if unique_classes.shape[0] < 2:
        # Degenerate case: only one class is available in this dataset split.
        # Train a constant classifier to keep pipeline end-to-end runnable.
        const_label = int(unique_classes[0]) if unique_classes.shape[0] == 1 else 1
        dummy = DummyClassifier(strategy="constant", constant=const_label)
        dummy.fit(X_df, y)

        scaler = StandardScaler().fit(X_df)
        y_pred = dummy.predict(X_df)
        metrics = evaluate_model(y, y_pred, y_proba=None)
        results_df = pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "model": "DummyConstant",
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                }
            ]
        )
        return {
            "best_model": dummy,
            "best_scaler": scaler,
            "best_model_name": "DummyConstant",
            "results_df": results_df,
            "X_test": X_df,
            "y_test": y,
            "all_models": {"DummyConstant": dummy},
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1),
        "MLP": MLPClassifier(max_iter=5000, random_state=RANDOM_STATE),
    }

    trained_models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            except Exception:
                y_proba = None

        metrics = evaluate_model(y_test, y_pred, y_proba=y_proba)
        rows.append(
            {
                "dataset": dataset_name,
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            }
        )
        trained_models[name] = model

        if metrics["f1_macro"] > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_name = name

    results_df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    assert best_name is not None

    return {
        "best_model": trained_models[best_name],
        "best_scaler": scaler,
        "best_model_name": best_name,
        "results_df": results_df,
        "X_test": X_test,
        "y_test": y_test,
        "all_models": trained_models,
    }


def apply_filter(X: np.ndarray | pd.DataFrame, model: Any, scaler: StandardScaler) -> np.ndarray:
    """
    Apply trained Binary-Filter model (returns 1=accepted, 0=rejected).
    """
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.mean(numeric_only=True)).fillna(0.0)

    # Keep scaler schema when available.
    if hasattr(scaler, "feature_names_in_"):
        X_df = X_df.reindex(columns=list(scaler.feature_names_in_)).fillna(0.0)

    X_scaled = scaler.transform(X_df)
    return model.predict(X_scaled).astype(int)
