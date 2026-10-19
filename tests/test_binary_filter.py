from __future__ import annotations

import numpy as np
import pandas as pd

from src.binary_filter import create_filter_features, create_filter_target, select_binary_filter


def meta_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match|summaryStats|team1|totalMatchesPlayed": [10, 3, np.nan],
            "match|summaryStats|team2|totalMatchesPlayed": [8, 12, 5],
            "match|summaryStats|team1|winPercentage": [60.0, 50.0, 40.0],
            "match|summaryStats|team2|winPercentage": [40.0, 50.0, 70.0],
            "match|powerRanks|team1|powerRank": [100.0, 90.0, 80.0],
            "match|powerRanks|team2|powerRank": [80.0, 95.0, np.inf],
            "match|headToHeadStats|team1|totalMatchesPlayed": [2, 0, np.nan],
        }
    )


def test_filter_features() -> None:
    feats = create_filter_features(meta_frame())

    assert list(feats.columns) == [
        "min_matches_played",
        "win_pct_diff",
        "power_rank_diff",
        "h2h_matches",
        "has_h2h",
        "max_power_rank",
    ]
    assert feats.loc[0, "min_matches_played"] == 8
    assert feats.loc[0, "win_pct_diff"] == 20.0
    assert feats["has_h2h"].tolist() == [1, 0, 0]
    assert not feats.isna().any().any()


def test_filter_features_without_impute_keeps_nan() -> None:
    feats = create_filter_features(meta_frame(), impute=False)
    assert np.isnan(feats.loc[2, "min_matches_played"])
    assert np.isnan(feats.loc[2, "power_rank_diff"])


def test_filter_target() -> None:
    meta = pd.DataFrame({"noPredictionReason": ["", "  ", "LowMatches", None]})
    assert create_filter_target(meta).tolist() == [1, 1, 0, 1]


def test_select_binary_filter_uses_validation() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int).to_numpy()
    result = select_binary_filter(X.iloc[:150], y[:150], X.iloc[150:], y[150:])

    assert result["best_model_name"] in result["all_models"]
    assert result["results_df"]["val_f1_macro"].iloc[0] > 0.8
