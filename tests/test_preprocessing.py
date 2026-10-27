from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.preprocessing import preprocess


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.POWER_RANK_COLUMNS[0]: [10.0, {"$numberDouble": "Infinity"}, 30.0],
            config.POWER_RANK_COLUMNS[1]: [5.0, 15.0, None],
            "match|summaryStats|team1|kills": [1.0, np.nan, 3.0],
            "match|summaryStats|team1|pandaScoreTeamId": [1, 2, 3],
            "match|actualWinnerId": [1, 2, None],
            "y": [1.0, 0.0, None],
        }
    )


def test_drops_leaky_columns_and_unlabelled_rows() -> None:
    df = preprocess(raw_frame())

    assert len(df) == 2
    assert not any("pandaScoreTeamId" in c for c in df.columns)
    assert "match|actualWinnerId" not in df.columns


def test_infinity_power_rank_becomes_nan_then_mean() -> None:
    df = preprocess(raw_frame())
    assert df[config.POWER_RANK_COLUMNS[0]].tolist() == [10.0, 20.0]


def test_no_impute_keeps_nan_for_train_only_fitting() -> None:
    df = preprocess(raw_frame(), impute=False)

    assert np.isnan(df[config.POWER_RANK_COLUMNS[0]].iloc[1])
    assert np.isnan(df["match|summaryStats|team1|kills"].iloc[1])
