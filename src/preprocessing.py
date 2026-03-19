from __future__ import annotations

import numpy as np
import pandas as pd

import config


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw matches dataframe:
    1) Convert {$numberDouble: 'Infinity'} to NaN for powerRank columns
    2) Cast integer columns to float
    3) Drop unnecessary columns
    4) Remove rows without target `y`
    5) Fill remaining NaNs with column means
    """

    df = df.copy()

    def convert_infinity(x):
        if isinstance(x, dict) and x.get("$numberDouble") == "Infinity":
            return np.nan
        return x

    # 1) Infinity conversion + numeric coercion + fill NaN with mean
    df[config.POWER_RANK_COLUMNS] = df[config.POWER_RANK_COLUMNS].map(convert_infinity)
    for col in config.POWER_RANK_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)

    # 2) Convert integer columns to float for consistent sklearn behavior.
    int_columns = df.select_dtypes(include=["int64"]).columns
    if len(int_columns) > 0:
        df[int_columns] = df[int_columns].astype(float)

    # 3) Drop columns not used for modeling.
    df.drop(columns=config.COLUMNS_TO_DROP, inplace=True, errors="ignore")

    # 4) Remove rows without target.
    df = df[df["y"].notna()]

    # 5) Fill NaN with means (as in the original notebook).
    # TODO: consider median/KNN/drop if this hurts performance.
    df = df.fillna(df.mean(numeric_only=True))

    return df

