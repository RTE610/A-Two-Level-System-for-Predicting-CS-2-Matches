from __future__ import annotations

import numpy as np
import pandas as pd

import config


def preprocess(df: pd.DataFrame, impute: bool = True, require_target: bool = True) -> pd.DataFrame:
    """
    Preprocess the raw matches dataframe:
    1) Convert {$numberDouble: 'Infinity'} to NaN for powerRank columns
    2) Cast integer columns to float
    3) Drop unnecessary columns
    4) Remove rows without target `y`
    5) Fill remaining NaNs with column means

    With `impute=False` steps that compute statistics over the whole dataset (mean filling)
    are skipped, so imputation can be fitted on the training split only (v2.0 pipeline).
    With `require_target=False` rows without a known winner are kept (inference on future matches).
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
        if impute:
            df[col] = df[col].fillna(df[col].mean())

    # 2) Convert integer columns to float for consistent sklearn behavior.
    int_columns = df.select_dtypes(include=["int64"]).columns
    if len(int_columns) > 0:
        df[int_columns] = df[int_columns].astype(float)

    # 3) Drop columns not used for modeling.
    df.drop(columns=config.COLUMNS_TO_DROP, inplace=True, errors="ignore")

    # 3.1) Drop leakage/technical columns by pattern.
    cols_to_drop = [col for col in df.columns if any(pattern in col for pattern in config.EXCLUDE_PATTERNS)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")
        print(f"Removed {len(cols_to_drop)} columns: {cols_to_drop[:5]}...")

    # 4) Remove rows without target.
    if require_target:
        df = df[df["y"].notna()]

    # 5) Fill NaN with means (as in the original notebook).
    # TODO: consider median/KNN/drop if this hurts performance.
    if impute:
        df = df.fillna(df.mean(numeric_only=True))

    return df
