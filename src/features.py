from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import pandas as pd


def euclidean_distance(x: float, y: float) -> float:
    """Euclidean distance for scalars."""

    return float(np.sqrt((x - y) ** 2))


def manhattan_distance(x: float, y: float) -> float:
    """Manhattan distance for scalars."""

    return float(np.abs(x - y))


def squared_euclidean_distance(x: float, y: float) -> float:
    """Squared euclidean distance for scalars."""

    return float((x - y) ** 2)


def log_euclidean_distance(x: float, y: float) -> float:
    """Absolute difference between natural logs (returns 0 for non-positive values)."""

    if x > 0 and y > 0:
        return float(np.abs(np.log(x) - np.log(y)))
    return 0.0


def bray_curtis_distance(x: float, y: float) -> float:
    """Bray-Curtis distance for scalars."""

    denom = np.abs(x) + np.abs(y)
    if denom != 0:
        return float(np.abs(x - y) / denom)
    return 0.0


def sorensen_distance(x: float, y: float) -> float:
    """Sorensen distance for scalars."""

    denom = np.abs(x) + np.abs(y)
    if denom != 0:
        return float(2.0 * np.abs(x - y) / denom)
    return 0.0


def mahalanobis_distance(x: float, y: float, var: float) -> float:
    """Mahalanobis-like distance for scalars (requires `var`)."""

    if var > 0:
        return float(np.abs(x - y) / np.sqrt(var))
    return 0.0


def relative_distance(x: float, y: float) -> float:
    """Relative difference (abs(x-y)/abs(y)) for scalars."""

    if y != 0:
        return float(np.abs(x - y) / np.abs(y))
    return 0.0


def cosine_distance(x: float, y: float) -> float:
    """Cosine distance-like measure for scalars (1 - sign-aligned similarity)."""

    if x != 0 and y != 0:
        return float(1.0 - ((x * y) / (np.abs(x) * np.abs(y))))
    return 0.0


def exponential_distance(x: float, y: float) -> float:
    """Exponential decay based on absolute difference."""

    return float(np.exp(-np.abs(x - y)))


def difference_distance(x: float, y: float) -> float:
    """Difference distance (x - y) - useful for augmentation when swapping teams."""

    return float(x - y)


def _wrap_distance_func(
    distance_func: Callable[..., float],
    distance_var: float | None,
) -> Callable[[float, float], float]:
    """
    Ensure distance_func can be called as distance_func(x, y).
    If it requires an extra `var` parameter, provide it via `distance_var`.
    """

    try:
        sig = inspect.signature(distance_func)
        n_params = len(sig.parameters)
    except Exception:
        n_params = 2

    def _safe_call_2(x: float, y: float) -> float:
        try:
            return float(distance_func(x, y))
        except Exception:
            return float(np.nan)

    if n_params <= 2:
        return _safe_call_2

    if n_params >= 3:
        if distance_var is None:
            raise ValueError("distance_func requires `distance_var` (3+ params).")

        def _safe_call_3(x: float, y: float) -> float:
            try:
                return float(distance_func(x, y, distance_var))
            except Exception:
                return float(np.nan)

        return _safe_call_3

    # Fallback
    return _safe_call_2


def _to_float_scalar(value: Any) -> float:
    """
    Convert common JSON-like scalar wrappers to float.
    Returns NaN for unsupported values.
    """
    if value is None:
        return float(np.nan)

    if isinstance(value, dict):
        if "$numberDouble" in value:
            raw = value.get("$numberDouble")
            if raw in {"Infinity", "+Infinity", "-Infinity"}:
                return float(np.nan)
            try:
                return float(raw)
            except Exception:
                return float(np.nan)
        if "$numberInt" in value:
            try:
                return float(int(value.get("$numberInt")))
            except Exception:
                return float(np.nan)
        if "$numberLong" in value:
            try:
                return float(int(value.get("$numberLong")))
            except Exception:
                return float(np.nan)
        return float(np.nan)

    try:
        return float(value)
    except Exception:
        return float(np.nan)


def compute_team_distances_augmented(
    df: pd.DataFrame,
    distance_func: Callable[..., float],
    distance_var: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute team-to-team distance features and augment by swapping teams.

    Returns:
      (orig_team1, orig_team2, orig_distances,
       swap_team1, swap_team2, swap_distances,
       aug_team1, aug_team2, aug_distances)
    """

    distance = _wrap_distance_func(distance_func, distance_var=distance_var)

    # Identify team1 and team2 columns based on naming pattern.
    team1_cols = sorted([col for col in df.columns if "team1" in col])
    team2_cols = sorted([col for col in df.columns if "team2" in col])

    if len(team1_cols) != len(team2_cols):
        raise ValueError("team1/team2 column counts do not match.")

    # ---------------------
    # Original ordering
    # ---------------------
    orig_team1_data: dict[str, pd.Series] = {}
    orig_team2_data: dict[str, pd.Series] = {}
    orig_distance_data: dict[str, list[float]] = {}

    for col1, col2 in zip(team1_cols, team2_cols):
        orig_team1_data[col1] = df[col1]
        orig_team2_data[col2] = df[col2]

        dist_col_name = col1.replace("team1", "teams_distance")

        a = df[col1].map(_to_float_scalar).to_numpy()
        b = df[col2].map(_to_float_scalar).to_numpy()
        orig_distance_data[dist_col_name] = [distance(x, y) for x, y in zip(a, b)]

    orig_team1_df = pd.DataFrame(orig_team1_data, index=df.index)
    orig_team2_df = pd.DataFrame(orig_team2_data, index=df.index)
    orig_distances = pd.DataFrame(orig_distance_data, index=df.index)
    orig_distances["y"] = df["y"]

    # ---------------------
    # Swapped ordering (augment)
    # ---------------------
    swapped_team1_data: dict[str, pd.Series] = {}
    swapped_team2_data: dict[str, pd.Series] = {}
    swapped_distance_data: dict[str, list[float]] = {}

    for col1, col2 in zip(team1_cols, team2_cols):
        swapped_team1_col = col2.replace("team2", "team1")
        swapped_team2_col = col1.replace("team1", "team2")

        swapped_team1_data[swapped_team1_col] = df[col2]
        swapped_team2_data[swapped_team2_col] = df[col1]

        dist_col_name = col1.replace("team1", "teams_distance")

        a = df[col2].map(_to_float_scalar).to_numpy()
        b = df[col1].map(_to_float_scalar).to_numpy()
        swapped_distance_data[dist_col_name] = [distance(x, y) for x, y in zip(a, b)]

    swapped_team1_df = pd.DataFrame(swapped_team1_data, index=df.index)
    swapped_team2_df = pd.DataFrame(swapped_team2_data, index=df.index)
    swapped_distances = pd.DataFrame(swapped_distance_data, index=df.index)
    swapped_distances["y"] = df["y"].apply(lambda y: 0 if y == 1 else 1)

    # ---------------------
    # Augmented dataset
    # ---------------------
    augmented_team1_df = pd.concat([orig_team1_df, swapped_team1_df], ignore_index=True)
    augmented_team2_df = pd.concat([orig_team2_df, swapped_team2_df], ignore_index=True)
    augmented_distances = pd.concat([orig_distances, swapped_distances], ignore_index=True)

    return (
        orig_team1_df,
        orig_team2_df,
        orig_distances,
        swapped_team1_df,
        swapped_team2_df,
        swapped_distances,
        augmented_team1_df,
        augmented_team2_df,
        augmented_distances,
    )

