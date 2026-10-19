"""Train/validation/test splitting and uncertainty estimates for v2.0 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

IntArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SplitIndices:
    train: IntArray
    val: IntArray
    test: IntArray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _check_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        raise ValueError("Expected three positive fractions for train/val/test.")
    total = float(sum(fractions))
    if not np.isclose(total, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {total}.")
    return float(fractions[0]), float(fractions[1]), float(fractions[2])


def _cut(order: IntArray, fractions: tuple[float, float, float]) -> SplitIndices:
    n = len(order)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    if n_train == 0 or n_val == 0 or n - n_train - n_val <= 0:
        raise ValueError(f"Not enough rows ({n}) for fractions {fractions}.")
    return SplitIndices(
        train=order[:n_train],
        val=order[n_train : n_train + n_val],
        test=order[n_train + n_val :],
    )


def temporal_split(
    dates: Sequence[str] | pd.Series,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
) -> SplitIndices:
    """
    Positional indices split by match date: oldest -> train, newest -> test.

    Ties are broken by original position, so the split is deterministic.
    """
    parsed = pd.to_datetime(pd.Series(list(dates)), utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{int(parsed.isna().sum())} rows have unparseable dates.")
    order = np.asarray(np.argsort(parsed.to_numpy(), kind="stable"), dtype=np.int64)
    return _cut(order, _check_fractions(fractions))


def random_split(n_rows: int, fractions: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 42) -> SplitIndices:
    """v1.0-style shuffled split, kept only as a comparison baseline."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows).astype(np.int64)
    return _cut(order, _check_fractions(fractions))


def date_ranges(dates: Sequence[str] | pd.Series, split: SplitIndices) -> dict[str, tuple[str, str]]:
    parsed = pd.to_datetime(pd.Series(list(dates)), utc=True)
    out: dict[str, tuple[str, str]] = {}
    for name in ("train", "val", "test"):
        part = parsed.iloc[getattr(split, name)]
        out[name] = (part.min().date().isoformat(), part.max().date().isoformat())
    return out


def bootstrap_accuracy_ci(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Accuracy with a percentile bootstrap confidence interval: (point, low, high)."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    if correct.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(correct, size=(n_resamples, correct.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(correct.mean()), float(low), float(high)
