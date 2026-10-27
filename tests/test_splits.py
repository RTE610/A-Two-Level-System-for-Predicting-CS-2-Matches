from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.splits import bootstrap_accuracy_ci, date_ranges, random_split, temporal_split


def shuffled_dates(n: int, seed: int = 0) -> list[str]:
    days = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    return [d.isoformat() for d in rng.permutation(days)]


def test_temporal_split_is_chronological() -> None:
    dates = shuffled_dates(100)
    split = temporal_split(dates, (0.7, 0.15, 0.15))
    parsed = pd.to_datetime(pd.Series(dates), utc=True)

    assert split.sizes() == {"train": 70, "val": 15, "test": 15}
    assert parsed.iloc[split.train].max() < parsed.iloc[split.val].min()
    assert parsed.iloc[split.val].max() < parsed.iloc[split.test].min()


def test_split_partitions_all_rows_without_overlap() -> None:
    for split in (temporal_split(shuffled_dates(50)), random_split(50)):
        all_idx = np.concatenate([split.train, split.val, split.test])
        assert sorted(all_idx.tolist()) == list(range(50))


def test_random_split_is_reproducible() -> None:
    a, b = random_split(40, seed=1), random_split(40, seed=1)
    assert np.array_equal(a.test, b.test)


@pytest.mark.parametrize("fractions", [(0.5, 0.5), (0.7, 0.2, 0.2), (1.0, 0.0, 0.0)])
def test_invalid_fractions(fractions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        temporal_split(shuffled_dates(20), fractions)


def test_unparseable_dates_rejected() -> None:
    with pytest.raises(ValueError):
        temporal_split(["2025-01-01", "not a date", "2025-01-03"] * 5)


def test_date_ranges() -> None:
    dates = shuffled_dates(20)
    ranges = date_ranges(dates, temporal_split(dates))
    assert ranges["train"][0] == "2025-01-01"
    assert ranges["test"][1] == "2025-01-20"


def test_bootstrap_ci_contains_point_estimate() -> None:
    y_true = np.array([1, 0] * 100)
    y_pred = y_true.copy()
    y_pred[:60] = 1 - y_pred[:60]
    point, low, high = bootstrap_accuracy_ci(y_true, y_pred)

    assert point == pytest.approx(0.7)
    assert low < point < high
    assert high - low < 0.2


def test_bootstrap_ci_empty() -> None:
    assert all(np.isnan(v) for v in bootstrap_accuracy_ci([], []))
