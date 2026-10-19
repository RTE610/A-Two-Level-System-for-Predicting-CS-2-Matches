from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_matches
from src.features import compute_team_distances_augmented, difference_distance
from src.splits import temporal_split
from tests.fixtures import synthetic_matches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_temporal  # noqa: E402


def write(tmp_path: Path, n: int = 160) -> Path:
    path = tmp_path / "matches.json"
    path.write_text(json.dumps(synthetic_matches(n)))
    return path


def test_load_matches_builds_target(tmp_path: Path) -> None:
    df = load_matches(write(tmp_path, 20))

    assert len(df) == 20
    assert set(df["y"].dropna().unique()) <= {0, 1}
    assert "match|summaryStats|team1|winPercentage" in df.columns
    assert "match|summaryStats|team2|allAverages|gameAverages|adr" in df.columns


def test_difference_distance_augmentation_flips_target() -> None:
    df = pd.DataFrame({"a|team1|x": [3.0, 1.0], "a|team2|x": [1.0, 4.0], "y": [1, 0]})
    out = compute_team_distances_augmented(df, difference_distance)
    orig, swapped = out[2], out[5]

    assert orig["a|teams_distance|x"].tolist() == [2.0, -3.0]
    assert swapped["a|teams_distance|x"].tolist() == [-2.0, 3.0]
    assert swapped["y"].tolist() == [0, 1]


def test_train_imputer_uses_train_statistics_only() -> None:
    train = pd.DataFrame({"a": [1.0, 3.0], "b": [np.nan, np.nan]})
    test = pd.DataFrame({"a": [np.nan], "b": [5.0]})
    imp = run_temporal.TrainImputer().fit(train)

    assert imp.columns_ == ["a"]
    assert imp.transform(test).to_dict("list") == {"a": [2.0]}


def test_two_stage_end_to_end(tmp_path: Path) -> None:
    frames = run_temporal.build_frames(write(tmp_path))
    split = temporal_split(frames.dates)
    result, pipelines = run_temporal.evaluate_two_stage(frames, split, mode="ultra", out_dir=tmp_path / "out")

    test = result["test"]
    assert result["date_ranges"]["train"][1] <= result["date_ranges"]["test"][0]
    assert 0.0 <= test["coverage"] <= 1.0
    assert test["baseline_ci95"][0] <= test["baseline_accuracy"] <= test["baseline_ci95"][1]
    assert test["baseline_accuracy"] > 0.6
    assert (tmp_path / "out" / "metrics.json").exists()
    assert set(pipelines) == {"cs2-match-predictor", "cs2-binary-filter"}
