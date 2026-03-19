from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure repo root is on sys.path for `config` and `src.*` imports.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
from src.data_loader import load_matches  # noqa: E402
from src.features import compute_team_distances_augmented, difference_distance  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.training import read_best_pca_settings, retrain_with_best_pca, run_pca_sweep  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="CS2 match predictor training pipeline.")
    parser.add_argument("--json-path", required=True, help="Path to the input CS2 matches JSON.")
    parser.add_argument("--header-path", default=None, help="Optional path to headers txt file.")
    parser.add_argument(
        "--distance-set",
        choices=["orig", "aug"],
        default="orig",
        help="Which distance features to train on (original or augmented).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.MODELS_DIR),
        help="Directory where models/scaler will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("train")

    json_path = Path(args.json_path)
    header_path = Path(args.header_path) if args.header_path else None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.output_dir)
    run_dir = out_base / run_id
    dist_dir = run_dir / f"dist_{args.distance_set}"
    dist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading matches...")
    df = load_matches(json_path=json_path, header_path=header_path)
    logger.info("Preprocessing...")
    df = preprocess(df)

    logger.info("Computing distance features...")
    (
        _orig_team1,
        _orig_team2,
        orig_distances,
        _swap_team1,
        _swap_team2,
        _swap_distances,
        _aug_team1,
        _aug_team2,
        aug_distances,
    ) = compute_team_distances_augmented(df, difference_distance)

    distance_df = orig_distances if args.distance_set == "orig" else aug_distances
    dataset_path = dist_dir / "dataset.csv"
    logger.info("Saving distance dataset to %s", dataset_path)
    distance_df.to_csv(dataset_path, index=False)

    logger.info("Scaling and splitting...")
    features = distance_df.drop(columns=["y"])
    target = distance_df["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X = pd.DataFrame(X_scaled, columns=features.columns, index=features.index)
    y = target

    joblib.dump(scaler, dist_dir / "scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # PCA не може мати `n_components` > min(n_samples, n_features).
    pca_max = min(int(X_train.shape[1]), int(X_train.shape[0]), config.MAX_PCA_COMPONENTS)
    pca_range = list(range(1, pca_max + 1))
    logger.info("Running PCA sweep: n_components=%d..%d", pca_range[0], pca_range[-1])

    classifiers = config.get_classifiers(random_state=config.RANDOM_STATE)

    _ = run_pca_sweep(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classifiers=classifiers,
        pca_range=pca_range,
        output_dir=dist_dir,
    )

    best_settings_path = dist_dir / "best_setting_result.txt"
    best_settings = read_best_pca_settings(best_settings_path)

    logger.info("Retraining models with best PCA...")
    _ = retrain_with_best_pca(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classifiers=classifiers,
        best_pca_settings=best_settings,
        output_dir=dist_dir,
    )

    logger.info("Training finished. Output dir: %s", dist_dir)


if __name__ == "__main__":
    main()

