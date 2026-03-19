from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
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
    parser.add_argument(
        "--mode",
        choices=["ultra", "fast", "full"],
        default="fast",
        help="Training mode: fast (recommended) or full sweep.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU for supported models (XGBoost/LightGBM/CatBoost).",
    )
    parser.add_argument(
        "--pca-max",
        type=int,
        default=None,
        help="Optional PCA upper bound. Overrides mode defaults.",
    )
    parser.add_argument(
        "--pca-step",
        type=int,
        default=None,
        help="Optional PCA step. Overrides mode defaults.",
    )
    parser.add_argument(
        "--model-set",
        choices=["ultra", "fast", "full"],
        default=None,
        help="Optional model subset. Defaults to mode value.",
    )
    return parser.parse_args()


def _select_classifiers(classifiers: dict[str, object], model_set: str) -> dict[str, object]:
    """
    Choose model subset for faster experimentation.
    """
    if model_set == "full":
        return classifiers

    if model_set == "ultra":
        # Ultra-fast subset: minimal but representative models.
        keep = [
            "Logistic Regression",
            "Random Forest",
            "LightGBM",
        ]
        return {k: v for k, v in classifiers.items() if k in keep}

    # Fast subset: good speed/quality trade-off.
    keep = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]
    return {k: v for k, v in classifiers.items() if k in keep}


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

    # Sanitize feature matrix for sklearn/PCA:
    # - force numeric
    # - replace inf with NaN
    # - drop columns that are entirely NaN
    # - fill remaining NaN with column means
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = features.columns[features.isna().all()].tolist()
    if all_nan_cols:
        logger.warning("Dropping %d all-NaN feature columns before training.", len(all_nan_cols))
        features = features.drop(columns=all_nan_cols)

    features = features.fillna(features.mean(numeric_only=True))

    remaining_nan = int(features.isna().sum().sum())
    if remaining_nan > 0:
        # Final fallback: remove rows with unresolved NaN.
        logger.warning("Found %d NaN values after imputation; dropping affected rows.", remaining_nan)
        valid_mask = features.notna().all(axis=1)
        features = features.loc[valid_mask].copy()
        target = target.loc[valid_mask].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X = pd.DataFrame(X_scaled, columns=features.columns, index=features.index)
    y = target

    joblib.dump(scaler, dist_dir / "scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # PCA не може мати `n_components` > min(n_samples, n_features).
    mode = args.mode
    model_set = args.model_set or mode
    if mode == "ultra":
        default_pca_max = 10
        default_pca_step = 3
    elif mode == "fast":
        default_pca_max = 40
        default_pca_step = 2
    else:
        default_pca_max = config.MAX_PCA_COMPONENTS
        default_pca_step = 1

    pca_max_raw = args.pca_max if args.pca_max is not None else default_pca_max
    pca_step = args.pca_step if args.pca_step is not None else default_pca_step
    pca_step = max(1, int(pca_step))

    pca_max = min(int(X_train.shape[1]), int(X_train.shape[0]), int(pca_max_raw), config.MAX_PCA_COMPONENTS)
    pca_range = list(range(1, pca_max + 1, pca_step))
    if pca_range[-1] != pca_max:
        pca_range.append(pca_max)
    logger.info("Mode=%s, model_set=%s, use_gpu=%s", mode, model_set, args.use_gpu)
    logger.info(
        "Running PCA sweep: n_components=%d..%d (step=%d, points=%d)",
        pca_range[0],
        pca_range[-1],
        pca_step,
        len(pca_range),
    )

    classifiers_all = config.get_classifiers(
        random_state=config.RANDOM_STATE,
        fast=(mode in {"ultra", "fast"}),
        use_gpu=args.use_gpu,
    )
    classifiers = _select_classifiers(classifiers_all, model_set=model_set)
    logger.info("Using %d classifiers.", len(classifiers))

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

