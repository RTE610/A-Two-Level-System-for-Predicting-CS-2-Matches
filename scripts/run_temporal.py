"""
v2.0 evaluation of the two-stage system with an honest train/validation/test protocol.

- train: fit imputation, scaling, PCA, Binary-Filter and Match-Predictor
- validation: choose PCA components, the predictor classifier and the filter model
- test: touched once for the final report (accuracy with bootstrap 95% CI, coverage, ROC-AUC)

`--split random` reproduces a v1.0-style shuffled split on the same data for comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
from src.binary_filter import create_filter_target, select_binary_filter  # noqa: E402
from src.inference import ReasonThresholds, TrainImputer, TwoStageModel, build_features  # noqa: E402
from src.splits import SplitIndices, bootstrap_accuracy_ci, date_ranges, random_split, temporal_split  # noqa: E402
from src.tracking import FILTER_NAME, PREDICTOR_NAME, log_run  # noqa: E402
from src.training import run_pca_sweep  # noqa: E402

DATE_COL = "match|matchStartingDateTime|$date"
REASON_COL = "match|noPredictionReason"
log = logging.getLogger("run_temporal")


@dataclass
class Frames:
    dates: pd.Series
    filter_X: pd.DataFrame
    filter_y: np.ndarray
    dist_X: pd.DataFrame
    dist_X_swapped: pd.DataFrame
    y: np.ndarray
    headers: list[str]


def build_frames(json_path: Path, headers: list[str] | None = None) -> Frames:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    df_raw, features = build_features(records, require_target=True, headers=headers)
    reasons = df_raw[REASON_COL] if REASON_COL in df_raw.columns else pd.Series("", index=df_raw.index)
    return Frames(
        dates=df_raw[DATE_COL],
        filter_X=features.filter_X,
        filter_y=create_filter_target(pd.DataFrame({"noPredictionReason": reasons})),
        dist_X=features.dist_X,
        dist_X_swapped=features.dist_X_swapped,
        y=df_raw["y"].astype(int).to_numpy(),
        headers=[c for c in df_raw.columns if c != "y"],
    )


def pca_range(mode: str, n_features: int, n_samples: int) -> list[int]:
    max_, step = {"ultra": (10, 3), "fast": (40, 2), "full": (config.MAX_PCA_COMPONENTS, 1)}[mode]
    upper = min(n_features, n_samples, max_, config.MAX_PCA_COMPONENTS)
    comps = list(range(1, upper + 1, step))
    if comps[-1] != upper:
        comps.append(upper)
    return comps


def predictors(mode: str) -> dict[str, Any]:
    cls = config.get_classifiers(random_state=config.RANDOM_STATE, fast=mode != "full")
    keep = {
        "ultra": ["Logistic Regression", "Random Forest", "LightGBM"],
        "fast": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
    }.get(mode)
    return cls if keep is None else {k: v for k, v in cls.items() if k in keep}


def evaluate_two_stage(
    frames: Frames,
    split: SplitIndices,
    mode: str,
    out_dir: Path,
    extra_test: Frames | None = None,
) -> tuple[dict[str, Any], dict[str, Pipeline]]:
    tr, va, te = split.train, split.val, split.test
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Binary-Filter
    f_imp = TrainImputer().fit(frames.filter_X.iloc[tr])
    fX = {k: f_imp.transform(frames.filter_X.iloc[idx]) for k, idx in {"train": tr, "val": va}.items()}
    flt = select_binary_filter(fX["train"], frames.filter_y[tr], fX["val"], frames.filter_y[va])
    f_model, f_scaler = flt["best_model"], flt["best_scaler"]

    # Stage 2: Match-Predictor
    d_imp = TrainImputer().fit(frames.dist_X.iloc[tr])
    scaler = StandardScaler().fit(d_imp.transform(frames.dist_X.iloc[tr]))

    def prep(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(scaler.transform(d_imp.transform(X)), columns=d_imp.columns_)

    X_tr, X_va = prep(frames.dist_X.iloc[tr]), prep(frames.dist_X.iloc[va])
    y_tr, y_va = frames.y[tr], frames.y[va]
    accepted_val = f_model.predict(f_scaler.transform(fX["val"])).astype(bool)

    sweep = run_pca_sweep(
        X_train=X_tr,
        X_test=X_va,
        y_train=y_tr,
        y_test=y_va,
        classifiers=predictors(mode),
        pca_range=pca_range(mode, X_tr.shape[1], X_tr.shape[0]),
        output_dir=out_dir / "pca_sweep",
    )

    def make_bundle(name: str) -> TwoStageModel:
        return TwoStageModel(
            filter_imputer=f_imp,
            filter_scaler=f_scaler,
            filter_model=f_model,
            imputer=d_imp,
            scaler=scaler,
            pca=sweep["best_pcas"][name],
            model=sweep["best_models"][name],
            reasons=ReasonThresholds.fit(fX["train"], frames.filter_y[tr]),
            headers=frames.headers,
        )

    def symmetric_pred(bundle: TwoStageModel, f: Frames, idx: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        dist = f.dist_X if idx is None else f.dist_X.iloc[idx]
        swapped = f.dist_X_swapped if idx is None else f.dist_X_swapped.iloc[idx]
        proba = bundle.team1_win_proba(dist, swapped)
        return proba, (proba >= 0.5).astype(int)

    rows: list[dict[str, Any]] = []
    for name in sweep["best_models"]:
        _, pred_va = symmetric_pred(make_bundle(name), frames, va)
        rows.append(
            {
                "model": name,
                "pca_components": int(sweep["best_pcas"][name].n_components_),
                "val_baseline_accuracy": float(accuracy_score(y_va, pred_va)),
                "val_proposed_accuracy": float(accuracy_score(y_va[accepted_val], pred_va[accepted_val]))
                if accepted_val.any()
                else float("nan"),
            }
        )
    selection = pd.DataFrame(rows).sort_values("val_proposed_accuracy", ascending=False).reset_index(drop=True)
    selection.to_csv(out_dir / "model_selection_val.csv", index=False)
    best = str(selection.iloc[0]["model"])
    bundle = make_bundle(best)

    def report(f: Frames, idx: np.ndarray | None) -> dict[str, Any]:
        filter_X = f.filter_X if idx is None else f.filter_X.iloc[idx]
        y = f.y if idx is None else f.y[idx]
        filt_y = f.filter_y if idx is None else f.filter_y[idx]
        acc_mask = bundle.accept(filter_X).astype(bool)
        proba, pred = symmetric_pred(bundle, f, idx)
        base = bootstrap_accuracy_ci(y, pred)
        prop = bootstrap_accuracy_ci(y[acc_mask], pred[acc_mask])
        return {
            "n_matches": int(len(y)),
            "baseline_accuracy": base[0],
            "baseline_ci95": [base[1], base[2]],
            "proposed_accuracy": prop[0],
            "proposed_ci95": [prop[1], prop[2]],
            "coverage": float(acc_mask.mean()),
            "accepted_matches": int(acc_mask.sum()),
            "roc_auc_baseline": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else None,
            "roc_auc_proposed": float(roc_auc_score(y[acc_mask], proba[acc_mask]))
            if len(set(y[acc_mask])) > 1
            else None,
            "filter_accuracy": float(accuracy_score(filt_y, acc_mask.astype(int))),
        }

    result: dict[str, Any] = {
        "split_sizes": split.sizes(),
        "date_ranges": date_ranges(frames.dates, split),
        "filter_model": flt["best_model_name"],
        "predictor_model": best,
        "pca_components": int(bundle.pca.n_components_),
        "symmetric_probability": True,
        "test": report(frames, te),
    }
    if extra_test is not None:
        result["post_cutoff"] = report(extra_test, None)

    bundle.metadata = {
        "filter_model": flt["best_model_name"],
        "predictor_model": best,
        "pca_components": int(bundle.pca.n_components_),
        "train_until": result["date_ranges"]["train"][1],
        "test_proposed_accuracy": result["test"]["proposed_accuracy"],
    }
    flt["results_df"].to_csv(out_dir / "filter_selection_val.csv", index=False)
    joblib.dump(bundle, out_dir / "bundle.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pipelines = {
        PREDICTOR_NAME: Pipeline([("scaler", scaler), ("pca", bundle.pca), ("clf", bundle.model)]),
        FILTER_NAME: Pipeline([("scaler", f_scaler), ("clf", f_model)]),
    }
    return result, pipelines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-path", type=Path, default=ROOT_DIR / "data" / "matches_v2.json")
    parser.add_argument("--extra-test-json", type=Path, default=None, help="Optional out-of-period test set.")
    parser.add_argument("--split", choices=["temporal", "random", "both"], default="both")
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--mode", choices=["ultra", "fast", "full"], default="fast")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "v2")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (default: sqlite:///mlflow.db).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    log.info("Loading %s", args.json_path)
    frames = build_frames(args.json_path)
    extra = build_frames(args.extra_test_json, headers=frames.headers) if args.extra_test_json else None
    log.info("Labelled matches: %d, filter features: %d, predictor features: %d",
             len(frames.y), frames.filter_X.shape[1], frames.dist_X.shape[1])  # fmt: skip

    splits = {
        "temporal": lambda: temporal_split(frames.dates, args.fractions),
        "random": lambda: random_split(len(frames.y), args.fractions, seed=config.RANDOM_STATE),
    }
    names = ["temporal", "random"] if args.split == "both" else [args.split]
    summary: dict[str, Any] = {}
    for name in names:
        log.info("=== %s split ===", name)
        split = splits[name]()
        out_dir = args.output_dir / name
        summary[name], pipelines = evaluate_two_stage(
            frames, split, args.mode, out_dir, extra if name == "temporal" else None
        )
        log.info("%s: %s", name, json.dumps(summary[name]["test"], indent=None))
        if not args.no_mlflow:
            run_id = log_run(
                run_name=f"{name}-{args.mode}",
                params={
                    "split": name,
                    "fractions": list(args.fractions),
                    "mode": args.mode,
                    "distance": "difference",
                    "filter_model": summary[name]["filter_model"],
                    "predictor_model": summary[name]["predictor_model"],
                    "pca_components": summary[name]["pca_components"],
                    **{f"rows_{k}": v for k, v in split.sizes().items()},
                },
                metrics={k: v for k, v in summary[name].items() if k in ("test", "post_cutoff")},
                data_path=args.json_path,
                artifacts_dir=out_dir,
                models=pipelines if name == "temporal" else None,
                **({"tracking_uri": args.tracking_uri} if args.tracking_uri else {}),
            )
            log.info("MLflow run: %s", run_id)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
