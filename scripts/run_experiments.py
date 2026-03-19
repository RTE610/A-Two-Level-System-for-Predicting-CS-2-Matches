from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
from src.binary_filter import (  # noqa: E402
    apply_filter,
    create_filter_features,
    create_filter_target,
    train_binary_filter,
)
from src.data_loader import load_matches  # noqa: E402
from src.evaluation import compare_models  # noqa: E402
from src.features import compute_team_distances_augmented, difference_distance  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.training import run_pca_sweep  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_baseline_vs_proposed,
    plot_confusion_matrix,
    plot_coverage_vs_accuracy,
    plot_dataset_comparison,
    plot_feature_importance_values,
    plot_filter_model_comparison,
    plot_pca_accuracy_curves,
    plot_roc_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full two-stage CS2 experiments (Binary-Filter + Match-Predictor).")
    parser.add_argument("--json-60", type=str, required=True, help="Path to 60-day JSON.")
    parser.add_argument("--json-90", type=str, required=True, help="Path to 90-day JSON.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for tables/figures.")
    parser.add_argument("--distance-set", choices=["orig", "aug"], default="orig", help="Distance dataset type.")
    parser.add_argument("--mode", choices=["ultra", "fast", "full"], default="fast", help="Predictor mode.")
    parser.add_argument("--pca-max", type=int, default=None, help="Optional PCA max components.")
    parser.add_argument("--pca-step", type=int, default=None, help="Optional PCA step.")
    return parser.parse_args()


def _ensure_dirs(base_out: Path) -> tuple[Path, Path, Path]:
    tables = base_out / "tables"
    figures = base_out / "figures"
    models = base_out / "models"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return tables, figures, models


def _select_predictor_classifiers(mode: str) -> dict[str, Any]:
    cls = config.get_classifiers(
        random_state=config.RANDOM_STATE,
        fast=(mode in {"ultra", "fast"}),
        use_gpu=False,
    )
    if mode == "full":
        return cls
    if mode == "ultra":
        keep = ["Logistic Regression", "Random Forest", "LightGBM"]
    else:
        keep = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    return {k: v for k, v in cls.items() if k in keep}


def _get_pca_range(mode: str, n_features: int, n_samples: int, pca_max_arg: int | None, pca_step_arg: int | None) -> list[int]:
    if mode == "ultra":
        default_max, default_step = 10, 3
    elif mode == "fast":
        default_max, default_step = 40, 2
    else:
        default_max, default_step = config.MAX_PCA_COMPONENTS, 1

    pca_max_raw = pca_max_arg if pca_max_arg is not None else default_max
    step = max(1, int(pca_step_arg if pca_step_arg is not None else default_step))
    pca_max = min(n_features, n_samples, int(pca_max_raw), config.MAX_PCA_COMPONENTS)
    comps = list(range(1, pca_max + 1, step))
    if comps[-1] != pca_max:
        comps.append(pca_max)
    return comps


def _sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)
    return X


def _filter_target_from_df(df: pd.DataFrame) -> np.ndarray:
    meta_cols = pd.DataFrame(index=df.index)
    if "match|noPredictionReason" in df.columns:
        meta_cols["noPredictionReason"] = df["match|noPredictionReason"]
    elif "noPredictionReason" in df.columns:
        meta_cols["noPredictionReason"] = df["noPredictionReason"]
    else:
        meta_cols["noPredictionReason"] = ""
    return create_filter_target(meta_cols)


def _save_filter_importance(best_model: Any, feature_names: list[str], save_path: Path) -> None:
    if hasattr(best_model, "feature_importances_"):
        imp = np.asarray(best_model.feature_importances_)
    elif hasattr(best_model, "coef_"):
        imp = np.abs(np.asarray(best_model.coef_)).ravel()
    else:
        return
    plot_feature_importance_values(feature_names=feature_names, importances=imp, save_path=save_path, top_n=min(20, len(feature_names)))


def _save_predictor_importance(best_model: Any, best_pca: Any, original_feature_names: list[str], save_path: Path) -> None:
    if hasattr(best_model, "feature_importances_"):
        pc_imp = np.asarray(best_model.feature_importances_).ravel()
    elif hasattr(best_model, "coef_"):
        pc_imp = np.abs(np.asarray(best_model.coef_)).ravel()
    else:
        return
    if not hasattr(best_pca, "components_"):
        return
    if pc_imp.shape[0] != best_pca.components_.shape[0]:
        return
    orig_imp = np.abs(best_pca.components_.T @ pc_imp)
    plot_feature_importance_values(
        feature_names=original_feature_names,
        importances=orig_imp,
        save_path=save_path,
        top_n=min(20, len(original_feature_names)),
    )


def run_for_dataset(
    dataset_name: str,
    json_path: Path,
    args: argparse.Namespace,
    tables_dir: Path,
    figures_dir: Path,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"=== STEP 1: Loading data ({dataset_name}) ===")
    df_raw = load_matches(json_path=json_path, header_path=None)
    # IMPORTANT: build Binary-Filter target from raw data BEFORE preprocessing,
    # because preprocessing removes noPredictionReason as leakage for predictor stage.
    filter_y_raw = _filter_target_from_df(df_raw)
    df = preprocess(df_raw)

    print(f"=== STEP 2: Shared train/test split ({dataset_name}) ===")
    y_winner = df["y"].astype(int).to_numpy()
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=y_winner,
    )
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    print(f"=== STEP 3: Binary-Filter (meta features) ({dataset_name}) ===")
    # Align raw filter target to the post-preprocessing dataframe indices.
    filter_y = np.asarray(filter_y_raw)[df.index.to_numpy()]
    filter_y_train = filter_y[train_idx]
    filter_y_test = filter_y[test_idx]

    X_filter_all = create_filter_features(df)
    X_filter_train = X_filter_all.iloc[train_idx]
    X_filter_test = X_filter_all.iloc[test_idx]

    filter_meta = train_binary_filter(X_filter_train, filter_y_train, dataset_name=dataset_name)
    accepted_meta_mask = apply_filter(X_filter_test, filter_meta["best_model"], filter_meta["best_scaler"]) == 1
    coverage_meta = float(accepted_meta_mask.mean()) if len(accepted_meta_mask) else 0.0
    print(f"Coverage (meta features): {coverage_meta:.1%} ({accepted_meta_mask.sum()}/{len(accepted_meta_mask)})")

    filter_results_df = filter_meta["results_df"].copy()
    filter_results_df["filter_type"] = "meta"
    filter_results_df["coverage"] = coverage_meta
    filter_results_df.to_csv(tables_dir / f"binary_filter_results_{dataset_name}.csv", index=False)

    plot_filter_model_comparison(
        filter_results_df,
        figures_dir / f"fig_3_1_filter_model_comparison_{dataset_name}.png",
    )

    cm_filter = confusion_matrix(
        filter_y_test,
        apply_filter(X_filter_test, filter_meta["best_model"], filter_meta["best_scaler"]),
        labels=[0, 1],
    )
    plot_confusion_matrix(
        cm_filter,
        class_names=["Rejected", "Accepted"],
        save_path=figures_dir / f"fig_3_4_confusion_matrix_filter_{dataset_name}.png",
    )
    _save_filter_importance(
        filter_meta["best_model"],
        feature_names=list(X_filter_test.columns),
        save_path=figures_dir / f"fig_3_7_feature_importance_filter_{dataset_name}.png",
    )

    print(f"=== STEP 4: Match-Predictor baseline ({dataset_name}) ===")
    (
        _orig_t1,
        _orig_t2,
        orig_distances,
        _swap_t1,
        _swap_t2,
        _swap_distances,
        _aug_t1,
        _aug_t2,
        aug_distances,
    ) = compute_team_distances_augmented(df, difference_distance)

    dist_df = orig_distances if args.distance_set == "orig" else aug_distances
    X_all = _sanitize_features(dist_df.drop(columns=["y"]))
    y_all = dist_df["y"].astype(int).to_numpy()

    X_train = X_all.iloc[train_idx].copy()
    X_test = X_all.iloc[test_idx].copy()
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    predictors = _select_predictor_classifiers(args.mode)
    pca_range = _get_pca_range(args.mode, X_train_scaled.shape[1], X_train_scaled.shape[0], args.pca_max, args.pca_step)
    ds_model_dir = models_dir / dataset_name
    ds_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ds_model_dir / "predictor_scaler.pkl")

    sweep_res = run_pca_sweep(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        classifiers=predictors,
        pca_range=pca_range,
        output_dir=ds_model_dir,
    )
    pca_df = pd.read_csv(ds_model_dir / "pca_sweep_results.csv")
    plot_pca_accuracy_curves(pca_df, figures_dir / f"fig_3_3_pca_accuracy_curves_{dataset_name}.png")

    print(f"=== STEP 5: Match-Predictor proposed ({dataset_name}) ===")
    rows_baseline_proposed: list[dict[str, Any]] = []
    roc_items: list[dict[str, Any]] = []

    for model_name, model in sweep_res["best_models"].items():
        pca_obj = sweep_res["best_pcas"][model_name]

        X_test_pca = pca_obj.transform(X_test_scaled)
        y_pred_full = model.predict(X_test_pca)
        baseline_acc = float((y_pred_full == y_test).mean())

        if accepted_meta_mask.sum() > 0:
            y_pred_acc = model.predict(X_test_pca[accepted_meta_mask])
            proposed_acc = float((y_pred_acc == y_test[accepted_meta_mask]).mean())
        else:
            proposed_acc = np.nan

        rows_baseline_proposed.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "baseline_accuracy": baseline_acc,
                "proposed_accuracy": proposed_acc,
                "coverage": coverage_meta,
                "accepted_count": int(accepted_meta_mask.sum()),
                "test_count": int(len(accepted_meta_mask)),
                "filter_type": "meta",
            }
        )

        # ROC data.
        scores = None
        if hasattr(model, "predict_proba"):
            try:
                scores = model.predict_proba(X_test_pca)[:, 1]
            except Exception:
                scores = None
        elif hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X_test_pca)
            except Exception:
                scores = None
        if scores is not None:
            try:
                fpr, tpr, _ = roc_curve(y_test, scores)
                auc = roc_auc_score(y_test, scores)
                roc_items.append({"name": model_name, "fpr": fpr, "tpr": tpr, "auc": float(auc)})
            except Exception:
                pass

    baseline_vs_proposed_df = pd.DataFrame(rows_baseline_proposed).sort_values("proposed_accuracy", ascending=False)
    baseline_vs_proposed_df.to_csv(tables_dir / f"baseline_vs_proposed_{dataset_name}.csv", index=False)
    plot_baseline_vs_proposed(
        baseline_vs_proposed_df[["model", "baseline_accuracy", "proposed_accuracy"]],
        figures_dir / f"fig_3_2_baseline_vs_proposed_{dataset_name}.png",
    )

    if roc_items:
        plot_roc_curves(roc_items, figures_dir / f"fig_3_6_roc_curves_{dataset_name}.png")

    # Best predictor confusion matrix and feature importance.
    best_row = baseline_vs_proposed_df.iloc[0]
    best_model_name = str(best_row["model"])
    best_model = sweep_res["best_models"][best_model_name]
    best_pca = sweep_res["best_pcas"][best_model_name]
    X_test_pca_best = best_pca.transform(X_test_scaled)
    cm_best = confusion_matrix(y_test, best_model.predict(X_test_pca_best), labels=[0, 1])
    plot_confusion_matrix(
        cm_best,
        class_names=["Team2 win (0)", "Team1 win (1)"],
        save_path=figures_dir / f"fig_3_5_confusion_matrix_best_model_{dataset_name}.png",
    )
    _save_predictor_importance(
        best_model,
        best_pca,
        original_feature_names=list(X_train_scaled.columns),
        save_path=figures_dir / f"fig_3_8_feature_importance_predictor_{dataset_name}.png",
    )

    print(f"=== STEP 6: Binary-Filter with distance features ({dataset_name}) ===")
    # Use distance features directly for filter experiment.
    X_filter_dist_train = X_train.copy()
    X_filter_dist_test = X_test.copy()
    filter_dist = train_binary_filter(X_filter_dist_train, filter_y_train, dataset_name=dataset_name)
    accepted_dist_mask = apply_filter(X_filter_dist_test, filter_dist["best_model"], filter_dist["best_scaler"]) == 1
    coverage_dist = float(accepted_dist_mask.mean()) if len(accepted_dist_mask) else 0.0

    extra_rows: list[dict[str, Any]] = []
    for model_name, model in sweep_res["best_models"].items():
        pca_obj = sweep_res["best_pcas"][model_name]
        X_test_pca = pca_obj.transform(X_test_scaled)
        if accepted_dist_mask.sum() > 0:
            y_pred_acc = model.predict(X_test_pca[accepted_dist_mask])
            proposed_acc = float((y_pred_acc == y_test[accepted_dist_mask]).mean())
        else:
            proposed_acc = np.nan
        extra_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "baseline_accuracy": float((model.predict(X_test_pca) == y_test).mean()),
                "proposed_accuracy": proposed_acc,
                "coverage": coverage_dist,
                "accepted_count": int(accepted_dist_mask.sum()),
                "test_count": int(len(accepted_dist_mask)),
                "filter_type": "distance",
            }
        )

    combined = pd.concat([baseline_vs_proposed_df, pd.DataFrame(extra_rows)], ignore_index=True)
    combined.to_csv(tables_dir / f"baseline_vs_proposed_{dataset_name}.csv", index=False)

    return filter_results_df, combined


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    tables_dir, figures_dir, models_dir = _ensure_dirs(out_dir)

    all_filter_tables: list[pd.DataFrame] = []
    all_baseline_proposed: list[pd.DataFrame] = []

    for name, path in [("60d", Path(args.json_60)), ("90d", Path(args.json_90))]:
        print(f"\n========== DATASET {name} ==========")
        filt_df, bvp_df = run_for_dataset(
            dataset_name=name,
            json_path=path,
            args=args,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            models_dir=models_dir,
        )
        all_filter_tables.append(filt_df.assign(dataset=name))
        all_baseline_proposed.append(bvp_df.assign(dataset=name))

    print("\n=== STEP 7: Global comparison and visualizations ===")
    all_bvp = pd.concat(all_baseline_proposed, ignore_index=True)
    coverage_acc = all_bvp[["dataset", "model", "filter_type", "coverage", "proposed_accuracy"]].copy()
    coverage_acc.to_csv(tables_dir / "coverage_vs_accuracy.csv", index=False)
    plot_coverage_vs_accuracy(coverage_acc, figures_dir / "fig_3_9_coverage_vs_accuracy.png")

    # Best model summary per dataset/filter type.
    best_rows = (
        all_bvp.sort_values("proposed_accuracy", ascending=False)
        .groupby(["dataset", "filter_type"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_rows.to_csv(tables_dir / "best_models_summary.csv", index=False)

    # 60d vs 90d comparison figure (meta filter rows only).
    r60 = all_bvp[(all_bvp["dataset"] == "60d") & (all_bvp["filter_type"] == "meta")][["model", "proposed_accuracy"]].rename(
        columns={"proposed_accuracy": "accuracy"}
    )
    r90 = all_bvp[(all_bvp["dataset"] == "90d") & (all_bvp["filter_type"] == "meta")][["model", "proposed_accuracy"]].rename(
        columns={"proposed_accuracy": "accuracy"}
    )
    plot_dataset_comparison(r60, r90, figures_dir / "fig_3_10_60d_vs_90d_comparison.png")

    print(f"Experiment complete. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()

