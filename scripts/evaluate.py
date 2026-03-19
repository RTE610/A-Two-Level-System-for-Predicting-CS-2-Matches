from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import config
from src.evaluation import compare_models, evaluate_model, run_cross_validation
from src.training import read_best_pca_settings
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_cv_results,
    plot_feature_importance_values,
    plot_pca_accuracy_curves,
    plot_roc_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained CS2 models and generate results/figures.")
    parser.add_argument("--model-dir", required=True, help="Path to models/<run_id>/dist_orig or dist_aug")
    parser.add_argument("--output-dir", default=str(Path("./results")), help="Directory for tables/figures")
    parser.add_argument("--top-k", type=int, default=3, help="How many top models to use for confusion matrices")
    parser.add_argument("--cv-top-k", type=int, default=5, help="How many top models to run CV for")
    return parser.parse_args()


def _safe_model_name(clf_name: str) -> str:
    return clf_name.replace(" ", "_")


def _get_model_and_pca_paths(model_dir: Path, clf_name: str) -> tuple[Path, Path]:
    safe = _safe_model_name(clf_name)
    model_path = model_dir / f"{safe}_best_model.pkl"
    pca_path = model_dir / f"{safe}_best_pca.pkl"
    return model_path, pca_path


def _get_scores(model: Any, X: Any) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores).reshape(-1)
    return None


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = model_dir / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_path}. Please ensure train.py saved dataset.csv into the model dir."
        )

    dataset_df = pd.read_csv(dataset_path)
    if "y" not in dataset_df.columns:
        raise ValueError("dataset.csv must contain column 'y'.")

    X_raw = dataset_df.drop(columns=["y"])
    y = dataset_df["y"].astype(int).values

    # Pre-plot class distribution and correlations
    plot_class_distribution(y, figures_dir / "class_distribution.png")
    plot_correlation_heatmap(X_raw, figures_dir / "correlation_heatmap.png", top_n=30)

    scaler_path = model_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler at {scaler_path}")

    scaler: StandardScaler = joblib.load(scaler_path)
    # Reproduce train.py feature sanitization before scaler.transform:
    # - numeric coercion
    # - inf/-inf -> NaN
    # - drop all-NaN columns
    # - fill NaN with column means
    X_eval = X_raw.apply(pd.to_numeric, errors="coerce")
    X_eval = X_eval.replace([np.inf, -np.inf], np.nan)
    all_nan_cols = X_eval.columns[X_eval.isna().all()].tolist()
    if all_nan_cols:
        X_eval = X_eval.drop(columns=all_nan_cols)
    X_eval = X_eval.fillna(X_eval.mean(numeric_only=True))

    # Align columns exactly with scaler training schema.
    if hasattr(scaler, "feature_names_in_"):
        train_cols = list(scaler.feature_names_in_)
        # Remove unseen columns and add missing ones.
        X_eval = X_eval.reindex(columns=train_cols)
        # If some columns are still NaN after reindexing, fill with 0.0 fallback.
        X_eval = X_eval.fillna(0.0)

    X_scaled = scaler.transform(X_eval)

    # Recreate the same holdout split as in train.py (no stratify).
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    best_settings_path = model_dir / "best_setting_result.txt"
    if not best_settings_path.exists():
        raise FileNotFoundError(f"Missing {best_settings_path}")

    best_pca_settings = read_best_pca_settings(best_settings_path)

    classifiers = config.get_classifiers(random_state=config.RANDOM_STATE)

    # PCA sweep curve (optional)
    pca_sweep_csv = model_dir / "pca_sweep_results.csv"
    if pca_sweep_csv.exists():
        pca_sweep_df = pd.read_csv(pca_sweep_csv)
        plot_pca_accuracy_curves(pca_sweep_df, figures_dir / "pca_accuracy_curves.png")

    final_results: dict[str, dict[str, Any]] = {}
    roc_items: list[dict[str, Any]] = []

    # Evaluate each classifier on holdout test set
    for clf_name in classifiers.keys():
        model_path, pca_path = _get_model_and_pca_paths(model_dir, clf_name)
        if not model_path.exists() or not pca_path.exists():
            continue

        model = joblib.load(model_path)
        pca = joblib.load(pca_path)

        X_test_pca = pca.transform(X_test_scaled)
        y_pred = model.predict(X_test_pca)

        scores = _get_scores(model, X_test_pca)
        y_proba_for_metrics = None
        if scores is not None:
            # For roc_auc we can use scores; evaluate_model uses y_proba for log_loss too.
            # If model provides true probabilities, they are already in [0,1].
            if hasattr(model, "predict_proba"):
                y_proba_for_metrics = scores
            else:
                y_proba_for_metrics = scores  # roc_auc will work, log_loss will likely become None.

        metrics = evaluate_model(y_true=y_test, y_pred=y_pred, y_proba=y_proba_for_metrics)
        metrics["roc_auc"] = metrics.get("roc_auc")

        final_results[clf_name] = metrics

        if scores is not None:
            try:
                fpr, tpr, _ = roc_curve(y_test, scores)
                auc = roc_auc_score(y_test, scores)
                roc_items.append({"name": clf_name, "fpr": fpr, "tpr": tpr, "auc": float(auc)})
            except Exception:
                pass

    if final_results:
        comparison_df = compare_models(final_results)
        comparison_df.sort_values(by="f1_macro", ascending=False, inplace=True)
        comparison_df.to_csv(tables_dir / "final_model_comparison.csv", index=False)

    # Confusion matrices for top-k by f1_macro
    if final_results:
        ranked = sorted(final_results.keys(), key=lambda n: float(final_results[n].get("f1_macro", -1.0)), reverse=True)
        top_k = max(1, int(args.top_k))
        for i, clf_name in enumerate(ranked[:top_k]):
            cm = final_results[clf_name]["confusion_matrix"]
            plot_confusion_matrix(cm, class_names=["0", "1"], save_path=figures_dir / f"confusion_matrix_{_safe_model_name(clf_name)}.png")

    # ROC curves for all models that provide scores
    if roc_items:
        plot_roc_curves(roc_items, figures_dir / "roc_curves.png")

    # Feature importance for top models (approx original importance from PCA components importances)
    # We use the top tree-based models by f1_macro and approximate original feature importance:
    tree_like = [n for n in final_results.keys() if n in {"Random Forest", "XGBoost", "LightGBM", "CatBoost", "Gradient Boosting", "Decision Tree"}]
    ranked_tree = sorted(tree_like, key=lambda n: float(final_results[n].get("f1_macro", -1.0)), reverse=True)
    top_for_importance = ranked_tree[: min(3, len(ranked_tree))]

    for clf_name in top_for_importance:
        model_path, pca_path = _get_model_and_pca_paths(model_dir, clf_name)
        if not model_path.exists() or not pca_path.exists():
            continue
        model = joblib.load(model_path)
        pca = joblib.load(pca_path)

        X_train_raw = X_raw  # for feature names only
        original_feature_names = list(X_train_raw.columns)

        # Importances in PCA space
        if hasattr(model, "feature_importances_"):
            pc_importances = np.asarray(model.feature_importances_).ravel()
        elif hasattr(model, "coef_"):
            pc_importances = np.abs(np.asarray(model.coef_)).ravel()
        else:
            continue

        # Approximate original feature importances via PCA loadings.
        # pca.components_ shape: (n_components, n_features_original)
        if hasattr(pca, "components_") and pc_importances.shape[0] == pca.components_.shape[0]:
            orig_importances = np.abs(pca.components_.T @ pc_importances)
            save_path = figures_dir / f"feature_importance_{_safe_model_name(clf_name)}.png"
            plot_feature_importance_values(
                feature_names=original_feature_names,
                importances=orig_importances,
                save_path=save_path,
                top_n=20,
            )

    # Cross-validation for top CV models by f1_macro
    if final_results:
        ranked = sorted(final_results.keys(), key=lambda n: float(final_results[n].get("f1_macro", -1.0)), reverse=True)
        cv_top_k = max(1, int(args.cv_top_k))
        top_models = ranked[:cv_top_k]

        classifiers_for_cv = {n: classifiers[n] for n in top_models if n in classifiers}

        # For CV we use raw features (unscaled) and rebuild scaling/PCA per fold.
        X_for_cv = X_eval
        y_for_cv = y

        cv_results = run_cross_validation(
            X=X_for_cv,
            y=y_for_cv,
            classifiers=classifiers_for_cv,
            best_pca_settings=best_pca_settings,
            n_splits=5,
            random_state=config.RANDOM_STATE,
        )

        # Save and plot summarized CV results.
        cv_rows = []
        for clf_name, metrics in cv_results.items():
            cv_rows.append({"model": clf_name, **metrics})
        pd.DataFrame(cv_rows).to_csv(tables_dir / "cv_results.csv", index=False)

        plot_cv_results(cv_results, figures_dir / "cv_boxplot.png")


if __name__ == "__main__":
    main()

