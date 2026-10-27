from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent_dir(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_pca_accuracy_curves(pca_sweep_results: pd.DataFrame, save_path: str | Path) -> None:
    """
    Plot accuracy vs n_components for each classifier.

    Expected columns: ['clf_name' (or 'model'), 'n_components', 'accuracy'].
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    df = pca_sweep_results.copy()
    if "clf_name" not in df.columns:
        if "model" in df.columns:
            df = df.rename(columns={"model": "clf_name"})
        else:
            raise ValueError("pca_sweep_results must have columns clf_name/model.")

    if "n_components" not in df.columns:
        raise ValueError("pca_sweep_results must include n_components column.")
    if "accuracy" not in df.columns:
        raise ValueError("pca_sweep_results must include accuracy column.")

    plt.figure(figsize=(12, 8))
    for clf_name, g in df.groupby("clf_name"):
        g = g.sort_values("n_components")
        plt.plot(g["n_components"], g["accuracy"], label=clf_name)

    plt.title("Залежність accuracy від кількості PCA компонентів")
    plt.xlabel("Кількість компонентів PCA")
    plt.ylabel("Точність (Accuracy)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Матриця плутанини")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Правильний клас")
    plt.xlabel("Передбачений клас")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(models_results: list[dict[str, Any]], save_path: str | Path) -> None:
    """
    models_results: list of {'name', 'fpr', 'tpr', 'auc'}
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    plt.figure(figsize=(10, 7))
    for item in models_results:
        plt.plot(item["fpr"], item["tpr"], label=f"{item['name']} (AUC={item['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.title("ROC-криві")
    plt.xlabel("Частота хибнопозитивних (FPR)")
    plt.ylabel("Частота істиннопозитивних (TPR)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _get_feature_importances(model: Any) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        coef = model.coef_
        return np.abs(np.asarray(coef)).ravel()
    if hasattr(model, "get_feature_importance"):
        # CatBoost returns array-like.
        return np.asarray(model.get_feature_importance())
    raise ValueError(f"Model {type(model)} has no known feature importance method.")


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    save_path: str | Path,
    top_n: int = 20,
) -> None:
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    importances = _get_feature_importances(model)
    if len(importances) != len(feature_names):
        # Fall back to min length to avoid hard crash.
        n = min(len(importances), len(feature_names))
        importances = importances[:n]
        feature_names = feature_names[:n]

    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_values = importances[idx]

    plt.figure(figsize=(12, 6))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.title("Найважливіші ознаки")
    plt.xlabel("Важливість")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance_values(
    feature_names: list[str],
    importances: np.ndarray,
    save_path: str | Path,
    top_n: int = 20,
) -> None:
    """
    Same as `plot_feature_importance`, but uses precomputed importances.
    Useful when approximating original feature importances from PCA-space importances.
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    importances = np.asarray(importances).ravel()
    if len(importances) != len(feature_names):
        n = min(len(importances), len(feature_names))
        importances = importances[:n]
        feature_names = feature_names[:n]

    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_values = importances[idx]

    plt.figure(figsize=(12, 6))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.title("Найважливіші ознаки")
    plt.xlabel("Важливість")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_distribution(y: np.ndarray | pd.Series, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    y_arr = np.asarray(y).astype(int)
    counts = np.bincount(y_arr, minlength=2)
    labels = ["Клас 0", "Клас 1"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=["#4C78A8", "#F58518"])
    plt.title("Розподіл класів")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(
    X: pd.DataFrame,
    save_path: str | Path,
    top_n: int = 30,
) -> None:
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    # Choose top_n by variance to keep heatmap readable.
    variances = X.var(numeric_only=True).sort_values(ascending=False)
    cols = list(variances.head(top_n).index)
    corr = X[cols].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    plt.yticks(range(len(cols)), cols, fontsize=7)
    plt.title("Кореляційна матриця (top фіч)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_results(cv_results: dict[str, dict[str, Any]], save_path: str | Path) -> None:
    """
    Expected format:
      cv_results[clf]['accuracy_mean'] and cv_results[clf]['accuracy_std'].
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    items = []
    for clf_name, metrics in cv_results.items():
        mean = metrics.get("accuracy_mean")
        std = metrics.get("accuracy_std")
        if mean is None:
            continue
        items.append((clf_name, float(mean), float(std) if std is not None else 0.0))

    if not items:
        return

    items.sort(key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in items]
    means = np.array([x[1] for x in items])
    stds = np.array([x[2] for x in items])

    plt.figure(figsize=(12, 6))
    plt.bar(labels, means, yerr=stds, capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Точність (Accuracy)")
    plt.title("Результати cross-validation (accuracy)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance_from_importances(
    importances: np.ndarray,
    feature_names: list[str],
    title: str,
    save_path: str | Path,
    top_n: int = 20,
) -> None:
    """
    Prompt-compatible wrapper: feature importances are passed directly as an array.
    """
    plot_feature_importance_values(
        feature_names=feature_names,
        importances=importances,
        save_path=save_path,
        top_n=top_n,
    )


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str | Path) -> None:
    """
    Prompt-compatible stub for grouped comparison plots.
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    if comparison_df.empty:
        return

    metric_cols = [c for c in comparison_df.columns if c not in {"model", "confusion_matrix"}]
    metric_cols = [c for c in metric_cols if isinstance(c, str)]
    metric_cols = metric_cols[:4]  # keep plot readable

    plt.figure(figsize=(10, 5))
    x = np.arange(len(comparison_df["model"]))
    width = 0.15
    for i, m in enumerate(metric_cols):
        plt.bar(x + i * width, comparison_df[m].values, width=width, label=m)
    plt.xticks(x + width * (len(metric_cols) - 1) / 2, comparison_df["model"], rotation=45, ha="right")
    plt.title("Порівняння моделей")
    plt.ylabel("Значення метрики")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dataset_comparison(
    results_60d: pd.DataFrame,
    results_90d: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """
    Prompt-compatible stub for comparing per-model metrics across datasets.
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    if results_60d.empty or results_90d.empty:
        return

    merged = results_60d.merge(results_90d, on="model", suffixes=("_60d", "_90d"))
    if merged.empty:
        return

    # Prefer F1 if present, otherwise accuracy.
    y_col = "f1_macro" if "f1_macro_60d" in merged.columns else "accuracy"
    if y_col == "f1_macro":
        vals60 = merged["f1_macro_60d"].values
        vals90 = merged["f1_macro_90d"].values
        metric_name = "F1 (macro)"
    else:
        vals60 = merged["accuracy_60d"].values
        vals90 = merged["accuracy_90d"].values
        metric_name = "Accuracy"

    labels = merged["model"].astype(str).values.tolist()
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vals60, width, label="60d", color="#4C78A8")
    plt.bar(x + width / 2, vals90, width, label="90d", color="#F58518")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(metric_name)
    plt.title("Порівняння 60d vs 90d")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_filter_model_comparison(results_df: pd.DataFrame, save_path: str | Path) -> None:
    """
    Plot Binary-Filter model comparison by F1 macro.
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)
    if results_df.empty:
        return

    df = results_df.sort_values("f1_macro", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["f1_macro"], color="#4C78A8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("F1 (macro)")
    plt.title("Порівняння моделей Binary-Filter")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_baseline_vs_proposed(comparison_df: pd.DataFrame, save_path: str | Path) -> None:
    """
    Grouped bars for baseline vs proposed accuracy by model.
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)
    if comparison_df.empty:
        return

    df = comparison_df.copy()
    labels = df["model"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, df["baseline_accuracy"], width, label="Baseline", color="#4C78A8")
    plt.bar(x + width / 2, df["proposed_accuracy"], width, label="Proposed", color="#F58518")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Baseline vs Proposed (з фільтром)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_coverage_vs_accuracy(df: pd.DataFrame, save_path: str | Path) -> None:
    """
    Scatter plot for coverage vs proposed accuracy.
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)
    if df.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(df["coverage"], df["proposed_accuracy"], s=60, alpha=0.8)
    for _, row in df.iterrows():
        label = f"{row.get('dataset', '')}:{row.get('model', '')}"
        plt.annotate(label, (row["coverage"], row["proposed_accuracy"]), fontsize=7)
    plt.xlabel("Coverage")
    plt.ylabel("Proposed accuracy")
    plt.title("Coverage vs Accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
