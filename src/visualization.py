from __future__ import annotations

from pathlib import Path
from typing import Any

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

    plt.title("Accuracy vs PCA components")
    plt.xlabel("PCA components")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
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
    plt.title("Confusion matrix")
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

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curves(models_results: list[dict[str, Any]], save_path: str | Path) -> None:
    """
    models_results: list of {'name', 'fpr', 'tpr', 'auc'}
    """
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    plt.figure(figsize=(10, 7))
    for item in models_results:
        plt.plot(item["fpr"], item["tpr"], label=f'{item["name"]} (AUC={item["auc"]:.3f})')

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.title("ROC curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _get_feature_importances(model: Any) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(getattr(model, "feature_importances_"))
    if hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
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
    plt.title("Top feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
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
    plt.title("Top feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_class_distribution(y: np.ndarray | pd.Series, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_parent_dir(save_path)

    y_arr = np.asarray(y).astype(int)
    counts = np.bincount(y_arr, minlength=2)
    labels = ["Class 0", "Class 1"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=["#4C78A8", "#F58518"])
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
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
    plt.title("Correlation heatmap (top features)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
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
    plt.ylabel("Accuracy")
    plt.title("Cross-validation accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

