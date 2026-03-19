from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute classification metrics for binary task.

    Returns scalar metrics + confusion matrix.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Precision/Recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    metrics["precision_macro"] = float(np.mean(precision))
    metrics["recall_macro"] = float(np.mean(recall))
    metrics["f1_macro"] = float(np.mean(f1))
    metrics["precision_class_0"] = float(precision[0])
    metrics["precision_class_1"] = float(precision[1])
    metrics["recall_class_0"] = float(recall[0])
    metrics["recall_class_1"] = float(recall[1])
    metrics["f1_class_0"] = float(f1[0])
    metrics["f1_class_1"] = float(f1[1])

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # ROC-AUC / log-loss if probabilities (or scores) are available.
    if y_proba is not None:
        y_proba = np.asarray(y_proba).reshape(-1)
        # AUC expects probability-like scores.
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None

        try:
            # log_loss expects probability distribution for both classes.
            # We reconstruct [1-p, p].
            p1 = np.clip(y_proba, 1e-15, 1 - 1e-15)
            p0 = 1 - p1
            metrics["log_loss"] = float(log_loss(y_true, np.vstack([p0, p1]).T, labels=[0, 1]))
        except Exception:
            metrics["log_loss"] = None
    else:
        metrics["roc_auc"] = None
        metrics["log_loss"] = None

    return metrics


def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Convert dict[model_name] -> metrics dict into a table.
    """

    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        flat = {"model": model_name}
        for k, v in metrics.items():
            if k == "confusion_matrix":
                continue
            flat[k] = v
        rows.append(flat)
    return pd.DataFrame(rows)


def run_cross_validation(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    classifiers: dict[str, Any],
    best_pca_settings: dict[str, int] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Stratified CV with per-classifier best PCA components.

    X must be *raw* features (unscaled). Scaling is fitted per-fold to avoid leakage.
    """

    if best_pca_settings is None:
        best_pca_settings = {}

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_metrics: dict[str, dict[str, list[float | None]]] = {
        clf_name: {
            "accuracy": [],
            "f1_macro": [],
            "precision_macro": [],
            "recall_macro": [],
            "roc_auc": [],
        }
        for clf_name in classifiers.keys()
    }

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        X_train_raw = X_arr[train_idx]
        X_val_raw = X_arr[val_idx]
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        for clf_name, clf in classifiers.items():
            n_features = X_train_scaled.shape[1]
            n_samples = X_train_scaled.shape[0]
            default_components = min(n_features, n_samples)
            n_components = int(best_pca_settings.get(clf_name, default_components))
            n_components = max(1, min(n_components, min(n_features, n_samples)))

            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca = pca.transform(X_val_scaled)

            clf_clone = clone(clf)
            clf_clone.fit(X_train_pca, y_train)

            y_pred = clf_clone.predict(X_val_pca)

            y_proba: np.ndarray | None = None
            if hasattr(clf_clone, "predict_proba"):
                try:
                    proba = clf_clone.predict_proba(X_val_pca)
                    y_proba = proba[:, 1]
                except Exception:
                    y_proba = None

            # ROC-AUC/log-loss in evaluate_model depend on y_proba
            metrics = evaluate_model(y_val, y_pred, y_proba=y_proba)

            cv_metrics[clf_name]["accuracy"].append(metrics["accuracy"])
            cv_metrics[clf_name]["f1_macro"].append(metrics["f1_macro"])
            cv_metrics[clf_name]["precision_macro"].append(metrics["precision_macro"])
            cv_metrics[clf_name]["recall_macro"].append(metrics["recall_macro"])
            cv_metrics[clf_name]["roc_auc"].append(metrics["roc_auc"])

    summarized: dict[str, dict[str, Any]] = {}
    for clf_name, met in cv_metrics.items():
        summarized[clf_name] = {}
        for metric_name, values in met.items():
            vals = [v for v in values if v is not None]
            if len(vals) == 0:
                summarized[clf_name][f"{metric_name}_mean"] = None
                summarized[clf_name][f"{metric_name}_std"] = None
            else:
                summarized[clf_name][f"{metric_name}_mean"] = float(np.mean(vals))
                summarized[clf_name][f"{metric_name}_std"] = float(np.std(vals))

    return summarized

