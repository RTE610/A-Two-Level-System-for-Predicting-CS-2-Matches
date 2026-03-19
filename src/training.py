from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def get_classifiers_result_string(components_range: list[int], pca_results: dict[str, list[float]]) -> list[str]:
    """
    Format PCA sweep results into lines compatible with `read_best_pca_settings`.
    """

    strings: list[str] = []
    for clf_name, scores in pca_results.items():
        best_idx = int(np.argmax(scores))
        best_accuracy = float(scores[best_idx])
        best_components = int(components_range[best_idx])
        strings.append(
            f"{clf_name}: Best Accuracy: {best_accuracy:.4f}, Best PCA Components: {best_components}"
        )
    return strings


def read_best_pca_settings(file_path: str | Path) -> dict[str, int]:
    """
    Parse best PCA components per classifier from a txt file.
    Expected line patterns (flexible):
      "Classifier: Best Accuracy: 0.7, Best PCA Components: 48"
      "Classifier: Best: 0.7, PCA: 48"
    """

    file_path = Path(file_path)
    best_pca: dict[str, int] = {}

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue

        clf_part = parts[0]
        pca_part = parts[1]

        clf_name = clf_part.split(":")[0].strip()
        m = re.search(r"(-?\d+)", pca_part)
        if not m:
            continue

        best_pca[clf_name] = int(m.group(1))

    return best_pca


def run_pca_sweep(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    classifiers: dict[str, Any],
    pca_range: list[int],
    output_dir: str | Path,
):
    """
    Train classifiers for each PCA(n_components) and save best models/PCA objects.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_results: dict[str, list[float]] = {clf_name: [] for clf_name in classifiers.keys()}
    best_models: dict[str, Any] = {}
    best_pcas: dict[str, Any] = {}
    best_scores: dict[str, float] = {clf_name: -np.inf for clf_name in classifiers.keys()}

    for n_components in pca_range:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        for clf_name, clf in classifiers.items():
            clf_clone = clone(clf)
            clf_clone.fit(X_train_pca, y_train)
            y_pred = clf_clone.predict(X_test_pca)
            acc = float(accuracy_score(y_test, y_pred))

            pca_results[clf_name].append(acc)

            if acc > best_scores[clf_name]:
                best_scores[clf_name] = acc
                best_models[clf_name] = clf_clone
                best_pcas[clf_name] = pca

    lines = get_classifiers_result_string(pca_range, pca_results)
    best_settings_path = output_dir / "best_setting_result.txt"
    best_settings_path.write_text("\n".join(lines), encoding="utf-8")

    # Persist full PCA sweep results for later plotting.
    sweep_rows: list[dict[str, Any]] = []
    for clf_name, acc_list in pca_results.items():
        for n_components, acc in zip(pca_range, acc_list):
            sweep_rows.append(
                {
                    "clf_name": clf_name,
                    "n_components": int(n_components),
                    "accuracy": float(acc),
                }
            )
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(output_dir / "pca_sweep_results.csv", index=False)

    for clf_name in classifiers.keys():
        safe_name = clf_name.replace(" ", "_")
        model_path = output_dir / f"{safe_name}_best_model.pkl"
        pca_path = output_dir / f"{safe_name}_best_pca.pkl"
        joblib.dump(best_models[clf_name], model_path)
        joblib.dump(best_pcas[clf_name], pca_path)

    return {
        "best_models": best_models,
        "best_pcas": best_pcas,
        "best_scores": best_scores,
        "lines": lines,
        "best_settings_path": str(best_settings_path),
    }


def retrain_with_best_pca(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    classifiers: dict[str, Any],
    best_pca_settings: dict[str, int],
    output_dir: str | Path,
):
    """
    Retrain each classifier with its best PCA(n_components) from the sweep.
    Saves `*_best_model.pkl` and `*_best_pca.pkl` into `output_dir`.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}

    for clf_name, clf in classifiers.items():
        optimal_components = int(best_pca_settings.get(clf_name, X_train.shape[1]))
        pca = PCA(n_components=optimal_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf_clone = clone(clf)
        clf_clone.fit(X_train_pca, y_train)

        y_pred = clf_clone.predict(X_test_pca)
        acc = float(accuracy_score(y_test, y_pred))

        safe_name = clf_name.replace(" ", "_")
        model_path = output_dir / f"{safe_name}_best_model.pkl"
        pca_path = output_dir / f"{safe_name}_best_pca.pkl"
        joblib.dump(clf_clone, model_path)
        joblib.dump(pca, pca_path)

        results[clf_name] = {"accuracy": acc, "pca_components": optimal_components}

    return results

