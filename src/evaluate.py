from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_predictions_and_probabilities(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return predictions, probabilities[:, 1]
        return predictions, probabilities[:, 0]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return predictions, np.asarray(scores)

    return predictions, None


def safe_roc_auc(y_true: pd.Series, y_score: np.ndarray | None) -> float | None:
    if y_score is None:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def evaluate_binary_classifier(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions, probabilities = get_predictions_and_probabilities(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": safe_roc_auc(y_test, probabilities),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def save_confusion_matrix_plot(model: Any, X_test: pd.DataFrame, y_test: pd.Series, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=[0, 1])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_roc_curve_plot(model: Any, X_test: pd.DataFrame, y_test: pd.Series, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
    finally:
        plt.close()


def save_model_comparison_plot(results: pd.DataFrame, output_path: str | Path, metric: str = "f1") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = results.sort_values(metric, ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(ordered["model"], ordered[metric])
    plt.title(f"Model Comparison by {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
