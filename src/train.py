from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import mlflow
except Exception:
    mlflow = None

from src.evaluate import (
    evaluate_binary_classifier,
    save_confusion_matrix_plot,
    save_model_comparison_plot,
    save_roc_curve_plot,
)
from src.preprocessing import (
    ALL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    clean_dataframe,
    get_feature_schema,
    split_features_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="outputs/model_output")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--search-level", choices=["quick", "standard"], default="quick")
    parser.add_argument(
        "--primary-metric",
        choices=["f1", "roc_auc", "recall", "accuracy"],
        default="f1",
    )
    return parser.parse_args()


def resolve_csv_path(data_path: str | Path) -> Path:
    path = Path(data_path)

    if path.is_file():
        return path

    if path.is_dir():
        csv_files = sorted(path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found under data path: {path}")
        return csv_files[0]

    raise FileNotFoundError(f"Data path does not exist: {path}")


def build_model_grid(
    search_level: str,
    random_state: int,
    y_train: pd.Series,
) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    quick = search_level == "quick"

    grids: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            {
                "classifier__C": [0.1, 1.0] if quick else [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear"],
            },
        ),
        "DecisionTree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            {
                "classifier__max_depth": [3, 5, None] if quick else [3, 5, 8, None],
                "classifier__min_samples_split": [2, 10] if quick else [2, 5, 10],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [100] if quick else [100, 200],
                "classifier__max_depth": [None, 8] if quick else [None, 8, 12],
                "classifier__min_samples_split": [2, 10],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "classifier__n_estimators": [100] if quick else [100, 200],
                "classifier__learning_rate": [0.05, 0.1] if quick else [0.03, 0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        ),
        "SVC": (
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
            {
                "classifier__C": [0.5, 1.0] if quick else [0.1, 0.5, 1.0, 2.0],
                "classifier__kernel": ["rbf", "linear"],
            },
        ),
    }

    if XGBClassifier is not None:
        counts = y_train.value_counts().to_dict()
        negatives = counts.get(0, 1)
        positives = counts.get(1, 1)
        scale_pos_weight = max(1.0, negatives / max(positives, 1))

        grids["XGBoost"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            ),
            {
                "classifier__n_estimators": [100] if quick else [100, 200],
                "classifier__max_depth": [3, 4] if quick else [3, 4, 5],
                "classifier__learning_rate": [0.05, 0.1] if quick else [0.03, 0.05, 0.1],
            },
        )

    return grids


def log_mlflow_metric(name: str, value: Any) -> None:
    if mlflow is None or value is None:
        return

    try:
        mlflow.log_metric(name, float(value))
    except Exception:
        pass


def log_mlflow_param(name: str, value: Any) -> None:
    if mlflow is None:
        return

    try:
        mlflow.log_param(name, value)
    except Exception:
        pass


def log_mlflow_artifact(path: str | Path) -> None:
    if mlflow is None:
        return

    try:
        mlflow.log_artifact(str(path))
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "model"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_csv_path(args.data_path)

    raw_df = pd.read_csv(csv_path)
    df = clean_dataframe(raw_df, require_target=True)
    X, y = split_features_target(df)

    if y.nunique() != 2:
        raise ValueError(
            f"Target column {TARGET_COLUMN} must be binary. "
            f"Found classes: {sorted(y.unique())}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    log_mlflow_param("data_path", str(csv_path))
    log_mlflow_param("rows_after_cleaning", len(df))
    log_mlflow_param("features", ",".join(ALL_FEATURES))
    log_mlflow_param("test_size", args.test_size)
    log_mlflow_param("search_level", args.search_level)
    log_mlflow_param("primary_metric", args.primary_metric)

    class_distribution = y.value_counts().sort_index().to_dict()
    class_distribution_path = metrics_dir / "class_distribution.json"

    with class_distribution_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in class_distribution.items()}, f, indent=2)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    refit_metric = args.primary_metric
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    model_grids = build_model_grid(args.search_level, args.random_state, y_train)

    results: list[dict[str, Any]] = []
    best_model_name = None
    best_model = None
    best_score = -np.inf
    best_grid = None

    for model_name, (classifier, param_grid) in model_grids.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )

        grid.fit(X_train, y_train)

        evaluation = evaluate_binary_classifier(grid.best_estimator_, X_test, y_test)

        row = {
            "model": model_name,
            "best_cv_score": float(grid.best_score_),
            "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
            **evaluation,
        }

        results.append(row)

        for metric_name, metric_value in evaluation.items():
            if isinstance(metric_value, (int, float)) and metric_value is not None:
                log_mlflow_metric(f"{model_name}_{metric_name}", metric_value)

        candidate_score = evaluation.get(refit_metric)
        if candidate_score is None:
            candidate_score = grid.best_score_

        if float(candidate_score) > best_score:
            best_score = float(candidate_score)
            best_model_name = model_name
            best_model = grid.best_estimator_
            best_grid = grid

    if best_model is None or best_model_name is None or best_grid is None:
        raise RuntimeError("No model was successfully trained.")

    results_df = pd.DataFrame(results).sort_values(args.primary_metric, ascending=False)
    comparison_path = metrics_dir / "model_comparison_results.csv"
    results_df.to_csv(comparison_path, index=False)

    best_metrics = evaluate_binary_classifier(best_model, X_test, y_test)

    metadata = {
        "project": "Student Depression Prediction",
        "target": TARGET_COLUMN,
        "features": ALL_FEATURES,
        "best_model_name": best_model_name,
        "primary_metric": args.primary_metric,
        "best_score": best_score,
        "best_params": best_grid.best_params_,
        "test_metrics": best_metrics,
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "training_location": os.environ.get("AZUREML_RUN_ID", "local_or_unknown"),
        "note": "Register the model/ folder from Azure ML Command Job output.",
    }

    model_path = model_dir / "model.pkl"
    metadata_path = model_dir / "model_metadata.json"
    schema_path = model_dir / "feature_schema.json"
    summary_path = output_dir / "run_summary.json"

    joblib.dump(best_model, model_path)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(get_feature_schema(), f, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_name": best_model_name,
                "best_score": best_score,
                "output_dir": str(output_dir),
                "model_dir": str(model_dir),
                "model_path": str(model_path),
                "comparison_path": str(comparison_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    confusion_path = figures_dir / "confusion_matrix.png"
    roc_path = figures_dir / "roc_curve.png"
    comparison_plot_path = figures_dir / f"model_comparison_{args.primary_metric}.png"

    save_confusion_matrix_plot(best_model, X_test, y_test, confusion_path)
    save_roc_curve_plot(best_model, X_test, y_test, roc_path)
    save_model_comparison_plot(results_df, comparison_plot_path, metric=args.primary_metric)

    log_mlflow_param("best_model_name", best_model_name)
    log_mlflow_param("best_params", json.dumps(best_grid.best_params_))

    for metric_name, metric_value in best_metrics.items():
        log_mlflow_metric(f"best_{metric_name}", metric_value)

    for artifact_path in [
        model_path,
        metadata_path,
        schema_path,
        summary_path,
        comparison_path,
        class_distribution_path,
        confusion_path,
        roc_path,
        comparison_plot_path,
    ]:
        log_mlflow_artifact(artifact_path)

    print("Training completed.")
    print(f"Best model: {best_model_name}")
    print(f"Best {args.primary_metric}: {best_score:.4f}")
    print(f"Model folder: {model_dir}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()