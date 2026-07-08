from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


model = None
metadata: dict[str, Any] = {}
feature_schema: dict[str, Any] = {}


def _find_file(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.exists():
        return direct

    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init() -> None:
    """Load model and optional metadata/schema files."""
    global model, metadata, feature_schema

    model_dir = Path(os.environ.get("AZUREML_MODEL_DIR", "."))
    model_path = _find_file(model_dir, "model.pkl")

    if model_path is None:
        raise FileNotFoundError(
            f"Could not find model.pkl under AZUREML_MODEL_DIR={model_dir}"
        )

    model = joblib.load(model_path)

    metadata = _load_json(_find_file(model_path.parent, "model_metadata.json"))
    feature_schema = _load_json(_find_file(model_path.parent, "feature_schema.json"))


def _parse_payload(raw_data: Any) -> pd.DataFrame:
    if isinstance(raw_data, str):
        payload = json.loads(raw_data)
    else:
        payload = raw_data

    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        return pd.DataFrame(payload["data"])

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if isinstance(payload, dict) and "input_data" in payload:
        input_data = payload["input_data"]
        if isinstance(input_data, dict) and {"columns", "data"}.issubset(input_data):
            return pd.DataFrame(input_data["data"], columns=input_data["columns"])

    if isinstance(payload, dict) and {"columns", "data"}.issubset(payload):
        return pd.DataFrame(payload["data"], columns=payload["columns"])

    raise ValueError(
        "Unsupported payload format. Use {'data': [{...}]} or "
        "{'columns': [...], 'data': [[...]]}."
    )


def _required_features() -> list[str]:
    if feature_schema.get("all_features"):
        return list(feature_schema["all_features"])

    if metadata.get("features"):
        return list(metadata["features"])
    
    return []


def _validate_columns(df: pd.DataFrame) -> None:
    required = _required_features()

    if not required:
        return

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"columns are missing: {set(missing)}")


def _positive_probability(estimator: Any, df: pd.DataFrame) -> list[float | None]:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(df)

        if probabilities.shape[1] > 1:
            return [float(x) for x in probabilities[:, 1]]

        return [float(x) for x in probabilities[:, 0]]

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(df)
        min_score = float(min(scores))
        max_score = float(max(scores))

        if max_score == min_score:
            return [0.5 for _ in scores]

        return [float((s - min_score) / (max_score - min_score)) for s in scores]

    return [None] * len(df)


def run(raw_data: Any) -> list[dict[str, Any]] | dict[str, Any]:
    try:
        if model is None:
            init()

        df = _parse_payload(raw_data)
        _validate_columns(df)

        required = _required_features()
        if required:
            df = df[required]

        predictions = model.predict(df)
        probabilities = _positive_probability(model, df)

        results: list[dict[str, Any]] = []

        for pred, prob in zip(predictions, probabilities):
            prediction = int(pred)
            label = "Depression risk" if prediction == 1 else "No depression risk"

            results.append(
                {
                    "prediction": prediction,
                    "label": label,
                    "probability": round(float(prob), 6) if prob is not None else None,
                }
            )

        return results

    except Exception as exc:
        return {"error": str(exc)}