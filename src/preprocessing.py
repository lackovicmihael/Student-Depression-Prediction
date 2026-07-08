from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Depression"

DROP_COLUMNS = ["id", "Name", "City", "Profession"]

NUMERIC_FEATURES = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Work/Study Hours",
    "Financial Stress",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COLUMN_ALIASES = {
    "Have you ever had suicidal thoughts?": "Have you ever had suicidal thoughts ?",
    "Have you ever had suicidal thoughts": "Have you ever had suicidal thoughts ?",
    "Family History of Mental illness": "Family History of Mental Illness",
    "family history of mental illness": "Family History of Mental Illness",
    "Work/Study Hour": "Work/Study Hours",
    "Work Study Hours": "Work/Study Hours",
    "Academic pressure": "Academic Pressure",
    "Work pressure": "Work Pressure",
    "Study satisfaction": "Study Satisfaction",
    "Job satisfaction": "Job Satisfaction",
    "Financial stress": "Financial Stress",
}


@dataclass(frozen=True)
class FeatureSchema:
    target: str
    numeric_features: list[str]
    categorical_features: list[str]
    all_features: list[str]
    dropped_columns: list[str]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and map common column-name variants to expected names."""
    result = df.copy()
    result.columns = [str(c).strip() for c in result.columns]
    result = result.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in result.columns})
    return result


def validate_input_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
    include_target: bool = True,
) -> None:
    """Raise a clear error if required columns are missing."""
    required = list(required_columns) if required_columns is not None else list(ALL_FEATURES)
    if include_target:
        required = required + [TARGET_COLUMN]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"columns are missing: {set(missing)}")


def clean_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Clean raw dataframe without introducing target leakage."""
    result = normalize_column_names(df)

    existing_drop_columns = [col for col in DROP_COLUMNS if col in result.columns]
    result = result.drop(columns=existing_drop_columns)

    validate_input_columns(result, include_target=require_target)

    result = result.drop_duplicates().reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        result[col] = result[col].astype("object")

    if require_target and TARGET_COLUMN in result.columns:
        result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
        result = result.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        result[TARGET_COLUMN] = result[TARGET_COLUMN].astype(int)

    return result


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into X and y."""
    validate_input_columns(df, include_target=True)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer used by all models."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_schema() -> dict:
    schema = FeatureSchema(
        target=TARGET_COLUMN,
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        all_features=list(ALL_FEATURES),
        dropped_columns=list(DROP_COLUMNS),
    )
    return schema.__dict__
