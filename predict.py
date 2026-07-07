"""
predict.py

Lightweight, inference-only module for the API repo. Loads the shared
model registry and all 5 model pipelines ONCE at import time, then
exposes:

  list_models() -> list          registry entries for the /models route
  preprocess(data, model_name)   raw listing -> encoded features for
                                  that specific model's expected columns
  predict(data, model_name)      raw listing -> predicted EUR price
  get_feature_importance(model_name) -> that model's own grouped
                                  importance, or None if unavailable

Expects, in the same folder:
    data_preprocessing.py
    models/models_registry.joblib
    models/<model_name>_pipeline.joblib   (one per registered model)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from data_preprocessing import PreprocessData

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

REGISTRY = joblib.load(MODEL_DIR / "models_registry.joblib")
MODELS = REGISTRY["models"]  # name -> {display_name, pipeline_file, feature_columns, ...}

# Load every pipeline once at startup -- not per request.
PIPELINES = {
    name: joblib.load(MODEL_DIR / info["pipeline_file"])
    for name, info in MODELS.items()
}


def list_models() -> list:
    """Registry entries for the dropdown: name, display name, test scores."""
    return [
        {
            "name": name,
            "display_name": info["display_name"],
            "test_scores": info["test_scores"],
            "has_feature_importance": info["feature_importance"] is not None,
        }
        for name, info in MODELS.items()
    ]


def get_feature_importance(model_name: str) -> list | None:
    """This model's own grouped feature importance ranking, or None if
    unavailable (e.g. SVM with a non-linear kernel)."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'.")
    return MODELS[model_name]["feature_importance"]


def preprocess(data: dict, model_name: str) -> pd.DataFrame:
    """
    Turns ONE raw (unencoded) listing dict into the exact encoded column
    format expected by the given model. Missing numeric values are
    handled by the pipeline's own median imputer, so partial listings
    are fine.

    "property_type" and "nearby_city" are the two exceptions: unlike the
    other encoding steps in data_preprocessing.py, PreprocessData raises
    a KeyError if these two are entirely absent. property_type is filled
    with a harmless default ("Apartment") when it isn't one of the
    model's expected feature columns anyway; nearby_city is filled with
    a sentinel that isn't in the known-cities table, so it falls back
    to the encoder's own average price when the client doesn't supply a
    recognized city.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'.")

    listing = dict(data)
    listing.setdefault("property_type", "Apartment")
    listing.setdefault("nearby_city", "Unknown")

    row = pd.DataFrame([listing])
    pre = PreprocessData(row)
    pre.preprocess_data(
        ordinal_specs=REGISTRY["ordinal_specs"],
        onehot_cols=REGISTRY["onehot_cols"],
        binary_cols=REGISTRY["binary_cols"],
        onehot_drop_first=False,
    )
    encoded = pre.get_data()
    return encoded.reindex(columns=MODELS[model_name]["feature_columns"], fill_value=0)


def predict(data: dict, model_name: str) -> float:
    """Raw listing dict + model name -> predicted EUR price."""
    if model_name not in PIPELINES:
        raise ValueError(f"Unknown model '{model_name}'.")

    X = preprocess(data, model_name)
    pred = PIPELINES[model_name].predict(X)[0]
    if MODELS[model_name]["log_target"]:
        pred = np.expm1(pred)
    return float(pred)
