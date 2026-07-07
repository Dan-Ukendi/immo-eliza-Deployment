"""
predict.py

Loads the trained Random Forest pipeline and its encoding metadata once
at import time, then exposes two plain Python functions (no CLI):

  preprocess(data: dict) -> pd.DataFrame   raw listing -> encoded features
  predict(data: dict)    -> float          raw listing -> predicted price

Expects, in the same folder:
    data_preprocessing.py
    model/random_forest_pipeline.joblib
    model/random_forest_metadata.joblib
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from data_preprocessing import PreprocessData

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

PIPELINE = joblib.load(MODEL_DIR / "random_forest_pipeline.joblib")
METADATA = joblib.load(MODEL_DIR / "random_forest_metadata.joblib")


def preprocess(data: dict) -> pd.DataFrame:
    """
    Turns ONE raw (unencoded) listing dict into the exact encoded column
    format the model was trained on. Missing numeric values are handled
    by the pipeline's own median imputer, so partial listings are fine.

    "property_type" and "nearby_city" are the two exceptions: unlike the
    other encoding steps in data_preprocessing.py, PreprocessData raises
    a KeyError if these two are entirely absent. property_type is filled
    with a harmless default ("Apartment") since it is dropped from the
    model's features anyway (see METADATA["drop_features"]). nearby_city
    is filled with a sentinel that isn't in the known-cities table, so it
    falls back to the encoder's own average price when the client
    doesn't supply a recognized city.
    """
    listing = dict(data)
    listing.setdefault("property_type", "Apartment")
    listing.setdefault("nearby_city", "Unknown")

    row = pd.DataFrame([listing])
    pre = PreprocessData(row)
    pre.preprocess_data(
        ordinal_specs=METADATA["ordinal_specs"],
        onehot_cols=METADATA["onehot_cols"],
        binary_cols=METADATA["binary_cols"],
        onehot_drop_first=False,
    )
    encoded = pre.get_data()
    return encoded.reindex(columns=METADATA["feature_columns"], fill_value=0)


def predict(data: dict) -> float:
    """Raw listing dict -> predicted EUR price (a single float)."""
    X = preprocess(data)
    pred = PIPELINE.predict(X)[0]
    if METADATA["log_target"]:
        pred = np.expm1(pred)
    return float(pred)
