"""
app.py

FastAPI app serving 5 price-prediction models (ridge, decision_tree,
random_forest, svm, xgboost) with model selection.

Routes
------
GET  /                    -> "alive" (health check)
GET  /models              -> list of models with display name + test scores
GET  /feature-importance  -> ?model=<name> -> that model's own grouped
                              feature importance, or null if unavailable
POST /predict              -> body {"model": "<name>", "data": {...}}
                              -> {"prediction": float | None, "status_code": int}

Run locally:
    uvicorn app:app --reload
Docs (auto-generated):
    http://127.0.0.1:8000/docs
"""

from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from predict import predict, list_models, get_feature_importance, MODELS


# ====================================================================
#  Request schema -- every listing field optional (partial listings
#  allowed); "model" selects which trained pipeline to use.
# ====================================================================
class ListingData(BaseModel):
    living_area_m2: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    building_year: Optional[int] = None
    property_type: Optional[str] = None
    property_subtype: Optional[str] = None
    province: Optional[str] = None
    region: Optional[str] = None
    kitchen_equipped: Optional[str] = None
    state_of_the_building: Optional[str] = None
    epc_score: Optional[str] = None
    nearby_city: Optional[str] = None
    furnished: Optional[int] = None
    has_garage: Optional[int] = None
    parking_count: Optional[int] = None
    has_elevator: Optional[int] = None
    facades: Optional[int] = None
    has_garden: Optional[int] = None
    garden_area_m2: Optional[float] = None
    has_terrace: Optional[int] = None
    total_area_m2: Optional[float] = None
    km_from_nearby_city: Optional[float] = None
    is_nearby_city_prestigious: Optional[int] = None
    floor_number: Optional[int] = None


class PredictRequest(BaseModel):
    model: str
    data: ListingData


# ====================================================================
#  App
# ====================================================================
app = FastAPI(title="Immo Eliza Price Prediction API")


@app.get("/")
def alive():
    return "alive"


@app.get("/models")
def models():
    return {"models": list_models()}


@app.get("/feature-importance")
def feature_importance(model: str):
    if model not in MODELS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown model '{model}'.", "status_code": 404},
        )
    return {"model": model, "top_features": get_feature_importance(model)}


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    if request.model not in MODELS:
        return JSONResponse(
            status_code=404,
            content={"prediction": None, "status_code": 404, "error": f"Unknown model '{request.model}'."},
        )

    raw_listing = request.data.model_dump(exclude_none=True)
    if not raw_listing:
        return JSONResponse(
            status_code=400,
            content={"prediction": None, "status_code": 400},
        )

    try:
        price = predict(raw_listing, request.model)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"prediction": None, "status_code": 500},
        )

    return {"prediction": price, "status_code": 200}
