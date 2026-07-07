"""
app.py

FastAPI app for the Immo Eliza deployment mission.

Routes
------
GET  /         -> "alive" (health check)
POST /predict  -> body {"data": {...listing fields...}}
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

from predict import predict


# ====================================================================
#  Request schema -- every field optional (partial listings allowed).
#  Adjust/rename fields here if the model's feature set changes.
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
    data: ListingData


# ====================================================================
#  App
# ====================================================================
app = FastAPI(title="Immo Eliza Price Prediction API")


@app.get("/")
def alive():
    return "alive"


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    raw_listing = request.data.model_dump(exclude_none=True)

    if not raw_listing:
        return JSONResponse(
            status_code=400,
            content={"prediction": None, "status_code": 400},
        )

    try:
        price = predict(raw_listing)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"prediction": None, "status_code": 500},
        )

    return {"prediction": price, "status_code": 200}
