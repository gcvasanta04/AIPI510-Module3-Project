"""FastAPI service. Runs locally with uvicorn and on AWS Lambda via Mangum.

Local:   uvicorn src.serve.app:app --reload
Lambda:  handler = src.serve.app.handler
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))


class Listing(BaseModel):
    """One listing to price. Field names match the training columns."""
    neighbourhood: str = Field(examples=["Capitol Hill, Lincoln Park"])
    room_type: Literal["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    accommodates: int = Field(ge=1, le=16, examples=[4])
    bedrooms: float | None = Field(default=None, ge=0, le=10, examples=[2])
    beds: float | None = Field(default=None, ge=0, le=20, examples=[2])
    bathrooms: float | None = Field(default=None, ge=0, le=10, examples=[1])
    shared_bath: bool = False
    minimum_nights: int = Field(default=2, ge=1, le=365)
    number_of_reviews: int = Field(default=0, ge=0)
    review_scores_rating: float | None = Field(default=None, ge=0, le=5)
    host_is_superhost: bool = False
    instant_bookable: bool = False


class Prediction(BaseModel):
    predicted_price: float
    range_low: float
    range_high: float
    range_coverage: float
    currency: str = "USD"
    model_type: str
    mlflow_run_id: str


@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(MODEL_DIR / "model.joblib")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return model, meta


app = FastAPI(title="DC Airbnb Price API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "ok"}
    except FileNotFoundError:
        raise HTTPException(503, "Model not found. Run training first.")


@app.get("/metadata")
def metadata():
    """Allowed categories, numeric ranges and metrics — used to build the front-end form."""
    _, meta = load_artifacts()
    return {k: meta[k] for k in
            ["model_type", "trained_at", "metrics", "categories", "numeric_ranges", "interval"]}


@app.post("/predict", response_model=Prediction)
def predict(listing: Listing):
    model, meta = load_artifacts()
    known = meta["categories"]["neighbourhood"]
    if listing.neighbourhood not in known:
        raise HTTPException(422, f"Unknown neighbourhood. Choose one of GET /metadata categories.")

    X = pd.DataFrame([listing.model_dump()])[meta["feature_columns"]]
    raw = float(model.predict(X)[0])
    iv = meta["interval"]
    if meta["log_target"]:
        price = np.expm1(raw)
        low, high = np.expm1(raw + iv["low_resid"]), np.expm1(raw + iv["high_resid"])
    else:
        price, low, high = raw, raw + iv["low_resid"], raw + iv["high_resid"]

    return Prediction(
        predicted_price=round(float(price), 2),
        range_low=round(max(float(low), 0.0), 2),
        range_high=round(float(high), 2),
        range_coverage=iv["coverage"],
        model_type=meta["model_type"],
        mlflow_run_id=meta["mlflow_run_id"],
    )


# AWS Lambda entry point (API Gateway HTTP API -> Lambda -> FastAPI)
handler = Mangum(app, lifespan="off")
