"""FastAPI mock-up for serving the selected traffic-volume model."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "shap_random_forest.joblib"

FEATURE_COLUMNS = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "is_holiday",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "weather_main_Clear",
    "weather_main_Clouds",
    "weather_main_Drizzle",
    "weather_main_Fog",
    "weather_main_Haze",
    "weather_main_Mist",
    "weather_main_Rain",
    "weather_main_Smoke",
    "weather_main_Snow",
    "weather_main_Squall",
    "weather_main_Thunderstorm",
]

WEATHER_CATEGORIES = [
    "Clear",
    "Clouds",
    "Drizzle",
    "Fog",
    "Haze",
    "Mist",
    "Rain",
    "Smoke",
    "Snow",
    "Squall",
    "Thunderstorm",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

try:
    MODEL = joblib.load(MODEL_PATH)
    LOGGER.info("Loaded deployment model from %s", MODEL_PATH)
except (FileNotFoundError, OSError, ValueError) as exc:
    MODEL = None
    LOGGER.error("Unable to load deployment model: %s", exc)


class TrafficInput(BaseModel):
    """Validated raw inputs used to create the model features."""

    date_time: datetime
    temp: float = Field(
        ...,
        ge=200,
        le=350,
        description="Temperature in Kelvin",
    )
    rain_1h: float = Field(
        default=0,
        ge=0,
        description="Rainfall during the previous hour in millimetres",
    )
    snow_1h: float = Field(
        default=0,
        ge=0,
        description="Snowfall during the previous hour in millimetres",
    )
    clouds_all: int = Field(
        ...,
        ge=0,
        le=100,
        description="Cloud coverage percentage",
    )
    is_holiday: bool = False
    weather_main: Literal[
        "Clear",
        "Clouds",
        "Drizzle",
        "Fog",
        "Haze",
        "Mist",
        "Rain",
        "Smoke",
        "Snow",
        "Squall",
        "Thunderstorm",
    ]


class TrafficPrediction(BaseModel):
    """Prediction returned by the deployment API."""

    predicted_traffic_volume: int
    raw_prediction: float
    unit: str
    model_version: str


app = FastAPI(
    title="Smart City Traffic Prediction API",
    description=(
        "Deployment mock-up serving the selected Random Forest traffic model."
    ),
    version="1.0.0",
)


def create_model_input(request: TrafficInput) -> pd.DataFrame:
    """Convert validated raw inputs into the model's 23 engineered features."""
    hour = request.date_time.hour
    day_of_week = request.date_time.weekday()
    month = request.date_time.month

    features = {
        "temp": request.temp,
        "rain_1h": request.rain_1h,
        "snow_1h": request.snow_1h,
        "clouds_all": request.clouds_all,
        "is_holiday": int(request.is_holiday),
        "is_weekend": int(day_of_week >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7),
        "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }

    for category in WEATHER_CATEGORIES:
        features[f"weather_main_{category}"] = int(
            request.weather_main == category
        )

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


@app.get("/")
def root() -> dict[str, str]:
    """Describe the API."""
    return {
        "message": "Smart City Traffic Prediction API",
        "documentation": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the selected model loaded successfully."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="The traffic prediction model is unavailable.",
        )

    return {
        "status": "healthy",
        "model": "Random Forest for SHAP",
        "model_version": "random_forest_shap_v1",
    }


@app.post("/predict", response_model=TrafficPrediction)
def predict_traffic(request: TrafficInput) -> TrafficPrediction:
    """Predict hourly traffic volume from validated traffic conditions."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="The traffic prediction model is unavailable.",
        )

    try:
        model_input = create_model_input(request)
        raw_prediction = float(MODEL.predict(model_input)[0])
        rounded_prediction = max(0, round(raw_prediction))
    except (ValueError, TypeError) as exc:
        LOGGER.error("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Unable to generate a traffic prediction.",
        ) from exc

    LOGGER.info(
        "Generated traffic prediction of %d vehicles per hour",
        rounded_prediction,
    )

    return TrafficPrediction(
        predicted_traffic_volume=rounded_prediction,
        raw_prediction=round(raw_prediction, 2),
        unit="vehicles per hour",
        model_version="random_forest_shap_v1",
    )