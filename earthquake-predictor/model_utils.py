from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "earthquake_model.joblib"
USGS_MONTH_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
APP_USER_AGENT = "earthquake-risk-demo/1.0 (local educational project)"
MODEL_VERSION = "geo-gradient-boosting-v2"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().title() for col in df.columns]
    required = {"Latitude", "Longitude", "Depth", "Magnitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    df = df[["Latitude", "Longitude", "Depth", "Magnitude"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df = df[
        df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
        & df["Depth"].between(0, 800)
        & df["Magnitude"].between(0, 10)
    ]
    return df.reset_index(drop=True)


def engineer_features(x: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(x, columns=["Latitude", "Longitude", "Depth"]) if not isinstance(x, pd.DataFrame) else x.copy()
    lat_rad = np.radians(df["Latitude"].to_numpy())
    lon_rad = np.radians(df["Longitude"].to_numpy())
    depth = df["Depth"].to_numpy()

    return pd.DataFrame(
        {
            "latitude": df["Latitude"],
            "longitude": df["Longitude"],
            "depth": df["Depth"],
            "sin_latitude": np.sin(lat_rad),
            "cos_latitude": np.cos(lat_rad),
            "sin_longitude": np.sin(lon_rad),
            "cos_longitude": np.cos(lon_rad),
            "depth_log": np.log1p(depth),
            "depth_sqrt": np.sqrt(np.clip(depth, 0, None)),
            "latitude_longitude": df["Latitude"] * df["Longitude"],
            "latitude_depth": df["Latitude"] * df["Depth"],
            "longitude_depth": df["Longitude"] * df["Depth"],
            "dist_himalaya_zone": np.sqrt(
                (df["Latitude"] - 30.5) ** 2 + ((df["Longitude"] - 81.0) * np.cos(lat_rad)) ** 2
            ),
            "dist_andaman_zone": np.sqrt(
                (df["Latitude"] - 12.0) ** 2 + ((df["Longitude"] - 93.0) * np.cos(lat_rad)) ** 2
            ),
            "dist_gujarat_zone": np.sqrt(
                (df["Latitude"] - 23.5) ** 2 + ((df["Longitude"] - 70.0) * np.cos(lat_rad)) ** 2
            ),
            "dist_northeast_zone": np.sqrt(
                (df["Latitude"] - 26.5) ** 2 + ((df["Longitude"] - 94.0) * np.cos(lat_rad)) ** 2
            ),
        }
    )


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(engineer_features, validate=False)),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=500,
                    learning_rate=0.035,
                    max_depth=3,
                    subsample=0.85,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save_model() -> dict[str, Any]:
    df = load_dataset()
    x = df[["Latitude", "Longitude", "Depth"]]
    y = df["Magnitude"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "records": int(len(df)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "default_depth": float(df["Depth"].median()),
        "model_name": "GradientBoostingRegressor with geospatial feature engineering",
        "model_version": MODEL_VERSION,
    }
    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    return metrics


def load_or_train_model() -> tuple[Pipeline, dict[str, Any]]:
    if not MODEL_PATH.exists():
        train_and_save_model()

    bundle = joblib.load(MODEL_PATH)
    if bundle.get("metrics", {}).get("model_version") != MODEL_VERSION:
        train_and_save_model()
        bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["metrics"]


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(a))


def nearby_stats(df: pd.DataFrame, lat: float, lon: float, radius_km: float = 250) -> dict[str, Any]:
    distances = haversine_km(lat, lon, df["Latitude"].to_numpy(), df["Longitude"].to_numpy())
    nearby = df.assign(distance_km=distances).sort_values("distance_km")
    in_radius = nearby[nearby["distance_km"] <= radius_km]

    if in_radius.empty:
        nearest = nearby.iloc[0]
        return {
            "count": 0,
            "max_magnitude": None,
            "avg_magnitude": None,
            "nearest_km": float(nearest["distance_km"]),
            "suggested_depth": float(nearest["Depth"]),
            "events": nearby.head(20),
        }

    return {
        "count": int(len(in_radius)),
        "max_magnitude": float(in_radius["Magnitude"].max()),
        "avg_magnitude": float(in_radius["Magnitude"].mean()),
        "nearest_km": float(in_radius["distance_km"].min()),
        "suggested_depth": float(in_radius["Depth"].median()),
        "events": in_radius.head(50),
    }


def fetch_recent_usgs_events(timeout: int = 8) -> pd.DataFrame:
    try:
        response = requests.get(USGS_MONTH_FEED, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return pd.DataFrame(columns=["Latitude", "Longitude", "Depth", "Magnitude", "Place", "Time"])

    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if len(coords) < 3 or props.get("mag") is None:
            continue

        rows.append(
            {
                "Latitude": float(coords[1]),
                "Longitude": float(coords[0]),
                "Depth": float(coords[2]),
                "Magnitude": float(props["mag"]),
                "Place": props.get("place") or "Unknown place",
                "Time": pd.to_datetime(props.get("time"), unit="ms", utc=True, errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def reverse_geocode(lat: float, lon: float) -> str:
    try:
        response = requests.get(
            NOMINATIM_REVERSE_URL,
            params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 12, "addressdetails": 1},
            headers={"User-Agent": APP_USER_AGENT},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("display_name") or "Address not found for these coordinates."
    except requests.RequestException:
        return "Address lookup is unavailable right now."


def geocode_place(query: str) -> dict[str, Any] | None:
    if not query.strip():
        return None

    try:
        response = requests.get(
            NOMINATIM_SEARCH_URL,
            params={"format": "jsonv2", "q": query, "limit": 1},
            headers={"User-Agent": APP_USER_AGENT},
            timeout=8,
        )
        response.raise_for_status()
        results = response.json()
    except requests.RequestException:
        return None

    if not results:
        return None

    result = results[0]
    return {
        "lat": float(result["lat"]),
        "lon": float(result["lon"]),
        "address": result.get("display_name", query),
    }


def estimate_risk(
    predicted_magnitude: float,
    historical: dict[str, Any],
    recent: dict[str, Any],
) -> tuple[int, str]:
    magnitude_score = 45 / (1 + math.exp(-(predicted_magnitude - 3.6) * 1.4))

    history_count = historical["count"]
    history_max = historical["max_magnitude"] or 0
    history_score = min(30, math.log1p(history_count) * 5 + max(0, history_max - 4) * 4)

    recent_count = recent["count"]
    recent_max = recent["max_magnitude"] or 0
    recent_score = min(25, recent_count * 3.5 + max(0, recent_max - 3.5) * 5)

    score = int(round(min(100, max(0, magnitude_score + history_score + recent_score))))
    if score >= 70:
        label = "High"
    elif score >= 40:
        label = "Moderate"
    else:
        label = "Low"
    return score, label


def recent_nearby_stats(recent_df: pd.DataFrame, lat: float, lon: float, radius_km: float = 300) -> dict[str, Any]:
    if recent_df.empty:
        return {"count": 0, "max_magnitude": None, "nearest_km": None, "events": recent_df}

    distances = haversine_km(lat, lon, recent_df["Latitude"].to_numpy(), recent_df["Longitude"].to_numpy())
    nearby = recent_df.assign(distance_km=distances).sort_values("distance_km")
    in_radius = nearby[nearby["distance_km"] <= radius_km]
    if in_radius.empty:
        return {
            "count": 0,
            "max_magnitude": None,
            "nearest_km": float(nearby.iloc[0]["distance_km"]),
            "events": nearby.head(10),
        }

    return {
        "count": int(len(in_radius)),
        "max_magnitude": float(in_radius["Magnitude"].max()),
        "nearest_km": float(in_radius["distance_km"].min()),
        "events": in_radius.head(25),
    }
