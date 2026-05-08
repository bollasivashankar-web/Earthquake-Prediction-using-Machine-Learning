from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_js_eval import get_geolocation

from model_utils import (
    estimate_risk,
    fetch_recent_usgs_events,
    geocode_place,
    load_dataset,
    load_or_train_model,
    nearby_stats,
    recent_nearby_stats,
    reverse_geocode,
)


st.set_page_config(page_title="Earthquake Risk Checker", page_icon="", layout="wide")


st.markdown(
    """
    <style>
    :root {
        color-scheme: light dark;
    }
    .block-container {
        max-width: 1160px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, p, label {
        letter-spacing: 0;
    }
    .hero {
        border: 1px solid rgba(128, 139, 153, 0.35);
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
        background:
            radial-gradient(circle at top right, rgba(214, 123, 49, 0.22), transparent 32%),
            linear-gradient(135deg, rgba(36, 88, 120, 0.24), rgba(128, 139, 153, 0.08));
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.03;
    }
    .hero p {
        max-width: 760px;
        margin: 0;
        color: inherit;
        opacity: 0.82;
        font-size: 1rem;
    }
    .stMetric {
        border: 1px solid rgba(128, 139, 153, 0.45);
        border-radius: 8px;
        padding: 12px 14px;
        background: rgba(128, 139, 153, 0.10);
        min-height: 112px;
    }
    .stMetric [data-testid="stMetricLabel"],
    .stMetric [data-testid="stMetricValue"],
    .stMetric [data-testid="stMetricDelta"] {
        color: inherit;
    }
    .address-box, .note {
        border: 1px solid rgba(128, 139, 153, 0.35);
        border-radius: 8px;
        padding: 13px 15px;
        margin: 12px 0;
        background: rgba(128, 139, 153, 0.08);
    }
    .address-box span {
        opacity: 0.72;
        display: block;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .note {
        border-left: 4px solid #d47b31;
        background: rgba(212, 123, 49, 0.16);
    }
    .status-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 16px;
    }
    .status-pill {
        border: 1px solid rgba(128, 139, 153, 0.32);
        border-radius: 8px;
        padding: 10px 12px;
        background: rgba(128, 139, 153, 0.08);
    }
    .status-pill strong {
        display: block;
        font-size: 0.82rem;
    }
    .status-pill span {
        opacity: 0.75;
        font-size: 0.9rem;
    }
    button[kind="primary"], button[kind="secondary"] {
        border-radius: 6px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 139, 153, 0.25);
    }
    @media (max-width: 800px) {
        .hero { padding: 18px; }
        .status-row { grid-template-columns: 1fr; }
        .stMetric { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_or_train_model()


@st.cache_data(ttl=3600)
def get_dataset():
    return load_dataset()


@st.cache_data(ttl=900)
def get_recent_events():
    return fetch_recent_usgs_events()


def query_float(name: str, default: float) -> float:
    try:
        return float(st.query_params.get(name, default))
    except (TypeError, ValueError):
        return default


def set_location(lat: float, lon: float, address: str | None = None) -> None:
    st.query_params["lat"] = f"{lat:.6f}"
    st.query_params["lon"] = f"{lon:.6f}"
    if address:
        st.query_params["address"] = address
    elif "address" in st.query_params:
        del st.query_params["address"]
    st.rerun()


def render_map(lat: float, lon: float, historical_events: pd.DataFrame, recent_events: pd.DataFrame) -> None:
    selected = pd.DataFrame([{"Latitude": lat, "Longitude": lon, "Magnitude": "Selected area", "Depth": "", "Place": ""}])
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=historical_events.head(90),
            get_position="[Longitude, Latitude]",
            get_radius=30000,
            get_fill_color=[214, 123, 49, 115],
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=recent_events.head(90),
            get_position="[Longitude, Latitude]",
            get_radius=42000,
            get_fill_color=[45, 145, 210, 165],
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=selected,
            get_position="[Longitude, Latitude]",
            get_radius=58000,
            get_fill_color=[236, 238, 241, 235],
            get_line_color=[20, 28, 35],
            line_width_min_pixels=2,
            stroked=True,
            pickable=True,
        ),
    ]
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=5, pitch=0),
            layers=layers,
            tooltip={"text": "Magnitude: {Magnitude}\nDepth: {Depth} km\nDistance: {distance_km} km\n{Place}"},
        ),
        use_container_width=True,
    )


model, model_metrics = get_model()
dataset = get_dataset()
recent_df = get_recent_events()

with st.sidebar:
    st.header("Check Area")
    place_query = st.text_input("Search place", placeholder="Example: Guwahati, India")
    if st.button("Find place", type="primary", use_container_width=True):
        result = geocode_place(place_query)
        if result:
            set_location(result["lat"], result["lon"], result["address"])
        else:
            st.warning("No matching place was found.")

    if st.button("Use current location", use_container_width=True):
        st.session_state["request_browser_location"] = True

    if st.session_state.get("request_browser_location"):
        location = get_geolocation(component_key="browser_location")
        if location and location.get("coords"):
            coords = location["coords"]
            set_location(float(coords["latitude"]), float(coords["longitude"]))
        else:
            st.caption("Allow location permission in the browser when prompted.")

    lat = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=query_float("lat", 28.6139),
        step=0.0001,
        format="%.6f",
    )
    lon = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=query_float("lon", 77.2090),
        step=0.0001,
        format="%.6f",
    )

    if st.button("Apply coordinates", use_container_width=True):
        set_location(lat, lon)

    historical_preview = nearby_stats(dataset, lat, lon)
    depth = st.number_input(
        "Depth estimate in km",
        min_value=0.0,
        max_value=800.0,
        value=float(historical_preview["suggested_depth"]),
        step=1.0,
        help="Suggested from the nearest historical earthquake records.",
    )
    threshold = st.slider("Concern magnitude", 2.0, 7.0, 4.0, 0.1)

st.markdown(
    """
    <section class="hero">
        <h1>Earthquake Risk Checker</h1>
        <p>Search a place, use your current location, or enter coordinates to estimate local seismic risk from trained historical patterns and live earthquake activity.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

prediction_input = pd.DataFrame([{"Latitude": lat, "Longitude": lon, "Depth": depth}])
predicted_magnitude = float(model.predict(prediction_input)[0])
historical = nearby_stats(dataset, lat, lon)
recent = recent_nearby_stats(recent_df, lat, lon)
risk_score, risk_label = estimate_risk(predicted_magnitude, historical, recent)
chance_against_threshold = max(0, min(100, int(round(risk_score + (predicted_magnitude - threshold) * 12))))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Risk estimate", f"{risk_score}%", risk_label)
col2.metric("Expected magnitude", f"{predicted_magnitude:.2f}")
col3.metric(f"Chance near M{threshold:.1f}+", f"{chance_against_threshold}%")
col4.metric("Recent nearby events", str(recent["count"]))

address = st.query_params.get("address")
query_lat = query_float("lat", lat)
query_lon = query_float("lon", lon)
if not address or abs(query_lat - lat) > 0.00001 or abs(query_lon - lon) > 0.00001:
    address = reverse_geocode(lat, lon)

st.markdown(
    f"""
    <div class="address-box">
        <span>Selected address</span>
        {address}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status-row">
        <div class="status-pill"><strong>Model</strong><span>{model_metrics.get("model_name", "trained regressor")}</span></div>
        <div class="status-pill"><strong>Training quality</strong><span>MAE {model_metrics["mae"]:.3f}, R2 {model_metrics["r2"]:.3f}</span></div>
        <div class="status-pill"><strong>Historical signal</strong><span>{historical["count"]} records within 250 km</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="note">Educational risk estimate only. Exact earthquake prediction by time and place is not currently reliable; follow official alerts and emergency guidance for safety decisions.</div>',
    unsafe_allow_html=True,
)

st.subheader("Map")
render_map(lat, lon, historical["events"], recent["events"])
