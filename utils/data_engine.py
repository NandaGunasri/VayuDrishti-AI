"""
utils/data_engine.py  — VayuDrishti AI v2.0
Data layer: CPCB AQI · OpenWeatherMap · Synthetic fallback
Production-hardened: all external calls wrapped, all DataFrames validated.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)


from config.constants import CITIES, CITY_BASE_AQI, POLLUTION_SOURCES
from utils.helpers import ensure_pollutants, validate_df, safe_val

logger = logging.getLogger(__name__)

# ── Weather defaults (per-city realistic proxies) ─────────────────────────────
_WEATHER_DEFAULTS: dict[str, dict] = {
    "Chennai":       {"temp": 34, "humidity": 72, "wind_speed": 5.2},
    "Vijayawada":    {"temp": 38, "humidity": 65, "wind_speed": 4.8},
    "Visakhapatnam": {"temp": 32, "humidity": 78, "wind_speed": 6.1},
    "Delhi":         {"temp": 40, "humidity": 45, "wind_speed": 4.0},
    "Mumbai":        {"temp": 30, "humidity": 80, "wind_speed": 7.0},
    "Hyderabad":     {"temp": 36, "humidity": 60, "wind_speed": 4.5},
    "Bengaluru":     {"temp": 26, "humidity": 68, "wind_speed": 3.8},
}

_FALLBACK_WEATHER = {
    "temp": 33.0, "humidity": 65, "pressure": 1010,
    "wind_speed": 5.0, "wind_deg": 180, "rainfall": 0.0,
    "description": "Haze", "icon": "50d",
    "visibility": 8.0, "source": "simulated",
}


def _city_weather_fallback(city: str) -> dict:
    """Return a city-specific simulated weather dict."""
    np.random.seed((hash(city) + datetime.now().hour) % 2 ** 31)
    base = _WEATHER_DEFAULTS.get(city, {"temp": 33, "humidity": 65, "wind_speed": 5})
    return {
        "temp":        round(base["temp"] + np.random.normal(0, 1.5), 1),
        "humidity":    int(np.clip(base["humidity"] + np.random.normal(0, 5), 20, 100)),
        "pressure":    int(np.random.uniform(1005, 1015)),
        "wind_speed":  round(max(0.3, base["wind_speed"] + np.random.normal(0, 1)), 1),
        "wind_deg":    int(np.random.uniform(0, 360)),
        "rainfall":    round(max(0.0, np.random.normal(0, 0.3)), 1),
        "description": np.random.choice(["Clear Sky", "Partly Cloudy", "Haze", "Mist"]),
        "icon":        "02d",
        "visibility":  round(np.random.uniform(5, 15), 1),
        "source":      "simulated",
    }


# ── OpenWeatherMap ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_weather(city: str, api_key: str = "") -> dict:
    """
    Fetch current weather from OpenWeatherMap.
    Returns city-specific simulated weather on any failure.
    Never raises; never exposes API keys.
    """
    if not api_key:
        try:
            api_key = st.session_state.get("OWM_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        api_key = os.getenv("OWM_API_KEY", "")
    if api_key.startswith("your_") or "placeholder" in api_key.lower():
        api_key = ""

    city_info = CITIES.get(city, {})
    if not city_info:
        return _city_weather_fallback(city)

    if api_key:
        try:
            r = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": city_info["lat"], "lon": city_info["lon"],
                    "appid": api_key, "units": "metric",
                },
                timeout=8,
            )
            if r.status_code == 200:
                d = r.json()
                return {
                    "temp":        d["main"]["temp"],
                    "humidity":    d["main"]["humidity"],
                    "pressure":    d["main"]["pressure"],
                    "wind_speed":  d["wind"]["speed"],
                    "wind_deg":    d["wind"].get("deg", 0),
                    "rainfall":    d.get("rain", {}).get("1h", 0.0),
                    "description": d["weather"][0]["description"].title(),
                    "icon":        d["weather"][0]["icon"],
                    "visibility":  d.get("visibility", 10000) / 1000,
                    "source":      "live",
                }
        except Exception as exc:
            logger.warning("fetch_weather(%s): %s — using simulated data", city, exc)

    return _city_weather_fallback(city)


@st.cache_data(ttl=3600)
def fetch_weather_forecast(city: str, api_key: str = "") -> pd.DataFrame:
    """
    72-hour hourly weather forecast.
    Falls back to synthetic forecast on any failure.
    """
    if not api_key:
        try:
            api_key = st.session_state.get("OWM_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        api_key = os.getenv("OWM_API_KEY", "")
    if api_key.startswith("your_") or "placeholder" in api_key.lower():
        api_key = ""
    city_info = CITIES.get(city, {})
    rows: list[dict] = []

    if api_key and city_info:
        try:
            r = requests.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={
                    "lat": city_info["lat"], "lon": city_info["lon"],
                    "appid": api_key, "units": "metric", "cnt": 24,
                },
                timeout=8,
            )
            if r.status_code == 200:
                for item in r.json().get("list", []):
                    rows.append({
                        "timestamp":  datetime.fromtimestamp(item["dt"]),
                        "temp":       item["main"]["temp"],
                        "humidity":   item["main"]["humidity"],
                        "wind_speed": item["wind"]["speed"],
                        "wind_deg":   item["wind"].get("deg", 0),
                        "rainfall":   item.get("rain", {}).get("3h", 0.0),
                        "pressure":   item["main"]["pressure"],
                    })
                if rows:
                    return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning("fetch_weather_forecast(%s): %s — using synthetic", city, exc)

    return _synthetic_weather_forecast(city)


def _synthetic_weather_forecast(city: str) -> pd.DataFrame:
    """Generate 72h synthetic weather forecast."""
    np.random.seed(hash(city) % 2 ** 31)
    base = _WEATHER_DEFAULTS.get(city, {"temp": 33, "humidity": 65, "wind_speed": 5})
    now = datetime.now()
    rows = []
    for h in range(72):
        t = now + timedelta(hours=h)
        rows.append({
            "timestamp":  t,
            "temp":       round(base["temp"] + 3 * np.sin(h * np.pi / 12) + np.random.normal(0, 1.5), 1),
            "humidity":   int(np.clip(base["humidity"] + 10 * np.sin(h * np.pi / 24) + np.random.normal(0, 4), 20, 100)),
            "wind_speed": round(max(0.5, base["wind_speed"] + 2 * np.sin(h * np.pi / 18) + np.random.normal(0, 0.8)), 1),
            "wind_deg":   int((h * 15 + 180) % 360),
            "rainfall":   round(max(0.0, np.random.normal(0, 0.25)), 1),
            "pressure":   int(1010 + np.random.normal(0, 3)),
        })
    return pd.DataFrame(rows)


# ── Station registry ───────────────────────────────────────────────────────────
STATIONS_DB: dict[str, list[dict]] = {
    "Chennai": [
        {"name": "Alandur Bus Depot", "zone": "Commercial",  "lat": 12.9990, "lon": 80.2004, "near_highway": True,  "near_industry": False},
        {"name": "Arumbakkam",        "zone": "Residential", "lat": 13.0695, "lon": 80.2133, "near_highway": False, "near_industry": False},
        {"name": "Manali Industrial", "zone": "Industrial",  "lat": 13.1669, "lon": 80.2614, "near_highway": False, "near_industry": True},
        {"name": "Velachery",         "zone": "Residential", "lat": 12.9815, "lon": 80.2209, "near_highway": True,  "near_industry": False},
        {"name": "Kodungaiyur",       "zone": "Industrial",  "lat": 13.1321, "lon": 80.2720, "near_highway": False, "near_industry": True},
        {"name": "Perungudi",         "zone": "Commercial",  "lat": 12.9620, "lon": 80.2430, "near_highway": True,  "near_industry": False},
    ],
    "Vijayawada": [
        {"name": "Auto Nagar",    "zone": "Industrial",  "lat": 16.4898, "lon": 80.6011, "near_highway": False, "near_industry": True},
        {"name": "Krishnalanka", "zone": "Residential", "lat": 16.5090, "lon": 80.6205, "near_highway": False, "near_industry": False},
        {"name": "PWD Grounds",  "zone": "Commercial",  "lat": 16.5220, "lon": 80.6356, "near_highway": True,  "near_industry": False},
        {"name": "Gunadala",     "zone": "Residential", "lat": 16.5316, "lon": 80.6185, "near_highway": False, "near_industry": False},
    ],
    "Visakhapatnam": [
        {"name": "GVMC Office",      "zone": "Commercial", "lat": 17.7231, "lon": 83.3012, "near_highway": True,  "near_industry": False},
        {"name": "Steel Plant Area", "zone": "Industrial", "lat": 17.6819, "lon": 83.1987, "near_highway": False, "near_industry": True},
        {"name": "Bheemunipatnam",   "zone": "Coastal",    "lat": 17.8880, "lon": 83.4467, "near_highway": False, "near_industry": False},
        {"name": "Pedagantyada",     "zone": "Industrial", "lat": 17.7641, "lon": 83.2218, "near_highway": False, "near_industry": True},
    ],
    "Delhi": [
        {"name": "Anand Vihar",  "zone": "Residential", "lat": 28.6469, "lon": 77.3152, "near_highway": True,  "near_industry": False},
        {"name": "ITO",          "zone": "Commercial",  "lat": 28.6289, "lon": 77.2440, "near_highway": True,  "near_industry": False},
        {"name": "Okhla Phase 2","zone": "Industrial",  "lat": 28.5411, "lon": 77.2703, "near_highway": False, "near_industry": True},
        {"name": "Punjabi Bagh", "zone": "Residential", "lat": 28.6721, "lon": 77.1316, "near_highway": False, "near_industry": False},
        {"name": "Jahangirpuri", "zone": "Residential", "lat": 28.7336, "lon": 77.1632, "near_highway": False, "near_industry": False},
    ],
    "Hyderabad": [
        {"name": "Bollaram Ind Area",  "zone": "Industrial",  "lat": 17.5014, "lon": 78.3956, "near_highway": False, "near_industry": True},
        {"name": "ICRISAT Patancheru", "zone": "Industrial",  "lat": 17.5136, "lon": 78.2646, "near_highway": False, "near_industry": True},
        {"name": "Sanathnagar",        "zone": "Residential", "lat": 17.4492, "lon": 78.4230, "near_highway": True,  "near_industry": False},
        {"name": "Zoo Park",           "zone": "Residential", "lat": 17.3561, "lon": 78.4527, "near_highway": False, "near_industry": False},
    ],
}


def _get_stations(city: str) -> list[dict]:
    if city in STATIONS_DB:
        return STATIONS_DB[city]
    info = CITIES.get(city, {"lat": 20.0, "lon": 78.0})
    return [{"name": f"{city} Central", "zone": "Central",
             "lat": info["lat"], "lon": info["lon"],
             "near_highway": True, "near_industry": False}]


# ── CPCB AQI ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_aqi_data(city: str, api_key: str = "") -> pd.DataFrame:
    """
    Fetch AQI data from CPCB API.
    Always returns a validated DataFrame — never raises.
    """
    if not api_key:
        try:
            api_key = st.session_state.get("CPCB_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        api_key = os.getenv("CPCB_API_KEY", "")
    if api_key.startswith("your_") or "placeholder" in api_key.lower():
        api_key = ""
    if api_key:
        try:
            r = requests.get(
                "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
                params={"api-key": api_key, "format": "json", "limit": 200, "filters[city]": city},
                timeout=10,
            )
            if r.status_code == 200:
                records = r.json().get("records", [])
                if records:
                    raw = pd.DataFrame(records)
                    try:
                        if "pollutant_id" in raw.columns and "pollutant_avg" in raw.columns:
                             raw["pollutant_id"] = raw["pollutant_id"].replace({
                                 "OZONE": "O3", "Ozone": "O3",
                                 "PM2.5": "PM2.5", "PM10": "PM10",
                                 "NO2": "NO2", "SO2": "SO2", "CO": "CO", "NH3": "NH3", "Pb": "Pb"
                             })
                             raw["pollutant_avg"] = pd.to_numeric(raw["pollutant_avg"], errors="coerce")
                             if "station" not in raw.columns:
                                 raw["station"] = "Unknown Station"
                             if "last_update" in raw.columns:
                                 raw["timestamp"] = pd.to_datetime(raw["last_update"], errors="coerce")
                             else:
                                 raw["timestamp"] = datetime.now()
                             if "latitude" in raw.columns and "longitude" in raw.columns:
                                 raw["lat"] = pd.to_numeric(raw["latitude"], errors="coerce")
                                 raw["lon"] = pd.to_numeric(raw["longitude"], errors="coerce")
                             index_cols = ["station", "timestamp"]
                             for c in ["lat", "lon"]:
                                 if c in raw.columns:
                                     index_cols.append(c)
                             pivoted = raw.pivot_table(
                                 index=index_cols,
                                 columns="pollutant_id",
                                 values="pollutant_avg",
                                 aggfunc="mean"
                             ).reset_index()
                             if "AQI" not in pivoted.columns:
                                 pm25_val = pivoted["PM2.5"] if "PM2.5" in pivoted.columns else pd.Series(60.0, index=pivoted.index)
                                 pm10_val = pivoted["PM10"] if "PM10" in pivoted.columns else pd.Series(100.0, index=pivoted.index)
                                 pivoted["AQI"] = pm25_val.fillna(60.0) * 1.2 + pm10_val.fillna(100.0) * 0.4
                                 pivoted["AQI"] = pivoted["AQI"].clip(20, 500)
                             stations_lookup = {s["name"]: s for s in _get_stations(city)}
                             city_coords = CITIES.get(city, {"lat": 20.0, "lon": 78.0})
                             if "lat" not in pivoted.columns:
                                 pivoted["lat"] = pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("lat", city_coords["lat"]))
                             else:
                                 pivoted["lat"] = pivoted["lat"].fillna(pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("lat", city_coords["lat"])))
                             if "lon" not in pivoted.columns:
                                 pivoted["lon"] = pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("lon", city_coords["lon"]))
                             else:
                                 pivoted["lon"] = pivoted["lon"].fillna(pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("lon", city_coords["lon"])))
                             if "zone" not in pivoted.columns:
                                 pivoted["zone"] = pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("zone", "Central"))
                             else:
                                 pivoted["zone"] = pivoted["zone"].fillna(pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("zone", "Central")))
                             if "near_highway" not in pivoted.columns:
                                 pivoted["near_highway"] = pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("near_highway", False))
                             else:
                                 pivoted["near_highway"] = pivoted["near_highway"].fillna(pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("near_highway", False)))
                             if "near_industry" not in pivoted.columns:
                                 pivoted["near_industry"] = pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("near_industry", False))
                             else:
                                 pivoted["near_industry"] = pivoted["near_industry"].fillna(pivoted["station"].map(lambda x: stations_lookup.get(x, {}).get("near_industry", False)))
                             return validate_df(pivoted, context=f"CPCB/{city}")
                        return validate_df(ensure_pollutants(raw), context=f"CPCB/{city}")
                    except ValueError:
                        pass  # fall through to synthetic
        except Exception as exc:
            logger.warning("fetch_aqi_data(%s): %s — using synthetic data", city, exc)

    return _synthetic_aqi(city)


def _synthetic_aqi(city: str) -> pd.DataFrame:
    """
    Generate 72h realistic synthetic AQI data.
    All pollutant columns are guaranteed to exist and be numeric.
    """
    np.random.seed(hash(city) % 2 ** 31)
    b = CITY_BASE_AQI.get(city, 100)
    stations = _get_stations(city)
    rows: list[dict] = []
    now = datetime.now()

    for s in stations:
        zone_boost = {"Industrial": 35, "Commercial": 15, "Residential": 0,
                      "Coastal": -10, "Central": 10}.get(s["zone"], 0)
        for h in range(72):
            t = now - timedelta(hours=72 - h)
            hour_factor = 1 + 0.35 * np.sin((t.hour - 8) * np.pi / 12)
            aqi  = max(20, int((b + zone_boost) * hour_factor + np.random.normal(0, 10)))
            pm25 = max(5,  int(aqi * 0.58 + np.random.normal(0, 4)))
            pm10 = max(10, int(aqi * 1.05 + np.random.normal(0, 7)))
            no2  = max(5,  int(30 + zone_boost * 0.4 + np.random.normal(0, 6)))
            so2  = max(2,  int(12 + (30 if s["near_industry"] else 0) + np.random.normal(0, 3)))
            co   = round(max(0.1, 0.9 + np.random.normal(0, 0.2)), 2)
            o3   = max(5,  int(45 + np.random.normal(0, 10)))
            nh3  = max(1,  int(10 + np.random.normal(0, 3)))
            rows.append({
                "station":       s["name"],
                "zone":          s["zone"],
                "lat":           s["lat"] + np.random.normal(0, 0.0005),
                "lon":           s["lon"] + np.random.normal(0, 0.0005),
                "near_highway":  s["near_highway"],
                "near_industry": s["near_industry"],
                "timestamp":     t,
                "hour":          t.hour,
                "day_of_week":   t.weekday(),
                "AQI":   aqi,  "PM2.5": pm25, "PM10": pm10,
                "NO2":   no2,  "SO2":   so2,  "CO":   co,
                "O3":    o3,   "NH3":   nh3,
            })

    df = pd.DataFrame(rows)
    return ensure_pollutants(df)   # guarantees all columns present & numeric


# ── Source attribution ─────────────────────────────────────────────────────────
def compute_source_attribution(row: pd.Series, weather: dict) -> dict[str, float]:
    """
    Compute pollution source contribution %.
    Uses safe_val() so missing columns never crash attribution.
    Returns dict summing to 100.
    """
    pm25     = safe_val(row, "PM2.5", 60.0)
    pm10     = safe_val(row, "PM10",  100.0)
    no2      = safe_val(row, "NO2",   40.0)
    so2      = safe_val(row, "SO2",   20.0)
    pm_ratio = pm25 / max(pm10, 1.0)

    wind     = float(weather.get("wind_speed", 5.0))
    is_ind   = bool(row.get("near_industry", False)) if hasattr(row, "get") else False
    is_hwy   = bool(row.get("near_highway",  False)) if hasattr(row, "get") else False
    hour     = int(safe_val(row, "hour", datetime.now().hour))

    scores = {
        "Traffic":       (0.4 if is_hwy else 0.2) * (1 + no2 / 80) * (1.3 if 7 <= hour <= 10 or 17 <= hour <= 20 else 0.8),
        "Industrial":    (0.5 if is_ind else 0.1) * (1 + so2 / 40) * (1.2 if 9 <= hour <= 18 else 0.6),
        "Construction":  0.15 * (1 - pm_ratio) * (1.4 if 8 <= hour <= 17 else 0.3) * (0.5 if wind > 8 else 1.0),
        "Waste Burning": 0.10 * (1.5 if 18 <= hour <= 22 or 5 <= hour <= 8 else 0.4),
        "Dust":          0.12 * (1 - pm_ratio) * (1.3 if wind > 6 else 0.7),
        "Biomass":       0.08 * (1.4 if 5 <= hour <= 9 else 0.5),
    }
    total = sum(scores.values()) or 1.0
    return {k: round(v / total * 100, 1) for k, v in scores.items()}


def get_ward_risk_scores(df: pd.DataFrame, weather: dict) -> pd.DataFrame:
    """
    Compute ward/zone risk scores.
    Defensively handles any missing columns via safe_val().
    """
    try:
        df = ensure_pollutants(df)
        latest = df.groupby("station").last().reset_index()
    except Exception as exc:
        logger.error("get_ward_risk_scores: groupby failed — %s", exc)
        return pd.DataFrame()

    rows: list[dict] = []
    for _, r in latest.iterrows():
        try:
            attr     = compute_source_attribution(r, weather)
            dominant = max(attr, key=attr.get)
            aqi_v    = safe_val(r, "AQI", 100.0)
            risk     = min(100, int(aqi_v / 5 + (10 if r.get("near_industry") else 0)))
            rows.append({
                "station":          r.get("station", "Unknown"),
                "zone":             r.get("zone",    "General"),
                "lat":              safe_val(r, "lat", 0.0),
                "lon":              safe_val(r, "lon", 0.0),
                "AQI":              aqi_v,
                "PM2.5":            safe_val(r, "PM2.5"),
                "PM10":             safe_val(r, "PM10"),
                "NO2":              safe_val(r, "NO2"),
                "SO2":              safe_val(r, "SO2"),
                "O3":               safe_val(r, "O3"),
                "CO":               safe_val(r, "CO"),
                "near_highway":     bool(r.get("near_highway", False)),
                "near_industry":    bool(r.get("near_industry", False)),
                "dominant_source":  dominant,
                "risk_score":       risk,
                **{f"src_{k}": v for k, v in attr.items()},
            })
        except Exception as exc:
            logger.warning("get_ward_risk_scores: skipped row — %s", exc)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)
