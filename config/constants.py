"""
VayuDrishti AI — Central configuration and constants
ET AI Hackathon 2026 | PS5: Urban Air Quality Intelligence
"""
from datetime import datetime

# ── City registry ────────────────────────────────────────────────────────────
CITIES = {
    "Chennai":       {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu",      "lang": "Tamil",   "pop": 10_971_108},
    "Vijayawada":    {"lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh",  "lang": "Telugu",  "pop": 1_048_240},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh",  "lang": "Telugu",  "pop": 2_035_922},
    "Tirupati":      {"lat": 13.6288, "lon": 79.4192, "state": "Andhra Pradesh",  "lang": "Telugu",  "pop": 459_985},
    "Coimbatore":    {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu",      "lang": "Tamil",   "pop": 2_151_466},
    "Madurai":       {"lat":  9.9252, "lon": 78.1198, "state": "Tamil Nadu",      "lang": "Tamil",   "pop": 1_561_129},
    "Guntur":        {"lat": 16.3067, "lon": 80.4365, "state": "Andhra Pradesh",  "lang": "Telugu",  "pop": 743_354},
    "Nellore":       {"lat": 14.4426, "lon": 79.9865, "state": "Andhra Pradesh",  "lang": "Telugu",  "pop": 564_584},
    "Delhi":         {"lat": 28.6139, "lon": 77.2090, "state": "Delhi",           "lang": "Hindi",   "pop": 32_941_309},
    "Mumbai":        {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra",     "lang": "Hindi",   "pop": 20_961_472},
    "Hyderabad":     {"lat": 17.3850, "lon": 78.4867, "state": "Telangana",       "lang": "Telugu",  "pop": 10_534_418},
    "Bengaluru":     {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka",       "lang": "Hindi",   "pop": 13_608_000},
}

# ── AQI thresholds (India CPCB) ──────────────────────────────────────────────
AQI_SCALE = [
    (50,  "Good",         "#16a34a", "🟢", 0),
    (100, "Satisfactory", "#65a30d", "🟡", 1),
    (200, "Moderate",     "#ca8a04", "🟠", 2),
    (300, "Poor",         "#ea580c", "🔴", 3),
    (400, "Very Poor",    "#dc2626", "🔴", 4),
    (500, "Severe",       "#7c3aed", "☠️", 5),
]

# ── Source types for attribution ─────────────────────────────────────────────
POLLUTION_SOURCES = {
    "Traffic":       {"icon": "🚗", "color": "#ef4444", "pm25_factor": 0.35, "pm10_factor": 0.28},
    "Industrial":    {"icon": "🏭", "color": "#f97316", "pm25_factor": 0.30, "pm10_factor": 0.35},
    "Construction":  {"icon": "🏗️", "color": "#eab308", "pm25_factor": 0.10, "pm10_factor": 0.25},
    "Waste Burning": {"icon": "🔥", "color": "#8b5cf6", "pm25_factor": 0.15, "pm10_factor": 0.08},
    "Dust":          {"icon": "🌫️", "color": "#6b7280", "pm25_factor": 0.05, "pm10_factor": 0.03},
    "Biomass":       {"icon": "🌾", "color": "#84cc16", "pm25_factor": 0.05, "pm10_factor": 0.01},
}

# ── Pollutants and WHO limits ─────────────────────────────────────────────────
POLLUTANTS = {
    "PM2.5": {"unit": "μg/m³", "who_limit": 15,  "cpcb_limit": 60},
    "PM10":  {"unit": "μg/m³", "who_limit": 45,  "cpcb_limit": 100},
    "NO2":   {"unit": "μg/m³", "who_limit": 40,  "cpcb_limit": 80},
    "SO2":   {"unit": "μg/m³", "who_limit": 20,  "cpcb_limit": 80},
    "CO":    {"unit": "mg/m³", "who_limit": 4.0, "cpcb_limit": 2.0},
    "O3":    {"unit": "μg/m³", "who_limit": 100, "cpcb_limit": 180},
}

# ── Realistic base AQI per city (from CPCB 2024 data) ───────────────────────
CITY_BASE_AQI = {
    "Delhi": 218, "Mumbai": 145, "Hyderabad": 130, "Bengaluru": 105,
    "Visakhapatnam": 110, "Chennai": 95, "Guntur": 92, "Vijayawada": 85,
    "Nellore": 80, "Madurai": 88, "Coimbatore": 70, "Tirupati": 75,
}

# ── Vulnerable population groups ─────────────────────────────────────────────
VULNERABLE_GROUPS = {
    "Children (0-14)":      {"multiplier": 1.40, "icon": "👧", "threshold": 100},
    "Elderly (60+)":        {"multiplier": 1.30, "icon": "👴", "threshold": 100},
    "Pregnant Women":       {"multiplier": 1.25, "icon": "🤰", "threshold": 80},
    "Asthma Patients":      {"multiplier": 1.50, "icon": "🫁", "threshold": 80},
    "Outdoor Workers":      {"multiplier": 1.60, "icon": "👷", "threshold": 100},
    "Heart Disease Patients":{"multiplier": 1.35, "icon": "❤️", "threshold": 100},
}

# ── What-if simulation parameters ────────────────────────────────────────────
WHATIF_SCENARIOS = {
    "Traffic -20%":         {"traffic": 0.80, "industrial": 1.00, "construction": 1.00, "rainfall": 0},
    "Industrial -15%":      {"traffic": 1.00, "industrial": 0.85, "construction": 1.00, "rainfall": 0},
    "Light Rainfall":       {"traffic": 1.00, "industrial": 1.00, "construction": 0.90, "rainfall": 1},
    "Odd-Even Traffic":     {"traffic": 0.50, "industrial": 1.00, "construction": 1.00, "rainfall": 0},
    "Construction Ban":     {"traffic": 1.00, "industrial": 1.00, "construction": 0.00, "rainfall": 0},
    "Industrial Shutdown":  {"traffic": 1.00, "industrial": 0.00, "construction": 1.00, "rainfall": 0},
    "All Interventions":    {"traffic": 0.70, "industrial": 0.75, "construction": 0.50, "rainfall": 1},
}

APP_VERSION = "2.0.0"
APP_NAME = "VayuDrishti AI"
HACKATHON = "ET AI Hackathon 2026 | PS5"
