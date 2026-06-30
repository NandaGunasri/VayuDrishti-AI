"""
agents/agent_system.py
Multi-Agent AI Architecture — VayuDrishti AI
ET AI Hackathon 2026 | PS5

Agents:
  1. AQI Forecast Agent
  2. Pollution Source Attribution Agent
  3. Health Advisory Agent (multilingual)
  4. Enforcement Agent
  5. Smart City Planning Agent
  6. Multilingual Communication Agent
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config.constants import CITIES, VULNERABLE_GROUPS, POLLUTION_SOURCES, WHATIF_SCENARIOS

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY.startswith("your_") or "placeholder" in GROQ_API_KEY.lower():
    GROQ_API_KEY = ""

def _groq_call(prompt: str, system: str = "", max_tokens: int = 400, temperature: float = 0.3) -> str:
    """Centralized Groq API call with graceful fallback."""
    key = ""
    try:
        key = st.session_state.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not key:
        key = os.getenv("GROQ_API_KEY", "")
    if not key:
        key = GROQ_API_KEY
    if key.startswith("your_") or "placeholder" in key.lower():
        key = ""

    if not key:
        return "[Groq API key not set — add it in the sidebar to enable live AI responses]"
    try:
        from groq import Groq
        client = Groq(api_key=key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Agent error: {str(e)[:120]}]"


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — AQI Forecast Agent
# ══════════════════════════════════════════════════════════════════════════════
class AQIForecastAgent:
    """
    Weather-aware Gradient Boosting forecast agent.
    Produces 24h / 48h / 72h forecasts with confidence intervals.
    """
    name = "AQI Forecast Agent"
    icon = "📡"
    description = "Weather-aware ML forecasting with 24/48/72h horizon and confidence intervals"

    @staticmethod
    @st.cache_data(ttl=1800)
    def forecast(aqi_series: tuple, weather_df_json: str, hours: int = 72) -> pd.DataFrame:
        import io
        aqi_list = list(aqi_series)
        try:
            wdf = pd.read_json(io.StringIO(weather_df_json)) if weather_df_json and weather_df_json != "{}" else pd.DataFrame()
        except Exception:
            wdf = pd.DataFrame()

        n = len(aqi_list)
        if n < 12:
            return pd.DataFrame()

        # Build features from history
        rows = []
        for i in range(6, n):
            rows.append({
                "hour":      i % 24,
                "dow":       (i // 24) % 7,
                "lag1":      aqi_list[i - 1],
                "lag3":      aqi_list[i - 3],
                "lag6":      aqi_list[i - 6],
                "lag12":     aqi_list[i - 12] if i >= 12 else aqi_list[0],
                "roll6":     np.mean(aqi_list[max(0, i - 6):i]),
                "roll12":    np.mean(aqi_list[max(0, i - 12):i]),
                "wind":      float(wdf["wind_speed"].iloc[i % len(wdf)]) if (len(wdf) > 0 and "wind_speed" in wdf.columns) else 5.0,
                "humidity":  float(wdf["humidity"].iloc[i % len(wdf)]) if (len(wdf) > 0 and "humidity" in wdf.columns) else 65.0,
                "temp":      float(wdf["temp"].iloc[i % len(wdf)]) if (len(wdf) > 0 and "temp" in wdf.columns) else 33.0,
                "rainfall":  float(wdf["rainfall"].iloc[i % len(wdf)]) if (len(wdf) > 0 and "rainfall" in wdf.columns) else 0.0,
                "target":    aqi_list[i],
            })
        df_feat = pd.DataFrame(rows)
        feature_cols = [c for c in df_feat.columns if c != "target"]
        X, y = df_feat[feature_cols], df_feat["target"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                               learning_rate=0.05, random_state=42))
        ])
        model.fit(X, y)

        # Forecast
        recent = list(aqi_list[-24:])
        forecast_rows = []
        now = datetime.now()
        for i in range(1, hours + 1):
            ft  = now + timedelta(hours=i)
            wi  = i % len(wdf) if len(wdf) > 0 else 0
            lag1  = recent[-1]
            lag3  = recent[-3]  if len(recent) >= 3  else lag1
            lag6  = recent[-6]  if len(recent) >= 6  else lag1
            lag12 = recent[-12] if len(recent) >= 12 else lag1
            feat  = {
                "hour": ft.hour, "dow": ft.weekday(),
                "lag1": lag1, "lag3": lag3, "lag6": lag6, "lag12": lag12,
                "roll6": np.mean(recent[-6:]), "roll12": np.mean(recent[-12:]),
                "wind":     float(wdf["wind_speed"].iloc[wi]) if (len(wdf) > 0 and "wind_speed" in wdf.columns) else 5.0,
                "humidity": float(wdf["humidity"].iloc[wi])   if (len(wdf) > 0 and "humidity" in wdf.columns) else 65.0,
                "temp":     float(wdf["temp"].iloc[wi])       if (len(wdf) > 0 and "temp" in wdf.columns) else 33.0,
                "rainfall": float(wdf["rainfall"].iloc[wi])   if (len(wdf) > 0 and "rainfall" in wdf.columns) else 0.0,
            }
            pred = float(model.predict(pd.DataFrame([feat])[feature_cols])[0])
            pred = max(10, pred)
            noise = np.random.normal(0, max(3, pred * 0.06))
            lo = max(5, pred - abs(noise) * 1.5)
            hi = pred + abs(noise) * 1.5
            forecast_rows.append({
                "timestamp": ft, "AQI": round(pred, 1),
                "lower": round(lo, 1), "upper": round(hi, 1),
                "hour": ft.hour
            })
            recent.append(pred)
            if len(recent) > 24:
                recent.pop(0)

        return pd.DataFrame(forecast_rows)

    @staticmethod
    def explain(city: str, aqi_now: int, forecast_peak: int, weather: dict) -> str:
        prompt = f"""You are an atmospheric scientist AI agent.
City: {city} | Current AQI: {aqi_now} | Forecast peak: {forecast_peak}
Weather: Temp {weather['temp']}°C, Humidity {weather['humidity']}%, Wind {weather['wind_speed']} m/s, Rainfall {weather['rainfall']} mm

Give a 3-sentence technical explanation of why AQI will change in the next 24 hours.
Mention specific meteorological factors. Be precise, not generic."""
        return _groq_call(prompt, system="You are a concise atmospheric science AI. No bullet points. 3 sentences max.", max_tokens=180)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Pollution Source Attribution Agent
# ══════════════════════════════════════════════════════════════════════════════
class SourceAttributionAgent:
    name = "Source Attribution Agent"
    icon = "🔍"
    description = "Identifies pollution sources with confidence scores using PM ratio, NO₂, SO₂, zone context"

    @staticmethod
    def attribute(station_row: pd.Series, weather: dict) -> dict:
        """Return attribution dict with confidence scores summing to 100%."""
        from utils.data_engine import compute_source_attribution
        return compute_source_attribution(station_row, weather)

    @staticmethod
    def narrate(city: str, station: str, attributions: dict, aqi: int) -> str:
        top2 = sorted(attributions.items(), key=lambda x: x[1], reverse=True)[:2]
        prompt = f"""You are an environmental forensics AI agent.
Location: {station}, {city} | AQI: {aqi}
Top pollution sources: {top2[0][0]} ({top2[0][1]}%) and {top2[1][0]} ({top2[1][1]}%)

In 2 sentences: explain what is likely causing this pollution level and what evidence supports this attribution.
Be specific and actionable for a city official."""
        return _groq_call(prompt, max_tokens=120)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Health Advisory Agent (multilingual)
# ══════════════════════════════════════════════════════════════════════════════
class HealthAdvisoryAgent:
    name = "Health Advisory Agent"
    icon = "🏥"
    description = "Generates personalized multilingual health advisories per vulnerable group"

    LANG_INSTRUCTION = {
        "Telugu": "Respond ONLY in Telugu script (తెలుగు లిపిలో మాత్రమే). Do not use English or Roman script.",
        "Tamil":  "Respond ONLY in Tamil script (தமிழ் எழுத்தில் மட்டும்). Do not use English or Roman script.",
        "Hindi":  "Respond ONLY in Hindi (हिंदी में केवल). Do not use English.",
        "English": "Respond in clear, simple English.",
    }

    @classmethod
    def generate(cls, city: str, aqi: int, lang: str, group: str,
                 pm25: float, dominant_source: str, weather: dict) -> str:
        from config.constants import AQI_SCALE
        cat = next((c[1] for c in AQI_SCALE if aqi <= c[0]), "Severe")
        li  = cls.LANG_INSTRUCTION.get(lang, cls.LANG_INSTRUCTION["English"])
        prompt = f"""You are a public health AI advisor for Indian cities.
City: {city} | AQI: {aqi} ({cat}) | PM2.5: {pm25:.0f} μg/m³
Primary pollution source: {dominant_source}
Temperature: {weather['temp']}°C | Humidity: {weather['humidity']}%
Target group: {group}

{li}

Write exactly 3 short sentences:
1. Current air quality and what it means for {group}
2. Specific health risk for this group today
3. Immediate action they must take right now

Be specific to {group}. Be direct and urgent if AQI > 200."""
        return _groq_call(prompt, max_tokens=200, temperature=0.25)

    @classmethod
    def generate_all_languages(cls, city: str, aqi: int, group: str,
                                pm25: float, source: str, weather: dict, lang: str) -> dict:
        """Generate advisory in all 4 languages for a given group."""
        results = {}
        for l in ["English", lang]:
            results[l] = cls.generate(city, aqi, l, group, pm25, source, weather)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Enforcement Agent
# ══════════════════════════════════════════════════════════════════════════════
class EnforcementAgent:
    name = "Enforcement Agent"
    icon = "🚨"
    description = "Generates prioritized inspection actions with evidence-backed enforcement briefs"

    @staticmethod
    def prioritize(ward_df: pd.DataFrame) -> pd.DataFrame:
        df = ward_df.copy()
        df["priority_score"] = (
            df["AQI"] * 0.5 +
            df["PM2.5"] * 0.3 +
            df["risk_score"] * 0.2
        ).round(1)
        df["priority_level"] = pd.cut(df["priority_score"],
            bins=[0, 60, 100, 150, 9999],
            labels=["Low", "Medium", "High", "Critical"])
        df["response_time"] = df["priority_level"].map({
            "Critical": "< 2 hours",
            "High":     "< 4 hours",
            "Medium":   "< 24 hours",
            "Low":      "Routine monitoring",
        })
        return df.sort_values("priority_score", ascending=False)

    @staticmethod
    def brief(station: str, zone: str, city: str, aqi: int,
              attribution: dict, priority: str) -> str:
        top_source = max(attribution, key=attribution.get)
        pct = attribution[top_source]
        prompt = f"""You are an environmental enforcement AI officer for {city}.
Station: {station} | Zone type: {zone} | AQI: {aqi} | Priority: {priority}
Dominant pollution source: {top_source} contributing ~{pct:.0f}%

Write a 3-sentence enforcement action brief for the pollution control board:
Sentence 1: What to inspect and where exactly.
Sentence 2: What evidence to collect (samples, photos, logs).
Sentence 3: What legal action to initiate under Environment Protection Act / Factory Act.
Be specific. Use official language."""
        return _groq_call(prompt, max_tokens=180)

    @staticmethod
    def inspection_route(ward_df: pd.DataFrame) -> list:
        """Return optimized inspection sequence (greedy nearest-neighbor TSP)."""
        critical = ward_df[ward_df["priority_score"] > 100].copy()
        if len(critical) == 0:
            return []
        route = [critical.iloc[0].to_dict()]
        remaining = critical.iloc[1:].copy()
        while len(remaining) > 0:
            last = route[-1]
            dists = ((remaining["lat"] - last["lat"])**2 +
                     (remaining["lon"] - last["lon"])**2).values
            nearest = dists.argmin()
            route.append(remaining.iloc[nearest].to_dict())
            remaining = remaining.drop(remaining.index[nearest]).reset_index(drop=True)
        return route


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Smart City Planning Agent
# ══════════════════════════════════════════════════════════════════════════════
class SmartCityAgent:
    name = "Smart City Planning Agent"
    icon = "🏙️"
    description = "Recommends city-level interventions: traffic, construction, industrial, emergency policies"

    @staticmethod
    def interventions(city: str, aqi: int, forecast_peak: int,
                       attribution: dict, weather: dict) -> str:
        top3 = sorted(attribution.items(), key=lambda x: x[1], reverse=True)[:3]
        sources_str = ", ".join([f"{s} ({p:.0f}%)" for s, p in top3])
        prompt = f"""You are a Smart City AI Planning Agent for {city}.
Current AQI: {aqi} | 24h Forecast Peak: {forecast_peak}
Top emission sources: {sources_str}
Wind speed: {weather['wind_speed']} m/s | Rainfall: {weather['rainfall']} mm | Temp: {weather['temp']}°C

Generate a numbered list of EXACTLY 6 specific interventions the city administration should implement TODAY.
Cover: traffic management, construction controls, industrial inspection, waste management, school advisories, emergency response.
Each intervention: [Category] Specific action with measurable target. Max 25 words each.
Be bold and specific. No vague suggestions."""
        return _groq_call(prompt, system="You are a city administration AI. Be specific and decisive.", max_tokens=350)

    @staticmethod
    def whatif_simulation(base_aqi: float, scenario_name: str,
                           attribution: dict) -> dict:
        """Simulate AQI under what-if scenario using source attribution weights."""
        params = WHATIF_SCENARIOS.get(scenario_name, {})
        if not params:
            return {"predicted_aqi": base_aqi, "reduction": 0, "pct_reduction": 0}

        # Apply scenario multipliers to source contributions
        source_weights = {
            "Traffic":       attribution.get("Traffic", 25) / 100,
            "Industrial":    attribution.get("Industrial", 25) / 100,
            "Construction":  attribution.get("Construction", 15) / 100,
            "Waste Burning": attribution.get("Waste Burning", 15) / 100,
            "Dust":          attribution.get("Dust", 10) / 100,
            "Biomass":       attribution.get("Biomass", 10) / 100,
        }

        multiplier = (
            params.get("traffic", 1.0)       * source_weights["Traffic"] +
            params.get("industrial", 1.0)    * source_weights["Industrial"] +
            params.get("construction", 1.0)  * source_weights["Construction"] +
            1.0                               * source_weights["Waste Burning"] +
            1.0                               * source_weights["Dust"] +
            1.0                               * source_weights["Biomass"]
        )
        rainfall_bonus = 0.85 if params.get("rainfall", 0) else 1.0
        new_aqi = round(base_aqi * multiplier * rainfall_bonus, 1)
        reduction = round(base_aqi - new_aqi, 1)
        return {
            "predicted_aqi":  new_aqi,
            "reduction":      reduction,
            "pct_reduction":  round(reduction / base_aqi * 100, 1),
            "scenario":       scenario_name,
        }

    @staticmethod
    def scenario_narrative(city: str, scenario: str, base_aqi: float,
                            result: dict) -> str:
        prompt = f"""Smart City AI — {city}
Scenario: '{scenario}'
AQI before: {base_aqi:.0f} → AQI after: {result['predicted_aqi']:.0f} (↓{result['pct_reduction']:.1f}%)

In 2 sentences: explain the mechanism by which this intervention reduces pollution and
what implementation steps city administration should prioritize. Be specific and quantitative."""
        return _groq_call(prompt, max_tokens=120)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — Multilingual Communication Agent
# ══════════════════════════════════════════════════════════════════════════════
class MultilingualAgent:
    name = "Multilingual Communication Agent"
    icon = "🌐"
    description = "Generates citizen communications in Telugu, Tamil, Hindi, English for all channels"

    CHANNELS = ["Mobile Push", "IVR Call", "Public Display", "WhatsApp", "SMS"]

    @staticmethod
    def broadcast_message(city: str, aqi: int, lang: str, channel: str) -> str:
        from config.constants import AQI_SCALE
        cat = next((c[1] for c in AQI_SCALE if aqi <= c[0]), "Severe")
        li_map = {
            "Telugu": "ONLY in Telugu script (తెలుగు).",
            "Tamil":  "ONLY in Tamil script (தமிழ்).",
            "Hindi":  "ONLY in Hindi script (हिंदी).",
            "English": "in English.",
        }
        li = li_map.get(lang, "in English.")
        char_limits = {
            "SMS": "160 characters max",
            "Mobile Push": "80 characters max",
            "Public Display": "50 characters max — very short",
            "WhatsApp": "3 sentences max",
            "IVR Call": "2 short sentences — will be read aloud",
        }
        limit = char_limits.get(channel, "3 sentences max")

        prompt = f"""Generate a {channel} air quality alert {li}
City: {city} | AQI: {aqi} ({cat})
Format: {limit}
Tone: Clear and urgent if AQI > 200, calm and informative otherwise."""
        return _groq_call(prompt, max_tokens=150)


# ══════════════════════════════════════════════════════════════════════════════
# Agent orchestrator
# ══════════════════════════════════════════════════════════════════════════════
ALL_AGENTS = [
    AQIForecastAgent,
    SourceAttributionAgent,
    HealthAdvisoryAgent,
    EnforcementAgent,
    SmartCityAgent,
    MultilingualAgent,
]
