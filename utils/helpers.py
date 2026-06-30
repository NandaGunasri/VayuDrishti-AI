"""
utils/helpers.py
Production-grade defensive helpers for VayuDrishti AI
──────────────────────────────────────────────────────
Provides:
  • safe_col()          — read a column or return a default Series
  • safe_mean()         — mean of a column, NaN-safe
  • safe_val()          — scalar from a Series or row, NaN-safe
  • ensure_pollutants() — guarantee all required columns exist on a DataFrame
  • validate_df()       — full DataFrame health-check + auto-repair
  • safe_metric()       — Streamlit metric that never raises
  • safe_chart()        — Streamlit plotly_chart that never raises
  • status_badge()      — green/amber/red API status indicator
  • ui_error()          — user-friendly error card (no tracebacks)
  • mask_key()          — hide API keys from display
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ── Pollutant defaults (CPCB national average proxies) ───────────────────────
POLLUTANT_DEFAULTS: dict[str, float] = {
    "AQI":   100.0,
    "PM2.5":  60.0,
    "PM10":  100.0,
    "NO2":    40.0,
    "SO2":    20.0,
    "CO":      1.0,
    "O3":     50.0,
    "NH3":    10.0,
    "Pb":      0.5,
}

# Columns that must be numeric
NUMERIC_COLS = set(POLLUTANT_DEFAULTS.keys())

# Minimum columns required for the dashboard to function
REQUIRED_COLS = {"station", "zone", "lat", "lon", "AQI", "PM2.5", "PM10"}


# ── Column-level helpers ──────────────────────────────────────────────────────

def safe_col(df: pd.DataFrame, col: str, default: float | None = None) -> pd.Series:
    """
    Return df[col] if it exists and is not all-NaN.
    Otherwise return a Series filled with the default value
    (looks up POLLUTANT_DEFAULTS if default is None).
    Never raises KeyError.
    """
    if col in df.columns:
        series = df[col]
        if series.notna().any():
            return series.fillna(_default_for(col, default))
    fill = _default_for(col, default)
    return pd.Series([fill] * len(df), index=df.index, name=col)


def safe_mean(df: pd.DataFrame, col: str, default: float | None = None) -> float:
    """Return the mean of a column. Falls back to default on any failure."""
    try:
        s = safe_col(df, col, default)
        val = float(s.mean())
        return val if np.isfinite(val) else _default_for(col, default)
    except Exception:
        return _default_for(col, default)


def safe_val(
    row: pd.Series | dict,
    col: str,
    default: float | None = None,
) -> float:
    """Return row[col] as float, never raises."""
    try:
        v = row[col] if isinstance(row, dict) else row.get(col, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return _default_for(col, default)
        return float(v)
    except Exception:
        return _default_for(col, default)


def _default_for(col: str, override: float | None) -> float:
    if override is not None:
        return override
    return POLLUTANT_DEFAULTS.get(col, 0.0)


# ── DataFrame validation & repair ─────────────────────────────────────────────

def ensure_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that every pollutant column exists on df.
    Missing columns are added with realistic defaults.
    Existing columns have NaNs filled.
    Returns the (possibly modified) DataFrame — does NOT mutate in-place.
    """
    df = df.copy()
    for col, default in POLLUTANT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            logger.info("ensure_pollutants: added missing column '%s' with default %s", col, default)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def validate_df(df: pd.DataFrame, context: str = "dataframe") -> pd.DataFrame:
    """
    Full health-check + auto-repair:
      1. Empty DataFrame → raise ValueError (caller should fall back to synthetic)
      2. Missing required columns → filled with defaults
      3. Duplicate rows → dropped
      4. Numeric columns coerced and NaN-filled
      5. Returns cleaned copy
    """
    if df is None or df.empty:
        raise ValueError(f"{context} is empty — triggering fallback")

    df = df.copy()

    # Drop full-duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info("validate_df: dropped %d duplicate rows", before - len(df))

    # Ensure required non-pollutant columns exist with sensible placeholders
    for col in REQUIRED_COLS:
        if col not in df.columns:
            if col in POLLUTANT_DEFAULTS:
                df[col] = POLLUTANT_DEFAULTS[col]
            else:
                df[col] = "Unknown" if col in {"station", "zone"} else 0.0

    # Coerce and fill all numeric pollutants
    df = ensure_pollutants(df)

    # Extract time components if timestamp is present
    if "timestamp" in df.columns:
        from datetime import datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "hour" not in df.columns:
            df["hour"] = df["timestamp"].dt.hour.fillna(datetime.now().hour).astype(int)
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)

    return df


# ── Streamlit UI helpers ──────────────────────────────────────────────────────

def safe_metric(
    col: Any,
    label: str,
    value: Any,
    sub: str = "",
    color: str = "inherit",
) -> None:
    """
    Render a KPI metric card. Never raises.
    Falls back to a plain st.metric on any rendering error.
    """
    try:
        col.markdown(
            f"""<div class="kpi-card">
  <div class="kpi-lbl">{label}</div>
  <div class="kpi-val" style="color:{color}">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>""",
            unsafe_allow_html=True,
        )
    except Exception:
        try:
            col.metric(label, str(value), sub)
        except Exception:
            pass


def safe_chart(fig: Any, container: Any = None, **kwargs: Any) -> None:
    """
    Render a Plotly figure. Shows a friendly placeholder on any error.
    """
    target = container if container is not None else st
    try:
        if fig is None:
            raise ValueError("Figure is None")
        target.plotly_chart(fig, use_container_width=True, **kwargs)
    except Exception as exc:
        logger.warning("safe_chart: chart could not render — %s", exc)
        target.info("📊 Chart temporarily unavailable. Data is still being processed.")


def ui_error(message: str, detail: str = "", kind: str = "warning") -> None:
    """
    Show a friendly Streamlit error card.
    kind: 'info' | 'warning' | 'error' | 'success'
    Never shows tracebacks or internal paths.
    """
    icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}
    icon = icons.get(kind, "⚠️")
    text = f"{icon} **{message}**"
    if detail:
        text += f"\n\n_{detail}_"
    getattr(st, kind, st.warning)(text)


def status_badge(label: str, connected: bool, detail: str = "") -> None:
    """
    Render a green (connected) or amber (unavailable) API status chip.
    Never reveals key values.
    """
    if connected:
        badge_html = (
            f'<span style="background:#dcfce7;color:#166534;padding:3px 10px;'
            f'border-radius:20px;font-size:12px;font-weight:500">'
            f'✅ {label} — Connected</span>'
        )
    else:
        badge_html = (
            f'<span style="background:#fef9c3;color:#92400e;padding:3px 10px;'
            f'border-radius:20px;font-size:12px;font-weight:500">'
            f'⚡ {label} — Simulated</span>'
        )
    if detail:
        badge_html += f' <span style="font-size:11px;color:#94a3b8">{detail}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)


def mask_key(key: str) -> str:
    """
    Return a display-safe representation of an API key.
    Never exposes more than the first 4 chars.
    """
    if not key:
        return ""
    visible = key[:4]
    return f"{visible}{'•' * 8}"


def api_key_status(env_var: str, label: str) -> bool:
    """
    Show API key status badge; return True if key is present.
    Reads from session_state first (sidebar input), then env.
    """
    import os
    key = st.session_state.get(env_var, "") or os.getenv(env_var, "")
    connected = bool(key)
    status_badge(label, connected)
    return connected


# ── Pollutant display helpers ─────────────────────────────────────────────────

def pollutant_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a display-safe pollutant summary table from any DataFrame.
    Always returns a DataFrame with all standard pollutants as rows.
    """
    rows = []
    for col, default in POLLUTANT_DEFAULTS.items():
        val = safe_mean(df, col, default)
        rows.append({
            "Pollutant": col,
            "Current":   round(val, 1),
        })
    return pd.DataFrame(rows)


def get_pollutant_values(df: pd.DataFrame) -> dict[str, float]:
    """Return {pollutant: mean_value} dict, always complete, never raises."""
    return {col: safe_mean(df, col) for col in POLLUTANT_DEFAULTS}
