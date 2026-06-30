"""
utils/map_builder.py
Folium map construction utilities — VayuDrishti AI
"""
import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
from config.constants import AQI_SCALE, POLLUTION_SOURCES


def aqi_meta(aqi: int) -> tuple:
    for threshold, cat, color, emoji, _ in AQI_SCALE:
        if aqi <= threshold:
            return cat, color, emoji
    return "Severe", "#7c3aed", "☠️"


# ── AQI marker map ─────────────────────────────────────────────────────────────
def build_station_map(ward_df: pd.DataFrame, city_lat: float, city_lon: float,
                      show_attribution: bool = True) -> folium.Map:
    m = folium.Map(
        location=[city_lat, city_lon],
        zoom_start=12,
        tiles="CartoDB positron"
    )

    for _, row in ward_df.iterrows():
        lat = row.get("lat")
        lon = row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        aqi  = int(row["AQI"])
        cat, color, emoji = aqi_meta(aqi)
        radius = max(14, min(32, aqi / 10))

        # Build attribution HTML
        attr_html = ""
        if show_attribution:
            src_cols = [c for c in row.index if c.startswith("src_")]
            if src_cols:
                top_srcs = sorted(
                    [(c.replace("src_", ""), row[c]) for c in src_cols],
                    key=lambda x: x[1], reverse=True
                )[:3]
                attr_rows = "".join([
                    f'<tr><td>{POLLUTION_SOURCES.get(s,{}).get("icon","●")} {s}</td>'
                    f'<td style="text-align:right"><b>{p:.0f}%</b></td></tr>'
                    for s, p in top_srcs
                ])
                attr_html = f"""
                <hr style="margin:6px 0">
                <b style="font-size:11px">Source Attribution</b>
                <table style="width:100%;font-size:11px;margin-top:4px">{attr_rows}</table>"""

        popup_html = f"""
        <div style="font-family:'Segoe UI',sans-serif;min-width:200px;padding:4px">
            <div style="font-size:15px;font-weight:700;color:#1e293b">{row['station']}</div>
            <div style="color:#64748b;font-size:11px;margin:2px 0">{row.get('zone','—')} Zone</div>
            <div style="font-size:28px;font-weight:700;color:{color};margin:6px 0">
                {emoji} {aqi} <span style="font-size:13px">AQI — {cat}</span>
            </div>
            <table style="width:100%;font-size:12px;border-collapse:collapse">
                <tr><td>PM2.5</td><td style="text-align:right"><b>{row.get('PM2.5','—'):.0f}</b> μg/m³</td></tr>
                <tr><td>PM10</td> <td style="text-align:right"><b>{row.get('PM10','—'):.0f}</b> μg/m³</td></tr>
                <tr><td>NO₂</td>  <td style="text-align:right"><b>{row.get('NO2','—'):.0f}</b> μg/m³</td></tr>
                <tr><td>SO₂</td>  <td style="text-align:right"><b>{row.get('SO2','—'):.0f}</b> μg/m³</td></tr>
            </table>
            {attr_html}
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.80,
            weight=2,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['station']} — AQI {aqi} {emoji}"
        ).add_to(m)

        # Risk score ring (outer)
        rs = int(row.get("risk_score", 0))
        if rs > 70:
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius + 6,
                color=color,
                fill=False,
                weight=1.5,
                opacity=0.4,
                tooltip="High risk zone"
            ).add_to(m)

    _add_legend(m)
    return m


# ── Heatmap layer ─────────────────────────────────────────────────────────────
def build_heatmap(ward_df: pd.DataFrame, city_lat: float, city_lon: float,
                  metric: str = "AQI") -> folium.Map:
    m = folium.Map(location=[city_lat, city_lon], zoom_start=12, tiles="CartoDB dark_matter")

    heat_data = [[row["lat"], row["lon"], float(row[metric])]
                 for _, row in ward_df.iterrows()
                 if not pd.isna(row[metric]) and not pd.isna(row.get("lat")) and not pd.isna(row.get("lon"))]

    if heat_data:
        HeatMap(
            heat_data,
            radius=35, blur=20, max_zoom=14,
            gradient={0.0: "blue", 0.4: "lime", 0.6: "yellow",
                      0.8: "orange", 1.0: "red"}
        ).add_to(m)

    return m


# ── Source attribution map ─────────────────────────────────────────────────────
def build_source_map(ward_df: pd.DataFrame, city_lat: float, city_lon: float) -> folium.Map:
    m = folium.Map(location=[city_lat, city_lon], zoom_start=12, tiles="CartoDB positron")

    for _, row in ward_df.iterrows():
        lat = row.get("lat")
        lon = row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        dom = row.get("dominant_source", "Traffic")
        info = POLLUTION_SOURCES.get(dom, {"color": "#6b7280", "icon": "●"})
        aqi  = int(row["AQI"])

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(12, min(28, aqi / 11)),
            color=info["color"],
            fill=True,
            fill_color=info["color"],
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{row['station']}</b><br>"
                f"Dominant: {info['icon']} {dom}<br>"
                f"AQI: {aqi}", max_width=160),
            tooltip=f"{info['icon']} {dom} — {row['station']}"
        ).add_to(m)

    # Source legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:12px 16px;border-radius:10px;border:1px solid #ccc;
                font-family:sans-serif;font-size:12px;min-width:160px">
        <b style="font-size:13px">Dominant Source</b><br><br>
    """ + "".join([
        f'<div style="margin:3px 0"><span style="color:{v["color"]};font-size:16px">●</span> '
        f'{v["icon"]} {k}</div>'
        for k, v in POLLUTION_SOURCES.items()
    ]) + "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── Inspection routing map ─────────────────────────────────────────────────────
def build_inspection_route_map(route: list, city_lat: float, city_lon: float) -> folium.Map:
    m = folium.Map(location=[city_lat, city_lon], zoom_start=12, tiles="CartoDB positron")
    if not route:
        return m

    coords = [[r["lat"], r["lon"]] for r in route if not pd.isna(r.get("lat")) and not pd.isna(r.get("lon"))]
    if len(coords) >= 2:
        folium.PolyLine(coords, color="#7c3aed", weight=3, opacity=0.8,
                        tooltip="Optimal inspection route").add_to(m)

    for i, r in enumerate(route):
        lat = r.get("lat")
        lon = r.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        _, color, emoji = aqi_meta(int(r["AQI"]))
        folium.Marker(
            location=[lat, lon],
            popup=f"Stop {i+1}: {r['station']}<br>AQI: {r['AQI']:.0f}",
            icon=folium.DivIcon(
                html=f'<div style="background:{color};color:white;border-radius:50%;'
                     f'width:26px;height:26px;display:flex;align-items:center;'
                     f'justify-content:center;font-weight:bold;font-size:12px;'
                     f'border:2px solid white">{i+1}</div>',
                icon_size=(26, 26), icon_anchor=(13, 13)
            )
        ).add_to(m)
    return m


# ── Legend helper ──────────────────────────────────────────────────────────────
def _add_legend(m: folium.Map):
    legend_html = """
    <div style="position:fixed;bottom:30px;right:30px;z-index:1000;background:rgba(255,255,255,0.95);
                padding:12px 16px;border-radius:10px;border:1px solid #ddd;
                font-family:sans-serif;font-size:12px">
        <b style="font-size:13px">CPCB AQI Scale</b><br><br>
        <span style="color:#16a34a">●</span> 0–50 Good<br>
        <span style="color:#65a30d">●</span> 51–100 Satisfactory<br>
        <span style="color:#ca8a04">●</span> 101–200 Moderate<br>
        <span style="color:#ea580c">●</span> 201–300 Poor<br>
        <span style="color:#dc2626">●</span> 301–400 Very Poor<br>
        <span style="color:#7c3aed">●</span> 400+ Severe<br><br>
        <i style="color:#94a3b8;font-size:10px">Ring = High risk zone</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
