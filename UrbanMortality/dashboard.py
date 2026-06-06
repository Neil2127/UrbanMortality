#!/usr/bin/env python3
"""
UrbanMortality — Streamlit interactive dashboard
Run:  streamlit run dashboard.py
"""
import os
import subprocess
import sys
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd

st.set_page_config(
    page_title="UrbanMortality · Chicago",
    layout="wide",
    page_icon="🦅",
)

RISK_PAL = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
INT_PAL  = {
    "Anti-collision window film":                 "#c0392b",
    "Reduced night lighting":                     "#9b59b6",
    "Vegetation buffer":                          "#d35400",
    "Wildlife crossing corridor":                 "#8e44ad",
}

# ── Data (auto-run pipeline if cache missing) ────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXECUTABLE = sys.executable

if not os.path.exists(os.path.join(HERE, "cache", "grid.geojson")):
    with st.spinner("Running data pipeline for the first time (~2 min)…"):
        subprocess.run([PYTHON_EXECUTABLE, "main.py"], check=True, cwd=HERE)

@st.cache_resource
def load_data():
    grid_path = os.path.join(HERE, "cache", "grid.geojson")
    mort_path = os.path.join(HERE, "cache", "incidents.geojson")

    grid = gpd.read_file(grid_path)
    required_columns = {
        "best_intervention", "expected_reduction", "intervention_risk",
        "baseline_risk", "intervention_rankings",
    }

    if not required_columns.issubset(grid.columns):
        with st.spinner("Detected stale cached data. Rebuilding pipeline…"):
            subprocess.run([PYTHON_EXECUTABLE, "main.py"], check=True, cwd=HERE)
        grid = gpd.read_file(grid_path)

    mort = gpd.read_file(mort_path)
    grid["glass_frac"] = grid["glass_frac"].astype(float).round(3)
    grid["risk_p"]     = grid["risk_p"].astype(float).round(3)
    return grid, mort

grid, mort = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦅 UrbanMortality")
    st.caption("Chicago, IL · Mississippi Flyway")

    layer = st.radio("Colour layer", ["Mortality Risk", "Intervention"])
    risk_filter = st.multiselect(
        "Show risk levels", ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    st.divider()
    st.markdown("**Risk breakdown**")
    for r, c in RISK_PAL.items():
        n = (grid["risk"] == r).sum()
        st.markdown(
            f"<span style='color:{c};font-size:18px'>■</span> **{r}**: {n} cells",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Intervention types**")
    for iv, c in INT_PAL.items():
        n = (grid["best_intervention"] == iv).sum()
        if n:
            st.markdown(
                f"<span style='color:{c};font-size:13px'>■</span> {iv}: **{n}**",
                unsafe_allow_html=True,
            )

    st.divider()
    st.metric("Bird incidents",   int((mort["kind"] == "bird").sum()))
    st.metric("Mammal incidents", int((mort["kind"] == "mammal").sum()))

    if st.button("🔄 Refresh data"):
        subprocess.run([PYTHON_EXECUTABLE, "main.py"], check=True, cwd=HERE)
        st.cache_resource.clear()
        st.rerun()

# ── Build Folium map ─────────────────────────────────────────────────────────────
grid_f = grid[grid["risk"].isin(risk_filter)]

m = folium.Map(
    location=[41.875, -87.640], zoom_start=13,
    tiles="CartoDB positron", control_scale=True,
)

def style_risk(feat):
    r = feat["properties"].get("risk", "Low")
    return {"fillColor": RISK_PAL.get(r, "#aaa"),
            "fillOpacity": 0.65, "color": "white", "weight": 0.4}

def style_int(feat):
    iv = feat["properties"].get("best_intervention")
    if iv is None:
        iv = feat["properties"].get("intervention", "No action")
    return {"fillColor": INT_PAL.get(iv, "#aaa"),
            "fillOpacity": 0.65, "color": "white", "weight": 0.4}

folium.GeoJson(
    grid_f.__geo_interface__,
    style_function=style_risk if layer == "Mortality Risk" else style_int,
    tooltip=folium.GeoJsonTooltip(
        fields=["risk", "risk_p", "best_intervention", "expected_reduction",
                "bldg_n", "glass_frac", "road_segs", "mort_n"],
        aliases=["Risk", "Probability", "Best intervention", "Expected reduction (%)",
                 "Buildings", "Glass fraction", "Road segs", "Incidents"],
        localize=True,
    ),
    name="Grid cells",
).add_to(m)

# Mortality markers
bird_grp   = folium.FeatureGroup("🐦 Bird incidents",   show=True)
mammal_grp = folium.FeatureGroup("🦊 Mammal incidents", show=True)

for _, row in mort.iterrows():
    pt = [row.geometry.y, row.geometry.x]
    if row["kind"] == "bird":
        folium.CircleMarker(
            pt, radius=3, color="#1a237e", fill=True,
            fill_color="#3949ab", fill_opacity=0.8, weight=1,
            tooltip="Bird mortality",
        ).add_to(bird_grp)
    else:
        folium.CircleMarker(
            pt, radius=4, color="#4a0072", fill=True,
            fill_color="#9c27b0", fill_opacity=0.8, weight=1,
            tooltip="Mammal mortality",
        ).add_to(mammal_grp)

bird_grp.add_to(m)
mammal_grp.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# ── Page layout ──────────────────────────────────────────────────────────────────
st.title("UrbanMortality — Chicago, IL")
st.caption(
    "Wildlife mortality risk · Bird–glass building collisions & urban road mortality · "
    "Mississippi Flyway · Data: iNaturalist + OpenStreetMap"
)

map_col, tbl_col = st.columns([3, 1])

with map_col:
    st_folium(m, width=900, height=620, returned_objects=[])

with tbl_col:
    st.markdown("#### Top high-risk blocks")
    top = (
        grid[grid["risk"] == "High"]
        .sort_values("risk_p", ascending=False)
        .head(15)[[
            "cy", "cx", "risk_p", "baseline_risk", "intervention_risk",
            "expected_reduction", "best_intervention", "glass_frac", "mort_n",
        ]]
        .rename(columns={
            "cy": "lat", "cx": "lon",
            "risk_p": "p(risk)", "baseline_risk": "baseline risk",
            "intervention_risk": "intervention risk",
            "expected_reduction": "expected reduction",
            "best_intervention": "best intervention",
            "glass_frac": "glass%", "mort_n": "incidents",
        })
        .reset_index(drop=True)
    )
    top["glass%"] = (top["glass%"] * 100).round(1).astype(str) + "%"
    top["expected reduction"] = top["expected reduction"].round(1).astype(str) + "%"
    top["baseline risk"] = top["baseline risk"].round(3)
    top["intervention risk"] = top["intervention risk"].round(3)
    st.dataframe(top, use_container_width=True, height=570)
