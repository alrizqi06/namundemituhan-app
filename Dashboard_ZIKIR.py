"""
Wind Power Plant (PLTB) Potential Assessment Dashboard
=======================================================
A Streamlit application for analyzing offshore wind data from NetCDF files
to evaluate the viability of a wind power plant at a specific location.

Accepts any number of .nc files (any NetCDF format version, from ERA5,
CMEMS, WRF, NOAA/NCEP, or other CF-style sources), auto-detecting whether
each uploaded file contains the U (eastward) wind component, the V
(northward) component, or both, from its metadata.

Tech Stack: streamlit, xarray, numpy, folium, streamlit-folium, plotly
"""

import io
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Energy Assessment",
    page_icon=":material/air:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Offshore / Industrial palette
# Signature element: glowing teal accent on dark navy
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: #d1dce8;
    }

    .stApp {
        background:
            radial-gradient(ellipse 1200px 600px at 15% -10%, rgba(0, 212, 255, 0.07), transparent 60%),
            radial-gradient(ellipse 900px 500px at 100% 10%, rgba(0, 196, 122, 0.05), transparent 55%),
            #0a0f1c;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1420px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1a2d 0%, #0a1422 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #a8bfd4 !important;
    }
    [data-testid="stSidebar"] h3 {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 1rem !important;
        letter-spacing: 0.02em;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #16293f;
        margin-bottom: 0.8rem !important;
    }

    /* ── Page title ── */
    .dashboard-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.6rem;
        padding-bottom: 0.9rem;
    }
    .dashboard-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        letter-spacing: -0.02em;
        color: #e8f4fd;
        line-height: 1.15;
    }
    .dashboard-subtitle {
        font-size: 0.82rem;
        color: #4f7c9c;
        font-weight: 500;
        margin-top: 0.3rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .title-accent {
        background: linear-gradient(120deg, #00d4ff 0%, #00ffc8 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        filter: drop-shadow(0 0 14px rgba(0, 212, 255, 0.35));
    }
    .title-icon-badge {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(0,196,122,0.10));
        border: 1px solid #1e3a5f;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        flex-shrink: 0;
    }
    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(0, 196, 122, 0.10);
        border: 1px solid #1e4a30;
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.74rem;
        color: #7ad4a8;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 600;
    }
    .live-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #00c47a;
        box-shadow: 0 0 0 0 rgba(0,196,122,0.6);
        animation: pulse-dot 1.8s infinite;
    }
    @keyframes pulse-dot {
        0%   { box-shadow: 0 0 0 0 rgba(0,196,122,0.55); }
        70%  { box-shadow: 0 0 0 7px rgba(0,196,122,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,196,122,0); }
    }

    /* ── KPI Cards ── */
    .kpi-card {
        position: relative;
        background: linear-gradient(150deg, #122544 0%, #0c1a2e 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.05rem 1.3rem;
        text-align: center;
        overflow: hidden;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #00ffc8);
        opacity: 0.85;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #2a5a8a;
        box-shadow: 0 10px 28px -10px rgba(0, 212, 255, 0.22);
    }
    .kpi-card.accent-amber::before { background: linear-gradient(90deg, #f0a500, #ffcf4d); }
    .kpi-card.accent-green::before { background: linear-gradient(90deg, #00c47a, #5dffc0); }
    .kpi-card.accent-mono::before  { background: linear-gradient(90deg, #3a5a7a, #5a8aaa); }
    .kpi-icon {
        font-size: 1.1rem;
        opacity: 0.85;
        margin-bottom: 0.15rem;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.1rem;
        font-weight: 600;
        color: #00d4ff;
        text-shadow: 0 0 14px rgba(0, 212, 255, 0.28);
        line-height: 1;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #5a8aaa;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    .kpi-unit {
        font-size: 0.88rem;
        color: #7baac8;
        margin-left: 3px;
    }

    /* ── Section headers ── */
    .section-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #e8f4fd;
        font-weight: 600;
        margin-bottom: 0.6rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #16293f;
    }
    .section-label::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 2px;
        background: linear-gradient(135deg, #00d4ff, #00ffc8);
        flex-shrink: 0;
        box-shadow: 0 0 8px rgba(0, 212, 255, 0.5);
    }
    .section-sub {
        font-size: 0.82rem;
        color: #6a93b0;
        margin: -0.3rem 0 0.9rem 0;
        line-height: 1.5;
    }

    /* ── Info / status banners ── */
    .info-box {
        background: #0c2340;
        border: 1px solid #1e4a7a;
        border-left: 4px solid #00d4ff;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #a8bfd4;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: #1f1800;
        border: 1px solid #4a3a00;
        border-left: 4px solid #f0a500;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #d4b87a;
        margin-bottom: 1rem;
    }
    .success-box {
        background: #0c2318;
        border: 1px solid #1e4a30;
        border-left: 4px solid #00c47a;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #7ad4a8;
        margin-bottom: 1rem;
    }

    /* ── Status tags (text-based, replaces colour-dot emoji) ── */
    .status-tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.18rem 0.55rem;
        border-radius: 5px;
        margin-right: 0.4rem;
        vertical-align: middle;
    }
    .status-tag--good { background: rgba(0,196,122,0.16); color: #5dffc0; border: 1px solid #1e4a30; }
    .status-tag--mid  { background: rgba(240,165,0,0.16); color: #ffcf4d; border: 1px solid #4a3a00; }
    .status-tag--low  { background: rgba(58,90,122,0.22); color: #9bc4de; border: 1px solid #1e3a5f; }

    /* ── Coordinate badge ── */
    .coord-badge {
        display: inline-block;
        background: #0c2340;
        border: 1px solid #00d4ff;
        border-radius: 4px;
        padding: 0.3rem 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #00d4ff;
    }

    /* ── Plotly chart container tweak ── */
    .stPlotlyChart {
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Buttons (e.g. leaderboard coordinate links) ── */
    div[data-testid="stButton"] button {
        background: #0c2340;
        border: 1px solid #00d4ff;
        border-radius: 4px;
        color: #00d4ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        padding: 0.3rem 0.6rem;
        transition: background 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        background: #123257;
        border-color: #00d4ff;
        color: #80eaff;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.35);
    }
    div[data-testid="stButton"] button:focus:not(:active) {
        border-color: #00d4ff;
        color: #00d4ff;
    }

    /* ── Divider ── */
    hr {
        border-color: #16293f;
        margin: 1.1rem 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #0c1827;
        padding: 5px;
        border-radius: 12px;
        border: 1px solid #16293f;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background: transparent;
        border-radius: 8px;
        color: #6a93b0;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 1.1rem;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.06);
        color: #a8bfd4;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.16), rgba(0,196,122,0.08)) !important;
        color: #00d4ff !important;
        box-shadow: inset 0 0 0 1px rgba(0,212,255,0.35);
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #0c1827;
        border: 1px solid #16293f;
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.88rem;
        color: #a8bfd4;
    }

    /* ── Metrics / misc widgets ── */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* ── Fade-in for content blocks ── */
    .block-container > div { animation: fadeUp 0.35s ease-out; }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0a0f1c; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: #2a5a8a; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Plotly shared theme
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0b1827",
    font=dict(family="Space Grotesk, sans-serif", color="#a8bfd4", size=12),
    xaxis=dict(
        gridcolor="#1a2d45",
        zerolinecolor="#1a2d45",
        linecolor="#1e3a5f",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="#1a2d45",
        zerolinecolor="#1a2d45",
        linecolor="#1e3a5f",
        tickfont=dict(size=11),
    ),
    margin=dict(l=50, r=20, t=55, b=50),
)
TEAL = "#00d4ff"
TEAL_DARK = "#007fa6"
ACCENT2 = "#f0a500"


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _find_var(ds: xr.Dataset, candidates, attrs_keywords=None):
    """
    Case-insensitive lookup of the first matching variable name in ds.

    Priority order:
    1. Exact (case-insensitive) short-name match against `candidates`.
    2. CF-attrs fallback: scans standard_name / long_name for any keyword
       in `attrs_keywords` (a separate, tighter set than `candidates`).
       Using a dedicated keyword set prevents short candidates like "v"
       from spuriously matching "v_component_of_wind" in a U-component
       file, which would cause that file to be classified as "both" when
       it actually only contains one component.
    """
    lower_vars = {var.lower(): var for var in ds.data_vars}
    match = next((lower_vars[k] for k in candidates if k in lower_vars), None)
    if match is not None:
        return match

    # Attrs fallback — only fires when exclusive keyword set is provided
    if attrs_keywords:
        for var in ds.data_vars:
            attrs_text = " ".join(
                str(ds[var].attrs.get(k, "")) for k in ("standard_name", "long_name")
            ).lower()
            if attrs_text and any(kw in attrs_text for kw in attrs_keywords):
                return var
    return None


# Variable-name candidates pooled from multiple data sources so the
# dashboard isn't tied to ERA5 alone — covers ERA5/CDS, CMEMS, WRF,
# NOAA/NCEP reanalysis, and generic CF-style exports.
U_VAR_CANDIDATES = [
    "u10", "u100", "u_component_of_wind", "uwnd", "ugrd",
    "eastward_wind", "ua", "wind_u", "u_wind", "xwind", "u10m",
]
V_VAR_CANDIDATES = [
    "v10", "v100", "v_component_of_wind", "vwnd", "vgrd",
    "northward_wind", "va", "wind_v", "v_wind", "ywind", "v10m",
]

# Exclusive attrs keywords for the fallback matcher — prevents cross-contamination
_U_ATTRS_KEYWORDS = {"eastward", "u_component", "uwnd", "u-wind", "u wind", "zonal"}
_V_ATTRS_KEYWORDS = {"northward", "v_component", "vwnd", "v-wind", "v wind", "meridional"}


def _find_time_dim(da: xr.DataArray):
    """
    Detect the name of the time coordinate/dimension on a DataArray.
    ERA5/CDS exports increasingly use 'valid_time' instead of 'time',
    so this checks common aliases (case-insensitive) rather than
    assuming the dimension is literally called 'time'.
    """
    candidates = ["time", "valid_time", "forecast_time", "datetime", "date"]
    lower_coords = {c.lower(): c for c in da.coords}
    found = next((lower_coords[k] for k in candidates if k in lower_coords), None)
    if found is not None:
        return found
    # Fallback: any coordinate with a datetime64 dtype
    for c in da.coords:
        if np.issubdtype(da[c].dtype, np.datetime64):
            return c
    return None


# Backends tried in order. Together they cover the full spread of the
# .nc format — classic NETCDF3 / 64-bit offset (scipy), NETCDF4 /
# NETCDF4_CLASSIC (netcdf4), and HDF5-backed NetCDF4 (h5netcdf) — so a
# file isn't rejected just because it was written by an older tool.
NETCDF_ENGINES = ["netcdf4", "h5netcdf", "scipy"]


def open_netcdf_any_format(path: str) -> xr.Dataset:
    """
    Open a .nc file trying multiple backend engines in turn, so any
    NetCDF format version/flavour can be read — not just whichever
    single engine happens to be installed or compatible with that file.
    """
    errors = []
    for engine in NETCDF_ENGINES:
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as err:
            errors.append(f"{engine}: {err}")
            continue
    raise RuntimeError(
        "Could not open this file with any supported NetCDF backend "
        f"({', '.join(NETCDF_ENGINES)}). Details:\n" + "\n".join(errors)
    )


def classify_netcdf_dataset(ds: xr.Dataset):
    """
    Inspect a dataset's variables/metadata and determine whether it
    contains a U (eastward) wind component, a V (northward) component,
    both, or neither. Returns (kind, u_key, v_key) where kind is one
    of 'both', 'u', 'v', 'unknown'.
    """
    u_key = _find_var(ds, U_VAR_CANDIDATES, _U_ATTRS_KEYWORDS)
    v_key = _find_var(ds, V_VAR_CANDIDATES, _V_ATTRS_KEYWORDS)
    if u_key and v_key:
        kind = "both"
    elif u_key:
        kind = "u"
    elif v_key:
        kind = "v"
    else:
        kind = "unknown"
    return kind, u_key, v_key



def extract_file_metadata(filename: str, size_bytes: int, ds: xr.Dataset, kind: str, u_key, v_key):
    """
    Pull a human-readable metadata summary out of an opened dataset —
    detected role, variable(s), dimensions, time coverage, and spatial
    extent — so the user can see exactly what the dashboard found in
    each uploaded file before it gets combined into the analysis.
    """
    meta = {
        "filename": filename,
        "size_mb": size_bytes / (1024 ** 2),
        "kind": kind,
        "u_key": u_key,
        "v_key": v_key,
        "n_vars": len(ds.data_vars),
        "all_vars": list(ds.data_vars),
        "conventions": ds.attrs.get("Conventions", ds.attrs.get("conventions", "n/a")),
        "source": ds.attrs.get("source", ds.attrs.get("institution", "n/a")),
        "time_range": "n/a",
        "n_timesteps": 0,
        "lat_range": "n/a",
        "lon_range": "n/a",
    }

    t_name = _find_time_dim(ds)
    if t_name is not None:
        try:
            t_vals = ds[t_name].values
            meta["n_timesteps"] = len(t_vals)
            if len(t_vals) > 0:
                meta["time_range"] = f"{np.min(t_vals)} → {np.max(t_vals)}"
        except Exception:
            pass

    lat_key = next((k for k in ds.coords if k.lower() in ["lat", "latitude"]), None)
    lon_key = next((k for k in ds.coords if k.lower() in ["lon", "longitude"]), None)
    if lat_key is not None:
        try:
            meta["lat_range"] = f"{float(ds[lat_key].min()):.2f}° to {float(ds[lat_key].max()):.2f}°"
        except Exception:
            pass
    if lon_key is not None:
        try:
            meta["lon_range"] = f"{float(ds[lon_key].min()):.2f}° to {float(ds[lon_key].max()):.2f}°"
        except Exception:
            pass

    return meta


def build_uv_from_records(file_records):
    """
    Combine an arbitrary number of classified files into a single U
    DataArray and a single V DataArray.

    Supports any mix of:
    • one file containing both U and V,
    • separate U-only / V-only files,
    • many files per component split across time (e.g. one file per
      month/year), which are concatenated and de-duplicated on 'time'.

    This is what lets the dashboard accept large batches of input
    files rather than being limited to exactly one or two uploads.
    """
    u_arrays, v_arrays = [], []

    for rec in file_records:
        ds = rec["ds"]
        if rec["kind"] in ("u", "both") and rec["u_key"]:
            da = ds[rec["u_key"]]
            t = _find_time_dim(da)
            if t and t != "time":
                da = da.rename({t: "time"})
            u_arrays.append(da)
        if rec["kind"] in ("v", "both") and rec["v_key"]:
            da = ds[rec["v_key"]]
            t = _find_time_dim(da)
            if t and t != "time":
                da = da.rename({t: "time"})
            v_arrays.append(da)

    if not u_arrays:
        raise KeyError(
            "No U (eastward) wind component was detected in any uploaded file. "
            "Check that at least one file contains a recognisable U-wind variable."
        )
    if not v_arrays:
        raise KeyError(
            "No V (northward) wind component was detected in any uploaded file. "
            "Check that at least one file contains a recognisable V-wind variable."
        )

    u = u_arrays[0] if len(u_arrays) == 1 else xr.concat(u_arrays, dim="time")
    v = v_arrays[0] if len(v_arrays) == 1 else xr.concat(v_arrays, dim="time")

    # Multiple files may overlap in time (e.g. re-uploads or overlapping
    # batches) — sort chronologically and drop duplicate timestamps.
    if "time" in u.coords and u.sizes.get("time", 0) > 1:
        u = u.sortby("time").drop_duplicates("time", keep="first")
    if "time" in v.coords and v.sizes.get("time", 0) > 1:
        v = v.sortby("time").drop_duplicates("time", keep="first")

    u, v = xr.align(u, v, join="inner")
    if u.size == 0 or v.size == 0:
        raise ValueError(
            "The uploaded U and V data do not share overlapping coordinates "
            "(time/lat/lon). Please check that they cover the same domain and period."
        )
    return u, v


def compute_wind(u: xr.DataArray, v: xr.DataArray):
    """
    Compute wind speed (m/s) and meteorological wind direction (degrees
    from North, clockwise) from already-combined U/V DataArrays.

    File ingestion, U/V detection across any number of uploaded files,
    and time-dimension standardisation all happen upstream in
    `build_uv_from_records()` — this function just does the physics.

    Returns (u, v, ws, wd) as DataArrays sharing a 'time' dimension.
    """
    t_name = _find_time_dim(u)
    if t_name and t_name != "time":
        u = u.rename({t_name: "time"})
        v = v.rename({t_name: "time"}) if t_name in v.coords else v

    ws = np.sqrt(u**2 + v**2)
    ws.name = "wind_speed"

    # Meteorological convention: direction FROM which wind blows
    wd = (270 - np.degrees(np.arctan2(v, u))) % 360
    wd.name = "wind_direction"

    return u, v, ws, wd


def get_lat_lon_bounds(ds: xr.Dataset):
    """Extract lat/lon coordinate names and their min/max bounds."""
    lat_key = next((k for k in ds.coords if k.lower() in ["lat", "latitude"]), None)
    lon_key = next((k for k in ds.coords if k.lower() in ["lon", "longitude"]), None)

    if lat_key is None or lon_key is None:
        raise KeyError(
            f"Could not find lat/lon coordinates. Available coords: {list(ds.coords)}"
        )

    lat_vals = ds[lat_key].values
    lon_vals = ds[lon_key].values

    return (
        lat_key, lon_key,
        float(lat_vals.min()), float(lat_vals.max()),
        float(lon_vals.min()), float(lon_vals.max()),
    )


def is_within_bounds(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Return True if (lat, lon) falls inside the dataset's spatial extent."""
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


# ─────────────────────────────────────────────
# Time / season filtering helpers
# ─────────────────────────────────────────────

# Indonesian monsoon season definitions (by month number)
SEASON_MONTHS = {
    "Musim Barat (Des–Feb)": [12, 1, 2],
    "Peralihan 1 (Mar–Mei)": [3, 4, 5],
    "Musim Timur (Jun–Agu)": [6, 7, 8],
    "Peralihan 2 (Sep–Nov)": [9, 10, 11],
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def filter_by_period(da: xr.DataArray, period_mode: str, selected_month: int = None,
                     selected_season: str = None, selected_day: int = None):
    """
    Filter a time-indexed DataArray according to the chosen period mode.

    period_mode: "All Data" | "Monthly" | "Seasonal" | "Daily"
    selected_month: 1-12, used when period_mode == "Monthly" or "Daily"
    selected_season: key into SEASON_MONTHS, used when period_mode == "Seasonal"
    selected_day: 1-31, used when period_mode == "Daily" (combined with selected_month)

    Returns the filtered DataArray (same dims, fewer 'time' entries).
    """
    if period_mode == "Monthly" and selected_month is not None:
        months = da["time"].dt.month
        return da.sel(time=(months == selected_month))
    elif period_mode == "Seasonal" and selected_season is not None:
        months = da["time"].dt.month
        target_months = SEASON_MONTHS[selected_season]
        mask = months.isin(target_months)
        return da.sel(time=mask)
    elif period_mode == "Daily" and selected_month is not None and selected_day is not None:
        months = da["time"].dt.month
        days = da["time"].dt.day
        mask = (months == selected_month) & (days == selected_day)
        return da.sel(time=mask)
    return da


def get_available_years(da: xr.DataArray):
    """Return a sorted list of distinct years present in a DataArray's 'time' coord."""
    return sorted(set(int(y) for y in da["time"].dt.year.values))


def filter_by_year(da: xr.DataArray, selected_year):
    """
    Filter a time-indexed DataArray down to a single year.
    selected_year: int, or None/"All Years" to pass the data through unchanged.
    """
    if selected_year is None or selected_year == "All Years":
        return da
    years = da["time"].dt.year
    return da.sel(time=(years == int(selected_year)))


# ─────────────────────────────────────────────
# Folium map builder
# ─────────────────────────────────────────────

def build_folium_map(lat_min, lat_max, lon_min, lon_max, poi=None):
    """
    Build a Folium map centred on the data domain with:
    • A dark CartoDB basemap
    • A dashed cyan rectangle showing the data boundary
    • An optional POI marker
    """
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        width="100%",
        height=480,
    )

    # Data-boundary rectangle
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="#00d4ff",
        weight=2,
        dash_array="8 4",
        fill=True,
        fill_color="#00d4ff",
        fill_opacity=0.06,
        tooltip=folium.Tooltip(
            f"<b>Data Domain</b><br>"
            f"Lat: {lat_min:.2f}° – {lat_max:.2f}°<br>"
            f"Lon: {lon_min:.2f}° – {lon_max:.2f}°"
        ),
    ).add_to(m)

    # Corner markers for visual clarity
    for lat, lon in [(lat_min, lon_min), (lat_min, lon_max),
                     (lat_max, lon_min), (lat_max, lon_max)]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#00d4ff",
            fill=True,
            fill_opacity=0.9,
            weight=1,
        ).add_to(m)

    # POI marker
    if poi:
        folium.Marker(
            location=poi,
            tooltip=f"POI: {poi[0]:.4f}°, {poi[1]:.4f}°",
            icon=folium.Icon(color="red", icon="wind", prefix="fa"),
        ).add_to(m)

    return m


# ─────────────────────────────────────────────
# Wind power estimation
# ─────────────────────────────────────────────

def estimate_power_curve(ws_series: np.ndarray, rotor_diameter_m: float, air_density: float,
                          cp: float, cut_in: float, rated_speed: float, cut_out: float,
                          rated_power_kw: float = None):
    """
    Estimate wind power output from a wind-speed time series using a simplified
    turbine power curve:
      • below cut_in or above cut_out  -> 0 kW
      • cut_in to rated_speed          -> theoretical power: P = 0.5 * rho * A * v^3 * Cp
                                           (capped at rated_power if provided)
      • rated_speed to cut_out         -> constant rated power

    Returns a dict with the per-timestep power array (kW) and summary stats,
    plus capacity factor (%) relative to rated power.
    """
    area = np.pi * (rotor_diameter_m / 2) ** 2  # swept rotor area, m^2

    # Theoretical power at rated speed sets the "rated_power" if not given
    theo_rated_w = 0.5 * air_density * area * (rated_speed ** 3) * cp
    rated_power_w = (rated_power_kw * 1000) if rated_power_kw else theo_rated_w

    v = np.asarray(ws_series, dtype=float)
    power_w = np.zeros_like(v)

    in_curve = (v >= cut_in) & (v < rated_speed)
    power_w[in_curve] = 0.5 * air_density * area * (v[in_curve] ** 3) * cp
    power_w[in_curve] = np.minimum(power_w[in_curve], rated_power_w)

    at_rated = (v >= rated_speed) & (v <= cut_out)
    power_w[at_rated] = rated_power_w

    # above cut_out or below cut_in stays 0 (safety shutdown / insufficient wind)

    power_kw = power_w / 1000.0
    rated_power_kw_final = rated_power_w / 1000.0

    mean_power_kw = float(np.mean(power_kw))
    capacity_factor_pct = float(mean_power_kw / rated_power_kw_final * 100) if rated_power_kw_final > 0 else 0.0
    annual_energy_mwh = float(mean_power_kw * 8760 / 1000)  # extrapolated to a full year

    return {
        "power_kw_series": power_kw,
        "rated_power_kw": rated_power_kw_final,
        "mean_power_kw": mean_power_kw,
        "max_power_kw": float(np.max(power_kw)),
        "capacity_factor_pct": capacity_factor_pct,
        "annual_energy_mwh": annual_energy_mwh,
        "swept_area_m2": area,
        "pct_time_producing": float(np.sum(power_kw > 0) / len(power_kw) * 100),
    }

def extrapolate_wind_speed(ws_series: np.ndarray, ref_height: float = 10.0,
                            hub_height: float = 100.0, alpha: float = 1/7):
    """
    Power-law wind shear extrapolation: v_hub = v_ref * (h_hub / h_ref)^alpha.
    Default alpha = 1/7 (Hellmann exponent, open sea / offshore).
    Returns (ws_hub_series, shear_factor).
    """
    factor = (hub_height / ref_height) ** alpha
    return ws_series * factor, factor




def compute_leaderboard(ws_field: xr.DataArray, lat_key: str, lon_key: str, top_n: int = 5):
    """
    Rank every grid point in the (already period-filtered) wind-speed field
    by its time-mean wind speed, and return the top_n locations as a list of
    dicts: lat, lon, mean_ws, max_ws, p75_ws.

    Used to surface the most promising PLTB candidate sites across the whole
    study domain, not just the user-selected POI.
    """
    mean_ws_2d = ws_field.mean(dim="time", skipna=True)
    max_ws_2d = ws_field.max(dim="time", skipna=True)
    p75_ws_2d = ws_field.quantile(0.75, dim="time", skipna=True)

    df = mean_ws_2d.to_dataframe(name="mean_ws").reset_index()
    df = df.dropna(subset=["mean_ws"])
    if df.empty:
        return []

    max_df = max_ws_2d.to_dataframe(name="max_ws").reset_index()
    p75_df = p75_ws_2d.to_dataframe(name="p75_ws").reset_index()
    # quantile() adds a 'quantile' coord column; drop it before merging
    p75_df = p75_df.drop(columns=[c for c in p75_df.columns if c == "quantile"], errors="ignore")

    df = df.merge(max_df, on=[lat_key, lon_key], how="left")
    df = df.merge(p75_df, on=[lat_key, lon_key], how="left")
    df = df.sort_values("mean_ws", ascending=False).head(top_n)

    return [
        {
            "lat": float(row[lat_key]),
            "lon": float(row[lon_key]),
            "mean_ws": float(row["mean_ws"]),
            "max_ws": float(row["max_ws"]),
            "p75_ws": float(row["p75_ws"]),
        }
        for _, row in df.iterrows()
    ]

# ─────────────────────────────────────────────
# Site Screening Map
# ─────────────────────────────────────────────

def compute_screening_grid(ws_field: xr.DataArray, lat_key: str, lon_key: str,
                            hub_height: float = 100.0, alpha: float = 1/7):
    """
    Compute per-grid-point statistics across the entire domain for the
    Automated Site Screening Map:
      • mean_ws_10m  — time-mean wind speed at 10 m (raw ERA5)
      • mean_ws_hub  — extrapolated to hub_height via power law
      • p75_ws_hub   — 75th-percentile wind speed at hub height
      • cf_proxy     — simplified capacity-factor proxy (fraction of time
                       wind speed at hub > 3.5 m/s), useful as a site
                       suitability signal without a full power-curve call
      • tier         — "Excellent" / "Moderate" / "Low" based on hub WS
    Returns a flat list of dicts (one per grid point) suitable for
    building a Plotly scatter_mapbox or GeoJSON overlay.
    """
    shear = (hub_height / 10.0) ** alpha

    # Time-mean and 75th-percentile at 10 m
    mean_10m = ws_field.mean(dim="time", skipna=True)
    p75_10m  = ws_field.quantile(0.75, dim="time", skipna=True)
    # Fraction of hours with WS > cut-in (3.5 m/s) at 10m — proxy for CF
    productive_frac = (ws_field > (3.5 / shear)).mean(dim="time", skipna=True)

    df_mean = mean_10m.to_dataframe(name="mean_ws_10m").reset_index()
    df_p75  = p75_10m.to_dataframe(name="p75_ws_10m").reset_index()
    df_cf   = productive_frac.to_dataframe(name="cf_proxy").reset_index()

    # Drop the 'quantile' level that xarray quantile() adds
    df_p75 = df_p75.drop(columns=[c for c in df_p75.columns if c == "quantile"], errors="ignore")

    df = df_mean.merge(df_p75, on=[lat_key, lon_key], how="left")
    df = df.merge(df_cf,  on=[lat_key, lon_key], how="left")
    df = df.dropna(subset=["mean_ws_10m"])

    df["mean_ws_hub"] = df["mean_ws_10m"] * shear
    df["p75_ws_hub"]  = df["p75_ws_10m"]  * shear
    df["cf_proxy_pct"] = df["cf_proxy"] * 100.0

    # Tier classification on hub-height mean WS
    def _tier(v):
        if v >= 7.0:   return "Excellent"
        if v >= 5.0:   return "Moderate"
        return "Low"
    df["tier"] = df["mean_ws_hub"].apply(_tier)

    return df, lat_key, lon_key


def build_screening_map(df, lat_key: str, lon_key: str,
                         hub_height: float,
                         show_tiers: list,
                         color_metric: str = "mean_ws_hub",
                         poi=None):
    """
    Interactive Plotly Scattermapbox Site Screening Map.
    Each grid point is a circle coloured by `color_metric` and filtered
    by `show_tiers`. Top-10 sites are highlighted with larger markers.
    Note: Scattermapbox only supports "circle" symbol; star differentiation
    is achieved via size + colour instead.
    """
    TIER_COLORS = {
        "Excellent": "#00d4ff",
        "Moderate":  "#f0a500",
        "Low":       "#3a5a7a",
    }
    METRIC_LABELS = {
        "mean_ws_hub":   f"Mean Wind Speed @ {hub_height:.0f} m (m/s)",
        "p75_ws_hub":    f"P75 Wind Speed @ {hub_height:.0f} m (m/s)",
        "cf_proxy_pct":  "Productive Hours (%)",
        "mean_ws_10m":   "Mean Wind Speed @ 10 m (m/s)",
    }

    filtered = df[df["tier"].isin(show_tiers)].copy() if show_tiers else df.copy()

    # Mark top-10 sites
    top10_idx = set(df.nlargest(10, color_metric).index)
    filtered["_is_top10"] = filtered.index.isin(top10_idx)

    fig = go.Figure()

    cmin = float(df[color_metric].min())
    cmax = float(df[color_metric].max())

    COLORSCALE = [
        [0.0,  "#0b1827"],
        [0.25, "#0e4a6e"],
        [0.5,  "#0082b0"],
        [0.75, "#00aad4"],
        [1.0,  "#00d4ff"],
    ]

    # ── Regular grid points (not top-10) per tier
    for tier in ["Excellent", "Moderate", "Low"]:
        sub = filtered[(filtered["tier"] == tier) & (~filtered["_is_top10"])]
        if sub.empty:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=sub[lat_key],
            lon=sub[lon_key],
            mode="markers",
            name=tier,
            marker=dict(
                size=9,
                color=sub[color_metric],
                colorscale=COLORSCALE,
                cmin=cmin,
                cmax=cmax,
                showscale=(tier == "Excellent"),
                colorbar=dict(
                    title=dict(
                        text=METRIC_LABELS.get(color_metric, color_metric),
                        font=dict(color="#a8bfd4", size=11),
                    ),
                    tickfont=dict(color="#a8bfd4", size=10),
                    outlinecolor="#1e3a5f",
                    bgcolor="rgba(13,26,45,0.85)",
                    x=1.01,
                ),
                opacity=0.80,
            ),
            customdata=np.stack([
                sub["mean_ws_10m"].round(2),
                sub["mean_ws_hub"].round(2),
                sub["p75_ws_hub"].round(2),
                sub["cf_proxy_pct"].round(1),
                sub["tier"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{lat:.4f}°, %{lon:.4f}°</b><br>"
                "Tier: %{customdata[4]}<br>"
                f"Mean Wind Speed @ 10 m: %{{customdata[0]}} m/s<br>"
                f"Mean Wind Speed @ {hub_height:.0f} m: %{{customdata[1]}} m/s<br>"
                f"P75 Wind Speed @ {hub_height:.0f} m: %{{customdata[2]}} m/s<br>"
                "Productive hours: %{customdata[3]}%"
                "<extra></extra>"
            ),
        ))

    # ── Top-10 sites — larger gold markers, labelled
    top10_sub = filtered[filtered["_is_top10"]]
    if not top10_sub.empty:
        top10_sub = top10_sub.sort_values(color_metric, ascending=False).reset_index(drop=True)
        fig.add_trace(go.Scattermapbox(
            lat=top10_sub[lat_key],
            lon=top10_sub[lon_key],
            mode="markers+text",
            name="Top-10 Sites",
            text=[f"#{i+1}" for i in range(len(top10_sub))],
            textfont=dict(size=9, color="#ffffff"),
            textposition="top right",
            marker=dict(
                size=18,
                color="#ffd700",
                opacity=1.0,
            ),
            customdata=np.stack([
                top10_sub["mean_ws_10m"].round(2),
                top10_sub["mean_ws_hub"].round(2),
                top10_sub["p75_ws_hub"].round(2),
                top10_sub["cf_proxy_pct"].round(1),
                top10_sub["tier"],
            ], axis=-1),
            hovertemplate=(
                "<b>Top Site — %{lat:.4f}°, %{lon:.4f}°</b><br>"
                "Tier: %{customdata[4]}<br>"
                f"Mean Wind Speed @ {hub_height:.0f} m: %{{customdata[1]}} m/s<br>"
                f"P75 Wind Speed @ {hub_height:.0f} m: %{{customdata[2]}} m/s<br>"
                "Productive hours: %{customdata[3]}%"
                "<extra></extra>"
            ),
        ))

    # ── Selected POI
    if poi:
        fig.add_trace(go.Scattermapbox(
            lat=[poi[0]], lon=[poi[1]],
            mode="markers",
            name="Selected POI",
            marker=dict(size=16, color="#ff4444", opacity=1.0),
            hovertemplate=(
                f"<b>Selected POI</b><br>"
                f"Lat: {poi[0]:.4f}°, Lon: {poi[1]:.4f}°<extra></extra>"
            ),
        ))

    center_lat = (df[lat_key].min() + df[lat_key].max()) / 2
    center_lon = (df[lon_key].min() + df[lon_key].max()) / 2

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=540,
        legend=dict(
            orientation="v",
            yanchor="top", y=0.98,
            xanchor="left", x=0.01,
            font=dict(size=11, color="#a8bfd4"),
            bgcolor="rgba(13,26,45,0.85)",
            bordercolor="#1e3a5f",
            borderwidth=1,
        ),
        uirevision="screening_map",
    )
    return fig



def screening_summary_stats(df, lat_key, lon_key, hub_height):
    """Return summary counts and key metrics for the domain screening."""
    total = len(df)
    counts = df["tier"].value_counts().to_dict()
    exc = counts.get("Excellent", 0)
    mod = counts.get("Moderate", 0)
    low = counts.get("Low", 0)
    best_row = df.loc[df["mean_ws_hub"].idxmax()]
    return {
        "total_points": total,
        "excellent_n": exc,
        "moderate_n": mod,
        "low_n": low,
        "excellent_pct": exc / total * 100 if total else 0,
        "moderate_pct": mod / total * 100 if total else 0,
        "low_pct": low / total * 100 if total else 0,
        "best_lat": float(best_row[lat_key]),
        "best_lon": float(best_row[lon_key]),
        "best_mean_ws_hub": float(best_row["mean_ws_hub"]),
        "best_p75_hub": float(best_row["p75_ws_hub"]),
        "domain_mean_hub": float(df["mean_ws_hub"].mean()),
        "domain_max_hub": float(df["mean_ws_hub"].max()),
    }


def screening_csv_bytes(df, lat_key, lon_key, hub_height):
    """Serialize the screening grid to CSV bytes for download."""
    import io as _io
    buf = _io.StringIO()
    cols = [lat_key, lon_key, "mean_ws_10m", "mean_ws_hub",
            "p75_ws_hub", "cf_proxy_pct", "tier"]
    rename = {
        lat_key: "latitude",
        lon_key: "longitude",
        "mean_ws_10m": "mean_ws_10m_ms",
        f"mean_ws_hub": f"mean_ws_{hub_height:.0f}m_ms",
        f"p75_ws_hub":  f"p75_ws_{hub_height:.0f}m_ms",
        "cf_proxy_pct": "productive_hours_pct",
        "tier": "tier",
    }
    out = df[cols].rename(columns=rename)
    out.to_csv(buf, index=False, float_format="%.4f")
    return buf.getvalue().encode()




    """
    Build a Folium map centred on the data domain with:
    • A dark CartoDB basemap
    • A dashed cyan rectangle showing the data boundary
    • An optional POI marker
    """
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        width="100%",
        height=480,
    )

    # Data-boundary rectangle
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="#00d4ff",
        weight=2,
        dash_array="8 4",
        fill=True,
        fill_color="#00d4ff",
        fill_opacity=0.06,
        tooltip=folium.Tooltip(
            f"<b>Data Domain</b><br>"
            f"Lat: {lat_min:.2f}° – {lat_max:.2f}°<br>"
            f"Lon: {lon_min:.2f}° – {lon_max:.2f}°"
        ),
    ).add_to(m)

    # Corner markers for visual clarity
    for lat, lon in [(lat_min, lon_min), (lat_min, lon_max),
                     (lat_max, lon_min), (lat_max, lon_max)]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#00d4ff",
            fill=True,
            fill_opacity=0.9,
            weight=1,
        ).add_to(m)

    # POI marker
    if poi:
        folium.Marker(
            location=poi,
            tooltip=f"POI: {poi[0]:.4f}°, {poi[1]:.4f}°",
            icon=folium.Icon(color="red", icon="wind", prefix="fa"),
        ).add_to(m)

    return m


# ─────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────

def plot_time_series(ws_series: np.ndarray, time_vals):
    """Plotly line chart of wind speed over time with rolling average."""
    ws_smooth = np.convolve(ws_series, np.ones(24) / 24, mode="same") if len(ws_series) >= 24 else ws_series

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_vals, y=ws_series,
        mode="lines",
        name="Wind Speed",
        line=dict(color=TEAL, width=1.2),
        opacity=0.55,
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Wind Speed: %{y:.2f} m/s<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=time_vals, y=ws_smooth,
        mode="lines",
        name="24-hr Rolling Avg",
        line=dict(color=ACCENT2, width=2),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Avg Wind Speed: %{y:.2f} m/s<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Wind Speed Time Series", font=dict(color="#e8f4fd", size=14), pad=dict(b=18)),
        xaxis_title="Date / Time",
        yaxis_title="Wind Speed (m/s)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def plot_wind_rose(ws_series: np.ndarray, wd_series: np.ndarray):
    """
    Plotly Wind Rose: stacked bar polar chart bucketing direction × speed class.
    """
    dir_bins = np.arange(0, 361, 22.5)
    dir_labels = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    speed_bins = [0, 3, 6, 9, 12, 15, np.inf]
    speed_labels = ["0–3", "3–6", "6–9", "9–12", "12–15", "≥15"]
    palette = ["#1a3d5c", "#0e5e8a", "#0082b0", "#00aad4", "#00d4ff", "#80eaff"]

    fig = go.Figure()
    for i, (lo, hi) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
        mask_speed = (ws_series >= lo) & (ws_series < hi)
        r_vals = []
        for j in range(len(dir_labels)):
            d_lo = dir_bins[j]
            d_hi = dir_bins[j + 1]
            if j == 0:
                mask_dir = (wd_series >= 348.75) | (wd_series < d_hi)
            else:
                mask_dir = (wd_series >= d_lo) & (wd_series < d_hi)
            count = np.sum(mask_speed & mask_dir)
            r_vals.append(count / len(ws_series) * 100)  # percentage

        fig.add_trace(go.Barpolar(
            r=r_vals,
            theta=dir_labels,
            name=f"{speed_labels[i]} m/s",
            marker_color=palette[i],
            marker_line_color="rgba(0,0,0,0.3)",
            marker_line_width=0.5,
            opacity=0.9,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="#0b1827",
            angularaxis=dict(
                # Compass orientation: N at top (90°), clockwise rotation so
                # E sits on the right, S at bottom, W on the left — matching
                # standard compass convention instead of Plotly's default
                # mathematical convention (0° = right, counter-clockwise).
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=10, color="#a8bfd4"),
                linecolor="#1e3a5f",
                gridcolor="#1a2d45",
            ),
            radialaxis=dict(
                ticksuffix="%",
                tickfont=dict(size=9, color="#5a8aaa"),
                gridcolor="#1a2d45",
                linecolor="#1e3a5f",
            ),
        ),
        title=dict(text="Wind Rose (Frequency by Direction & Speed)", font=dict(color="#e8f4fd", size=14), pad=dict(b=18)),
        legend=dict(
            orientation="v", yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            font=dict(size=10, color="#a8bfd4"),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family="Space Grotesk, sans-serif", color="#a8bfd4"),
        margin=dict(l=40, r=120, t=75, b=40),
    )
    return fig


def plot_histogram(ws_series: np.ndarray):
    """
    Plotly histogram of wind speed with Weibull PDF overlay (estimated via MLE).
    """
    from scipy.stats import weibull_min

    fig = go.Figure()

    # Histogram bars
    fig.add_trace(go.Histogram(
        x=ws_series,
        nbinsx=30,
        name="Observed Freq.",
        marker=dict(color=TEAL, opacity=0.7, line=dict(color="#0b1827", width=0.6)),
        histnorm="probability density",
        hovertemplate="Wind Speed: %{x:.1f} m/s<br>Density: %{y:.4f}<extra></extra>",
    ))

    # Weibull fit
    try:
        ws_clean = ws_series[ws_series > 0]
        shape, loc, scale = weibull_min.fit(ws_clean, floc=0)
        x_fit = np.linspace(0, ws_clean.max() * 1.1, 300)
        y_fit = weibull_min.pdf(x_fit, shape, loc, scale)
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit,
            mode="lines",
            name=f"Weibull fit (k={shape:.2f}, λ={scale:.2f})",
            line=dict(color=ACCENT2, width=2.5, dash="dash"),
            hovertemplate="Wind Speed: %{x:.1f} m/s<br>PDF: %{y:.4f}<extra></extra>",
        ))
    except Exception:
        pass  # scipy may not be installed; silently skip

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Wind Speed Distribution (Weibull Assessment)", font=dict(color="#e8f4fd", size=14), pad=dict(b=18)),
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Probability Density",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.05,
    )
    return fig


def plot_quiver_map(u_2d: np.ndarray, v_2d: np.ndarray, ws_2d: np.ndarray,
                     lats_arr: np.ndarray, lons_arr: np.ndarray, title_suffix: str = ""):
    """
    Spatial quiver plot of wind direction & speed over the study domain.
    u_2d, v_2d, ws_2d: 2D arrays shaped (lat, lon) — typically a time-mean.
    lats_arr, lons_arr: 1D coordinate arrays matching the grid.

    Background: heatmap of wind speed (m/s).
    Foreground: hand-built arrows (line + small triangular arrowhead) whose
    length is normalised against the grid spacing (NOT raw degrees-per-m/s),
    so arrows stay readable and proportionate regardless of the dataset's
    spatial resolution or wind-speed magnitude.
    """
    lon_grid, lat_grid = np.meshgrid(lons_arr, lats_arr)

    # Subsample the grid for arrows so the plot doesn't get too cluttered
    # on high-resolution datasets — aim for roughly 12-15 arrows per axis.
    n_lat, n_lon = lat_grid.shape
    lat_step = max(1, n_lat // 14)
    lon_step = max(1, n_lon // 14)

    lat_sub = lat_grid[::lat_step, ::lon_step]
    lon_sub = lon_grid[::lat_step, ::lon_step]
    u_sub = u_2d[::lat_step, ::lon_step]
    v_sub = v_2d[::lat_step, ::lon_step]
    ws_sub = np.sqrt(u_sub**2 + v_sub**2)

    # Flatten + drop NaNs (masked/land cells)
    flat_mask = ~(np.isnan(u_sub) | np.isnan(v_sub))
    lon_flat = lon_sub[flat_mask]
    lat_flat = lat_sub[flat_mask]
    u_flat = u_sub[flat_mask]
    v_flat = v_sub[flat_mask]
    ws_flat = ws_sub[flat_mask]

    # Background heatmap of wind speed
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=lons_arr, y=lats_arr, z=ws_2d,
        colorscale=[
            [0.0, "#0b1827"], [0.2, "#0e5e8a"], [0.4, "#0082b0"],
            [0.6, "#00aad4"], [0.8, "#00d4ff"], [1.0, "#80eaff"],
        ],
        colorbar=dict(
            title=dict(text="m/s", font=dict(color="#a8bfd4")),
            tickfont=dict(color="#a8bfd4"),
            outlinecolor="#1e3a5f",
        ),
        hovertemplate="Lat: %{y:.2f}°<br>Lon: %{x:.2f}°<br>Wind Speed: %{z:.2f} m/s<extra></extra>",
        zsmooth="best",
    ))

    if len(lon_flat) > 0 and ws_flat.max() > 0:
        # Reference arrow length: a fraction of the (subsampled) grid
        # spacing, so neighbouring arrows never overlap regardless of how
        # large the raw wind speeds are. Direction comes from u/v; length
        # is then scaled by each point's relative speed (faster = longer),
        # capped at max_len so the fastest point doesn't collide with its
        # neighbour.
        lat_spacing = np.median(np.diff(np.unique(lat_sub))) if len(np.unique(lat_sub)) > 1 else 1.0
        lon_spacing = np.median(np.diff(np.unique(lon_sub))) if len(np.unique(lon_sub)) > 1 else 1.0
        cell_spacing = min(abs(lat_spacing), abs(lon_spacing))
        max_len = cell_spacing * 0.8  # leave headroom so arrows don't touch

        speed_norm = ws_flat / ws_flat.max()
        # sqrt-scale so slow-wind arrows are still visible (not squashed to ~0)
        arrow_len = max_len * (0.25 + 0.75 * np.sqrt(speed_norm))

        # Unit direction vectors
        mag = np.sqrt(u_flat**2 + v_flat**2)
        mag_safe = np.where(mag == 0, 1, mag)
        dir_lon = (u_flat / mag_safe) * arrow_len
        dir_lat = (v_flat / mag_safe) * arrow_len

        end_lon = lon_flat + dir_lon
        end_lat = lat_flat + dir_lat

        # Build all shaft segments as one Scatter trace (None-separated)
        # for performance — avoids one trace per arrow.
        shaft_x, shaft_y = [], []
        head_x, head_y = [], []
        head_size = arrow_len * 0.35
        for i in range(len(lon_flat)):
            shaft_x += [lon_flat[i], end_lon[i], None]
            shaft_y += [lat_flat[i], end_lat[i], None]

            # Arrowhead: two short lines splayed ±25° from the shaft direction
            theta = np.arctan2(dir_lat[i], dir_lon[i])
            for offset in (2.7, -2.7):  # radians offset ≈ 155° back from tip
                ang = theta + offset
                hx = end_lon[i] + head_size * np.cos(ang)
                hy = end_lat[i] + head_size * np.sin(ang)
                head_x += [end_lon[i], hx, None]
                head_y += [end_lat[i], hy, None]

        fig.add_trace(go.Scatter(
            x=shaft_x, y=shaft_y, mode="lines",
            line=dict(color="#e8f4fd", width=1.6),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=head_x, y=head_y, mode="lines",
            line=dict(color="#e8f4fd", width=1.6),
            hoverinfo="skip", showlegend=False,
        ))
        # Invisible markers at arrow origins carrying the real hover info
        fig.add_trace(go.Scatter(
            x=lon_flat, y=lat_flat, mode="markers",
            marker=dict(size=4, color="#e8f4fd", opacity=0.0),
            customdata=np.stack([ws_flat], axis=-1),
            hovertemplate="Lat: %{y:.2f}°<br>Lon: %{x:.2f}°<br>Wind Speed: %{customdata[0]:.2f} m/s<extra></extra>",
            showlegend=False,
        ))

    layout_base = {k: v_ for k, v_ in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")}
    fig.update_layout(
        **layout_base,
        title=dict(
            text=f"Spatial Wind Field — Speed &amp; Direction{title_suffix}",
            font=dict(color="#e8f4fd", size=14),
        ),
        xaxis=dict(
            title="Longitude (°)",
            gridcolor="#1a2d45", zerolinecolor="#1a2d45", linecolor="#1e3a5f",
        ),
        yaxis=dict(
            title="Latitude (°)",
            scaleanchor="x", scaleratio=1,
            gridcolor="#1a2d45", zerolinecolor="#1a2d45", linecolor="#1e3a5f",
        ),
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig




def main():
    # ── Header ────────────────────────────────
    st.markdown(
        """
        <div class="dashboard-header">
            <div style="display:flex; align-items:center; gap:0.9rem;">
                <div class="title-icon-badge">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2" stroke-linecap="round">
                        <path d="M3 8h9a3 3 0 1 0-3-3"/>
                        <path d="M3 13h13a3 3 0 1 1-3 3"/>
                        <path d="M3 18h7a2.5 2.5 0 1 0-2.5-2.5"/>
                    </svg>
                </div>
                <div>
                    <div class="dashboard-title">Wind Energy <span class="title-accent">Assessment</span></div>
                    <div class="dashboard-subtitle">Offshore Wind Power Plant Potential · NetCDF Analysis Suite</div>
                </div>
            </div>
            <div class="live-pill"><span class="live-dot"></span>Analysis Engine Ready</div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.markdown("### Data Input")
        show_info = st.toggle("Show upload instructions", value=True, key="show_upload_info")
        if show_info:
            st.markdown(
                '<div class="info-box">Upload any number of NetCDF (<code>.nc</code>) files — '
                'browse to select them or drag-and-drop them directly into the box below. '
                'The dashboard automatically reads each file\'s metadata and determines whether '
                'it contains the U (eastward) wind component, the V (northward) component, or '
                'both — so files can be split by time, by component, or combined in any '
                'combination. Not limited to ERA5: CMEMS, WRF, NOAA/NCEP, and other CF-style '
                'NetCDF sources are also recognized, and every .nc format version (classic '
                'NETCDF3, 64-bit offset, NETCDF4/HDF5) is supported.</div>',
                unsafe_allow_html=True,
            )
        uploaded_files = st.file_uploader(
            "Upload NetCDF wind data file(s)",
            type=["nc", "nc4", "netcdf", "cdf"],
            accept_multiple_files=True,
            help="Drag-and-drop or browse for one or many .nc files. Any combination of "
                 "U-only, V-only, or combined U+V files is accepted, from any NetCDF-producing "
                 "source.",
            key="wind_file_uploader",
        )
        st.caption(
            "Streamlit's default per-file upload limit is 200 MB. For larger files, raise "
            "`server.maxUploadSize` in your app's `.streamlit/config.toml`."
        )

        st.divider()
        st.markdown("### Turbine Parameters")
        st.markdown(
            '<div class="info-box" style="font-size:0.8rem;">Used to estimate the power potential '
            '(power curve) at the POI. Default values follow a medium-to-large-scale offshore '
            'wind turbine — adjust as needed.</div>',
            unsafe_allow_html=True,
        )
        rotor_diameter = st.number_input(
            "Rotor Diameter (m)", min_value=1.0, max_value=250.0, value=100.0, step=5.0,
            help="Diameter of the turbine's rotor blades, used to compute the swept area (A = π·r²) "
                 "that captures wind energy. Larger rotors capture more energy at a given wind speed.",
            key="rotor_diameter",
        )
        air_density = st.number_input(
            "Air Density (kg/m³)", min_value=0.8, max_value=1.5, value=1.225, step=0.005,
            format="%.3f",
            help="Mass of air per cubic metre. 1.225 kg/m³ is the standard sea-level value; "
                 "it decreases slightly with elevation and air temperature.",
            key="air_density",
        )
        cp_value = st.slider(
            "Power Coefficient (Cp)", min_value=0.10, max_value=0.50, value=0.40, step=0.01,
            help="The theoretical Betz limit is 0.593; modern turbines generally operate in the 0.35–0.45 range.",
            key="cp_value",
        )
        cut_in_speed = st.number_input(
            "Cut-in Speed (m/s)", min_value=0.0, max_value=10.0, value=3.5, step=0.5,
            help="The minimum wind speed at which the turbine starts generating power. "
                 "Below this speed, the turbine remains idle.",
            key="cut_in_speed",
        )
        rated_speed = st.number_input(
            "Rated Speed (m/s)", min_value=1.0, max_value=30.0, value=12.0, step=0.5,
            help="The wind speed at which the turbine reaches its rated (maximum) power output. "
                 "Output stays flat at rated power for speeds above this until cut-out.",
            key="rated_speed",
        )
        cut_out_speed = st.number_input(
            "Cut-out Speed (m/s)", min_value=5.0, max_value=40.0, value=25.0, step=0.5,
            help="The wind speed at which the turbine shuts down to avoid mechanical damage. "
                 "Above this speed, power output drops to zero.",
            key="cut_out_speed",
        )
        hub_height = st.number_input(
            "Hub height (m)", min_value=10.0, max_value=200.0, value=100.0, step=10.0,
            key="hub_height",
            help="Wind speed is extrapolated from 10 m ERA5 reference to hub height using "
                 "the power law with Hellmann exponent α = 1/7 (standard for open sea).",
        )

        st.divider()
        st.markdown("### How to use")
        st.markdown(
            """
            1. **Upload** one or more `.nc` files — browse or drag-and-drop, any U/V/combined mix.  
            2. **Review** the detected file roles and metadata shown above the map.  
            3. **Select** a period mode: All Data, Monthly, Seasonal, or Daily.  
            4. **Inspect** the cyan boundary on the map — your data domain.  
            5. **Click** anywhere inside the boundary to set your POI.  
            6. **Read** the charts, KPIs, leaderboard, and power potential estimates.
            """
        )
        st.divider()
        st.caption("Wind Energy Assessment · v1.3 · Built with Python")

        st.divider()
        show_dev = st.toggle("Show Developer info", value=False, key="show_developer_info")
        if show_dev:
            st.markdown("### Developer")
            st.markdown(
                '<div class="info-box" style="font-size:0.82rem; line-height:1.7;">'
                'M. Ravi Alrizqi Permana (12923026)<br>'
                'Kevin Aulia Aryasena (12923047)<br>'
                'Zaid Ahmad Shadiq (12923061)'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Contact Us")
        st.markdown(
            '<div class="info-box" style="font-size:0.82rem;">12923047@mahasiswa.itb.ac.id</div>',
            unsafe_allow_html=True,
        )

    # ── No files uploaded — landing state ─────
    if not uploaded_files:
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.markdown(
                """
                <div style="text-align:center; padding: 3.5rem 0 2.5rem;">
                    <div style="display:flex; justify-content:center; margin-bottom:0.4rem;">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.4" stroke-linecap="round" style="filter: drop-shadow(0 0 22px rgba(0,212,255,0.3));">
                            <path d="M2 16c1.5 1.5 3 1.5 4.5 0s3-1.5 4.5 0 3 1.5 4.5 0 3-1.5 4.5 0"/>
                            <path d="M2 11c1.5 1.5 3 1.5 4.5 0s3-1.5 4.5 0 3 1.5 4.5 0 3-1.5 4.5 0"/>
                        </svg>
                    </div>
                    <div style="font-size:1.35rem; color:#d1dce8; margin-top:1.1rem; font-weight:600;">
                        Upload one or more NetCDF wind files to begin
                    </div>
                    <div style="font-size:0.86rem; color:#5a8aaa; margin-top:0.5rem; max-width:480px; margin-left:auto; margin-right:auto;">
                        ERA5, CMEMS, WRF, NOAA/NCEP, or any CF-style source with U/V components —
                        browse or drag-and-drop, any number of files, any .nc format version
                    </div>
                    <div style="display:flex; gap:0.6rem; justify-content:center; flex-wrap:wrap; margin-top:1.6rem;">
                        <span class="live-pill" style="background:rgba(0,212,255,0.08); border-color:#1e4a7a; color:#7fd4f0;">Interactive POI Map</span>
                        <span class="live-pill" style="background:rgba(240,165,0,0.08); border-color:#4a3a00; color:#f0c97a;">Power Curve Estimator</span>
                        <span class="live-pill">Full-Domain Site Screening</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # ── Load, classify & parse every uploaded file ──
    tmp_paths = []
    file_records = []
    file_metadata = []
    load_issues = []

    with st.spinner(f"Reading and classifying {len(uploaded_files)} file(s)…"):
        for f in uploaded_files:
            try:
                # Streamlit's UploadedFile is an in-memory buffer; xarray/pandas
                # need a real path on disk. Each file gets its own temp path so
                # large batches never overwrite one another.
                suffix = os.path.splitext(f.name)[1].lower() or ".nc"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                tmp_paths.append(tmp_path)

                # Tries multiple backends so any .nc format version is readable.
                ds_i = open_netcdf_any_format(tmp_path)
                kind, u_key, v_key = classify_netcdf_dataset(ds_i)

                # Inspect this file's metadata to determine its role.
                file_metadata.append(extract_file_metadata(f.name, f.size, ds_i, kind, u_key, v_key))

                if kind == "unknown":
                    load_issues.append(
                        f"**{f.name}** — no recognisable U/V or wind speed/direction data found "
                        f"(variables present: {', '.join(ds_i.data_vars) or 'none'}). Skipped."
                    )
                    continue

                file_records.append(
                    {"name": f.name, "ds": ds_i, "kind": kind, "u_key": u_key, "v_key": v_key}
                )

            except Exception as err:
                load_issues.append(f"**{f.name}** — failed to open: {err}")

        if not file_records:
            st.error("None of the uploaded files contained recognisable wind data.")
            for msg in load_issues:
                st.markdown(f'<div class="warning-box">{msg}</div>', unsafe_allow_html=True)
            st.stop()

        try:
            u_combined, v_combined = build_uv_from_records(file_records)
            u, v, ws, wd = compute_wind(u_combined, v_combined)

            # Merge attrs from every contributing file (later files win on
            # key collisions) so global metadata isn't lost when a dataset
            # is split across many uploads.
            src_attrs = {}
            for rec in file_records:
                src_attrs.update(rec["ds"].attrs)

            # Build the combined dataset from u/v themselves (not the raw
            # per-file datasets) so the standardised 'time' dimension from
            # compute_wind is always what the rest of the dashboard sees.
            ds = xr.Dataset({u.name or "u": u, v.name or "v": v})
            if not ds.attrs:
                ds.attrs = src_attrs

            lat_key, lon_key, lat_min, lat_max, lon_min, lon_max = get_lat_lon_bounds(ds)

        except Exception as err:
            st.error(f"**Failed to combine uploaded files:** {err}")
            st.stop()

    if load_issues:
        with st.expander(f"{len(load_issues)} file(s) skipped or had issues", expanded=False):
            for msg in load_issues:
                st.markdown(f'<div class="warning-box">{msg}</div>', unsafe_allow_html=True)

    n_u = sum(1 for r in file_records if r["kind"] in ("u", "both"))
    n_v = sum(1 for r in file_records if r["kind"] in ("v", "both"))
    st.markdown(
        f'<div class="success-box">{len(file_records)} file(s) loaded successfully '
        f'({n_u} contributing U, {n_v} contributing V) · {len(ds.time)} time steps · '
        f'Lat {lat_min:.2f}°–{lat_max:.2f}° · Lon {lon_min:.2f}°–{lon_max:.2f}°</div>',
        unsafe_allow_html=True,
    )

    # ── Per-file metadata, read straight from each upload ──
    with st.expander(f"Detected file metadata — {len(file_metadata)} file(s)", expanded=False):
        def _role_label(m):
            if m["kind"] == "both" and m["u_key"] and m["v_key"] and m["u_key"] != m["v_key"]:
                return "U + V"
            elif m["kind"] in ("u", "both"):
                return "U only"
            elif m["kind"] == "v":
                return "V only"
            return "Unrecognised"
        role_label = {k: k for k in ["both","u","v","unknown"]}  # fallback unused
        table_rows = [
            {
                "File": m["filename"],
                "Size (MB)": round(m["size_mb"], 2),
                "Role": _role_label(m),
                "Variables": ", ".join(m["all_vars"]),
                "Time Steps": m["n_timesteps"],
                "Time Range": m["time_range"],
                "Lat Range": m["lat_range"],
                "Lon Range": m["lon_range"],
                "Source": m["source"],
            }
            for m in file_metadata
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # ── Year filter — only shown when the uploaded data spans more ──
    # ── than one calendar year ───────────────────────────────────────
    available_years = get_available_years(ds["time"])
    selected_year = "All Years"

    if len(available_years) > 1:
        st.markdown(
            '<div class="section-label">Data Year</div>'
            '<div class="section-sub">Your uploaded data spans multiple years — pick one to focus the analysis, or keep "All Years" to combine them.</div>',
            unsafe_allow_html=True,
        )
        year_options = ["All Years"] + [str(y) for y in available_years]
        chosen_year_label = st.selectbox(
            "Select year",
            options=year_options,
            key="year_select",
            label_visibility="collapsed",
        )
        selected_year = "All Years" if chosen_year_label == "All Years" else int(chosen_year_label)

        u = filter_by_year(u, selected_year)
        v = filter_by_year(v, selected_year)
        ws = filter_by_year(ws, selected_year)
        wd = filter_by_year(wd, selected_year)

        n_years = len(available_years)
        year_span = f"{available_years[0]}–{available_years[-1]}"
        if selected_year == "All Years":
            st.caption(f"Combining {n_years} years of data ({year_span}) — {len(ws['time'])} time steps total.")
        else:
            st.caption(f"Showing {selected_year} only — {len(ws['time'])} of the full {year_span} dataset's time steps.")

    # ── Period filter: All Data / Monthly / Seasonal / Daily ──
    st.markdown(
        '<div class="section-label">Data Period Filter</div>'
        '<div class="section-sub">Narrow the analysis window before exploring the map, KPIs, and charts below.</div>',
        unsafe_allow_html=True,
    )
    pf_col1, pf_col2, pf_col3 = st.columns([1, 1, 2], gap="medium")

    with pf_col1:
        period_mode = st.radio(
            "Display mode",
            options=["All Data", "Monthly", "Seasonal", "Daily"],
            horizontal=False,
            key="period_mode",
            label_visibility="collapsed",
        )

    selected_month = None
    selected_season = None
    selected_day = None

    with pf_col2:
        if period_mode in ("Monthly", "Daily"):
            available_months = sorted(set(int(m) for m in ws["time"].dt.month.values))
            month_options = [MONTH_NAMES[m - 1] for m in available_months]
            chosen_label = st.selectbox("Select month", options=month_options, key="month_select")
            selected_month = available_months[month_options.index(chosen_label)]
        elif period_mode == "Seasonal":
            available_seasons = list(SEASON_MONTHS.keys())
            selected_season = st.selectbox("Select season", options=available_seasons, key="season_select")

    with pf_col3:
        if period_mode == "Daily" and selected_month is not None:
            # Find available days for the selected month
            month_mask = ws["time"].dt.month.values == selected_month
            available_days = sorted(set(int(d) for d in ws["time"].dt.day.values[month_mask]))
            if available_days:
                day_options = [str(d) for d in available_days]
                chosen_day_label = st.selectbox(
                    f"Select day ({MONTH_NAMES[selected_month - 1]})",
                    options=day_options,
                    key="day_select",
                )
                selected_day = int(chosen_day_label)
            else:
                st.warning("No days available for the selected month.")

    # Apply the period filter to u, v, ws, wd (all share the 'time' dim)
    u_f = filter_by_period(u, period_mode, selected_month, selected_season, selected_day)
    v_f = filter_by_period(v, period_mode, selected_month, selected_season, selected_day)
    ws_f = filter_by_period(ws, period_mode, selected_month, selected_season, selected_day)
    wd_f = filter_by_period(wd, period_mode, selected_month, selected_season, selected_day)

    if len(ws_f["time"]) == 0:
        st.warning("No data found for the selected period. Showing all data instead.")
        u_f, v_f, ws_f, wd_f = u, v, ws, wd
    else:
        if period_mode == "All Data":
            period_label = "all available data"
        elif period_mode == "Monthly":
            period_label = f"{MONTH_NAMES[selected_month - 1]}" if selected_month else "selected month"
        elif period_mode == "Seasonal":
            period_label = selected_season or "selected season"
        elif period_mode == "Daily":
            period_label = (
                f"{MONTH_NAMES[selected_month - 1]} {selected_day}"
                if selected_month and selected_day else "selected day"
            )
        else:
            period_label = "selected period"
        st.caption(f"Showing {len(ws_f['time'])} of {len(ws['time'])} time steps — {period_label}.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Layout: map + right panel ─────────────
    col_map, col_info = st.columns([3, 1], gap="medium")

    with col_map:
        st.markdown(
            '<div class="section-label">Interactive Data Domain Map</div>'
            '<div class="section-sub">Click anywhere inside the cyan boundary to set your Point of Interest (POI).</div>',
            unsafe_allow_html=True,
        )

        # Retrieve POI from previous interaction (session state)
        poi = st.session_state.get("poi", None)
        folium_map = build_folium_map(lat_min, lat_max, lon_min, lon_max, poi=poi)

        map_output = st_folium(
            folium_map,
            use_container_width=True,
            height=460,
            returned_objects=["last_clicked"],
            key="wind_map",
        )

        # Process click
        if map_output and map_output.get("last_clicked"):
            click_lat = map_output["last_clicked"]["lat"]
            click_lon = map_output["last_clicked"]["lng"]
            clicked = [click_lat, click_lon]

            if is_within_bounds(click_lat, click_lon, lat_min, lat_max, lon_min, lon_max):
                # Only rerun if this is actually a NEW click (avoids infinite
                # rerun loops, since st_folium keeps returning the last click
                # on every subsequent rerun too).
                if st.session_state.get("poi") != clicked:
                    st.session_state["poi"] = clicked
                    # Force an immediate rerun so build_folium_map() draws the
                    # red pin at the new location right away — without this,
                    # the map object already sent to the browser keeps showing
                    # the OLD pin position until some unrelated rerun happens.
                    st.rerun()
            else:
                st.markdown(
                    '<div class="warning-box">Clicked outside the data domain. '
                    'Please click within the cyan boundary.</div>',
                    unsafe_allow_html=True,
                )

    with col_info:
        st.markdown('<div class="section-label">Dataset Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card accent-mono" style="margin-bottom:0.6rem;">
            <div class="kpi-label">Time Steps (Filtered)</div>
            <div class="kpi-value" style="font-size:1.5rem;">{len(ws_f['time'])}</div>
        </div>
        <div class="kpi-card accent-mono" style="margin-bottom:0.6rem;">
            <div class="kpi-label">Lat Range</div>
            <div class="kpi-value" style="font-size:1.05rem;">{lat_min:.2f}° – {lat_max:.2f}°</div>
        </div>
        <div class="kpi-card accent-mono" style="margin-bottom:0.6rem;">
            <div class="kpi-label">Lon Range</div>
            <div class="kpi-value" style="font-size:1.05rem;">{lon_min:.2f}° – {lon_max:.2f}°</div>
        </div>
        """, unsafe_allow_html=True)

        if poi:
            st.markdown('<div class="section-label" style="margin-top:0.4rem;">Selected POI</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="coord-badge">{poi[0]:.4f}°, {poi[1]:.4f}°</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="info-box" style="margin-top:0.4rem;">Click on the map inside the cyan boundary to select a Point of Interest.</div>',
                unsafe_allow_html=True,
            )

    # ── Gateway: an analysis POI is required beyond this point ─
    if not poi:
        st.info("Select a Point of Interest on the map above to unlock wind analysis, power estimation, and site screening.")
        st.stop()

    # ── Extract nearest grid-point time series ─
    with st.spinner("Extracting nearest grid point…"):
        try:
            ws_point = ws_f.sel({lat_key: poi[0], lon_key: poi[1]}, method="nearest")
            wd_point = wd_f.sel({lat_key: poi[0], lon_key: poi[1]}, method="nearest")

            # Resolve actual lat/lon of the snapped grid point
            snapped_lat = float(ws_point[lat_key].values)
            snapped_lon = float(ws_point[lon_key].values)

            ws_vals = ws_point.values.flatten().astype(float)
            wd_vals = wd_point.values.flatten().astype(float)
            time_vals = ws_f["time"].values

            # Remove NaN
            valid = ~np.isnan(ws_vals) & ~np.isnan(wd_vals)
            ws_vals = ws_vals[valid]
            wd_vals = wd_vals[valid]
            time_vals = time_vals[valid]

        except Exception as err:
            st.error(f"**Data extraction failed:** {err}")
            st.stop()

    # ── Wind shear extrapolation to hub height ─
    ws_hub, shear_factor = extrapolate_wind_speed(ws_vals, ref_height=10.0, hub_height=hub_height)
    mean_ws_10m = float(np.mean(ws_vals))
    mean_ws_hub = float(np.mean(ws_hub))

    st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)

    tab_overview, tab_power, tab_screening, tab_charts = st.tabs(
        ["Overview & KPIs", "Power Potential", "Site Screening", "Wind Charts"]
    )

    # ════════════════════════════════════════════
    # TAB 1 — Overview & KPIs
    # ════════════════════════════════════════════
    with tab_overview:
        # ── KPI Row ───────────────────────────────
        st.markdown(
            '<div class="section-label">Wind Potential KPIs — Nearest Grid Point</div>',
            unsafe_allow_html=True,
        )
        mean_ws = mean_ws_hub          # assessment uses hub-height (more realistic)
        max_ws  = float(np.max(ws_hub))
        p75_ws  = float(np.percentile(ws_hub, 75))
        calm_pct = float(np.sum(ws_vals < 3.5) / len(ws_vals) * 100)  # calm still at 10m

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(
                f'<div class="kpi-card accent-mono">'
                f'<div class="kpi-label">Mean Wind Speed @ 10 m</div>'
                f'<div class="kpi-value">{mean_ws_10m:.2f}<span class="kpi-unit">m/s</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Mean Wind Speed @ {hub_height:.0f} m hub</div>'
                f'<div class="kpi-value">{mean_ws_hub:.2f}<span class="kpi-unit">m/s</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Max Wind Speed @ Hub</div>'
                f'<div class="kpi-value">{max_ws:.2f}<span class="kpi-unit">m/s</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with kpi4:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">P75 @ Hub</div>'
                f'<div class="kpi-value">{p75_ws:.2f}<span class="kpi-unit">m/s</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with kpi5:
            st.markdown(
                f'<div class="kpi-card accent-amber">'
                f'<div class="kpi-label">Calm Hours (&lt; 3.5 m/s)</div>'
                f'<div class="kpi-value">{calm_pct:.1f}<span class="kpi-unit">%</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="info-box" style="font-size:0.82rem; margin-top:0.9rem;">'
            f'<b>Wind shear extrapolation</b> (Power Law, α = 1/7, offshore): '
            f'{mean_ws_10m:.2f} m/s at 10 m → <b>{mean_ws_hub:.2f} m/s at {hub_height:.0f} m</b> hub height '
            f'(factor {shear_factor:.3f}). Assessment and power estimation use hub-height wind speed.</div>',
            unsafe_allow_html=True,
        )

        # Potential assessment banner — based on hub-height mean WS
        if mean_ws >= 7.0:
            assessment = ('<span class="status-tag status-tag--good">Excellent Potential</span> '
                          "Mean wind speed ≥ 7 m/s at hub height is highly "
                          "favourable for commercial-scale wind power development.")
            box_cls = "success-box"
        elif mean_ws >= 5.0:
            assessment = ('<span class="status-tag status-tag--mid">Moderate Potential</span> '
                          "Mean wind speed 5–7 m/s at hub height may be "
                          "viable with modern turbines and favourable capacity factors.")
            box_cls = "warning-box"
        else:
            assessment = ('<span class="status-tag status-tag--low">Low Potential</span> '
                          "Mean wind speed < 5 m/s at hub height is generally "
                          "insufficient for economically viable large-scale wind power.")
            box_cls = "info-box"

        st.markdown(f'<div class="{box_cls}" style="margin-top:0.8rem;">{assessment}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.8rem; color:#3a6a8a; margin-bottom:0.2rem;">'
            f'Snapped to nearest grid point: '
            f'<span class="coord-badge">{snapped_lat:.4f}°, {snapped_lon:.4f}°</span></div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════
    # TAB 2 — Power Potential Estimation
    # ════════════════════════════════════════════
    with tab_power:
        st.markdown('<div class="section-label">Power Potential Estimation — POI</div>', unsafe_allow_html=True)

        # Use hub-height wind speed for power estimation (more realistic)
        power_result = estimate_power_curve(
            ws_hub,
            rotor_diameter_m=rotor_diameter,
            air_density=air_density,
            cp=cp_value,
            cut_in=cut_in_speed,
            rated_speed=rated_speed,
            cut_out=cut_out_speed,
        )

        # Determine energy label and value based on selected period mode
        if period_mode == "Monthly":
            energy_label = "Est. Energy / Month"
            # hours in the selected month across all years in dataset
            n_hours_period = len(ws_vals)
            energy_period_mwh = power_result["mean_power_kw"] * n_hours_period / 1000
            energy_value = f"{energy_period_mwh:.0f}"
            energy_unit = "MWh"
        elif period_mode == "Seasonal":
            energy_label = "Est. Energy / Season"
            n_hours_period = len(ws_vals)
            energy_period_mwh = power_result["mean_power_kw"] * n_hours_period / 1000
            energy_value = f"{energy_period_mwh:.0f}"
            energy_unit = "MWh"
        elif period_mode == "Daily":
            energy_label = "Est. Energy / Day"
            n_hours_period = len(ws_vals)
            energy_period_mwh = power_result["mean_power_kw"] * n_hours_period / 1000
            energy_value = f"{energy_period_mwh:.2f}"
            energy_unit = "MWh"
        else:
            energy_label = "Est. Energy / Year"
            energy_value = f"{power_result['annual_energy_mwh']:.0f}"
            energy_unit = "MWh"

        pw1, pw2, pw3, pw4 = st.columns(4)
        with pw1:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Mean Power Output</div>'
                f'<div class="kpi-value" style="font-size:1.7rem;">{power_result["mean_power_kw"]:.1f}'
                f'<span class="kpi-unit">kW</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with pw2:
            st.markdown(
                f'<div class="kpi-card accent-mono">'
                f'<div class="kpi-label">Rated Power</div>'
                f'<div class="kpi-value" style="font-size:1.7rem;">{power_result["rated_power_kw"]:.0f}'
                f'<span class="kpi-unit">kW</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with pw3:
            st.markdown(
                f'<div class="kpi-card accent-green">'
                f'<div class="kpi-label">Capacity Factor</div>'
                f'<div class="kpi-value" style="font-size:1.7rem;">{power_result["capacity_factor_pct"]:.1f}'
                f'<span class="kpi-unit">%</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with pw4:
            st.markdown(
                f'<div class="kpi-card accent-amber">'
                f'<div class="kpi-label">{energy_label}</div>'
                f'<div class="kpi-value" style="font-size:1.7rem;">{energy_value}'
                f'<span class="kpi-unit">{energy_unit}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="info-box" style="margin-top:0.8rem; font-size:0.82rem;">'
            f'Estimate based on hub-height ({hub_height:.0f} m) wind speed using power-law shear (α=1/7). '
            f'Rotor {rotor_diameter:.0f} m (swept area {power_result["swept_area_m2"]:.0f} m²), '
            f'ρ = {air_density:.3f} kg/m³, Cp = {cp_value:.2f}, cut-in {cut_in_speed:.1f} m/s, '
            f'rated {rated_speed:.1f} m/s, cut-out {cut_out_speed:.1f} m/s. '
            f'Turbine produces power during {power_result["pct_time_producing"]:.1f}% of the analysed period. '
            f'Annual energy is extrapolated from mean power to 8,760 hr/yr — '
            f'indicative figure only, not a substitute for a full technical feasibility study.</div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════
    # TAB 3 — Site Screening (full domain + leaderboard)
    # ════════════════════════════════════════════
    with tab_screening:
        # ── Automated Site Screening Map ─────────────────────────────────────────
        st.markdown(
            '<div class="section-label">Automated Site Screening Map — Full Domain</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box">' 
            'Every grid point in the study domain is scored and colour-coded by wind potential ' 
            'at hub height, using the turbine hub height set in the sidebar. ' 
            '<b>Gold stars</b> mark the top-10 candidate sites. ' 
            'Use the tier filter and colour metric controls to focus the view, ' 
            'then download the full grid as CSV for GIS or report use.</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Computing site screening grid for entire domain…"):
            try:
                scr_df, scr_lat_key, scr_lon_key = compute_screening_grid(
                    ws_f, lat_key, lon_key, hub_height=hub_height
                )
                scr_stats = screening_summary_stats(scr_df, scr_lat_key, scr_lon_key, hub_height)
                screening_ok = True
            except Exception as scr_err:
                st.warning(f"Site screening could not be computed: {scr_err}")
                screening_ok = False

        if screening_ok:
            # ── Domain-wide KPI banner ──────────────────
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            for col, label, val, unit in [
                (sc1, "Total Grid Points",    f"{scr_stats['total_points']:,}",        ""),
                (sc2, "Excellent Sites",      f"{scr_stats['excellent_n']:,}",          f"({scr_stats['excellent_pct']:.1f}%)"),
                (sc3, "Moderate Sites",       f"{scr_stats['moderate_n']:,}",           f"({scr_stats['moderate_pct']:.1f}%)"),
                (sc4, "Mean Wind Speed @ Hub", f"{scr_stats['domain_mean_hub']:.2f}",   "m/s"),
                (sc5, "Best Wind Speed @ Hub", f"{scr_stats['domain_max_hub']:.2f}",    "m/s"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="kpi-card">' 
                        f'<div class="kpi-label">{label}</div>' 
                        f'<div class="kpi-value" style="font-size:1.4rem;">{val}' 
                        f'<span class="kpi-unit" style="font-size:0.8rem;"> {unit}</span></div>' 
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Best site callout
            st.markdown(
                f'<div class="success-box" style="margin-top:0.5rem;">' 
                f'<b>Best candidate site</b> in domain: ' 
                f'<span class="coord-badge">{scr_stats["best_lat"]:.4f}°, {scr_stats["best_lon"]:.4f}°</span> ' 
                f'— mean Wind Speed <b>{scr_stats["best_mean_ws_hub"]:.2f} m/s</b> ' 
                f'· P75 <b>{scr_stats["best_p75_hub"]:.2f} m/s</b> at {hub_height:.0f} m hub height.</div>',
                unsafe_allow_html=True,
            )

            # ── Map controls row ───────────────────────
            ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1], gap="medium")
            with ctrl1:
                tier_filter = st.multiselect(
                    "Show tiers",
                    options=["Excellent", "Moderate", "Low"],
                    default=["Excellent", "Moderate", "Low"],
                    key="scr_tier_filter",
                )
            with ctrl2:
                _metric_keys   = ["mean_ws_hub", "p75_ws_hub", "cf_proxy_pct", "mean_ws_10m"]
                _metric_labels = [
                    f"Mean Wind Speed @ {hub_height:.0f} m hub",
                    f"P75 Wind Speed @ {hub_height:.0f} m hub",
                    "Productive Hours (%)",
                    "Mean Wind Speed @ 10 m (raw)",
                ]
                _metric_idx = st.selectbox(
                    "Colour by",
                    options=range(len(_metric_keys)),
                    format_func=lambda i: _metric_labels[i],
                    key="scr_color_metric",
                )
                color_key = _metric_keys[_metric_idx]
            with ctrl3:
                st.markdown("<br>", unsafe_allow_html=True)
                csv_bytes = screening_csv_bytes(scr_df, scr_lat_key, scr_lon_key, hub_height)
                st.download_button(
                    label="Export CSV",
                    data=csv_bytes,
                    file_name=f"site_screening_{period_label.replace(' ','_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download full screening grid with coordinates, wind statistics, and tier for every grid point.",
                )

            # ── Screening Map ──────────────────────────
            scr_fig = build_screening_map(
                scr_df, scr_lat_key, scr_lon_key,
                hub_height=hub_height,
                show_tiers=tier_filter,
                color_metric=color_key,
                poi=poi,
            )
            st.plotly_chart(scr_fig, use_container_width=True, config={"displayModeBar": True})

            # ── Tier distribution bar chart ────────────
            tier_col, hist_col = st.columns([1, 2], gap="medium")
            with tier_col:
                st.markdown(
                    '<div class="section-label" style="margin-top:0.5rem;">Tier Distribution</div>',
                    unsafe_allow_html=True,
                )
                tier_counts = scr_df["tier"].value_counts().reindex(
                    ["Excellent", "Moderate", "Low"], fill_value=0
                )
                tier_fig = go.Figure(go.Bar(
                    x=tier_counts.index.tolist(),
                    y=tier_counts.values.tolist(),
                    marker_color=["#00d4ff", "#f0a500", "#3a5a7a"],
                    text=[f"{v:,}\n({v/scr_stats['total_points']*100:.1f}%)" for v in tier_counts.values],
                    textposition="outside",
                    textfont=dict(color="#a8bfd4", size=11),
                ))
                tier_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0b1827",
                    font=dict(family="Space Grotesk, sans-serif", color="#a8bfd4"),
                    xaxis=dict(gridcolor="#1a2d45", linecolor="#1e3a5f", tickfont=dict(size=12, color="#a8bfd4")),
                    yaxis=dict(gridcolor="#1a2d45", linecolor="#1e3a5f", title="Number of Grid Points"),
                    margin=dict(l=40, r=10, t=20, b=40),
                    height=280,
                )
                st.plotly_chart(tier_fig, use_container_width=True, config={"displayModeBar": False})

            with hist_col:
                st.markdown(
                    '<div class="section-label" style="margin-top:0.5rem;">' 
                    f'Distribution of Hub-Height Wind Speed (all {scr_stats["total_points"]:,} grid points)' 
                    '</div>',
                    unsafe_allow_html=True,
                )
                hist_fig = go.Figure()
                for tier, color in [("Excellent","#00d4ff"),("Moderate","#f0a500"),("Low","#3a5a7a")]:
                    sub = scr_df[scr_df["tier"] == tier]["mean_ws_hub"]
                    if sub.empty:
                        continue
                    hist_fig.add_trace(go.Histogram(
                        x=sub, name=tier,
                        marker_color=color, opacity=0.75,
                        nbinsx=40,
                        hovertemplate=f"{tier}<br>Wind Speed: %{{x:.1f}} m/s<br>Count: %{{y}}<extra></extra>",
                    ))
                # Vertical lines for thresholds
                for v, label, color in [
                    (5.0, "Moderate threshold (5 m/s)", "#f0a500"),
                    (7.0, "Excellent threshold (7 m/s)", "#00d4ff"),
                ]:
                    hist_fig.add_vline(x=v, line_dash="dash", line_color=color, line_width=1.5,
                        annotation_text=label,
                        annotation=dict(font_size=9, font_color=color, textangle=-90))
                hist_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0b1827",
                    font=dict(family="Space Grotesk, sans-serif", color="#a8bfd4"),
                    xaxis=dict(gridcolor="#1a2d45", linecolor="#1e3a5f",
                               title=f"Mean Wind Speed @ {hub_height:.0f} m (m/s)"),
                    yaxis=dict(gridcolor="#1a2d45", linecolor="#1e3a5f", title="Grid Point Count"),
                    barmode="stack",
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                                font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=50, r=10, t=30, b=50),
                    height=280,
                )
                st.plotly_chart(hist_fig, use_container_width=True, config={"displayModeBar": False})

    
        # ── Leaderboard: Top-5 most potential locations ──
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Leaderboard — Top 5 Most Potential Grid Points</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Ranking of all grid points in the study domain by time-mean wind speed '
            '(filtered to the active period) — helps identify alternative PLTB candidate sites '
            'with higher potential than the currently selected POI. '
            'Click any coordinate below to jump your POI straight to that point.</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Computing location rankings…"):
            leaderboard = compute_leaderboard(ws_f, lat_key, lon_key, top_n=5)

        if leaderboard:
            rank_labels = ["1", "2", "3", "4", "5"]

            header_cols = st.columns([0.5, 2.0, 1.3, 1.3, 1.3, 1.1])
            for col, label in zip(
                header_cols,
                ["Rank", "Coordinates", "Mean Wind Speed (m/s)", "Max Wind Speed (m/s)", "P75 Wind Speed (m/s)", "Category"],
            ):
                col.markdown(f"<span style='font-size:0.8rem; color:#5a8aaa; text-transform:uppercase; "
                             f"letter-spacing:0.06em;'>{label}</span>", unsafe_allow_html=True)

            for i, e in enumerate(leaderboard):
                row_cols = st.columns([0.5, 2.0, 1.3, 1.3, 1.3, 1.1])
                row_cols[0].markdown(f"**{rank_labels[i]}**")

                if row_cols[1].button(
                    f"{e['lat']:.4f}°, {e['lon']:.4f}°",
                    key=f"leaderboard_jump_{i}",
                    help="Click to move your POI to this grid point",
                    use_container_width=True,
                ):
                    st.session_state["poi"] = [e["lat"], e["lon"]]
                    st.rerun()

                row_cols[2].markdown(f"{e['mean_ws']:.2f}")
                row_cols[3].markdown(f"{e['max_ws']:.2f}")
                row_cols[4].markdown(f"{e['p75_ws']:.2f}")
                # Tier based on 10 m mean WS adjusted for hub-height context:
                # 10m equivalents of hub-height thresholds at 100m (factor ~1.56 for α=1/7):
                # Excellent hub ≥7 m/s → 10m ≥4.5 m/s; Moderate hub 5-7 → 10m 3.2-4.5 m/s
                category = "Excellent" if e["mean_ws"] >= 4.5 else "Moderate" if e["mean_ws"] >= 3.2 else "Low"
                row_cols[5].markdown(category)
        else:
            st.info("Not enough data to build a leaderboard for the selected period.")

    # ════════════════════════════════════════════
    # TAB 4 — Wind Analysis Charts
    # ════════════════════════════════════════════
    with tab_charts:
        st.markdown('<div class="section-label">Wind Analysis Charts</div>', unsafe_allow_html=True)

        # Row 1: Time Series (full width)
        st.plotly_chart(
            plot_time_series(ws_vals, time_vals),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Row 2: Wind Rose + Histogram side by side
        col_rose, col_hist = st.columns(2, gap="medium")

        with col_rose:
            st.plotly_chart(
                plot_wind_rose(ws_vals, wd_vals),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_hist:
            st.plotly_chart(
                plot_histogram(ws_vals),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Raw data expander ─────────────────────
        with st.expander("Dataset Variables & Metadata", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Data Variables**")
                for var in ds.data_vars:
                    st.code(f"{var}: {ds[var].dims} — {ds[var].attrs.get('long_name', 'n/a')}", language="text")
            with col_d2:
                st.markdown("**Global Attributes**")
                for k, v_attr in list(ds.attrs.items())[:12]:
                    st.code(f"{k}: {v_attr}", language="text")

    # ── Cleanup temp files (every uploaded file) ──
    for p in tmp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass


if __name__ == "__main__":
    main()
