"""
Wind Power Plant (PLTB) Potential Assessment Dashboard
=======================================================
A Streamlit application for analyzing offshore wind data from NetCDF files
to evaluate the viability of a wind power plant at a specific location.

Tech Stack: streamlit, xarray, numpy, folium, streamlit-folium, plotly
"""

import os
import tempfile
import warnings
import numpy as np
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
    page_icon="🌬️",
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
        background-color: #0b1120;
        color: #d1dce8;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1a2d;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #a8bfd4 !important;
    }

    /* ── Page title ── */
    .dashboard-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: -0.02em;
        color: #e8f4fd;
        line-height: 1.2;
    }
    .dashboard-subtitle {
        font-size: 0.9rem;
        color: #5a8aaa;
        font-weight: 400;
        margin-top: 0.2rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .title-accent {
        color: #00d4ff;
        text-shadow: 0 0 18px rgba(0, 212, 255, 0.45);
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #0f2240 0%, #0d1a2d 100%);
        border: 1px solid #1e3a5f;
        border-top: 3px solid #00d4ff;
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #00d4ff;
        text-shadow: 0 0 12px rgba(0, 212, 255, 0.3);
        line-height: 1;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #5a8aaa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.5rem;
    }
    .kpi-unit {
        font-size: 0.9rem;
        color: #7baac8;
        margin-left: 3px;
    }

    /* ── Section headers ── */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #00d4ff;
        font-weight: 600;
        margin-bottom: 0.4rem;
        border-left: 3px solid #00d4ff;
        padding-left: 0.6rem;
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
        border-color: #1e3a5f;
        margin: 1.2rem 0;
    }

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

def _find_var(ds: xr.Dataset, candidates):
    """Case-insensitive lookup of the first matching variable name in ds."""
    lower_vars = {v.lower(): v for v in ds.data_vars}
    return next((lower_vars[k] for k in candidates if k in lower_vars), None)


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


def compute_wind(ds_u: xr.Dataset, ds_v: xr.Dataset = None):
    """
    Detect U/V variable names (case-insensitive), compute wind speed (m/s)
    and meteorological wind direction (degrees from North, clockwise).

    Accepts either:
    • a single dataset that already contains both U and V (ds_v=None), or
    • two separate datasets — one with the U component, one with the V
      component — which are aligned/merged on their shared coordinates.

    Whatever the original time dimension is called (e.g. 'time' or the
    newer ERA5/CDS 'valid_time'), it is standardised to 'time' on the
    returned arrays so the rest of the dashboard can rely on `ds.time`.

    Returns (u, v, ws, wd) as DataArrays with a unified 'time' dimension.
    """
    u_candidates = ["u10", "u", "u_component_of_wind", "uwnd"]
    v_candidates = ["v10", "v", "v_component_of_wind", "vwnd"]

    if ds_v is None:
        # Backward-compatible single-file path
        u_key = _find_var(ds_u, u_candidates)
        v_key = _find_var(ds_u, v_candidates)

        if u_key is None or v_key is None:
            raise KeyError(
                f"Could not find U and V wind components. Available variables: {list(ds_u.data_vars)}"
            )

        u = ds_u[u_key]
        v = ds_u[v_key]
    else:
        # Two-file path: U comes from ds_u, V comes from ds_v
        u_key = _find_var(ds_u, u_candidates)
        v_key = _find_var(ds_v, v_candidates)

        if u_key is None:
            raise KeyError(
                f"Could not find a U wind component in the U file. Available variables: {list(ds_u.data_vars)}"
            )
        if v_key is None:
            raise KeyError(
                f"Could not find a V wind component in the V file. Available variables: {list(ds_v.data_vars)}"
            )

        u = ds_u[u_key]
        v = ds_v[v_key]

        # Standardise each one's time dimension name to 'time' BEFORE
        # aligning, so two files using different time-dimension names
        # (e.g. one 'time', one 'valid_time') can still be matched.
        u_time = _find_time_dim(u)
        v_time = _find_time_dim(v)
        if u_time and u_time != "time":
            u = u.rename({u_time: "time"})
        if v_time and v_time != "time":
            v = v.rename({v_time: "time"})

        # Align the two arrays onto their common coordinates (time/lat/lon)
        # so that mismatched-but-overlapping grids still work, and so a
        # clear error is raised if the files don't actually overlap.
        u, v = xr.align(u, v, join="inner")
        if u.size == 0 or v.size == 0:
            raise ValueError(
                "The U and V files do not share overlapping coordinates "
                "(time/lat/lon). Please check that both files cover the "
                "same domain and period."
            )

    # Standardise the time dimension name on the final arrays too,
    # covering the single-file path and any case missed above.
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
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>WS: %{y:.2f} m/s<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=time_vals, y=ws_smooth,
        mode="lines",
        name="24-hr Rolling Avg",
        line=dict(color=ACCENT2, width=2),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Avg WS: %{y:.2f} m/s<extra></extra>",
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
        hovertemplate="WS: %{x:.1f} m/s<br>Density: %{y:.4f}<extra></extra>",
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
            hovertemplate="WS: %{x:.1f} m/s<br>PDF: %{y:.4f}<extra></extra>",
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
        hovertemplate="Lat: %{y:.2f}°<br>Lon: %{x:.2f}°<br>WS: %{z:.2f} m/s<extra></extra>",
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
            hovertemplate="Lat: %{y:.2f}°<br>Lon: %{x:.2f}°<br>WS: %{customdata[0]:.2f} m/s<extra></extra>",
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
        <div style="margin-bottom:0.5rem;">
            <div class="dashboard-title">Wind Energy <span class="title-accent">Assessment</span></div>
            <div class="dashboard-subtitle">Offshore Wind Power Plant Potential · NetCDF Analysis Suite</div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Data Input")
        st.markdown(
            '<div class="info-box">Upload two NetCDF files — one containing the U (eastward) '
            'wind component, one containing the V (northward) component. Both stay loaded '
            'independently and are combined automatically. If a single file already contains '
            'both U and V, you can upload it in either slot and leave the other empty.</div>',
            unsafe_allow_html=True,
        )
        u_file = st.file_uploader(
            "Upload U-component file (.nc)",
            type=["nc", "nc4", "netcdf"],
            help="ERA5 or similar reanalysis / model output with the U (eastward) wind component.",
            key="u_file_uploader",
        )
        v_file = st.file_uploader(
            "Upload V-component file (.nc)",
            type=["nc", "nc4", "netcdf"],
            help="ERA5 or similar reanalysis / model output with the V (northward) wind component. "
                 "Optional if your U file already contains both components.",
            key="v_file_uploader",
        )

        st.divider()
        st.markdown("### 🔧 Turbine Parameters")
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

        st.divider()
        st.markdown("### 📌 How to use")
        st.markdown(
            """
            1. **Upload** the U-component `.nc` file.  
            2. **Upload** the V-component `.nc` file (both stay loaded together).  
            3. **Select** a period mode: All Data, Monthly, Seasonal, or Daily.  
            4. **Inspect** the cyan boundary on the map — your data domain.  
            5. **Click** anywhere inside the boundary to set your POI.  
            6. **Read** the charts, KPIs, leaderboard, and power potential estimates.
            """
        )
        st.divider()
        st.caption("Wind Energy Assessment · v1.3 · Built with Python")

    # ── No file uploaded — landing state ──────
    if u_file is None and v_file is None:
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.markdown(
                """
                <div style="text-align:center; padding: 4rem 0;">
                    <div style="font-size:5rem;">🌊</div>
                    <div style="font-size:1.3rem; color:#5a8aaa; margin-top:1rem;">
                        Upload a NetCDF wind file to begin
                    </div>
                    <div style="font-size:0.85rem; color:#2a4a6a; margin-top:0.5rem;">
                        ERA5, CMEMS, WRF, or any dataset with U/V components
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # ── Require at least the U file (or a combined file) ──
    if u_file is None:
        st.warning("Please upload at least the U-component file to continue (V-only is not enough).")
        st.stop()

    # ── Load & parse dataset(s) ───────────────
    tmp_paths = []
    with st.spinner("Reading NetCDF file(s)…"):
        try:
            # Streamlit's UploadedFile is a BytesIO object; xarray needs a real path.
            # Each uploaded file gets its own temp path so the U and V files never
            # overwrite one another — both stay independently readable.
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_u:
                tmp_u.write(u_file.getvalue())
                u_tmp_path = tmp_u.name
            tmp_paths.append(u_tmp_path)

            ds_u = xr.open_dataset(u_tmp_path, engine="netcdf4")

            if v_file is not None:
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_v:
                    tmp_v.write(v_file.getvalue())
                    v_tmp_path = tmp_v.name
                tmp_paths.append(v_tmp_path)

                ds_v = xr.open_dataset(v_tmp_path, engine="netcdf4")
                u, v, ws, wd = compute_wind(ds_u, ds_v)
                src_attrs = {**ds_v.attrs, **ds_u.attrs}
            else:
                # Single file already contains both U and V
                u, v, ws, wd = compute_wind(ds_u)
                src_attrs = ds_u.attrs

            # Build the combined dataset from u/v themselves (not the raw
            # ds_u/ds_v) so the standardised 'time' dimension from
            # compute_wind is always what the rest of the dashboard sees —
            # even when the source file calls it 'valid_time' or similar.
            ds = xr.Dataset({u.name or "u": u, v.name or "v": v})
            if not ds.attrs:
                ds.attrs = src_attrs

            lat_key, lon_key, lat_min, lat_max, lon_min, lon_max = get_lat_lon_bounds(ds)

        except Exception as err:
            st.error(f"**Failed to load dataset:** {err}")
            st.stop()

    file_label = u_file.name if v_file is None else f"{u_file.name} + {v_file.name}"
    st.markdown(
        f'<div class="success-box">✅ File loaded — <b>{file_label}</b> · '
        f'{len(ds.time)} time steps · '
        f'Lat {lat_min:.2f}°–{lat_max:.2f}° · Lon {lon_min:.2f}°–{lon_max:.2f}°</div>',
        unsafe_allow_html=True,
    )

    # ── Period filter: All Data / Monthly / Seasonal / Daily ──
    st.markdown('<div class="section-label">Data Period Filter</div>', unsafe_allow_html=True)
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
            available_months = sorted(set(int(m) for m in ds["time"].dt.month.values))
            month_options = [MONTH_NAMES[m - 1] for m in available_months]
            chosen_label = st.selectbox("Select month", options=month_options, key="month_select")
            selected_month = available_months[month_options.index(chosen_label)]
        elif period_mode == "Seasonal":
            available_seasons = list(SEASON_MONTHS.keys())
            selected_season = st.selectbox("Select season", options=available_seasons, key="season_select")

    with pf_col3:
        if period_mode == "Daily" and selected_month is not None:
            # Find available days for the selected month
            month_mask = ds["time"].dt.month.values == selected_month
            available_days = sorted(set(int(d) for d in ds["time"].dt.day.values[month_mask]))
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
        st.caption(f"📅 Showing {len(ws_f['time'])} of {len(ws['time'])} time steps — {period_label}.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Layout: map + right panel ─────────────
    col_map, col_info = st.columns([3, 1], gap="medium")

    with col_map:
        st.markdown('<div class="section-label">Interactive Data Domain Map — click to set POI</div>', unsafe_allow_html=True)

        # Retrieve POI from previous interaction (session state)
        poi = st.session_state.get("poi", None)
        folium_map = build_folium_map(lat_min, lat_max, lon_min, lon_max, poi=poi)

        map_output = st_folium(
            folium_map,
            use_container_width=True,
            height=480,
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
                    '<div class="warning-box">⚠️ Clicked outside the data domain. '
                    'Please click within the cyan boundary.</div>',
                    unsafe_allow_html=True,
                )

    with col_info:
        st.markdown('<div class="section-label">Dataset Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:0.6rem;">
            <div class="kpi-label">Time Steps (Terfilter)</div>
            <div class="kpi-value" style="font-size:1.6rem;">{len(ws_f['time'])}</div>
        </div>
        <div class="kpi-card" style="margin-bottom:0.6rem;">
            <div class="kpi-label">Lat Range</div>
            <div class="kpi-value" style="font-size:1.1rem;">{lat_min:.2f}° – {lat_max:.2f}°</div>
        </div>
        <div class="kpi-card" style="margin-bottom:0.6rem;">
            <div class="kpi-label">Lon Range</div>
            <div class="kpi-value" style="font-size:1.1rem;">{lon_min:.2f}° – {lon_max:.2f}°</div>
        </div>
        """, unsafe_allow_html=True)

        if poi:
            st.markdown('<div class="section-label" style="margin-top:1rem;">Selected POI</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="coord-badge">📍 {poi[0]:.4f}°, {poi[1]:.4f}°</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="info-box" style="margin-top:1rem;">Click on the map inside the cyan boundary to select a Point of Interest.</div>',
                unsafe_allow_html=True,
            )

    # ── Analysis section (only if POI is set) ─
    if not poi:
        st.info("Select a Point of Interest on the map to view wind analysis.")
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

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────
    st.markdown('<div class="section-label">Wind Potential KPIs — Nearest Grid Point</div>', unsafe_allow_html=True)
    mean_ws = float(np.mean(ws_vals))
    max_ws = float(np.max(ws_vals))
    p75_ws = float(np.percentile(ws_vals, 75))
    calm_pct = float(np.sum(ws_vals < 3.5) / len(ws_vals) * 100)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Mean Wind Speed</div>'
            f'<div class="kpi-value">{mean_ws:.2f}<span class="kpi-unit">m/s</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Max Wind Speed</div>'
            f'<div class="kpi-value">{max_ws:.2f}<span class="kpi-unit">m/s</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">75th Percentile</div>'
            f'<div class="kpi-value">{p75_ws:.2f}<span class="kpi-unit">m/s</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Calm Hours ( < 3.5 m/s)</div>'
            f'<div class="kpi-value">{calm_pct:.1f}<span class="kpi-unit">%</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Potential assessment banner
    if mean_ws >= 7.0:
        assessment = ("🟢 <b>Excellent potential</b> — Mean wind speed ≥ 7 m/s is highly favourable "
                      "for commercial-scale wind power development.")
        box_cls = "success-box"
    elif mean_ws >= 5.0:
        assessment = ("🟡 <b>Moderate potential</b> — Mean wind speed 5–7 m/s may be viable with "
                      "modern turbines and favourable capacity factors.")
        box_cls = "warning-box"
    else:
        assessment = ("🔴 <b>Low potential</b> — Mean wind speed < 5 m/s is generally insufficient "
                      "for economically viable large-scale wind power.")
        box_cls = "info-box"

    st.markdown(f'<div class="{box_cls}" style="margin-top:0.8rem;">{assessment}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.8rem; color:#3a6a8a; margin-bottom:1rem;">'
        f'Snapped to nearest grid point: '
        f'<span class="coord-badge">{snapped_lat:.4f}°, {snapped_lon:.4f}°</span></div>',
        unsafe_allow_html=True,
    )

    # ── Power Potential Estimation at POI ──────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Power Potential Estimation — POI</div>', unsafe_allow_html=True)

    power_result = estimate_power_curve(
        ws_vals,
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
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Rated Power</div>'
            f'<div class="kpi-value" style="font-size:1.7rem;">{power_result["rated_power_kw"]:.0f}'
            f'<span class="kpi-unit">kW</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with pw3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Capacity Factor</div>'
            f'<div class="kpi-value" style="font-size:1.7rem;">{power_result["capacity_factor_pct"]:.1f}'
            f'<span class="kpi-unit">%</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with pw4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{energy_label}</div>'
            f'<div class="kpi-value" style="font-size:1.7rem;">{energy_value}'
            f'<span class="kpi-unit">{energy_unit}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="info-box" style="margin-top:0.8rem; font-size:0.82rem;">'
        f'Estimate based on a simplified turbine power curve: rotor {rotor_diameter:.0f} m '
        f'(swept area {power_result["swept_area_m2"]:.0f} m²), ρ = {air_density:.3f} kg/m³, '
        f'Cp = {cp_value:.2f}, cut-in {cut_in_speed:.1f} m/s, rated {rated_speed:.1f} m/s, '
        f'cut-out {cut_out_speed:.1f} m/s. Turbine produces power during '
        f'{power_result["pct_time_producing"]:.1f}% of the analysed period. '
        f'Annual energy is extrapolated from mean power over the filtered period to 8,760 hr/yr — '
        f'indicative figure only, not a substitute for a full technical feasibility study.</div>',
        unsafe_allow_html=True,
    )

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
            ["Rank", "Coordinates", "Mean WS (m/s)", "Max WS (m/s)", "P75 WS (m/s)", "Category"],
        ):
            col.markdown(f"<span style='font-size:0.8rem; color:#5a8aaa; text-transform:uppercase; "
                         f"letter-spacing:0.06em;'>{label}</span>", unsafe_allow_html=True)

        for i, e in enumerate(leaderboard):
            row_cols = st.columns([0.5, 2.0, 1.3, 1.3, 1.3, 1.1])
            row_cols[0].markdown(f"**{rank_labels[i]}**")

            if row_cols[1].button(
                f"📍 {e['lat']:.4f}°, {e['lon']:.4f}°",
                key=f"leaderboard_jump_{i}",
                help="Click to move your POI to this grid point",
                use_container_width=True,
            ):
                st.session_state["poi"] = [e["lat"], e["lon"]]
                st.rerun()

            row_cols[2].markdown(f"{e['mean_ws']:.2f}")
            row_cols[3].markdown(f"{e['max_ws']:.2f}")
            row_cols[4].markdown(f"{e['p75_ws']:.2f}")
            category = "Excellent" if e["mean_ws"] >= 7.0 else "Moderate" if e["mean_ws"] >= 5.0 else "Low"
            row_cols[5].markdown(category)
    else:
        st.info("Not enough data to build a leaderboard for the selected period.")


    # ── Charts ────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
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
    with st.expander("🗂️ Dataset Variables & Metadata", expanded=False):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Data Variables**")
            for var in ds.data_vars:
                st.code(f"{var}: {ds[var].dims} — {ds[var].attrs.get('long_name', 'n/a')}", language="text")
        with col_d2:
            st.markdown("**Global Attributes**")
            for k, v_attr in list(ds.attrs.items())[:12]:
                st.code(f"{k}: {v_attr}", language="text")

    # ── Cleanup temp files (both U and V, if present) ──
    for p in tmp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass


if __name__ == "__main__":
    main()
