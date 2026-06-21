# Wind Energy Assessment Dashboard

A Streamlit dashboard for analyzing offshore wind data from NetCDF files to evaluate the viability of a wind power plant (PLTB) at a specific location.

Upload any number of `.nc` files — the app auto-detects whether each file contains the U (eastward) wind component, the V (northward) component, or both, so files can be split by time, by component, or combined in any combination. It's not limited to ERA5: CMEMS, WRF, NOAA/NCEP, and other CF-style NetCDF sources are supported, including every `.nc` format version (classic NETCDF3, 64-bit offset, NETCDF4/HDF5).

**Live demo:** [windenergyassessment.streamlit.app](https://windenergyassessment.streamlit.app)

## Features

- Interactive map for selecting a Point of Interest (POI) inside the data domain
- Wind analysis charts — time series, wind rose, speed distribution with Weibull fit
- Power curve estimator based on configurable turbine parameters (rotor diameter, air density, power coefficient, cut-in/rated/cut-out speeds, hub height)
- Full-domain site screening with a Top-5 leaderboard of the most promising grid points
- CSV export of the site screening grid for GIS or report use

## Requirements

- Python 3.9+
- The packages listed in [`requirements.txt`](./requirements.txt)

## Running locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/alrizqi06/namundemituhan-app.git
   cd namundemituhan-app
   ```

2. **(Recommended) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run Dashboard_ZIKIR.py
   ```

   Streamlit will open the dashboard in your browser at `http://localhost:8501`.

5. **Upload data** — drag and drop one or more `.nc` wind data files into the sidebar uploader to begin the analysis.

> **Large files:** Streamlit's default per-file upload limit is 200 MB. To raise it, add a `.streamlit/config.toml` file with:
> ```toml
> [server]
> maxUploadSize = 1000
> ```

## Deploying your own copy

This app is built for [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Fork or push this repository to your own GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, branch, and set the main file path to `Dashboard_ZIKIR.py`.
4. Click **Deploy**.

## Project structure

```
namundemituhan-app/
├── Dashboard_ZIKIR.py   # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md
```

## Tech stack

- [Streamlit](https://streamlit.io/) — dashboard UI
- [xarray](https://docs.xarray.dev/) + [netCDF4](https://unidata.github.io/netcdf4-python/) / [h5netcdf](https://github.com/h5netcdf/h5netcdf) — NetCDF reading
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data wrangling
- [SciPy](https://scipy.org/) — Weibull distribution fitting
- [Plotly](https://plotly.com/python/) — interactive charts
- [Folium](https://python-visualization.github.io/folium/) + [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) — interactive map

## Developers

- M. Ravi Alrizqi Permana (12923026)
- Kevin Aulia Aryasena (12923047)
- Zaid Ahmad Shadiq (12923061)

## Contact

For questions or feedback: **12923047@mahasiswa.itb.ac.id**
