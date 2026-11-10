# aca_uncertainty

Exploring spatial uncertainty of the Allen Coral Atlas. Currently a **work in progress**.

## Setup

Use a conda environment with [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install) installed (for example the `gee` env), then install this package in editable mode:

```bash
pip install -e .
```

Authenticate once with `ee.Authenticate()` before calling `ee.Initialize()`.

N.B. uses `geemap` to interactively visualise Google Earth Engine (GEE)-style maps. This is pretty janky for Jupyter notebooks (returns an empty object), and sometimes requires quitting the notebook and reopening. Also requires declaring

```python
os.environ["USE_FOLIUM"] = "1"
```

before running code to avoid a `basemaps` **/** `xyz_to_folium name collision.`

## Package layout

| Path                                                                                                                                 | Role                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| `utils/config.py`                                                                                                                    | Case-study constants, GEE CSP asset IDs, S2 band list     |
| `utils/ee.py`                                                                                                                        | `clip01`, `reproject_to_grid`, `count_fc`, ...            |
| `confidence/aca.py`                                                                                                                  | ACA asset, calibration fragments, traffic-light agreement |
| `confidence/rf.py`                                                                                                                   | CSP pixel/segment RF workflow                             |
| `confidence/stack.py`                                                                                                                | Multi-component spatial confidence index                  |
| `confidence/spectra.py`, `proximity.py`                                                                                              | Spectral similarity & training geometry                   |
| `geo/aoi.py`, `geo/s2.py`                                                                                                            | AOI helpers & Sentinel-2 composites                       |
| `plotting/`                                                                                                                          | Palettes, map export, figure helpers                      |
| Local calibration GeoJSON/Excel belongs under `data/` (gitignored; `data/.gitkeep` keeps the folder). Figures default to `figures/`. |                                                           |
