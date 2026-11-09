### --- IMPORTS
from __future__ import annotations

# general
from pathlib import Path

# file paths
repo_dir = Path(__file__).resolve().parent.parent.parent

data_dir = repo_dir / "data"
figures_dir = repo_dir / "figures"

# Bora Bora case study
SCALE_M = 10
CS_AOI_SIDE_M = 12_000  # case-study AOI: square, side length in metres
BORA_BORA_COORDS = [-151.74041533691084, -16.50219901417235]

# Sentinel-2 satellite imagery composite
S2_COMPOSITE_DATE_START = "2019-06-01"
S2_COMPOSITE_DATE_END = "2021-12-31"
CLOUDY_PIXEL_PERCENTAGE_THRESHOLD = 35  # warning: arbitrary threshold!

S2_VIS_RGB = {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.12, "gamma": 1.1}

# Allen Coral Atlas
ACA_CODE_TO_CSP_NAME = {
    11: "Sand",
    12: "Rubble",
    13: "Rock",
    14: "Seagrass Dense",  # ACA: single "Seagrass" – using 'dense' colour
    15: "Coral/Algae",
    18: "Microalgal Mats",
}
ACA_CODES = list(ACA_CODE_TO_CSP_NAME.keys())
