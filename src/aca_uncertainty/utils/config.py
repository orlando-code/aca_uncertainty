### --- IMPORTS
from __future__ import annotations

from pathlib import Path

### --- PATHS
repo_dir = Path(__file__).resolve().parent.parent.parent

data_dir = repo_dir / "data"
figures_dir = repo_dir / "figures"

### --- CASE STUDY (Bora Bora)
SCALE_M = 10
CS_AOI_SIDE_M = 12_500  # case-study AOI: square, side length in metres
BORA_BORA_COORDS = [-151.74041533691084, -16.50219901417235]

### --- SENTINEL-2
S2_COMPOSITE_DATE_START = "2018-01-01"
S2_COMPOSITE_DATE_END = "2020-12-31"
CLOUDY_PIXEL_PERCENTAGE_THRESHOLD = 35  # warning: arbitrary threshold!

S2_VIS_RGB = {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.12, "gamma": 1.1}

S2_BANDS = ["B2", "B3", "B4", "B5", "B8"]
S2_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

# SCL (scene classification); see Sentinel-2 L2A docs
SCL_CLOUD_SHADOW = 3
SCL_CLOUD_CLASSES = (8, 9, 10)  # medium probability, high probability, thin cirrus


### --- ACA BENTHIC (codes and RF class names)
ACA_CODE_TO_CSP_NAME = {
    11: "Sand",
    12: "Rubble",
    13: "Rock",
    14: "Seagrass Dense",  # ACA: single "Seagrass" – using 'dense' colour
    15: "Coral/Algae",
    18: "Microalgal Mats",
}
ACA_CODES = list(ACA_CODE_TO_CSP_NAME.keys())

# ACA benthic code -> alternate calibration `Class_name` spellings in the CSP shapefile.
ACA_CLASS_NAME_VARIANTS: dict[int, tuple[str, ...]] = {
    14: ("Seagrass Dense", "Seagrass Sparse", "Seagrass"),
}


### --- CSP BENTHIC RF (Allen Coral Atlas Lyons et al. GEE assets: https://github.com/CoralMapping/gee-mapping-source/tree/master/global_reefs2/central_south_pacific) inferred from https://github.com/CoralMapping/gee-mapping-source/blob/master/global_reefs_modules/reef_params
CSP_ASSET_PREFIX = "projects/coral_atlas/central_south_pacific"  # still to be updated for access (uses S-2 data as fallback in meantime)
CSP_PIXELS = [f"{CSP_ASSET_PREFIX}/in_out/csp_pixels"]
CSP_SEGMENTS = [f"{CSP_ASSET_PREFIX}/in_out/csp_segmentation"]
CSP_LOW_TIDE_IMAGE = [
    f"{CSP_ASSET_PREFIX}/image/central_south_pacific_reef_low_tide_normalized_sr_august2018_oct2020_v2_shift_mosaic",
]
CSP_BENTHIC_TRAIN_LIBRARY = f"{CSP_ASSET_PREFIX}/training/csp_benthic_training"
# Reserved if switching from local calibration polygons to global train/val libraries.
CSP_BENTHIC_VAL_LIBRARY = f"{CSP_ASSET_PREFIX}/training/csp_benthic_validation"
CSP_EXPORT_SCALE_M = 5  # Planet Dove native
CSP_RF_CLASS_FIELD = "class_num"
CSP_BENTHIC_INPUT_BANDS = [
    "b1_p",
    "b2_p",
    "b3_p",
    "b4_p",
    "b1_savg_p",
    "b2_savg_p",
    "depth_stdDev_p",
]
CSP_RF_TREES = 20
CSP_RF_MIN_LEAF = 5
CSP_RF_SEED = 42
CSP_TRAIN_FRACTION = 0.8
# `auto` -> Dove CSP stack if readable; else S2 proxy (for acauncertainty project)
CSP_PIXSEGS_MODE = "auto"  # auto | csp | s2_proxy
# Unused in case-study workflow; kept for parity with ACA CSP asset list.
CSP_EXTENT_MASK = f"{CSP_ASSET_PREFIX}/in_out/csp_extent"
CSP_NOTREEF_MASK = f"{CSP_ASSET_PREFIX}/in_out/csp_notreef"
CSP_COAST_MASK = f"{CSP_ASSET_PREFIX}/in_out/csp_Coast_mask"
CSP_RF_PROBABILITIES = True  # MULTIPROBABILITY + Shannon entropy map

### --- TRAFFIC-LIGHT META CONFIDENCE (c_gt weights; 1 = certain, 0 = uncertain)
CONFIDENCE_C_AGREE = 1.0
CONFIDENCE_C_DISAGREE = 0.0
CONFIDENCE_C_NO_ACA = 0.0
