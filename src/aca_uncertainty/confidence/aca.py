"""Allen Coral Atlas vs region calibration comparison."""

### --- IMPORTS
from __future__ import annotations

# general
from typing import Literal

# spatial
import ee
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

# custom
from aca_uncertainty.geo.aoi import fragments_to_ee
from aca_uncertainty.plotting import config as plot_config
from aca_uncertainty.plotting.utils import opaque_fill as plot_utils_opaque_fill
from aca_uncertainty.utils import config
from aca_uncertainty.utils import ee as ee_utils

### --- CONSTANTS
ACA_ASSET = "ACA/reef_habitat/v2_0"
BENTHIC_BAND = "benthic"
ACA_SCALE_M = (
    config.CSP_EXPORT_SCALE_M
)  # TODO: replace with 3.7m? Will need to check original Planet data

AgreementStatus = Literal["agree", "disagree", "no_aca"]


### --- HELPERS


def remap_benthic_codes(img: ee.Image, *, band: str = BENTHIC_BAND) -> ee.Image:
    """Remap sparse ACA benthic codes to 1...n for ACA benthic palette display."""
    return (
        img.select(band)
        .remap(config.ACA_CODES, plot_config.aca_benthic_remap_to())
        .rename(band)
    )


def highlight_only(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    """Keep mismatches and ACA gaps (drop agreeing fragments from the map)."""
    return fc.filter(ee.Filter.neq("aca_status", "agree"))


### --- TRAFFIC LIGHT HELPERS
def _traffic_light_style_dict(
    status: AgreementStatus,
    *,
    base_fill: str,
    opaque_fill: str,
) -> dict:
    """
    Return formatting dictionary for traffic light style by status (agree, disagree, no_aca).
    """
    base_fill = str(base_fill or plot_config._missing)
    opaque_fill = str(opaque_fill or base_fill[:6] + "ff")
    if status == "no_aca":
        return {"color": "000000", "width": 8, "fillColor": "00000000"}
    if status == "disagree":
        return {"color": "ff0000", "width": 8, "fillColor": opaque_fill}
    return {"color": "00000000", "width": 1, "fillColor": "00000000"}


def _client_traffic_light_style(props: dict) -> dict:
    """
    Return formatting dictionary for traffic light style by client-side feature properties (class_name, opaqueFill, aca_status).
    """
    style0 = props.get("style") or {}
    base_fill = style0.get("fillColor", plot_config._missing)
    class_name = props.get("Class_name")
    fill_opaque = props.get("opaqueFill")
    if not fill_opaque and class_name:
        fill_opaque = plot_utils_opaque_fill(class_name)
    if not fill_opaque:
        fill_opaque = str(base_fill)[:6] + "ff"
    status = props.get("aca_status", "agree")
    if status not in ("agree", "disagree", "no_aca"):
        status = "agree"
    return _traffic_light_style_dict(
        status, base_fill=base_fill, opaque_fill=fill_opaque
    )


def traffic_light_collection(
    cs_fc: ee.FeatureCollection,
    aca_benthic: ee.Image,
    scale_m: int = ACA_SCALE_M,
    *,
    max_features: int | None = None,
) -> ee.FeatureCollection:
    """Per-fragment agreement; one batched getInfo for aca_status."""
    fc_status = cs_fc.map(
        lambda f: _aca_status_feature(f, aca_benthic, scale_m),
    )
    features = fc_status.getInfo().get("features", [])
    if max_features is not None:
        features = features[:max_features]

    styled = []
    for feat in features:
        props = dict(feat.get("properties") or {})
        props["tl_style"] = _client_traffic_light_style(props)
        styled.append(ee.Feature(ee.Geometry(feat["geometry"]), props))
    return ee.FeatureCollection(styled)


def traffic_light_from_gdf(
    gdf: gpd.GeoDataFrame,
    aoi: BaseGeometry,
    aca_benthic: ee.Image,
    *,
    scale_m: int = ACA_SCALE_M,
    min_area_m2: float = 50.0,
    max_features: int | None = None,
) -> tuple[gpd.GeoDataFrame, ee.FeatureCollection]:
    """Build exploded fragments + styled EE collection for maps."""
    frag = ee_utils.calibration_fragments_gdf(gdf, aoi, min_area_m2=min_area_m2)
    cs_fc = fragments_to_ee(frag)
    cs_tl = traffic_light_collection(
        cs_fc, aca_benthic, scale_m, max_features=max_features
    )
    return frag, cs_tl


def _aca_status_feature(f, aca_benthic: ee.Image, scale_m: int):
    """Server-side aca_status from ACA mode vs region class_num."""
    f = ee.Feature(f)
    geom = f.geometry()
    ref = ee.Number(f.get("class_num"))
    aca = aca_benthic.select(BENTHIC_BAND)
    aca_mapped = aca.updateMask(aca.neq(0))
    params = {
        "scale": scale_m,
        "maxPixels": int(1e9),
        "bestEffort": True,
        "tileScale": 4,
    }
    valid_n = ee.Number(
        aca_mapped.reduceRegion(ee.Reducer.count(), geom, **params).get(BENTHIC_BAND)
    )
    mode = ee.Number(
        aca_mapped.reduceRegion(ee.Reducer.mode(), geom, **params).get(BENTHIC_BAND)
    )
    no_aca = ee.Algorithms.If(valid_n, valid_n.eq(0), True)
    disagree = ee.Algorithms.If(no_aca, False, mode.neq(ref))
    status = ee.String(
        ee.Algorithms.If(
            no_aca, "no_aca", ee.Algorithms.If(disagree, "disagree", "agree")
        )
    )
    return f.set("aca_status", status)


### --- CLASSIFY FRAGMENTS INTO TRAFFIC LIGHT GROUPINGS
def classify_fragment_aca(
    aca_benthic: ee.Image,
    geometry: ee.Geometry,
    class_num: int,
    *,
    scale_m: int = ACA_SCALE_M,
) -> AgreementStatus:
    """Dominant mapped ACA class under one polygon fragment vs region class_num."""
    aca = aca_benthic.select(BENTHIC_BAND)
    aca_mapped = aca.updateMask(aca.neq(0))

    valid_info = aca_mapped.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        scale=scale_m,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    valid_n = (valid_info or {}).get(BENTHIC_BAND, 0) or 0
    if valid_n == 0:
        return "no_aca"

    mode_info = aca_mapped.reduceRegion(
        reducer=ee.Reducer.mode(),
        geometry=geometry,
        scale=scale_m,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    aca_mode = (mode_info or {}).get(BENTHIC_BAND)
    if aca_mode is None:
        return "no_aca"
    if int(aca_mode) != int(class_num):
        return "disagree"
    return "agree"


### --- AGREEMENT HELPERS


def region_class_raster(
    fragments_fc: ee.FeatureCollection,
    aoi: ee.Geometry,
    *,
    scale_m: int = ACA_SCALE_M,
) -> ee.Image:
    """Paint region class_num values (0 outside regions)."""
    return (
        ee.Image(0)
        .byte()
        .paint(fragments_fc, "class_num")
        .rename("region_class")
        .clip(aoi)
        .reproject(crs="EPSG:4326", scale=scale_m)
    )


### --- CALCULATE AGREEMENT SUMMARIES


def summarize_statuses(fc: ee.FeatureCollection) -> dict[str, int]:
    """Count agree / disagree / no_aca after traffic_light_collection."""
    counts: dict[str, int] = {"agree": 0, "disagree": 0, "no_aca": 0}
    for feat in fc.getInfo()["features"]:
        status = (feat.get("properties") or {}).get("aca_status", "agree")
        counts[status] = counts.get(status, 0) + 1
    return counts


def agreement_rasters(
    gdf: gpd.GeoDataFrame,
    aoi: BaseGeometry,
    aoi_ee: ee.Geometry,
    aca_benthic: ee.Image,
    *,
    scale_m: int = ACA_SCALE_M,
    min_area_m2: float = 50.0,
) -> tuple[ee.Image, ee.Image, ee.Image]:
    """Pixel-level agreement masks (scale_m) in AOI; return (disagree, no_aca, agree)."""
    frag = ee_utils.calibration_fragments_gdf(gdf, aoi, min_area_m2=min_area_m2)
    frag_fc = fragments_to_ee(frag)
    region = region_class_raster(frag_fc, aoi_ee, scale_m=scale_m)
    aca = (
        aca_benthic.select(BENTHIC_BAND)
        .clip(aoi_ee)
        .reproject(crs="EPSG:4326", scale=scale_m)
    )
    in_region = region.gt(0)
    # Match fragment logic: mapped ACA = benthic > 0 (0 and masked -> no ACA)
    has_aca = aca.gt(0)
    no_aca = in_region.And(has_aca.Not().unmask(1)).rename("no_aca")
    agree = in_region.And(has_aca).And(region.eq(aca)).rename("agree")
    disagree = in_region.And(has_aca).And(region.neq(aca)).rename("disagree")
    return disagree.selfMask(), no_aca.selfMask(), agree.selfMask()


def agreement_area_by_fragment_m2(
    frag_gdf: gpd.GeoDataFrame,
    cs_tl: ee.FeatureCollection,
) -> dict[str, float]:
    """Sum fragment areas (m^2) by aca_status — matches fragment agreement counts."""
    status_by_id: dict[int, str] = {}
    for feat in cs_tl.getInfo().get("features", []):
        props = feat.get("properties") or {}
        fid = props.get("fragment_id")
        if fid is not None:
            status_by_id[int(fid)] = props.get("aca_status", "agree")
    areas = {"agree": 0.0, "disagree": 0.0, "no_aca": 0.0}
    for _, row in frag_gdf.iterrows():
        fid = int(row["fragment_id"])
        status = status_by_id.get(fid, "agree")
        areas[status] = areas.get(status, 0.0) + float(row["area_m2"])
    areas["total_region"] = sum(areas[k] for k in ("agree", "disagree", "no_aca"))
    return areas


def agreement_area_m2(
    gdf: gpd.GeoDataFrame,
    aoi: BaseGeometry,
    aoi_ee: ee.Geometry,
    aca_benthic: ee.Image,
    *,
    scale_m: int = ACA_SCALE_M,
    min_area_m2: float = 50.0,
) -> dict[str, float]:
    """Areas (m^2) of agree / disagree / no_aca on the scale_m agreement grid."""
    disagree, no_aca, agree = agreement_rasters(
        gdf, aoi, aoi_ee, aca_benthic, scale_m=scale_m, min_area_m2=min_area_m2
    )

    def _count_px(mask: ee.Image) -> int:
        band = mask.bandNames().get(0)
        raw = mask.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi_ee,
            scale=scale_m,
            maxPixels=int(1e13),
            bestEffort=True,
            tileScale=4,
        )
        return int(ee.Number(raw.get(band)).getInfo() or 0)

    px = scale_m**2
    agree_n, disagree_n, no_aca_n = (_count_px(x) for x in (agree, disagree, no_aca))
    return {
        "agree": agree_n * px,
        "disagree": disagree_n * px,
        "no_aca": no_aca_n * px,
        "total_region": (agree_n + disagree_n + no_aca_n) * px,
    }
