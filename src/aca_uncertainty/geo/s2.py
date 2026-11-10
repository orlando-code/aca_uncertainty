"""Sentinel-2 SR composites and per-pixel scene / acquisition counts."""

### --- IMPORTS
from __future__ import annotations

import ee

# custom
from aca_uncertainty.plotting.config import INFERNO_HEXES
from aca_uncertainty.utils import config

### --- SCL MASKS


def mask_s2_scl(
    img: ee.Image,
    *,
    include_cloud_shadow: bool = True,
) -> ee.Image:
    """
    Mask cloudy/cirrus SCL pixels; optionally cloud shadow (class 3).

    Shadow is included in `mask_s2_clear` for the median composite but often
    labels dark lagoon / deep water as shadow. Use `include_cloud_shadow=False`
    for cloud-only clearing.
    """
    scl = img.select("SCL")
    bad = scl.eq(config.SCL_CLOUD_CLASSES[0])
    for c in config.SCL_CLOUD_CLASSES[1:]:
        bad = bad.Or(scl.eq(c))
    if include_cloud_shadow:
        bad = bad.Or(scl.eq(config.SCL_CLOUD_SHADOW))
    return img.updateMask(bad.Not())


def mask_s2_clear(img: ee.Image) -> ee.Image:
    """Drop cloud shadow and likely cloud / cirrus (SCL); keep water (+ reef) pixels."""
    return mask_s2_scl(img, include_cloud_shadow=True)


def mask_s2_clouds_only(img: ee.Image) -> ee.Image:
    """Cloud + cirrus only (SCL 8–10); keep pixels even if labelled cloud shadow."""
    return mask_s2_scl(img, include_cloud_shadow=False)


### --- BUILDING COMPOSITES


def build_s2_collection(
    region: ee.Geometry,
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    cloud_threshold: float | None = config.CLOUDY_PIXEL_PERCENTAGE_THRESHOLD,
    mask_fn=mask_s2_clear,
) -> ee.ImageCollection:
    """S2_SR scenes over `region`, scene-level cloud filter, per-pixel SCL mask."""
    date_start = date_start or config.S2_COMPOSITE_DATE_START
    date_end = date_end or config.S2_COMPOSITE_DATE_END
    return (
        ee.ImageCollection(config.S2_COLLECTION_ID)
        .filterBounds(region)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_fn)
        .select(config.S2_BANDS)
    )


def median_composite(ic: ee.ImageCollection, region: ee.Geometry) -> ee.Image:
    """Median SR composite in [0, 1], clipped to `region`. Dividing by 10000 to convert from digital number to reflectance from: https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/."""
    return ic.median().clip(region).divide(10000).rename(config.S2_BANDS)


### --- IMAGERY COUNTS


def observation_count(
    ic: ee.ImageCollection,
    *,
    band: str = "B4",
) -> ee.Image:
    """
    Per-pixel number of scenes where the pixel survives the collection's mask.

    This is not "how often was the weather clear?" It sums valid pixels after
    `mask_s2_clear` (cloud, cirrus, and cloud shadow). SCL shadow over dark
    water reduces counts and often traces reef / lagoon geometry. For orbit coverage
    use `acquisition_count`; for clouds without shadow use a collection built with
    `mask_s2_clouds_only`.
    """
    ref_band = band if band in config.S2_BANDS else config.S2_BANDS[0]

    def _one_count(img: ee.Image) -> ee.Image:
        return (
            ee.Image.constant(1)
            .toUint16()
            .rename("scene_count")
            .updateMask(img.select(ref_band).mask())
        )

    return ic.map(_one_count).sum().rename("scene_count")


def acquisition_count(ic: ee.ImageCollection) -> ee.Image:
    """
    Per-pixel number of scene footprints covering the pixel (no per-pixel SCL).

    Reflects Sentinel-2 revisit / overlap only; should not mirror reef bathymetry.
    Scene metadata filter (`CLOUDY_PIXEL_PERCENTAGE`) still applies to `ic`.
    """

    def _one(img: ee.Image) -> ee.Image:
        return (
            ee.Image.constant(1)
            .toUint16()
            .rename("acquisition_count")
            .clip(img.geometry())
        )

    return ic.map(_one).sum().rename("acquisition_count")


def scene_count_vis(max_scenes: int, *, band: str = "scene_count") -> dict:
    """Palette for mapping observation or acquisition counts."""
    return {
        "bands": [band],
        "min": 0,
        "max": max(1, max_scenes),
        "palette": INFERNO_HEXES,
    }
