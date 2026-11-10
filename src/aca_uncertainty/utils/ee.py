"""Earth Engine image utilities."""

### --- IMPORTS
from __future__ import annotations

# general
import ee

# custom
from aca_uncertainty.utils import config


def clip01(img: ee.Image) -> ee.Image:
    """Clamp band values to [0, 1]."""
    return img.clamp(0, 1)


def reproject_to_grid(
    img: ee.Image,
    aoi: ee.Geometry,
    *,
    scale_m: int | None = None,
) -> ee.Image:
    """Clip to AOI and reproject to the case-study metric grid."""
    scale_m = scale_m or config.SCALE_M
    return img.clip(aoi).reproject(crs="EPSG:4326", scale=scale_m)


def count_fc(fc: ee.FeatureCollection) -> int:
    """Return the number of features in `fc`."""
    return int(fc.size().getInfo())


def class_tag(class_name: str) -> str:
    """Return a tag for the class name by removing spaces and slashes."""
    return class_name.lower().replace(" ", "_").replace("/", "_")


def max_distance_kernel_m(scale_m: int) -> float:
    """EE `Image.distance` kernel is capped at 255 pixels in the image's projection."""
    return float(255 * scale_m)
