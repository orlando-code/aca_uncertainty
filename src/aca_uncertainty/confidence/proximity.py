"""Training geometry: distance to class footprints and area fractions in an AOI."""

### --- IMPORTS

# general
import ee

# custom
from aca_uncertainty.utils import config
from aca_uncertainty.utils import ee as utils_ee

### --- HELPERS


def class_footprint_mask(
    fc: ee.FeatureCollection,
    class_name: str,
    region: ee.Geometry,
    scale_m: int | None = None,
) -> ee.Image:
    """Binary mask (1 = class polygon) on a full-AOI grid at `scale_m`."""
    scale_m = scale_m or config.SCALE_M
    class_fc = fc.filter(ee.Filter.eq("Class_name", class_name)).filterBounds(region)
    mask = ee.Image.constant(0).byte().paint(class_fc, 1).gt(0).unmask(0).rename("mask")
    return mask.clipToBoundsAndScale(geometry=region, scale=scale_m)


def stack_class_distances(
    fc: ee.FeatureCollection,
    class_names: list[str],
    region: ee.Geometry,
    scale_m: int | None = None,
    kernel_radius_m: float | None = None,
) -> ee.Image:
    """Multi-band image: one `dist_<class>` band per class."""
    if not class_names:
        raise ValueError("class_names is empty")
    stacked = distance_to_class_m(
        fc, class_names[0], region, scale_m=scale_m, kernel_radius_m=kernel_radius_m
    )
    for name in class_names[1:]:
        stacked = stacked.addBands(
            distance_to_class_m(
                fc, name, region, scale_m=scale_m, kernel_radius_m=kernel_radius_m
            )
        )
    return stacked


### --- CALCULATE DISTANCES
def distance_to_class_m(
    fc: ee.FeatureCollection,
    class_name: str,
    region: ee.Geometry,
    scale_m: int | None = config.SCALE_M,
    kernel_radius_m: float | None = None,
) -> ee.Image:
    """
    Per-pixel distance (m) to the nearest rasterized cell of `class_name`.

    Pixels on the footprint are 0; farther pixels increase up to the kernel cap
    (`255 * scale_m`), then saturate at distance limit.
    """
    max_dist_m = utils_ee.max_distance_kernel_m(scale_m)
    kernel_radius_m = min(kernel_radius_m or max_dist_m, max_dist_m)
    mask = class_footprint_mask(fc, class_name, region, scale_m)
    kernel = ee.Kernel.euclidean(radius=kernel_radius_m, units="meters")
    tag = utils_ee.class_tag(class_name)
    dist = (
        mask.distance(kernel)
        .toFloat()
        .rename(f"dist_{tag}")
        .unmask(max_dist_m)
        .clipToBoundsAndScale(geometry=region, scale=scale_m)
    )
    return dist
