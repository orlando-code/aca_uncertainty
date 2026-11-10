"""Per-class Sentinel-2 spectral stats and pixel similarity layers."""

### --- IMPORTS
# spatial
import ee

# custom
from aca_uncertainty.utils import config

### --- CLASS SPECTRAL STATS


def reduce_class_spectral_stats(
    image: ee.Image,
    geometry: ee.Geometry,
    bands: list[str] | None = None,
) -> dict:
    """Per-band mean and stdDev of surface reflectance under `geometry`."""
    bands = bands or config.S2_BANDS
    return (
        image.select(bands)
        .reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True),
            geometry=geometry,
            scale=config.SCALE_M,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        .getInfo()
    )


### --- SIMILARITY LAYERS


def similarity_band_name(class_name: str) -> str:
    """EE band name for a class similarity layer."""
    tag = class_name.lower().replace(" ", "_").replace("/", "_")
    return f"sim_{tag}"


def class_similarity_image(image: ee.Image, class_name: str, stats: dict) -> ee.Image:
    """Pixel-wise similarity to calibration training spectra (diagonal Gaussian in band space)."""
    bands = config.S2_BANDS
    means = [stats.get(f"{b}_mean", 0) for b in bands]
    stds = [max(stats.get(f"{b}_stdDev", 1e-4), 1e-4) for b in bands]
    mean_img = ee.Image.constant(means).rename(bands)
    std_img = ee.Image.constant(stds).rename(bands)
    chi2 = (
        image.select(bands)
        .subtract(mean_img)
        .divide(std_img)
        .pow(2)
        .reduce(ee.Reducer.sum())
    )
    return chi2.multiply(-0.5).exp().rename(similarity_band_name(class_name))
