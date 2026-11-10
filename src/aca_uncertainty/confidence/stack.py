"""
Spatial confidence index on [0, 1]: low values = low confidence, high = high confidence.

Components combine multiplicatively within and across categories
(equivalent to a noisy-OR on uncertainties).

Epistemic: RF entropy, spectral similarity to calibration training spectra for the RF
predicted class at each pixel.
Meta: calibration vs ACA (traffic light), Sentinel-2 acquisition count.
Aleatoric: placeholder (C_ale = 1 until bathymetry / turbidity layers exist).
"""

### --- IMPORTS
from __future__ import annotations

# general
import math
from dataclasses import dataclass
from typing import Any

import ee
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

# custom
from aca_uncertainty.confidence import aca, spectra
from aca_uncertainty.geo import aoi
from aca_uncertainty.utils import config
from aca_uncertainty.utils import ee as ee_utils

### --- HELPERS


def category_confidence(*components: ee.Image) -> ee.Image:
    """
    Combine confidence components: C = prod_i c_i, each c_i in [0, 1].

    Masked pixels in a component are treated as c_i = 1 (that layer does not reduce C).
    """
    combined = ee.Image(1)
    for comp in components:
        c = ee_utils.clip01(comp.unmask(1))
        combined = combined.multiply(c)
    return ee_utils.clip01(combined)


def category_uncertainty(*components: ee.Image) -> ee.Image:
    """Noisy-OR uncertainty; U = 1 - category_confidence(1 - u_i) per component."""
    inverted = [
        ee_utils.clip01(ee.Image(1).subtract(ee_utils.clip01(c))) for c in components
    ]
    return ee_utils.clip01(ee.Image(1).subtract(category_confidence(*inverted)))


### --- EPISTEMIC
### --- EPISTEMIC: SPECTRA


def class_name_to_sim_band(class_name: str) -> str:
    """Band name for `build_similarity_stack` (same rule as `spectra.similarity_band_name`)."""
    return spectra.similarity_band_name(class_name)


def calibration_class_spectral_stats(
    s2_composite: ee.Image,
    calibration_gdf: gpd.GeoDataFrame,
    aoi_poly: BaseGeometry,
    class_names: list[str] | None = None,
) -> dict[str, dict]:
    """Per-class mean/stdDev under calibration footprints (for similarity maps)."""
    class_names = class_names or aoi.classes_in_aoi(calibration_gdf, aoi_poly)
    stats: dict[str, dict] = {}
    for name in class_names:
        geom = aoi.class_geom_in_aoi(calibration_gdf, name, aoi_poly)
        stats[name] = spectra.reduce_class_spectral_stats(s2_composite, geom)
    return stats


def build_similarity_stack(
    s2_composite: ee.Image,
    class_stats: dict[str, dict],
    *,
    class_names: list[str] | None = None,
) -> ee.Image:
    class_names = class_names or list(class_stats.keys())
    layers = [
        spectra.class_similarity_image(s2_composite, name, class_stats[name]).toFloat()
        for name in class_names
    ]
    return ee.Image(layers)


def resolve_class_similarity_bands(
    available_band_names: list[str] | set[str],
    *,
    code_to_name: dict[int, str] | None = None,
) -> dict[int, str]:
    """
    Map benthic class codes (e.g. RF `class_num`) to similarity band names in the stack.

    Skips codes with no calibration spectrum (those pixels keep similarity 0 -> low c_spec).
    """
    available = set(available_band_names)
    code_to_name = code_to_name or config.ACA_CODE_TO_CSP_NAME
    out: dict[int, str] = {}
    for code, aca_name in code_to_name.items():
        name_candidates = (aca_name, *config.ACA_CLASS_NAME_VARIANTS.get(code, ()))
        deduped: list[str] = []
        for name in name_candidates:
            if name not in deduped:
                deduped.append(name)
        for name in deduped:
            band = class_name_to_sim_band(name)
            if band in available:
                out[code] = band
                break
    return out


def similarity_for_predicted_class(
    similarity_stack: ee.Image,
    class_prediction: ee.Image,
    *,
    band: str = "benthic_pred",
    code_to_sim_band: dict[int, str] | None = None,
) -> ee.Image:
    """
    Similarity to training spectra for the class predicted at each pixel.

    `class_prediction` is typically the case-study RF map (`benthic_pred` / `class_num`).
    """
    if code_to_sim_band is None:
        available = similarity_stack.bandNames().getInfo() or []
        code_to_sim_band = resolve_class_similarity_bands(available)
    predicted = class_prediction.select(band)
    picked = ee.Image(0).toFloat()
    for code, sim_band in code_to_sim_band.items():
        picked = picked.where(
            predicted.eq(int(code)), similarity_stack.select(sim_band)
        )
    return picked.rename("similarity_pred")


def spectral_confidence_from_prediction(
    similarity_stack: ee.Image,
    class_prediction: ee.Image,
    *,
    band: str = "benthic_pred",
) -> ee.Image:
    sim = similarity_for_predicted_class(similarity_stack, class_prediction, band=band)
    return ee_utils.clip01(sim.rename("c_spec"))


def spectral_uncertainty_from_prediction(
    similarity_stack: ee.Image,
    class_prediction: ee.Image,
    *,
    band: str = "benthic_pred",
) -> ee.Image:
    return ee_utils.clip01(
        ee.Image(1).subtract(
            spectral_confidence_from_prediction(
                similarity_stack, class_prediction, band=band
            )
        )
    )


### --- EPISTEMIC: ENTROPY


def entropy_confidence(
    entropy_img: ee.Image,
    *,
    n_classes: int | None = None,
    band: str = "entropy",
) -> ee.Image:
    """RF vote entropy -> confidence: 1 - H/ln(K)."""
    k = n_classes if n_classes is not None else len(config.ACA_CODES)
    max_h = math.log(k) if k > 1 else 1.0
    return ee_utils.clip01(
        ee.Image(1).subtract(entropy_img.select(band).divide(max_h)).rename("c_ent")
    )


def entropy_uncertainty(
    entropy_img: ee.Image,
    *,
    n_classes: int | None = None,
    band: str = "entropy",
) -> ee.Image:
    return ee_utils.clip01(
        ee.Image(1).subtract(
            entropy_confidence(entropy_img, n_classes=n_classes, band=band)
        )
    )


### --- META
### --- META: ACQUISITION


def acquisition_confidence(
    acquisition_count: ee.Image,
    n_ref: int,
    *,
    band: str = "acquisition_count",
) -> ee.Image:
    """High revisit -> high confidence; *n_ref* fixed for cross-AOI comparability."""
    n_ref = max(1, int(n_ref))
    return ee_utils.clip01(acquisition_count.select(band).divide(n_ref).rename("c_img"))


def acquisition_uncertainty(
    acquisition_count: ee.Image,
    n_ref: int,
    *,
    band: str = "acquisition_count",
) -> ee.Image:
    return ee_utils.clip01(
        ee.Image(1).subtract(
            acquisition_confidence(acquisition_count, n_ref, band=band)
        )
    )


### --- META: GROUND TRUTH


def ground_truth_confidence(
    disagree: ee.Image,
    no_aca: ee.Image,
    agree: ee.Image,
    in_region: ee.Image,
    *,
    c_agree: float | None = None,
    c_disagree: float | None = None,
    c_no_aca: float | None = None,
) -> ee.Image:
    """Traffic-light agreement rasters -> meta confidence (masked outside *in_region*)."""
    if c_agree is None:
        c_agree = config.CONFIDENCE_C_AGREE
    if c_disagree is None:
        c_disagree = config.CONFIDENCE_C_DISAGREE
    if c_no_aca is None:
        c_no_aca = config.CONFIDENCE_C_NO_ACA
    c = (
        disagree.unmask(0)
        .multiply(c_disagree)
        .add(no_aca.unmask(0).multiply(c_no_aca))
        .add(agree.unmask(0).multiply(c_agree))
    )
    return ee_utils.clip01(c).rename("c_gt").updateMask(in_region)


def ground_truth_uncertainty(
    disagree: ee.Image,
    no_aca: ee.Image,
    agree: ee.Image,
    in_region: ee.Image,
    **kwargs: Any,
) -> ee.Image:
    return ee_utils.clip01(
        ee.Image(1).subtract(
            ground_truth_confidence(disagree, no_aca, agree, in_region, **kwargs)
        )
    )


### --- VALIDATION: ACA TRAFFIC-LIGHT AGREEMENT


def agreement_layers(
    calibration_gdf: gpd.GeoDataFrame,
    aoi_poly: BaseGeometry,
    aoi_ee: ee.Geometry,
    aca_benthic: ee.Image,
    *,
    agreement_scale_m: int = aca.ACA_SCALE_M,
    min_area_m2: float = 50.0,
) -> tuple[ee.Image, ee.Image, ee.Image, ee.Image]:
    """(disagree, no_aca, agree, in_region) on the metric grid."""
    disagree, no_aca, agree = aca.agreement_rasters(
        calibration_gdf,
        aoi_poly,
        aoi_ee,
        aca_benthic,
        scale_m=agreement_scale_m,
        min_area_m2=min_area_m2,
    )
    frag = aca.calibration_fragments_gdf(
        calibration_gdf, aoi_poly, min_area_m2=min_area_m2
    )
    frag_fc = aoi.fragments_to_ee(frag)
    region = aca.region_class_raster(frag_fc, aoi_ee, scale_m=agreement_scale_m)
    in_region = ee_utils.reproject_to_grid(region.gt(0), aoi_ee).rename("in_region")
    disagree = ee_utils.reproject_to_grid(disagree.unmask(0), aoi_ee).gt(0).selfMask()
    no_aca = ee_utils.reproject_to_grid(no_aca.unmask(0), aoi_ee).gt(0).selfMask()
    agree = ee_utils.reproject_to_grid(agree.unmask(0), aoi_ee).gt(0).selfMask()
    return disagree, no_aca, agree, in_region


### --- CONFIDENCE STACK


@dataclass
class ConfidenceStack:
    """Pixel-wise confidence index and components (Earth Engine images, low -> low confidence)."""

    c_ent: ee.Image
    c_spec: ee.Image
    c_img: ee.Image
    c_gt: ee.Image
    C_epi: ee.Image
    C_meta: ee.Image
    C_ale: ee.Image
    C_total: ee.Image
    similarity_pred: ee.Image

    def component_bands(self) -> ee.Image:
        return ee.Image.cat(
            [
                self.c_ent,
                self.c_spec,
                self.c_img,
                self.c_gt,
                self.C_epi,
                self.C_meta,
                self.C_ale,
                self.C_total,
            ]
        )

    def summary(
        self,
        region: ee.Geometry,
        *,
        scale_m: int | None = None,
    ) -> dict[str, float | None]:
        scale_m = scale_m or config.SCALE_M
        stats = (
            self.C_total.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.percentile([50, 90]), "", True
                ),
                geometry=region,
                scale=scale_m,
                maxPixels=int(1e9),
                bestEffort=True,
                tileScale=4,
            ).getInfo()
            or {}
        )
        return {
            "mean": stats.get("C_total_mean"),
            "p50": stats.get("C_total_p50"),
            "p90": stats.get("C_total_p90"),
        }


def build_confidence_stack(
    *,
    aoi_ee: ee.Geometry,
    aoi_poly: BaseGeometry,
    calibration_gdf: gpd.GeoDataFrame,
    aca_benthic: ee.Image,
    rf_prediction: ee.Image,
    s2_composite: ee.Image,
    class_stats: dict[str, dict],
    entropy_img: ee.Image,
    acquisition_count_img: ee.Image,
    n_acquisitions_ref: int,
    n_entropy_classes: int | None = None,
    agreement_scale_m: int = aca.ACA_SCALE_M,
    scale_m: int | None = None,
    reef_mask: ee.Image | None = None,
) -> ConfidenceStack:
    """Assemble epistemic + meta confidence and C_total (low values = low confidence)."""
    scale_m = scale_m or config.SCALE_M
    pred_on_grid = ee_utils.reproject_to_grid(rf_prediction, aoi_ee, scale_m=scale_m)
    sim_stack = build_similarity_stack(s2_composite, class_stats)
    c_spec = ee_utils.reproject_to_grid(
        spectral_confidence_from_prediction(sim_stack, pred_on_grid),
        aoi_ee,
        scale_m=scale_m,
    )
    similarity_pred = c_spec.rename("similarity_pred")

    c_ent = ee_utils.reproject_to_grid(
        entropy_confidence(entropy_img, n_classes=n_entropy_classes),
        aoi_ee,
        scale_m=scale_m,
    )
    c_img = ee_utils.reproject_to_grid(
        acquisition_confidence(acquisition_count_img, n_acquisitions_ref),
        aoi_ee,
        scale_m=scale_m,
    )

    disagree, no_aca, agree, in_region = agreement_layers(
        calibration_gdf,
        aoi_poly,
        aoi_ee,
        aca_benthic,
        agreement_scale_m=agreement_scale_m,
    )
    c_gt = ee_utils.reproject_to_grid(
        ground_truth_confidence(disagree, no_aca, agree, in_region),
        aoi_ee,
        scale_m=scale_m,
    )

    C_epi = category_confidence(c_ent, c_spec).rename("C_epi")
    C_meta = category_confidence(c_img, c_gt).rename("C_meta")
    C_ale = ee.Image(1).toFloat().rename("C_ale")
    C_total = category_confidence(C_meta, C_epi, C_ale).rename("C_total")

    if reef_mask is not None:
        mask = ee_utils.reproject_to_grid(reef_mask, aoi_ee, scale_m=scale_m).gt(0)
        c_ent = c_ent.updateMask(mask)
        c_spec = c_spec.updateMask(mask)
        c_img = c_img.updateMask(mask)
        c_gt = c_gt.updateMask(mask)
        C_epi = C_epi.updateMask(mask)
        C_meta = C_meta.updateMask(mask)
        C_total = C_total.updateMask(mask)
        similarity_pred = similarity_pred.updateMask(mask)

    return ConfidenceStack(
        c_ent=c_ent,
        c_spec=c_spec,
        c_img=c_img,
        c_gt=c_gt,
        C_epi=C_epi,
        C_meta=C_meta,
        C_ale=C_ale,
        C_total=C_total,
        similarity_pred=similarity_pred,
    )
