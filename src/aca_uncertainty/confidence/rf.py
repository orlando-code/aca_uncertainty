"""
RF benthic classification, currently implemented for the Central South Pacific region (ACA Lyons et al. workflow: see https://github.com/CoralMapping/gee-mapping-source/blob/master/global_reefs2/central_south_pacific/csp_classification). N.B. does not run any of the `map refinement` steps.

Mirrors the GEE `global_reefs` classification script: pixel (csp_pixels) + segment (csp_segments) stacks,
`smileRandomForest` on `benthic_train_library`, classify `pixsegs`.
"""

### --- IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ee
import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry

# custom
from aca_uncertainty.confidence import aca
from aca_uncertainty.geo import aoi, s2
from aca_uncertainty.utils import config

### --- ASSETS AND PIXSEGS
PixsegsMode = str  # "csp" | "s2_proxy"


def _asset_error(path: str, err: Exception) -> RuntimeError:
    """Error message if asset not available e.g. proprietary Planet Dove data"""
    return RuntimeError(
        f"Earth Engine asset not available: {path!r}\n"
        f"({err})\n"
        "The ACA CSP pixel/segment stacks live under "
        "'projects/coral_atlas/central_south_pacific/...'. "
        "Initialize EE with a project that can read those assets (e.g. the "
        "Allen Coral Atlas workflow project), or override paths in "
        "aca_uncertainty.config.\n"
        "Training can also use local calibration polygons via "
        "CaseStudyRF.fit(..., calibration_gdf=gdf) once pixsegs loads."
    )


def mosaic_collection(asset_paths: list[str]) -> ee.Image:
    """Merge a list of asset paths into a single image."""
    merged = ee.ImageCollection([])
    for path in asset_paths:
        merged = merged.merge(ee.ImageCollection(path))
    return merged.mosaic()


def csp_assets_available() -> bool:
    """True if `csp_pixels` ImageCollection is readable in the active EE project."""
    try:
        n = ee.ImageCollection(config.CSP_PIXELS[0]).limit(1).size().getInfo()
        return int(n or 0) > 0
    except ee.EEException:
        return False


def sample_scale_for_mode(mode: PixsegsMode) -> int:
    """Sample scale for the pixsegs mode (CSP or S-2 proxy)."""
    return config.CSP_EXPORT_SCALE_M if mode == "csp" else config.SCALE_M


def resolve_pixsegs_mode(mode: str | None = None) -> PixsegsMode:
    """Resolve the pixsegs mode from the config or the input mode."""
    mode = (mode or config.CSP_PIXSEGS_MODE).lower()
    if mode == "auto":
        return "csp" if csp_assets_available() else "s2_proxy"
    if mode not in ("csp", "s2_proxy"):
        raise ValueError(f"Unknown CSP_PIXSEGS_MODE: {mode!r}")
    return mode


def load_pixsegs_csp(aoi: ee.Geometry) -> ee.Image:
    """ACA Dove pixel + segment stack (5 m) and return as image clipped to AOI."""
    pix = mosaic_collection(config.CSP_PIXELS).regexpRename("(^.*$)", "$1_p")
    seg = mosaic_collection(config.CSP_SEGMENTS).regexpRename("(^.*$)", "$1_s")
    return pix.addBands(seg).select(config.CSP_BENTHIC_INPUT_BANDS).clip(aoi)


def load_pixsegs_s2_proxy(aoi: ee.Geometry) -> ee.Image:
    """
    Approximate ACA benthic RF predictors from Sentinel-2 (10 m).

    Not identical to Dove `csp_pixels`; use when `projects/coral_atlas/...` is
    unreadable. Band names match `CSP_BENTHIC_INPUT_BANDS` for the same RF code path.
    """
    ic = s2.build_s2_collection(aoi)
    med = ic.median()
    b1_p = med.select("B2").rename("b1_p")
    b2_p = med.select("B3").rename("b2_p")
    b3_p = med.select("B4").rename("b3_p")
    b4_p = med.select("B8").rename("b4_p")
    b1_savg_p = b1_p.focal_mean(3, "square", "pixels").rename("b1_savg_p")
    b2_savg_p = b2_p.focal_mean(3, "square", "pixels").rename("b2_savg_p")
    depth_stdDev_p = (
        ic.select("B4").reduce(ee.Reducer.stdDev()).rename("depth_stdDev_p")
    )
    return (
        ee.Image.cat([b1_p, b2_p, b3_p, b4_p, b1_savg_p, b2_savg_p, depth_stdDev_p])
        .clip(aoi)
        .reproject(crs=med.select(0).projection(), scale=config.SCALE_M)
    )


def load_pixsegs(
    aoi: ee.Geometry, *, mode: str | None = None
) -> tuple[ee.Image, PixsegsMode]:
    """Load the pixsegs image and resolve the mode (CSP or S-2 proxy)."""
    resolved = resolve_pixsegs_mode(mode)
    if resolved == "csp":
        try:
            img = load_pixsegs_csp(aoi)
            img.bandNames().getInfo()
            return img, resolved
        except ee.EEException as exc:
            raise _asset_error(config.CSP_PIXELS[0], exc) from exc
    return load_pixsegs_s2_proxy(aoi), resolved


def load_low_tide_image(aoi: ee.Geometry, *, mode: PixsegsMode) -> ee.Image:
    """If able, load the Planet Dove (low tide) image; else use the S-2 proxy."""
    if mode == "csp":
        return mosaic_collection(config.CSP_LOW_TIDE_IMAGE).clip(aoi)
    ic = s2.build_s2_collection(aoi)
    return (
        ic.median().select(["B4", "B3", "B2"], ["b3", "b2", "b1"]).clip(aoi)
    )  # median composite


def load_mapping_mask(aoi: ee.Geometry, pixsegs_mode: PixsegsMode) -> ee.Image:
    """
    Pixels where benthic RF may train and predict (excludes land and off-reef).

    Default: ACA `reef_mask` (public). With CSP assets: also exclude
    `csp_Coast_mask` and `csp_notreef` when readable.
    """
    reef = ee.Image("ACA/reef_habitat/v2_0").select("reef_mask").clip(aoi).gt(0)
    valid = reef
    if pixsegs_mode == "csp":
        for path, invert in (
            (config.CSP_NOTREEF_MASK, True),
            (config.CSP_COAST_MASK, True),
        ):
            try:
                layer = ee.ImageCollection(path).mosaic().clip(aoi).gt(0)
                valid = valid.And(layer.Not() if invert else layer)
            except ee.EEException:
                try:
                    layer = ee.Image(path).clip(aoi).gt(0)
                    valid = valid.And(layer.Not() if invert else layer)
                except ee.EEException:
                    pass
    return valid.rename("mapping_mask")


### --- TRAINING


def training_in_aoi(aoi: ee.Geometry) -> ee.FeatureCollection:
    """Sample ACA-style predictors under calibration polygons in the AOI."""
    try:
        fc = ee.FeatureCollection(config.CSP_BENTHIC_TRAIN_LIBRARY).filterBounds(aoi)
    except ee.EEException as exc:
        raise _asset_error(config.CSP_BENTHIC_TRAIN_LIBRARY, exc) from exc
    return fc.filter(ee.Filter.notNull(config.CSP_BENTHIC_INPUT_BANDS))


def training_from_calibration(
    calibration_gdf: gpd.GeoDataFrame,
    aoi_poly: BaseGeometry,
    pixsegs: ee.Image,
    *,
    sample_scale: int | None = None,
) -> ee.FeatureCollection:
    """
    Sample ACA-style predictors under calibration polygons in the AOI.

    Replicates '*_calval' / training_library points using `sampleRegions` on
    the pixel+segment stack (only AOI fragments, no global multipolygons).
    """
    frag = aca.calibration_fragments_gdf(calibration_gdf, aoi_poly)
    if frag.empty:
        raise ValueError("No calibration fragments in case-study AOI.")
    fc = aoi.fragments_to_ee(frag)
    sample_scale = sample_scale or config.CSP_EXPORT_SCALE_M
    return pixsegs.sampleRegions(
        collection=fc,
        properties=[config.CSP_RF_CLASS_FIELD, "Class_name"],
        scale=sample_scale,
        tileScale=4,
        geometries=False,
    ).filter(ee.Filter.notNull(config.CSP_BENTHIC_INPUT_BANDS))


def train_test_split(
    fc: ee.FeatureCollection,
    *,
    train_fraction: float = config.CSP_TRAIN_FRACTION,
    seed: int = config.CSP_RF_SEED,
) -> tuple[ee.FeatureCollection, ee.FeatureCollection]:
    """Split the training features into train and test sets."""
    split = fc.randomColumn("rf_split", seed=seed)
    train = split.filter(ee.Filter.lt("rf_split", train_fraction))
    test = split.filter(ee.Filter.gte("rf_split", train_fraction))
    return train, test


def make_rf_classifier(
    *,
    trees: int = config.CSP_RF_TREES,
    min_leaf: int = config.CSP_RF_MIN_LEAF,
    seed: int = config.CSP_RF_SEED,
) -> ee.Classifier:
    """Create a GEE RF classifier: https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest"""
    return ee.Classifier.smileRandomForest(
        numberOfTrees=trees,
        minLeafPopulation=min_leaf,
        variablesPerSplit=None,
        seed=seed,
    )


def train_benthic_classifier(
    training: ee.FeatureCollection,
    input_bands: list[str] | None = None,
    *,
    trees: int = config.CSP_RF_TREES,
    min_leaf: int = config.CSP_RF_MIN_LEAF,
    seed: int = config.CSP_RF_SEED,
    probabilities: bool = False,
) -> ee.Classifier:
    """Train a benthic RF classifier."""
    input_bands = input_bands or config.CSP_BENTHIC_INPUT_BANDS
    clf = make_rf_classifier(trees=trees, min_leaf=min_leaf, seed=seed)
    if probabilities:
        clf = clf.setOutputMode("MULTIPROBABILITY")
    return clf.train(training, config.CSP_RF_CLASS_FIELD, input_bands)


def classify(pixsegs: ee.Image, classifier: ee.Classifier) -> ee.Image:
    """Classify the pixsegs image using the classifier."""
    return pixsegs.classify(classifier).toUint8().rename("benthic_pred")


### --- CLASSIFICATION AND PROBABILITIES


def _raw_probability_array(raw: ee.Image) -> ee.Image:
    """MULTIPROBABILITY array band (one value per trained class)."""
    names = raw.bandNames()
    return ee.Image(
        ee.Algorithms.If(
            names.contains("classification"),
            raw.select("classification"),
            ee.Algorithms.If(
                names.contains("probability"),
                raw.select("probability"),
                raw.select(0),
            ),
        )
    )


def _probability_band_at(array_img: ee.Image, index: int, name: str) -> ee.Image:
    """One scalar probability band from a 1-D (or first-axis) MULTIPROBABILITY array."""
    return (
        array_img.arraySlice(0, index, index + 1)
        .arrayProject([0])
        .arrayGet([0])
        .rename(name)
    )


def unpack_multiprobability_bands(raw: ee.Image, class_codes: list[int]) -> ee.Image:
    """Explode the probability array into bands named `str(class_code)`."""
    if not class_codes:
        raise ValueError("class_codes required to unpack MULTIPROBABILITY image.")
    array_img = _raw_probability_array(raw)
    stacked = _probability_band_at(array_img, 0, str(class_codes[0]))
    for i, code in enumerate(class_codes[1:], start=1):
        stacked = stacked.addBands(_probability_band_at(array_img, i, str(code)))
    return stacked


def shannon_entropy(prob_img: ee.Image, *, eps: float = 1e-12) -> ee.Image:
    """Shannon entropy (nats) from unpacked per-class probability bands."""
    log_p = prob_img.max(eps).log()
    return (
        prob_img.multiply(log_p).reduce(ee.Reducer.sum()).multiply(-1).rename("entropy")
    )


def build_probability_layers(
    pixsegs: ee.Image,
    classifier_prob: ee.Classifier,
    aoi: ee.Geometry,
    *,
    mapping_mask: ee.Image | None = None,
) -> tuple[ee.Image, ee.Image]:
    """Raw MULTIPROBABILITY classify -> unpacked bands + scalar entropy."""
    raw = pixsegs.classify(classifier_prob).clip(aoi)
    codes = probability_class_codes(classifier_prob)
    probabilities = unpack_multiprobability_bands(raw, codes)
    entropy = shannon_entropy(probabilities)
    if mapping_mask is not None:
        probabilities = probabilities.updateMask(mapping_mask)
        entropy = entropy.updateMask(mapping_mask)
    return probabilities, entropy


def classify_probabilities(pixsegs: ee.Image, classifier: ee.Classifier) -> ee.Image:
    raw = pixsegs.classify(classifier)
    codes = probability_class_codes(classifier)
    return unpack_multiprobability_bands(raw, codes)


### --- EVALUATION AND METRICS


def confusion_matrix(
    test_fc: ee.FeatureCollection,
    classifier: ee.Classifier,
) -> ee.ConfusionMatrix:
    """Calculate GEE confusion matrix from classifier for test features"""
    labeled = test_fc.classify(classifier)
    return labeled.errorMatrix(
        config.CSP_RF_CLASS_FIELD,
        "classification",
    )


def confusion_to_arrays(matrix: ee.ConfusionMatrix) -> dict[str, Any]:
    """Client-side confusion table and overall accuracy."""
    info = matrix.getInfo()
    return {
        "matrix": info,
        "accuracy": matrix.accuracy().getInfo(),
        "kappa": matrix.kappa().getInfo(),
        "order": matrix.order().getInfo(),
    }


def active_confusion_classes(cm: dict[str, Any]) -> list[int]:
    """
    Return only class codes with any reference or predicted pixels in the hold-out confusion matrix.

    EE `errorMatrix` is square from min code to max code (here 0...18) corresponding to ACA geomorphic class codes, so unused ACA gaps (0–10, 16, 17) would appear as empty rows/columns unless filtered.
    """

    matrix = np.asarray(cm["matrix"], dtype=float)
    order = [int(x) for x in cm["order"]]
    support = matrix.sum(axis=0) + matrix.sum(axis=1)
    return [order[i] for i, s in enumerate(support) if s > 0]


def variable_importance(classifier: ee.Classifier) -> dict[str, float]:
    """Return variable importance from classifier explain()"""
    explain = ee.Dictionary(classifier.explain())
    importance = ee.Dictionary(explain.get("importance")).getInfo()
    return {str(k): float(v) for k, v in (importance or {}).items()}


### --- CASE STUDY FIT


@dataclass
class CaseStudyRF:
    """Trained RF and data clipped to the case-study AOI."""

    aoi: ee.Geometry
    pixsegs: ee.Image
    low_tide: ee.Image
    classifier: ee.Classifier
    train_fc: ee.FeatureCollection
    test_fc: ee.FeatureCollection
    prediction: ee.Image
    pixsegs_mode: PixsegsMode
    mapping_mask: ee.Image
    classifier_prob: ee.Classifier | None
    probabilities: ee.Image | None
    entropy: ee.Image | None

    @classmethod
    def fit(
        cls,
        aoi: ee.Geometry,
        *,
        aoi_poly: BaseGeometry | None = None,
        calibration_gdf: gpd.GeoDataFrame | None = None,
        pixsegs_mode: str | None = None,
        apply_land_mask: bool = True,
        compute_entropy: bool | None = None,
    ) -> CaseStudyRF:
        """Fit the RF classifier to the data within AOI."""
        compute_entropy = (
            config.CSP_RF_PROBABILITIES if compute_entropy is None else compute_entropy
        )
        pixsegs, resolved = load_pixsegs(aoi, mode=pixsegs_mode)
        mapping_mask = load_mapping_mask(aoi, resolved)
        if apply_land_mask:
            pixsegs = pixsegs.updateMask(mapping_mask)
        low_tide = load_low_tide_image(aoi, mode=resolved)
        if apply_land_mask:
            low_tide = low_tide.updateMask(mapping_mask)
        scale_m = sample_scale_for_mode(resolved)
        if calibration_gdf is not None:
            if aoi_poly is None:
                raise ValueError("aoi_poly required when using calibration_gdf")
            samples = training_from_calibration(
                calibration_gdf, aoi_poly, pixsegs, sample_scale=scale_m
            )
        else:
            samples = training_in_aoi(aoi)
        train_fc, test_fc = train_test_split(samples)
        classifier = train_benthic_classifier(train_fc)
        prediction = classify(pixsegs, classifier).clip(aoi)
        if apply_land_mask:
            prediction = prediction.updateMask(mapping_mask)

        classifier_prob = None
        probabilities = None
        entropy = None
        if compute_entropy:
            classifier_prob = train_benthic_classifier(train_fc, probabilities=True)
            mask = mapping_mask if apply_land_mask else None
            probabilities, entropy = build_probability_layers(
                pixsegs, classifier_prob, aoi, mapping_mask=mask
            )

        return cls(
            aoi=aoi,
            pixsegs=pixsegs,
            low_tide=low_tide,
            classifier=classifier,
            train_fc=train_fc,
            test_fc=test_fc,
            prediction=prediction,
            pixsegs_mode=resolved,
            mapping_mask=mapping_mask,
            classifier_prob=classifier_prob,
            probabilities=probabilities,
            entropy=entropy,
        )


def probability_class_codes(classifier: ee.Classifier) -> list[int]:
    """Return class codes in the MULTIPROBABILITY band / feature property order."""
    explain = ee.Dictionary(classifier.explain()).getInfo()
    classes = explain.get("classes")
    if not classes:
        raise ValueError(
            "Classifier explain() missing 'classes' (train prob RF first)."
        )
    return [int(c) for c in classes]


def _feature_class_probabilities(
    props: dict[str, Any], codes: list[int]
) -> dict[int, float]:
    """Return per-class vote shares from a `classify()` feature (MULTIPROBABILITY)."""
    for key in ("classification", "probability"):
        raw_arr = props.get(key)
        if isinstance(raw_arr, (list, tuple)):
            return {
                code: float(raw_arr[i])
                for i, code in enumerate(codes)
                if i < len(raw_arr) and raw_arr[i] is not None
            }
    probs: dict[int, float] = {}
    for code in codes:
        key = str(code)
        val = props.get(key, props.get(code))
        if val is not None:
            probs[code] = float(val)
    return probs


### --- HOLD-OUT SUMMARIES


def summarize_per_class_probabilities(
    prob_img: ee.Image,
    classifier_prob: ee.Classifier,
    region: ee.Geometry,
    *,
    scale_m: int | None = None,
) -> dict[int, dict[str, float | None]]:
    """Mean / percentiles of each class's ensemble probability over `region`."""
    scale_m = scale_m or config.SCALE_M
    reducer = ee.Reducer.mean().combine(ee.Reducer.percentile([50, 90]), "", True)
    out: dict[int, dict[str, float | None]] = {}
    for code in probability_class_codes(classifier_prob):
        band = str(code)
        stats = (
            prob_img.select(band)
            .reduceRegion(
                reducer=reducer,
                geometry=region,
                scale=scale_m,
                maxPixels=int(1e9),
                bestEffort=True,
                tileScale=4,
            )
            .getInfo()
            or {}
        )
        out[code] = {
            "mean": stats.get(f"{band}_mean"),
            "p50": stats.get(f"{band}_p50"),
            "p90": stats.get(f"{band}_p90"),
        }
    return out


def holdout_probability_records(
    test_fc: ee.FeatureCollection,
    classifier_prob: ee.Classifier,
) -> list[dict[str, Any]]:
    """
    Hold-out training pixels: ensemble probability of the reference class and winner.
    """
    codes = probability_class_codes(classifier_prob)
    labeled = test_fc.classify(classifier_prob)
    records: list[dict[str, Any]] = []
    for feat in labeled.getInfo().get("features", []):
        props = feat.get("properties") or {}
        ref = int(props[config.CSP_RF_CLASS_FIELD])
        probs = _feature_class_probabilities(props, codes)
        if not probs:
            continue
        winner = max(codes, key=lambda c: probs.get(c, float("-inf")))
        name = config.ACA_CODE_TO_CSP_NAME.get(ref, str(ref))
        records.append(
            {
                "class_num": ref,
                "Class_name": name,
                "p_true_class": probs.get(ref),
                "p_predicted_class": probs.get(winner),
                "predicted_class_num": winner,
                "correct": winner == ref,
            }
        )
    return records


def entropy_summary(
    entropy_img: ee.Image,
    region: ee.Geometry,
    *,
    scale_m: int | None = None,
) -> dict[str, float | None]:
    """Mean / percentiles of entropy within `region`."""
    scale_m = scale_m or config.SCALE_M
    ent = entropy_img.select("entropy")
    stats = (
        ent.reduceRegion(
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
        "mean": stats.get("entropy_mean"),
        "p50": stats.get("entropy_p50"),
        "p90": stats.get("entropy_p90"),
    }


def benthic_prediction_for_display(prediction: ee.Image) -> ee.Image:
    """RF class codes remapped for correct ACA colours on geemap."""
    return aca.remap_benthic_codes(prediction, band="benthic_pred")
