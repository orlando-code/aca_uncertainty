### --- IMPORTS
from __future__ import annotations

# general
import json

# spatial
import ee
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

# custom
from aca_uncertainty import config, s2

### --- GEOMETRY HELPERS


def largest_part(geom):
    if geom.geom_type == "Polygon":
        return geom
    return max((g for g in geom.geoms if not g.is_empty), key=lambda g: g.area)


def fragments_to_ee(gdf_frag: gpd.GeoDataFrame) -> ee.FeatureCollection:
    return ee.FeatureCollection(json.loads(gdf_frag.to_json()))


### --- CASE-STUDY AOI


def case_study_aoi_polygon(
    centre_lonlat: list[float] | None = None,
    side_m: float | None = None,
) -> BaseGeometry:
    """Case-study AOI as a square centred on `BORA_BORA_COORDS` (EPSG:4326)."""
    centre_lonlat = centre_lonlat or config.BORA_BORA_COORDS
    side_m = side_m if side_m is not None else config.CS_AOI_SIDE_M
    half = side_m / 2.0
    cx, cy = (
        gpd.GeoSeries(
            gpd.points_from_xy([centre_lonlat[0]], [centre_lonlat[1]]), crs=4326
        )
        .to_crs(3857)
        .iloc[0]
        .coords[0]
    )
    square_3857 = box(cx - half, cy - half, cx + half, cy + half)
    return gpd.GeoSeries([square_3857], crs=3857).to_crs(4326).iloc[0]


def case_study_aoi_ee(
    centre_lonlat: list[float] | None = None,
    side_m: float | None = None,
) -> ee.Geometry:
    """Earth Engine geometry for the case-study square AOI."""
    poly = case_study_aoi_polygon(centre_lonlat=centre_lonlat, side_m=side_m)
    return ee.Geometry(poly.__geo_interface__, None, False)


def aoi_wgs84_bounds(aoi_shapely: BaseGeometry) -> dict[str, float]:
    """Min/max lon/lat from a Shapely AOI polygon in EPSG:4326."""
    min_lon, min_lat, max_lon, max_lat = aoi_shapely.bounds
    return {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }


### --- CALIBRATION IN AOI


def class_geom_in_aoi(
    gdf: gpd.GeoDataFrame,
    class_name: str,
    aoi_shapely: BaseGeometry,
) -> ee.Geometry:
    """Class footprint clipped to the case-study AOI, as an EE geometry."""
    footprint = gdf.loc[gdf["Class_name"] == class_name, "geometry"].iloc[0]
    local = footprint.intersection(aoi_shapely)
    if local.is_empty:
        raise ValueError(f"{class_name!r} does not intersect case-study AOI.")
    return ee.Geometry(local.__geo_interface__, None, False)


def classes_in_aoi(
    gdf: gpd.GeoDataFrame,
    aoi_shapely: BaseGeometry,
) -> list[str]:
    """Class names with any polygon fragment inside the AOI (GeoPandas)."""
    hit = gdf.copy()
    hit["geometry"] = gdf.geometry.intersection(aoi_shapely)
    return sorted(
        hit.loc[hit.geometry.notna() & ~hit.geometry.is_empty, "Class_name"]
        .unique()
        .tolist()
    )


def classes_in_aoi_ee(
    fc: ee.FeatureCollection,
    aoi_ee: ee.Geometry,
) -> list[str]:
    """Class names with any fragment inside the AOI (Earth Engine)."""
    names = (
        fc.filterBounds(aoi_ee)
        .aggregate_array("Class_name")
        .distinct()
        .sort()
        .getInfo()
    )
    return names or []


### --- SAMPLING


def sample_class_pixels(
    image: ee.Image,
    gdf: gpd.GeoDataFrame,
    class_name: str,
    aoi_shapely: BaseGeometry,
    bands: list[str] | None = None,
    n_pixels: int = 400,
    seed: int = 7,
    tile_scale: int = 4,
) -> dict[str, list[float]]:
    """Uniform samples from imagery inside the class footprint within AOI."""
    bands = bands or s2.S2_BANDS
    region = class_geom_in_aoi(gdf, class_name, aoi_shapely)
    fc_samples = image.select(bands).sample(
        region=region,
        scale=config.SCALE_M,
        numPixels=n_pixels,
        seed=seed,
        geometries=False,
        tileScale=tile_scale,
    )
    features = fc_samples.getInfo().get("features", [])
    return {
        b: [f["properties"][b] for f in features if b in f["properties"]] for b in bands
    }
