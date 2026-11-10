### --- IMPORTS

# general
import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry


### --- CALCULATE AREA FRACTIONS
def class_area_fractions(
    gdf: gpd.GeoDataFrame,
    aoi_shapely: BaseGeometry,
    class_names: list[str],
) -> pd.DataFrame:
    """
    Area fraction of the AOI covered by each class footprint (projected metres).

    `fraction` = intersection(class, AOI).area / AOI.area (can sum to < 1 if
    unlabeled water remains in the circle).
    """
    aoi_3857 = gpd.GeoSeries([aoi_shapely], crs=4326).to_crs(3857).iloc[0]
    aoi_area_m2 = float(aoi_3857.area)
    rows = []
    for name in class_names:
        footprint = gdf.loc[gdf["Class_name"] == name, "geometry"]
        if footprint.empty:
            frac = 0.0
            area_m2 = 0.0
        else:
            local = footprint.iloc[0].intersection(aoi_shapely)
            if local.is_empty:
                frac = 0.0
                area_m2 = 0.0
            else:
                area_m2 = float(
                    gpd.GeoSeries([local], crs=4326).to_crs(3857).iloc[0].area
                )
                frac = area_m2 / aoi_area_m2
        rows.append(
            {
                "Class_name": name,
                "area_m2": area_m2,
                "aoi_area_m2": aoi_area_m2,
                "fraction": frac,
            }
        )
    return pd.DataFrame(rows).sort_values("Class_name").reset_index(drop=True)
