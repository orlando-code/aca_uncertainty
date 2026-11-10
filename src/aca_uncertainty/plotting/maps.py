"""Export geemap / Folium maps to `figures/`. N.B. having lots of issues here, still a work in progress"""

### --- IMPORTS
from __future__ import annotations

# general
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# spatial
import ee
from shapely.geometry.base import BaseGeometry

# custom
from aca_uncertainty.geo import aoi as geo_aoi
from aca_uncertainty.plotting.config import PLOT, PlotConfig

# Folium fit_bounds: [[south, west], [north, east]]
LatLonBounds = tuple[list[float], list[float]]


def _map_png_failure_types() -> tuple[type[BaseException], ...]:
    """Exception types Folium/Selenium PNG export is expected to raise."""
    types_: list[type[BaseException]] = [
        OSError,
        RuntimeError,
        ImportError,
        ValueError,
        TypeError,
    ]
    try:
        from selenium.common.exceptions import WebDriverException

        types_.append(WebDriverException)
    except ImportError:
        pass
    return tuple(types_)


_MAP_PNG_FAILURES = _map_png_failure_types()


@dataclass
class MapView:
    """
    View to apply immediately before export (not read from the live widget).

    Set one of `bounds`, `bounds_wgs84`, `shapely`, or `ee_object`.
    Optional `zoom` is used with `ee_object` when not fitting bounds.
    """

    bounds: LatLonBounds | None = None
    bounds_wgs84: dict[str, float] | None = None
    shapely: BaseGeometry | None = None
    ee_object: ee.ComputedObject | None = None
    zoom: int | None = None

    @classmethod
    def from_aoi_shapely(cls, aoi: BaseGeometry, *, zoom: int | None = None) -> MapView:
        return cls(bounds_wgs84=geo_aoi.aoi_wgs84_bounds(aoi), zoom=zoom)

    @classmethod
    def from_aoi_ee(cls, aoi: ee.Geometry, *, zoom: int | None = None) -> MapView:
        return cls(ee_object=aoi, zoom=zoom)


def wgs84_dict_to_folium_bounds(bb: dict[str, float]) -> LatLonBounds:
    return [
        [bb["min_lat"], bb["min_lon"]],
        [bb["max_lat"], bb["max_lon"]],
    ]


def apply_map_view(map_obj: Any, view: MapView | None = None) -> None:
    """Programmatic zoom; does not capture pan/zoom from the Jupyter iframe."""
    if view is None:
        return
    if view.bounds is not None:
        map_obj.fit_bounds(view.bounds)
        return
    if view.bounds_wgs84 is not None:
        map_obj.fit_bounds(wgs84_dict_to_folium_bounds(view.bounds_wgs84))
        return
    if view.shapely is not None:
        map_obj.fit_bounds(
            wgs84_dict_to_folium_bounds(geo_aoi.aoi_wgs84_bounds(view.shapely))
        )
        return
    if view.ee_object is not None:
        map_obj.center_object(view.ee_object, zoom=view.zoom)
        return


def _selenium_driver(size: tuple[int, int] = (1600, 900)):
    """Headless Chrome (Selenium Manager); fallback Firefox if Chrome unavailable."""
    from selenium import webdriver
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions

    chrome_opts = ChromeOptions()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--hide-scrollbars")
    chrome_opts.add_argument(f"--window-size={size[0]},{size[1]}")
    try:
        return webdriver.Chrome(options=chrome_opts)
    except WebDriverException:
        firefox_opts = FirefoxOptions()
        firefox_opts.add_argument("--headless")
        return webdriver.Firefox(options=firefox_opts)


def _save_map_png_folium(
    map_obj: Any,
    path: Path,
    *,
    wait_sec: int = 8,
    size: tuple[int, int] = (1600, 900),
) -> None:
    """Rasterise via Folium `_to_png` (Selenium + headless Chrome recommended)."""
    to_png = getattr(map_obj, "_to_png", None)
    if to_png is None:
        raise RuntimeError("Map object does not support Folium _to_png.")
    if hasattr(map_obj, "_png_image"):
        map_obj._png_image = None
    driver = _selenium_driver(size)
    img_data = to_png(delay=wait_sec, driver=driver, size=size)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.open(io.BytesIO(img_data)).save(path)
    except ImportError as exc:
        path.write_bytes(img_data)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise RuntimeError(
                "Install Pillow to convert Folium PNG bytes to an image file."
            ) from exc


def save_map(
    map_obj: Any,
    stem: str,
    *,
    plot: PlotConfig = PLOT,
    view: MapView | None = None,
    save_html: bool = True,
    save_png: bool = False,
    png_wait_sec: int = 12,
    png_size: tuple[int, int] = (1600, 900),
) -> dict[str, Path]:
    """
    Export a geemap Map.

    * **HTML** — always reliable (`map.to_html`).
    * **PNG** — Folium `_to_png` via headless **Chrome** (Selenium 4 manager);
      needs `pip install selenium` and Google Chrome installed.

    **Interactive vs export view:** Jupyter pan/zoom is not read from the widget.
    Pass `view=MapView.from_aoi_shapely(cs_aoi_poly)` (or bounds / EE geometry)
    so exports match a chosen extent. For a manual view, pan in the browser, use
    geemap's fullscreen screenshot / `to_image` (ipyleaflet), or save HTML and
    screenshot in a browser.
    """
    plot.ensure_dirs()
    apply_map_view(map_obj, view)
    out: dict[str, Path] = {}

    if save_html:
        html_path = plot.figure_path(stem, "html")
        if hasattr(map_obj, "to_html"):
            map_obj.to_html(str(html_path))
        else:
            map_obj.save(str(html_path))
        out["html"] = html_path

    if save_png:
        png_path = plot.figure_path(stem, "png")
        try:
            _save_map_png_folium(
                map_obj, png_path, wait_sec=png_wait_sec, size=png_size
            )
            out["png"] = png_path
        except _MAP_PNG_FAILURES as exc:
            warnings.warn(
                f"Map PNG export failed ({exc}). HTML was still saved. "
                "Install selenium, ensure Google Chrome is installed (Selenium 4 "
                "downloads chromedriver automatically), or open the HTML export and "
                "screenshot manually.",
                UserWarning,
                stacklevel=2,
            )

    return out
