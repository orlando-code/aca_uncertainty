"""Plot styling, export paths, and matplotlib defaults."""

### --- IMPORTS
from __future__ import annotations

# general
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# custom
from aca_uncertainty.utils import config

### --- COLOURS (aligned with ACA v2.0)
_palette = {
    "Sand": "ffffbecc",
    "Rubble": "e0d05dcc",
    "Rock": "b29c3acc",
    "Seagrass Dense": "668438cc",
    "Seagrass Sparse": "9aa688cc",
    "Coral/Algae": "ff6161cc",
    "Microalgal Mats": "9bcc4ecc",
}
_missing = "bdc3c7cc"

INFERNO_HEXES = ["#000004", "#57106e", "#bc3754", "#f98e09", "#fcffa4"]


def aca_benthic_remap_from() -> list[int]:
    """ACA benthic class codes in display order (see `config.ACA_CODES`)."""
    return list(config.ACA_CODES)


def aca_benthic_remap_to() -> list[int]:
    """Consecutive EE palette indices (1...n) for `Image.remap` visualization."""
    n = len(config.ACA_CODES)
    return list(range(1, n + 1))


def aca_benthic_palette() -> list[str]:
    """`#RRGGBB` colours for `config.ACA_CODES`, aligned with `_palette`."""
    return [
        f"#{_palette[config.ACA_CODE_TO_CSP_NAME[code]][:6]}"
        for code in config.ACA_CODES
    ]


def aca_benthic_vis_params(*, band: str = "benthic") -> dict[str, Any]:
    """
    Geemap / `Map.addLayer` params for a band already remapped with
    `aca.remap_benthic_codes` (or equivalent `remap` call).
    """
    n = len(config.ACA_CODES)
    return {
        "bands": [band],
        "min": 1,
        "max": n,
        "palette": aca_benthic_palette(),
    }


def aca_benthic_legend() -> dict[str, str]:
    """Legend entries for ACA benthic classes (label -> `#RRGGBB`)."""
    palette = aca_benthic_palette()
    return {
        f"{config.ACA_CODE_TO_CSP_NAME[code]} (ACA={code})": palette[i]
        for i, code in enumerate(config.ACA_CODES)
    }


### --- PLOT HANDLING
FigureFormat = Literal["png", "pdf", "svg"]


@dataclass
class PlotConfig:
    """Shared figure export settings for notebooks and scripts."""

    figures_dir: Path = field(default_factory=lambda: config.figures_dir)
    dpi: int = 300
    default_formats: tuple[FigureFormat, ...] = ("png",)
    figsize: tuple[float, float] = (8.0, 3.0)
    figsize_dual: tuple[float, float] = (8.0, 4.0)
    figsize_square: tuple[float, float] = (7.0, 6.0)
    bar_chart_height: float = 4.2
    bar_chart_min_width: float = 9.0
    bar_chart_width_per_class: float = 1.35
    tight_layout: bool = True
    grid_color: str = "k"
    xtick_rotation: float = 35.0
    xtick_rotation_dense: float = 45.0
    legend_fontsize: int = 8
    annotation_fontsize: int = 8
    bar_edgecolor: str = "#333333"
    bar_linewidth: float = 0.5
    bar_capsize: float = 3.0
    bar_error_elinewidth: float = 0.9
    matplotlib_rc: dict[str, Any] = field(
        default_factory=lambda: {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.1,
        }
    )

    @property
    def figure_dpi(self) -> int:
        """DPI for `plt.subplots` / on-screen figures (`matplotlib_rc['figure.dpi']`)."""
        return int(self.matplotlib_rc.get("figure.dpi", self.dpi))

    def figsize_class_bar_chart(self, n_class: int) -> tuple[float, float]:
        """Figure size for a confidence bar chart with `n_class` classes."""
        width = max(self.bar_chart_min_width, self.bar_chart_width_per_class * n_class)
        return (width, self.bar_chart_height)

    def subplots(
        self,
        *args: Any,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        **kwargs: Any,
    ):
        """Create a figure via matplotlib using this config's default size and DPI."""
        return plt.subplots(
            *args,
            figsize=figsize or self.figsize,
            dpi=dpi if dpi is not None else self.figure_dpi,
            **kwargs,
        )

    def _grid_kw(self) -> dict[str, Any]:
        return {
            "linestyle": self.matplotlib_rc.get("grid.linestyle", "--"),
            "alpha": float(self.matplotlib_rc.get("grid.alpha", 0.5)),
            "color": self.grid_color,
        }

    def style_grid(
        self,
        ax: Axes,
        *,
        visible: bool = True,
        axis: Literal["both", "x", "y"] = "both",
    ) -> None:
        """Apply shared grid styling (`matplotlib_rc` linestyle/alpha)."""
        ax.grid(visible, axis=axis, **self._grid_kw())

    def style_xticklabels(
        self,
        ax: Axes,
        *,
        rotation: float | None = None,
        ha: str = "right",
    ) -> None:
        plt.setp(
            ax.xaxis.get_majorticklabels(),
            rotation=rotation if rotation is not None else self.xtick_rotation,
            ha=ha,
        )

    def bar_error_kw(self, *, elinewidth: float | None = None) -> dict[str, Any]:
        lw = elinewidth if elinewidth is not None else self.bar_error_elinewidth
        return {"elinewidth": lw, "ecolor": self.bar_edgecolor}

    def finish_figure(
        self, fig: Figure, *, tight: bool | None = None, **kwargs: Any
    ) -> None:
        if tight if tight is not None else self.tight_layout:
            fig.tight_layout(**kwargs)

    def ensure_dirs(self) -> Path:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        return self.figures_dir

    def apply_matplotlib(self) -> None:
        rc = dict(self.matplotlib_rc)
        rc.setdefault("savefig.dpi", self.dpi)
        plt.rcParams.update(rc)

    def figure_path(self, stem: str, ext: FigureFormat | str = "png") -> Path:
        ext = ext.removeprefix(".")
        return self.ensure_dirs() / f"{stem}.{ext}"

    def save_figure(
        self,
        fig,
        stem: str,
        *,
        formats: list[FigureFormat] | None = None,
        tight: bool = True,
        **savefig_kw: Any,
    ) -> list[Path]:
        """Write *fig* under `figures/`; returns paths written."""
        formats = tuple(formats or self.default_formats)
        dpi = savefig_kw.pop("dpi", self.dpi)
        paths: list[Path] = []
        for ext in formats:
            path = self.figure_path(stem, ext)
            kw = dict(savefig_kw)
            if tight:
                kw.setdefault("bbox_inches", "tight")
            fig.savefig(path, format=ext, dpi=dpi, **kw)
            paths.append(path)
        return paths


# default for notebooks: `from aca_uncertainty.plotting.config import PLOT`
PLOT = PlotConfig()
