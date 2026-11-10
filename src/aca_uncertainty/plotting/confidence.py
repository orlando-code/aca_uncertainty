"""Confidence index: Earth Engine vis params and matplotlib summaries."""

### --- IMPORTS
from __future__ import annotations

# general
import math

# spatial
from typing import Any

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# custom
from aca_uncertainty.confidence import stack
from aca_uncertainty.plotting import utils as plot_utils
from aca_uncertainty.plotting.config import INFERNO_HEXES, PLOT
from aca_uncertainty.utils import config

### --- PLOTTING LABELS AND FORMAT CONFIG
UncertaintyFamily = str  # epistemic | meta | total

# band, legend label, uncertainty family (colour group)
CONFIDENCE_BAR_SPEC: tuple[tuple[str, str, UncertaintyFamily], ...] = (
    ("c_ent", r"$c_{\mathrm{ent}}$", "epistemic"),
    ("c_spec", r"$c_{\mathrm{spec}}$", "epistemic"),
    ("C_epi", r"$C_{\mathrm{epi}}$", "epistemic"),
    ("c_img", r"$c_{\mathrm{img}}$", "meta"),
    ("C_meta", r"$C_{\mathrm{meta}}$", "meta"),
    ("C_total", r"$C_{\mathrm{total}}$", "total"),
)

CONFIDENCE_BAR_METRICS: tuple[tuple[str, str], ...] = tuple(
    (band, label) for band, label, _family in CONFIDENCE_BAR_SPEC
)

CONFIDENCE_BAR_WIDTH = 0.09
CONFIDENCE_BAR_INNER_GAP = 0.02
CONFIDENCE_BAR_GROUP_GAP = 0.07
MISSING_BAR_Y = 0.03


### --- PLOT HELPERS
def confidence_vis(
    *,
    band: str | None = "C_total",  # default: total confidence layer
    image: ee.Image | None = None,
    min_val: float = 0.0,
    max_val: float = 1.0,
    palette: list[str] | None = None,
) -> dict[str, Any]:
    """geemap visualisation for confidence layers."""
    if band is None:
        if image is None:
            raise ValueError("Provide `band=` or `image=` for confidence_vis.")
        names = image.bandNames().getInfo() or []
        if not names:
            raise ValueError("Image has no bands.")
        band = str(names[0])

    palette = palette or INFERNO_HEXES
    return {
        "bands": [band],
        "min": min_val,
        "max": max_val,
        "palette": palette,
    }


def confidence_vis_for(img: ee.Image, **kwargs: Any) -> dict[str, Any]:
    """Vis params using the image's first band name (for geemap layers)."""
    return confidence_vis(band=None, image=img, **kwargs)


def _uncertainty_family_base_color(family: UncertaintyFamily) -> str:
    inferno = INFERNO_HEXES
    if family == "total":
        return inferno[1]
    if family == "epistemic":
        return inferno[2]
    if family == "meta":
        return inferno[3]
    if family == "aleatoric":
        return inferno[4]
    raise ValueError(f"Unknown uncertainty family: {family!r}")


def _bar_color_for_spec(
    spec: tuple[tuple[str, str, UncertaintyFamily], ...],
    index: int,
) -> str:
    """Colour confidence bars by type of uncertainty"""
    family = spec[index][2]
    if family == "total":
        return _uncertainty_family_base_color("total")
    indices = [i for i, row in enumerate(spec) if row[2] == family]
    rank = indices.index(index)
    n = len(indices)
    if n == 1:
        return _uncertainty_family_base_color(family)
    mix = 0.38 + 0.62 * (rank / (n - 1))
    return plot_utils.blend_hex(_uncertainty_family_base_color(family), mix)


def _bar_offsets_for_spec(
    spec: tuple[tuple[str, str, UncertaintyFamily], ...],
    *,
    bar_width: float = CONFIDENCE_BAR_WIDTH,
    inner_gap: float = CONFIDENCE_BAR_INNER_GAP,
    group_gap: float = CONFIDENCE_BAR_GROUP_GAP,
) -> list[float]:
    edges: list[float] = []
    pos = 0.0
    prev_family: UncertaintyFamily | None = None
    for _band, _label, family in spec:
        if prev_family is not None:
            pos += group_gap if family != prev_family else inner_gap
        edges.append(pos)
        pos += bar_width
        prev_family = family
    cluster_span = pos
    shift = cluster_span / 2.0 - bar_width / 2.0
    return [e - shift for e in edges]


### --- COMPUTE PER-CLASS PER-BAND CONFIDENCE STATISTICS


def confidence_by_predicted_class(
    stack: stack.ConfidenceStack,
    rf_prediction: ee.Image,
    region: ee.Geometry,
    class_codes: list[int],
    *,
    metrics: tuple[tuple[str, str], ...] | None = CONFIDENCE_BAR_METRICS,
    scale_m: int | None = config.SCALE_M,
    class_names: dict[int, str] | None = config.ACA_CODE_TO_CSP_NAME,
) -> pd.DataFrame:
    """
    Mean and SEM of confidence layers, grouped by RF-predicted class (pixels) in `region`.

    SEM uses per-class pixel stdDev / sqrt(n) from `reduceRegion` (spatial pixels).
    """
    metric_bands = [m[0] for m in metrics]
    pred = rf_prediction.select("benthic_pred")
    layers = stack.component_bands().select(metric_bands)
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), None, True)
        .combine(ee.Reducer.count(), None, True)
    )
    rows: list[dict[str, Any]] = []
    for code in tqdm(class_codes, desc="Calculating confidence by class"):
        code = int(code)  # class code
        stats = (  # mean, stdDev, count
            layers.updateMask(pred.eq(code))
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
        for band, _label in tqdm(metrics, desc="Calculating confidence by band"):
            mean = stats.get(f"{band}_mean")
            std = stats.get(f"{band}_stdDev")
            count = stats.get(f"{band}_count")
            if mean is None or count in (None, 0):
                continue
            n = float(count)  # number of pixels in class and band
            sem = (float(std) / math.sqrt(n)) if std is not None and n > 1 else 0.0
            rows.append(
                {
                    "class_num": code,
                    "Class_name": class_names.get(code, str(code)),
                    "metric": band,
                    "mean": float(mean),
                    "sem": float(sem),
                    "n_pixels": int(n),
                }
            )
    df = pd.DataFrame(rows)  # DataFrame of confidence stats by class and band
    if df.empty:
        print("No confidence stats found for any classes.")
        return df
    order = (
        df.groupby("class_num")["n_pixels"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    df["class_num"] = pd.Categorical(
        df["class_num"], categories=order, ordered=True
    )  # sort classes by number of pixels
    return df.sort_values(["class_num", "metric"])


### --- PLOTTING CONFIDENCE BY UNCERTAINTY CLASS
def plot_confidence_by_class(
    df: pd.DataFrame,
    *,
    metrics: tuple[tuple[str, str], ...] | None = CONFIDENCE_BAR_METRICS,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Grouped bar chart: confidence metrics x RF-predicted class, with SEM error bars."""
    if df.empty:
        raise ValueError("No per-class confidence stats (empty DataFrame).")

    spec_by_band = {row[0]: row for row in CONFIDENCE_BAR_SPEC}
    if metrics == CONFIDENCE_BAR_METRICS:
        spec = CONFIDENCE_BAR_SPEC
    else:
        spec = tuple(
            spec_by_band[band] for band, _label in metrics if band in spec_by_band
        )
        if not spec:
            raise ValueError("metrics bands not found in CONFIDENCE_BAR_SPEC.")

    class_order = [int(c) for c in df["class_num"].cat.categories]
    class_labels = [
        df.loc[df["class_num"] == c, "Class_name"].iloc[0] for c in class_order
    ]
    n_class = len(class_order)

    fig, ax = PLOT.subplots(figsize=figsize or PLOT.figsize_class_bar_chart(n_class))
    x = np.arange(n_class, dtype=float)

    for j, (band, label, _family) in enumerate(spec):
        sub = df.loc[df["metric"] == band].set_index("class_num")
        means = np.array(
            [sub.loc[c, "mean"] if c in sub.index else np.nan for c in class_order],
            dtype=float,
        )
        sems = np.array(
            [sub.loc[c, "sem"] if c in sub.index else 0.0 for c in class_order],
            dtype=float,
        )
        xs = x + _bar_offsets_for_spec(spec)[j]
        color = _bar_color_for_spec(spec, j)
        valid = np.isfinite(means)
        if valid.any():  # plot bars for valid classes (non-zero)
            ax.bar(
                xs[valid],
                means[valid],
                CONFIDENCE_BAR_WIDTH,
                yerr=sems[valid],
                capsize=PLOT.bar_capsize,
                label=label,
                color=color,
                edgecolor=PLOT.bar_edgecolor,
                linewidth=PLOT.bar_linewidth,
                error_kw=PLOT.bar_error_kw(),
            )
        missing = ~valid
        if missing.any():  # indicate missing classes (NaN) with crosses
            ax.scatter(
                xs[missing],
                np.full(int(missing.sum()), MISSING_BAR_Y),
                marker="x",
                s=22,
                c=color,
                linewidths=0.9,
                label="_nolegend",
                zorder=4,
                clip_on=False,
            )

    ylabels = [plot_utils.limit_line_length(label) for label in class_labels]
    ax.set_xticks(x, ylabels)
    ax.set_ylabel("Mean confidence (0–1)")
    ax.set_xlabel("Random Forest-predicted class", labelpad=10)
    ax.set_ylim(0.0, 1.0)
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(spec),  # number of metrics
        fontsize=PLOT.legend_fontsize,
    )
    PLOT.style_grid(ax, axis="y")
    PLOT.finish_figure(fig)
    return fig
