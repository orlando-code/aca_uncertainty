"""Random-forest workflow: geemap vis and matplotlib figures."""

### --- IMPORTS
from __future__ import annotations

# general
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# custom
from aca_uncertainty.confidence import rf
from aca_uncertainty.plotting import config as plot_config
from aca_uncertainty.utils import config


### --- PLOTTING HELPER FUNCTIONS
def entropy_vis(
    *,
    max_entropy: float | None = None,
    n_classes: int | None = None,
) -> dict[str, Any]:
    """Vis for entropy layer (0 = confident, high = spread across classes)."""
    if max_entropy is None:
        k = n_classes or len(config.ACA_CODES)
        max_entropy = math.log(k) if k > 1 else 1.0
    return {
        "bands": ["entropy"],
        "min": 0,
        "max": max_entropy,
        "palette": plot_config.INFERNO_HEXES,
    }


def class_probability_vis(*, class_code: int, max_prob: float = 1.0) -> dict[str, Any]:
    """Single-band RF vote probability (0–1) for geemap."""
    return {
        "bands": [str(class_code)],
        "min": 0,
        "max": max_prob,
        "palette": plot_config.INFERNO_HEXES,
    }


def benthic_prediction_vis() -> dict[str, Any]:
    """Vis params for `benthic_prediction_for_display` (remapped 1...n palette)."""
    return plot_config.aca_benthic_vis_params(band="benthic_pred")


### --- PLOTTING
def plot_holdout_class_probabilities(
    records: list[dict[str, Any]],
    *,
    class_names: dict[int, str] | None = config.ACA_CODE_TO_CSP_NAME,
) -> plt.Figure:
    """Boxplot of P(true class) on hold-out samples, one box per reference class."""
    df = pd.DataFrame(records)  # DataFrame of hold-out probability records
    if df.empty:
        raise ValueError("No hold-out probability records.")
    mean_p = df.groupby("class_num")["p_true_class"].mean()
    order = mean_p.sort_values(
        ascending=False
    ).index.tolist()  # sort classes by mean probability
    labels = [class_names.get(int(c), str(c)) for c in order]
    data = [df.loc[df["class_num"] == c, "p_true_class"].dropna().values for c in order]
    colors = [
        f"#{plot_config._palette.get(class_names.get(int(c), ''), plot_config._missing)[:6]}"
        for c in order
    ]  # set up by-class box colours

    fig, (box_ax, bar_ax) = plot_config.PLOT.subplots(
        1, 2, figsize=plot_config.PLOT.figsize_dual, sharey=True
    )

    # (left) boxplot of hold-out probabilities
    bp = box_ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):  # colour boxes
        patch.set_facecolor(color)

    # formatting
    box_ax.set_ylabel(r"$P$(reference class)")
    box_ax.set_xlabel("Hold-out reference class")
    plot_config.PLOT.style_xticklabels(box_ax)

    # (right) bar chart of mean hold-out probabilities
    mean_sorted = mean_p.reindex(order)  # mean probabilities sorted by class
    sem = df.groupby("class_num")["p_true_class"].sem().reindex(order).fillna(0.0)
    counts = df.groupby("class_num").size().reindex(order)
    x = np.arange(len(order))
    bar_ax.bar(
        x,
        mean_sorted,
        yerr=sem,
        capsize=plot_config.PLOT.bar_capsize,
        color=colors,
        edgecolor=plot_config.PLOT.bar_edgecolor,
        linewidth=plot_config.PLOT.bar_linewidth,
        error_kw=plot_config.PLOT.bar_error_kw(elinewidth=1.0),
    )
    # formatting
    bar_ax.set_xticks(x, labels)
    plot_config.PLOT.style_xticklabels(bar_ax)
    bar_ax.set_ylabel(r"Mean $P$(reference class)")
    for i, (m, err, n) in enumerate(zip(mean_sorted, sem, counts, strict=True)):
        if np.isfinite(m):  # add pixel validation pixel counts to bars
            bar_ax.text(
                i,
                float(m) + float(err) + 0.02,
                f"n={int(n)}",
                ha="center",
                fontsize=plot_config.PLOT.annotation_fontsize,
            )
    for ax in (box_ax, bar_ax):
        plot_config.PLOT.style_grid(ax)
    plot_config.PLOT.finish_figure(fig)
    return fig


def plot_confusion_matrix(
    cm: dict[str, Any],
    *,
    class_names: dict[int, str] | None = None,
) -> plt.Figure:
    """Heatmap from `confusion_to_arrays` output."""
    class_names = class_names or config.ACA_CODE_TO_CSP_NAME
    full = np.asarray(cm["matrix"], dtype=float)
    full_order = [int(x) for x in cm["order"]]
    idx = [
        i for i, c in enumerate(full_order) if c in set(rf.active_confusion_classes(cm))
    ]
    matrix = full[np.ix_(idx, idx)]
    order = [full_order[i] for i in idx]
    labels = [class_names.get(c, str(c)) for c in order]

    fig, ax = plot_config.PLOT.subplots(figsize=plot_config.PLOT.figsize_square)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    plot_config.PLOT.style_xticklabels(
        ax, rotation=plot_config.PLOT.xtick_rotation_dense
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference (CSP)")
    ax.set_title(
        f"RF hold-out (AOI only) — accuracy {cm['accuracy']:.2f}, κ {cm['kappa']:.2f}"
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=plot_config.PLOT.annotation_fontsize,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plot_config.PLOT.finish_figure(fig)
    return fig
