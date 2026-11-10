from pathlib import Path

import matplotlib.pyplot as plt

from aca_uncertainty.plotting.config import (
    PLOT,
    PlotConfig,
    aca_benthic_palette,
    aca_benthic_vis_params,
)
from aca_uncertainty.utils import config


def test_subplots_uses_config_dpi_and_figsize():
    cfg = PlotConfig(figsize=(6.0, 2.0), dpi=200)
    cfg.matplotlib_rc["figure.dpi"] = 111
    fig, _ax = cfg.subplots()
    try:
        assert fig.get_size_inches().tolist() == [6.0, 2.0]
        assert fig.dpi == 111
    finally:
        plt.close(fig)


def test_style_grid_uses_matplotlib_rc():
    cfg = PlotConfig()
    cfg.matplotlib_rc["grid.alpha"] = 0.22
    fig, ax = cfg.subplots()
    try:
        cfg.style_grid(ax)
        lines = ax.yaxis.get_gridlines()
        assert lines
        assert lines[0].get_alpha() == 0.22
    finally:
        plt.close(fig)


def test_figsize_class_bar_chart_scales_with_n_class():
    assert PLOT.figsize_class_bar_chart(3)[0] == PLOT.bar_chart_min_width
    wide = PLOT.figsize_class_bar_chart(20)
    assert wide[0] == PLOT.bar_chart_width_per_class * 20
    assert wide[1] == PLOT.bar_chart_height


def test_save_figure_tmp(tmp_path: Path):
    cfg = PlotConfig(figures_dir=tmp_path / "figures", default_formats=("png",))
    fig, ax = cfg.subplots()
    ax.plot([0, 1], [0, 1])
    paths = cfg.save_figure(fig, "smoke")
    plt.close(fig)
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".png"


def test_aca_benthic_vis_matches_codes():
    pal = aca_benthic_palette()
    vis = aca_benthic_vis_params(band="benthic")
    assert len(pal) == len(config.ACA_CODES)
    assert vis["min"] == 1
    assert vis["max"] == len(config.ACA_CODES)
    assert vis["palette"] == pal
    assert vis["bands"] == ["benthic"]
