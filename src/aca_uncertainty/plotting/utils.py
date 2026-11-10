### --- IMPORTS
from __future__ import annotations

# general
import numpy as np

# custom
from aca_uncertainty.plotting.config import _missing, _palette
from aca_uncertainty.utils import config
from aca_uncertainty.utils import ee as utils_ee


def opaque_fill(class_name: str) -> str:
    """RRGGBB + full alpha for traffic-light disagree fills."""
    return _palette.get(class_name, _missing)[:6] + "ff"


def limit_line_length(chars: str, n_chars: int = 20) -> str:
    """Limit the length of a string to a given number of characters."""
    # break a string on whitespace into multiple lines
    lines = [""]
    for i, word in enumerate(chars.split()):
        if i == 0:
            lines[0] = word
        elif len(lines[-1]) + len(word) > n_chars:
            lines.append(word)
        else:
            lines[-1] += f" {word}"
    return "\n".join(lines)


def max_distance_vis_m(scale_m: int | None = None) -> float:
    """Colour-scale maximum aligned with the EE distance kernel cap."""
    scale_m = scale_m or config.SCALE_M
    return utils_ee.max_distance_kernel_m(scale_m)


def blend_hex(hex_color: str, mix: float) -> str:
    """Blend two hex colours."""
    from matplotlib.colors import to_hex, to_rgb

    mix = float(np.clip(mix, 0.0, 1.0))
    r, g, b = to_rgb(hex_color)
    w = 1.0 - mix
    return to_hex((r * mix + w, g * mix + w, b * mix + w))
