### --- IMPORTS
from __future__ import annotations

# general
import math

import numpy as np

# custom
from aca_uncertainty.confidence import spectra as spectra_mod
from aca_uncertainty.confidence import stack
from aca_uncertainty.utils import config


def test_similarity_band_names_match():
    for name in config.ACA_CODE_TO_CSP_NAME.values():
        assert stack.class_name_to_sim_band(name) == spectra_mod.similarity_band_name(
            name
        )


def test_similarity_band_coral_algae():
    assert spectra_mod.similarity_band_name("Coral/Algae") == "sim_coral_algae"


def test_resolve_class_bands_skips_missing_seagrass():
    available = [
        "sim_coral_algae",
        "sim_microalgal_mats",
        "sim_rock",
        "sim_rubble",
        "sim_sand",
    ]
    resolved = stack.resolve_class_similarity_bands(available)
    assert 14 not in resolved
    assert resolved[15] == "sim_coral_algae"


def test_resolve_class_bands_seagrass_name_variant():
    available = ["sim_sand", "sim_seagrass_sparse"]
    resolved = stack.resolve_class_similarity_bands(available)
    assert resolved[14] == "sim_seagrass_sparse"


def test_category_confidence_product():
    """Category confidence is the product of components."""
    comps = np.array([0.2, 0.5, 0.0])
    assert math.isclose(float(np.prod(comps)), 0.0)


def test_noisy_or_matches_complement_of_confidence_product():
    """Uncertainty noisy-OR equals 1 - prod(c_i) when c_i = 1 - u_i."""
    us = np.array([0.2, 0.5, 0.0])
    cs = 1.0 - us
    u = 1.0 - float(np.prod(cs))
    assert math.isclose(u, 1 - (0.8 * 0.5 * 1.0))
