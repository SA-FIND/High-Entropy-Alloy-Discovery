import numpy as np
import pytest

from metaforge.composition import (
    crystal_type, project_to_bounded_simplex, round_on_bounded_simplex, vegard_lattice_parameter,
)


def test_projection_respects_bounds_for_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(10_000):
        y = rng.normal(0.2, 0.3, size=5)
        x = project_to_bounded_simplex(y)
        assert abs(x.sum() - 1.0) < 1e-9
        assert x.min() >= 0.05 - 1e-12 and x.max() <= 0.35 + 1e-12


def test_old_clip_and_renormalise_breaks_bounds_but_projection_does_not():
    y = np.array([0.9, 0.9, 0.0, 0.0, 0.0])
    old = np.clip(y, 0.05, 0.35); old = old / old.sum()
    assert old.max() > 0.35  # the bug seen in the GA output
    assert project_to_bounded_simplex(y).max() <= 0.35 + 1e-12


def test_feasible_point_is_unchanged():
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert np.allclose(project_to_bounded_simplex(x), x)


def test_infeasible_bounds_raise():
    with pytest.raises(ValueError):
        project_to_bounded_simplex([0.5, 0.5], lo=0.05, hi=0.35)


def test_rounding_keeps_sum_and_bounds():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        x = round_on_bounded_simplex(project_to_bounded_simplex(rng.random(5)))
        assert abs(x.sum() - 1.0) < 1e-9
        assert x.min() >= 0.05 - 1e-9 and x.max() <= 0.35 + 1e-9


def test_vegard_lattice_parameter_matches_published_mo_nb_ta_w():
    # Maresca & Curtin (2020) Table 2: a(Vegard) = 3.228 A for equiatomic MoNbTaW.
    a = vegard_lattice_parameter({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25})
    assert abs(a - 3.228) < 0.005


def test_crystal_type_from_vec():
    assert crystal_type({"Mo": 0.25, "Nb": 0.25, "Ta": 0.25, "W": 0.25}) == "BCC"
    assert crystal_type({"Co": 0.25, "Cr": 0.25, "Fe": 0.25, "Ni": 0.25}) == "FCC"
    assert crystal_type({"Al": 0.2, "Ti": 0.2, "Sc": 0.2, "Zr": 0.2, "V": 0.2}) == "BCC"
