import numpy as np
import pandas as pd

from metaforge.phase import PAIRS, leave_one_out, pair_caps, pair_products, violating_pairs, violation


def _alloys(rows):
    return pd.DataFrame({"f": [f for f, _ in rows], "single": [s for _, s in rows]})


def test_pair_products_are_symmetric_and_complete():
    f = {"Nb": 0.5, "Ta": 0.5}
    pp = pair_products(f)
    assert len(pp) == len(PAIRS) == 36
    assert np.isclose(pp.sum(), 0.25)


def test_caps_come_only_from_single_phase_alloys():
    A = _alloys([({"Nb": 0.5, "Ta": 0.5}, 1), ({"Zr": 0.5, "W": 0.5}, 0)])
    caps = pair_caps(A)
    assert np.isclose(caps[PAIRS.index(("Nb", "Ta"))], 0.25)
    assert caps[PAIRS.index(("Zr", "W"))] == 0.0


def test_unseen_pair_is_a_violation():
    A = _alloys([({"Nb": 0.5, "Ta": 0.5}, 1), ({"Zr": 0.5, "Nb": 0.5}, 1)])
    caps = pair_caps(A)
    f = {"Zr": 0.3, "Nb": 0.4, "W": 0.3}
    assert violation(f, caps) > 0
    assert "Zr-W" in violating_pairs(f, caps) and "Nb-W" in violating_pairs(f, caps)
    assert violation({"Nb": 0.6, "Ta": 0.4}, caps) <= 0


def test_leave_one_out_rejects_an_alloy_with_an_unseen_pair():
    # NbTa sets the Nb-Ta cap itself, so left out it exceeds the cap of Nb0.4Ta0.6 (0.24 < 0.25)
    A = _alloys([({"Nb": 0.5, "Ta": 0.5}, 1), ({"Nb": 0.4, "Ta": 0.6}, 1), ({"Zr": 0.5, "W": 0.5}, 0)])
    assert list(leave_one_out(A)) == [0, 1, 0]
