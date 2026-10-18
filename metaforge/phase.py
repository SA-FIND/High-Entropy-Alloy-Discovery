"""Phase-stability domain for refractory alloy design: pairwise chemistry caps.

Simple descriptors (VEC, size misfit, a Laves-pair index) separate single-phase BCC from
multiphase refractory alloys poorly in the MPEA data (cross-validated AUC about 0.63), because
reported phases depend on processing. A stricter, transparent rule is used instead: a candidate
may not contain any element pair (i, j) at a concentration product c_i c_j larger than in some
alloy reported as single-phase BCC. Pairs never seen together in a single-phase alloy
(e.g. Zr-W, Hf-W, Zr-Cr, which form Laves phases) get a cap of zero.

An alloy counts as single-phase only if every report of it in the dataset says "BCC"
(e.g. HfNbTaTiZr, reported as BCC and as BCC+HCP after annealing, counts as multiphase).
"""
import itertools

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ELEMENTS = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
PAIRS = list(itertools.combinations(ELEMENTS, 2))


def pair_products(f):
    """Concentration products c_i c_j for all element pairs, in PAIRS order."""
    return np.array([f.get(a, 0.0) * f.get(b, 0.0) for a, b in PAIRS])


def label_alloys(mpea_csv, elements=ELEMENTS):
    """One row per alloy (fractions, single-phase flag) from the MPEA dataset file."""
    d = pd.read_csv(mpea_csv)
    d.columns = [c.split(": ", 1)[-1] for c in d.columns]

    def fractions(formula):
        try:
            c = Composition(formula).fractional_composition
            return {e.symbol: float(c[e]) for e in c.elements}
        except Exception:
            return None

    d["f"] = d.FORMULA.map(fractions)
    d = d[d.f.map(lambda f: f is not None and set(f) <= set(elements)) & d.Microstructure.notna()].copy()
    d["key"] = d.f.map(lambda f: " ".join(f"{e}{c:.3f}" for e, c in sorted(f.items())))
    g = d.groupby("key")
    out = pd.DataFrame({"single": g.Microstructure.apply(lambda s: int((s == "BCC").all())),
                        "labels": g.Microstructure.apply(lambda s: "/".join(sorted(set(s))))})
    first = d.drop_duplicates("key").set_index("key")
    out["formula"] = first.FORMULA
    out["f"] = first.f
    return out


def pair_caps(alloys):
    """Largest c_i c_j over single-phase alloys, per pair (0 if never seen together)."""
    pp = np.array([pair_products(f) for f in alloys.f[alloys.single == 1]])
    return pp.max(0)


def violation(f, caps):
    """Largest excess of any pair product over its cap (<= 0 means inside the domain)."""
    return float(np.max(pair_products(f) - caps))


def violating_pairs(f, caps, tol=1e-9):
    return [f"{a}-{b}" for (a, b), v, c in zip(PAIRS, pair_products(f), caps) if v > c + tol]


def leave_one_out(alloys, tol=1e-9):
    """Leave-one-alloy-out predictions of the rule (1 = predicted single-phase)."""
    pp = np.array([pair_products(f) for f in alloys.f])
    y = alloys.single.values
    idx = np.arange(len(y))
    pred = np.array([int(np.all(pp[i] <= pp[(y == 1) & (idx != i)].max(0) + tol)) for i in idx])
    return pred
