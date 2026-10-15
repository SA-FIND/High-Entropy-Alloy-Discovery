"""
Multi-objective design of single-phase BCC refractory alloys (NSGA-II).

Space: Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W; 4-6 elements present, each 5-35 at.%.
Objectives:
  1. maximise the lower 90% conformal bound of yield strength at 1000 C divided by
     rule-of-mixtures density (GPa cm3/g), using the hybrid model of strength_models.py
     refitted on all single-phase BCC data (train + future, after validation);
  2. minimise VEC, a ductility proxy: Sheikh et al., J. Appl. Phys. 120 (2016) 164902
     found ductile RHEAs at VEC <= 4.4 and brittle ones at VEC >= 4.6.
Constraints:
  - size misfit delta = sqrt(sum c dV^2) / (3 V) <= 6.6 % (volume form, Baruffi et al. 2022);
  - domain of applicability: standardised Magpie distance to the nearest measured alloy no
    larger than the 95th percentile of leave-one-alloy-out nearest-neighbour distances.

Run (web venv): python scripts/design_nsga2.py
Writes results/design_pareto.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition, Element
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from metaforge.composition import VEC, project_to_bounded_simplex, round_on_bounded_simplex  # noqa: E402
from metaforge.strength_edge import ELEMENTS_LEE2021, predict_rom  # noqa: E402
import strength_models as sm  # noqa: E402

warnings.filterwarnings("ignore")
EL = sm.ELEMENTS
AMU_PER_A3_TO_G_CM3 = 1.66053906660
ep = ElementProperty.from_preset("magpie")


def density(f):
    m = sum(c * Element(e).atomic_mass for e, c in f.items())
    v = sum(c * ELEMENTS_LEE2021[e][0] for e, c in f.items())
    return float(m / v * AMU_PER_A3_TO_G_CM3)


def misfit_delta(f):
    v = {e: ELEMENTS_LEE2021[e][0] for e in f}
    vm = sum(c * v[e] for e, c in f.items())
    return float(np.sqrt(sum(c * (v[e] - vm) ** 2 for e, c in f.items())) / (3 * vm))


def vec(f):
    return float(sum(c * VEC[e] for e, c in f.items()))


def to_fractions(x):
    return {e: float(c) for e, c in zip(EL, x) if c > 0}


class BoundedRepair(Repair):
    """Keep 4-6 elements (largest), each projected into [0.05, 0.35] summing to 1."""

    def _do(self, problem, X, **kwargs):
        out = np.zeros_like(X)
        for i, x in enumerate(X):
            order = np.argsort(-x)
            k = int(np.clip((x >= 0.05).sum(), 4, 6))
            keep = order[:k]
            y = round_on_bounded_simplex(project_to_bounded_simplex(x[keep]))
            out[i, keep] = y
        return out


class HybridPredictor:
    def __init__(self, d, X):
        tr = np.arange(len(d))
        y, p = d.YS_MPa.values, d.physics.values
        self.res = sm.gbm().fit(X, y - p)
        self.X, self.y, self.p, self.groups = X, y, p, d.key.values
        self.scaler = StandardScaler().fit(X[:, :-2])
        alloys = d.drop_duplicates("key")
        Z = self.scaler.transform(X[alloys.index, :-2])
        dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        self.Z = Z
        self.doa_threshold = float(np.percentile(dist.min(1), 95))

    def features(self, f, T_C):
        mag = np.array(ep.featurize(Composition(f)))
        return np.concatenate([mag, [T_C, 0.0]])  # compression test

    def predict(self, f, T_C):
        xf = self.features(f, T_C)[None, :]
        phys = 1000 * float(predict_rom(f, T_C + 273.15))
        lo, hi = sm.interval(self.X, self.y, self.p, xf, np.array([phys]), self.groups) \
            if getattr(self, "_with_pi", False) else (None, None)
        return phys + float(self.res.predict(xf)[0]), phys, lo, hi

    def doa_distance(self, f):
        z = self.scaler.transform(np.array(ep.featurize(Composition(f)))[None, :])
        return float(np.sqrt(((self.Z - z) ** 2).sum(1)).min())


class Problem(ElementwiseProblem):
    def __init__(self, model, qhat_lo):
        super().__init__(n_var=len(EL), n_obj=2, n_ieq_constr=2, xl=0.0, xu=0.35)
        self.m, self.qlo = model, qhat_lo

    def _evaluate(self, x, out, *args, **kwargs):
        f = to_fractions(x)
        s, _, _, _ = self.m.predict(f, 1000.0)
        lcb = s - self.qlo
        out["F"] = [-(lcb / 1000.0) / density(f), vec(f)]
        out["G"] = [misfit_delta(f) - 0.066, self.m.doa_distance(f) - self.m.doa_threshold]


def lower_offset(d, X):
    """Typical distance from the hybrid point prediction to its conformal lower bound at
    high temperature, estimated out-of-fold on the data (used inside the GA for speed;
    exact conformal intervals are recomputed for the final Pareto set)."""
    hot = np.where(d.T_C.values >= 800)[0]
    y, p = d.YS_MPa.values, d.physics.values
    from sklearn.model_selection import GroupKFold
    offs = []
    for a, b in GroupKFold(n_splits=5).split(X, groups=d.key.values):
        pt = p[b] + sm.gbm().fit(X[a], y[a] - p[a]).predict(X[b])
        lo, _ = sm.interval(X[a], y[a], p[a], X[b], p[b], d.key.values[a])
        offs += list((pt - lo)[np.isin(b, hot)])
    return float(np.median(offs))


def main():
    d, X = sm.load()
    model = HybridPredictor(d, X)
    q_lo = lower_offset(d, X)
    print(f"DoA threshold {model.doa_threshold:.2f}; median lower-bound offset at >=800 C: {q_lo:.0f} MPa")
    res = minimize(Problem(model, q_lo), NSGA2(pop_size=80, repair=BoundedRepair(), eliminate_duplicates=True),
                   ("n_gen", 60), seed=1, verbose=False)
    rows = []
    for x in np.unique(np.round(res.X, 3), axis=0):
        f = to_fractions(x)
        r = {"alloy": "".join(f"{e}{100 * c:.1f}" for e, c in f.items()), "n_el": len(f), "VEC": round(vec(f), 2),
             "density": round(density(f), 2), "delta_pct": round(100 * misfit_delta(f), 2),
             "doa_dist": round(model.doa_distance(f), 2)}
        for T in (25.0, 1000.0, 1200.0):
            xf = model.features(f, T)[None, :]
            s, phys, _, _ = model.predict(f, T)
            lo, hi = sm.interval(X, d.YS_MPa.values, d.physics.values, xf, np.array([phys]), d.key.values)
            r.update({f"YS_{int(T)}C": round(s), f"lo_{int(T)}C": round(float(lo[0])),
                      f"hi_{int(T)}C": round(float(hi[0])), f"physics_{int(T)}C": round(phys)})
        r["spec_lcb_1000C"] = round(r["lo_1000C"] / 1000 / r["density"], 4)
        rows.append(r)
    P = pd.DataFrame(rows).sort_values("VEC")
    os.makedirs("results", exist_ok=True)
    P.to_csv(os.path.join("results", "design_pareto.csv"), index=False)
    pd.set_option("display.width", 200)
    print(P.to_string(index=False))


if __name__ == "__main__":
    main()
