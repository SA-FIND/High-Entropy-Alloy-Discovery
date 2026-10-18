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
    larger than the 95th percentile of leave-one-alloy-out nearest-neighbour distances;
  - phase stability: no element pair at a concentration product above that of any alloy
    reported as single-phase BCC (metaforge/phase.py; validated in scripts/phase_domain.py).
    Without it every Pareto alloy combined Zr with W, Mo and V (Laves formers);
  - high-temperature domain: the same distance limit, measured to alloys tested at >= 900 C.
    Ti-Zr-V alloys soften abruptly above ~600 C (NbTiVZr: 834 MPa at 600 C, 58 MPa at 1000 C);
    held out by alloy system, the hybrid over-predicts them 6-7x at 1000 C. Without this
    constraint the search favoured Ti-Zr-Hf-V-Mo alloys, a family tested only at 25 C.

Candidates whose predicted strength at 1000 C exceeds the largest strength measured at
>= 900 C in the data are flagged (extrapolation in the target, not only in composition).

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
from metaforge.phase import label_alloys, pair_caps, violating_pairs, violation  # noqa: E402
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
    """Hybrid model refitted on all single-phase BCC data, with conformalized quantile
    intervals calibrated once (out-of-fold, grouped by alloy) so that any new prediction
    and its 90% interval cost only a few model evaluations."""

    def __init__(self, d, X, alpha=0.10):
        y, p, g = d.YS_MPa.values, d.physics.values, d.key.values
        r = y - p
        self.res = sm.gbm().fit(X, r)
        self.q_lo = sm.gbm(quantile=0.05).fit(X, r)
        self.q_hi = sm.gbm(quantile=0.95).fit(X, r)
        from sklearn.model_selection import GroupKFold
        scores = np.empty(len(r))
        for a, b in GroupKFold(n_splits=5).split(X, groups=g):
            lo, hi = sm._quantile_band(X[a], r[a], X[b])
            scores[b] = np.maximum(lo - r[b], r[b] - hi)
        n = len(scores)
        self.qhat = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
        self.scaler = StandardScaler().fit(X[:, :-2])
        alloys = d.drop_duplicates("key")
        Z = self.scaler.transform(X[alloys.index, :-2])
        dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        self.Z = Z
        self.doa_threshold = float(np.percentile(dist.min(1), 95))
        # alloys tested at >= 900 C: the only ones that show where a chemistry softens
        self.Z_hot = Z[alloys.key.isin(set(d.key[d.T_C >= 900])).values]

    @staticmethod
    def magpie(f):
        return np.array(ep.featurize(Composition(f)))

    def predict(self, f, T_C, mag=None):
        """(point, lower, upper, physics) in MPa for a compression test at T_C."""
        mag = self.magpie(f) if mag is None else mag
        xf = np.concatenate([mag, [T_C, 0.0]])[None, :]
        phys = 1000 * float(predict_rom(f, T_C + 273.15))
        point = phys + float(self.res.predict(xf)[0])
        lo = phys + float(self.q_lo.predict(xf)[0]) - self.qhat
        hi = phys + float(self.q_hi.predict(xf)[0]) + self.qhat
        return point, lo, hi, phys

    def doa_distance(self, f, mag=None, hot=False):
        mag = self.magpie(f) if mag is None else mag
        z = self.scaler.transform(mag[None, :])
        return float(np.sqrt((((self.Z_hot if hot else self.Z) - z) ** 2).sum(1)).min())


class Problem(ElementwiseProblem):
    def __init__(self, model, caps):
        super().__init__(n_var=len(EL), n_obj=2, n_ieq_constr=4, xl=0.0, xu=0.35)
        self.m = model
        self.caps = caps

    def _evaluate(self, x, out, *args, **kwargs):
        f = to_fractions(x)
        mag = self.m.magpie(f)
        _, lcb, _, _ = self.m.predict(f, 1000.0, mag)
        out["F"] = [-(lcb / 1000.0) / density(f), vec(f)]
        out["G"] = [misfit_delta(f) - 0.066, self.m.doa_distance(f, mag) - self.m.doa_threshold,
                    violation(f, self.caps), self.m.doa_distance(f, mag, hot=True) - self.m.doa_threshold]


def main():
    d, X = sm.load()
    model = HybridPredictor(d, X)
    caps = pair_caps(label_alloys(os.path.join("data", "external", "mpea", "MPEA_dataset.csv")))
    ys_max_hot = float(d.YS_MPa[d.T_C >= 900].max())
    print(f"DoA threshold {model.doa_threshold:.2f}; conformal margin {model.qhat:.0f} MPa; "
          f"max measured YS at >= 900 C {ys_max_hot:.0f} MPa", flush=True)
    res = minimize(Problem(model, caps), NSGA2(pop_size=80, repair=BoundedRepair(), eliminate_duplicates=True),
                   ("n_gen", 60), seed=1, verbose=False)
    if res.X is None:
        raise SystemExit("no feasible design found")
    rows = []
    for x in np.unique(np.round(res.X, 3), axis=0):
        f = to_fractions(x)
        r = {"alloy": "".join(f"{e}{100 * c:.1f}" for e, c in f.items()), "n_el": len(f), "VEC": round(vec(f), 2),
             "density": round(density(f), 2), "delta_pct": round(100 * misfit_delta(f), 2),
             "doa_dist": round(model.doa_distance(f), 2), "doa_hot_dist": round(model.doa_distance(f, hot=True), 2),
             "phase_violations": " ".join(violating_pairs(f, caps))}
        for T in (25.0, 1000.0, 1200.0):
            s, lo, hi, phys = model.predict(f, T)
            r.update({f"YS_{int(T)}C": round(s), f"lo_{int(T)}C": round(lo),
                      f"hi_{int(T)}C": round(hi), f"physics_{int(T)}C": round(phys)})
        r["spec_lcb_1000C"] = round(r["lo_1000C"] / 1000 / r["density"], 4)
        r["beyond_measured_1000C"] = r["YS_1000C"] > ys_max_hot
        rows.append(r)
    P = pd.DataFrame(rows).sort_values("VEC")
    os.makedirs("results", exist_ok=True)
    P.to_csv(os.path.join("results", "design_pareto.csv"), index=False)
    pd.set_option("display.width", 200)
    print(P.to_string(index=False))


if __name__ == "__main__":
    main()
