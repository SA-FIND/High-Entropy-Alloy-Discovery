"""
Feed potential-computed inputs into the edge-strength model and score against experiment.

Input routes compared on the single-phase BCC training alloys:
  lee2021        elemental rule of mixtures with the literature table (Lee et al. 2021)
  <pot>_elements elemental rule of mixtures with bcc a, C_ij from each potential
                 (results/mlip_elements.csv). Alloys whose averaged C11 - C12 <= 0 (from
                 mechanically unstable bcc Ti/Zr/Hf) cannot be evaluated and are counted.
  <pot>_volumes+lit_Cij  literature C_ij with the potential's bcc atomic volumes (misfits)
  <pot>_sqs      alloy-level C_ij and lattice parameter from relaxed SQS supercells
                 (results/mlip_alloys.csv), misfit volumes from the potential's elemental
                 bcc volumes; only for the alloys computed there.

Run (web venv): python scripts/mlip_strength_inputs.py
Writes results/mlip_strength_inputs.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from metaforge.strength_edge import (ELEMENTS_LEE2021, rule_of_mixtures_inputs, yield_stress,  # noqa: E402
                                     zero_temperature)

warnings.filterwarnings("ignore")
EL = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]


def score(y, p):
    ok = np.isfinite(p)
    e = p[ok] - y[ok]
    return {"n": int(ok.sum()), "n_failed": int((~ok).sum()), "MAE": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "R2": float(1 - np.sum(e ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))}


def predict(f, T_C, a, c11, c12, c44, dv):
    if c11 - c12 <= 0 or c44 <= 0:
        return np.nan
    t0, eb = zero_temperature(a, c11, c12, c44, f, dv)
    return 1000 * float(yield_stress(t0, eb, T_C + 273.15))


def main():
    d = pd.read_csv(os.path.join("data", "processed", "rhea_strength.csv"))
    d = d[d.physics_domain & (d.holdout == "train")].reset_index(drop=True)
    fr = [{e: r[e] for e in EL if r[e] > 0} for _, r in d.iterrows()]
    y = d.YS_MPa.values
    rows = []

    p = np.array([predict(f, t, *rule_of_mixtures_inputs(f)) for f, t in zip(fr, d.T_C)])
    rows.append({"inputs": "lee2021", **score(y, p)})

    el_path = os.path.join("results", "mlip_elements.csv")
    if os.path.exists(el_path):
        E = pd.read_csv(el_path)
        for pot, g in E.groupby("potential"):
            table = {r.element: (r.V, r.C11, r.C12, r.C44) for r in g.itertuples()}
            if not set(EL) <= set(table):
                continue
            p = np.array([predict(f, t, *rule_of_mixtures_inputs(f, table)) for f, t in zip(fr, d.T_C)])
            rows.append({"inputs": f"{pot}_elements", **score(y, p)})
            # mixed: literature C_ij, potential atomic volumes (misfits). Lattice parameters are
            # accurate to ~1.5% while shear constants are strongly softened (Deng et al. 2025).
            mixed = {e: (table[e][0],) + tuple(ELEMENTS_LEE2021[e][1:]) for e in EL}
            pm = np.array([predict(f, t, *rule_of_mixtures_inputs(f, mixed)) for f, t in zip(fr, d.T_C)])
            rows.append({"inputs": f"{pot}_volumes+lit_Cij", **score(y, pm)})
            # same rows with the literature inputs, for a like-for-like comparison
            ok = np.isfinite(p)
            p_ref = np.array([predict(f, t, *rule_of_mixtures_inputs(f)) for f, t in zip(fr, d.T_C)])
            rows.append({"inputs": f"lee2021_on_{pot}_rows", **score(y[ok], p_ref[ok])})

    al_path = os.path.join("results", "mlip_alloys.csv")
    if os.path.exists(al_path) and os.path.exists(el_path):
        A = pd.read_csv(al_path); E = pd.read_csv(el_path)
        for pot, g in A.groupby("potential"):
            vols = {r.element: r.V for r in E[E.potential == pot].itertuples()}
            sel = d.key.isin(g.key)
            p, p_ref = [], []
            for f, t, k in zip(np.array(fr, dtype=object)[sel], d.T_C[sel], d.key[sel]):
                r = g[g.key == k].iloc[0]
                vm = sum(c * vols[e] for e, c in f.items())
                dv = {e: vols[e] - vm for e in f}
                p.append(predict(f, t, r.a, r.C11, r.C12, r.C44, dv))
                p_ref.append(predict(f, t, *rule_of_mixtures_inputs(f)))
            rows.append({"inputs": f"{pot}_sqs", **score(y[sel], np.array(p))})
            rows.append({"inputs": f"lee2021_on_{pot}_sqs_rows", **score(y[sel], np.array(p_ref))})

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join("results", "mlip_strength_inputs.csv"), index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
