"""
Phase-stability domain used by the design search (metaforge/phase.py).

Builds one phase label per refractory alloy (Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W only) from the
MPEA dataset, compares simple descriptors as single-phase classifiers (logistic regression,
repeated stratified 5-fold CV) with the pairwise-cap rule (leave-one-alloy-out), and writes the
caps used as a design constraint.

Run (web venv): python scripts/phase_domain.py
Writes results/phase_pair_caps.csv and results/phase_validation.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from metaforge.composition import VEC  # noqa: E402
from metaforge.phase import PAIRS, label_alloys, leave_one_out, pair_caps  # noqa: E402
from metaforge.strength_edge import ELEMENTS_LEE2021  # noqa: E402

SRC = os.path.join("data", "external", "mpea", "MPEA_dataset.csv")
# Binary Laves phases among the nine elements (C14/C15/C36), from the binary phase diagrams.
LAVES = [("Zr", "V"), ("Zr", "Cr"), ("Zr", "Mo"), ("Zr", "W"), ("Hf", "V"), ("Hf", "Cr"), ("Hf", "Mo"),
         ("Hf", "W"), ("Ti", "Cr"), ("Nb", "Cr"), ("Ta", "Cr"), ("Ta", "V")]


def delta(f):
    v = {e: ELEMENTS_LEE2021[e][0] for e in f}
    vm = sum(c * v[e] for e, c in f.items())
    return np.sqrt(sum(c * (v[e] - vm) ** 2 for e, c in f.items())) / (3 * vm)


def main():
    A = label_alloys(SRC)
    A["laves_index"] = A.f.map(lambda f: sum(f.get(a, 0) * f.get(b, 0) for a, b in LAVES))
    A["VEC"] = A.f.map(lambda f: sum(c * VEC[e] for e, c in f.items()))
    A["delta"] = A.f.map(delta)
    y = A.single.values
    print(f"{len(A)} alloys: {y.sum()} single-phase BCC, {(1 - y).sum()} multiphase")

    rows = []
    for feats in (["laves_index"], ["VEC", "delta"], ["laves_index", "VEC", "delta"]):
        aucs = []
        for seed in range(20):
            p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), A[feats].values, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(y, p))
        rows.append({"method": "logistic(" + "+".join(feats) + ")", "AUC_mean": np.mean(aucs), "AUC_sd": np.std(aucs)})

    pred = leave_one_out(A)
    tpr = ((pred == 1) & (y == 1)).sum() / (y == 1).sum()
    tnr = ((pred == 0) & (y == 0)).sum() / (y == 0).sum()
    rows.append({"method": "pair caps (leave-one-alloy-out)", "single_passed": tpr, "multiphase_rejected": tnr,
                 "balanced_accuracy": (tpr + tnr) / 2})
    V = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    V.to_csv(os.path.join("results", "phase_validation.csv"), index=False)
    print(V.round(3).to_string(index=False))
    print("multiphase alloys the rule misses:")
    print(A[(pred == 1) & (y == 0)][["formula", "labels"]].to_string())

    caps = pair_caps(A)
    C = pd.DataFrame({"pair": [f"{a}-{b}" for a, b in PAIRS], "cap": caps,
                      "laves_forming": [(a, b) in LAVES or (b, a) in LAVES for a, b in PAIRS]})
    C.to_csv(os.path.join("results", "phase_pair_caps.csv"), index=False)
    print("pairs never seen in a single-phase alloy:", ", ".join(C.pair[C.cap == 0]))


if __name__ == "__main__":
    main()
