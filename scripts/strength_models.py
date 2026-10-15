"""
Strength models for single-phase BCC refractory alloys, with honest validation.

Data: data/processed/rhea_strength.csv, physics_domain rows only.
Models (all predict yield strength in MPa at the test temperature):
  physics     Maresca-Curtin reduced edge model, elemental rule-of-mixtures inputs
              (Lee et al. 2021), strain rate 1e-3 /s. No fitted parameters.
  physics_lin physics prediction scaled by a fitted linear map (2 parameters).
  ml          HistGradientBoosting on 132 Magpie descriptors + temperature + test type.
  hybrid      same ML, trained on the residual (measured - physics).

Validation (training alloys only, holdout never touched until the end):
  grouped     5-fold GroupKFold by alloy (no alloy in train and test).
  system      5-fold GroupKFold by alloy system (element set), i.e. unseen chemistries.
  future      fit on all training alloys, score once on alloys first reported in 2019.
Uncertainty: 90% prediction intervals by conformalized quantile regression on the hybrid
residual (calibrated inside the training folds only); coverage is reported for each split.

Run (web venv): python scripts/strength_models.py [--inputs results/mlip_inputs.csv]
Writes results/strength_models.csv and results/strength_predictions.csv
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metaforge.strength_edge import predict_rom  # noqa: E402

warnings.filterwarnings("ignore")
ELEMENTS = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
DATA = os.path.join("data", "processed", "rhea_strength.csv")


def gbm(loss="squared_error", quantile=None):
    kw = dict(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
              min_samples_leaf=10, random_state=0)
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(loss=loss, **kw)


def load():
    d = pd.read_csv(DATA)
    d = d[d.physics_domain].reset_index(drop=True)
    d["fractions"] = [{e: r[e] for e in ELEMENTS if r[e] > 0} for _, r in d.iterrows()]
    d["system"] = d["fractions"].map(lambda f: "-".join(sorted(f)))
    d["physics"] = [1000 * float(predict_rom(f, t + 273.15)) for f, t in zip(d.fractions, d.T_C)]
    ep = ElementProperty.from_preset("magpie")
    mag = np.array([ep.featurize(Composition(f)) for f in d.fractions])
    X = np.column_stack([mag, d.T_C.values, (d.test == "T").astype(float).values])
    return d, X


def fit_predict(name, Xtr, ytr, ptr, Xte, pte):
    if name == "physics":
        return pte
    if name == "physics_lin":
        lr = LinearRegression().fit(ptr.reshape(-1, 1), ytr)
        return lr.predict(pte.reshape(-1, 1))
    if name == "ml":
        return gbm().fit(Xtr, ytr).predict(Xte)
    if name == "hybrid":
        return pte + gbm().fit(Xtr, ytr - ptr).predict(Xte)
    raise ValueError(name)


def _quantile_band(Xtr, rtr, Xte):
    return (gbm(quantile=0.05).fit(Xtr, rtr).predict(Xte),
            gbm(quantile=0.95).fit(Xtr, rtr).predict(Xte))


def interval(Xtr, ytr, ptr, Xte, pte, gtr, alpha=0.10):
    """90% interval for the hybrid model by conformalized quantile regression
    (Romano et al., NeurIPS 2019): quantile bands on the physics residual, widened by
    the (1-alpha) quantile of out-of-fold conformity scores computed inside the
    training set only (grouped by alloy)."""
    r = ytr - ptr
    scores = np.empty(len(r))
    for a, b in GroupKFold(n_splits=5).split(Xtr, groups=gtr):
        lo, hi = _quantile_band(Xtr[a], r[a], Xtr[b])
        scores[b] = np.maximum(lo - r[b], r[b] - hi)
    n = len(scores)
    qhat = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    lo, hi = _quantile_band(Xtr, r, Xte)
    return pte + lo - qhat, pte + hi + qhat


def metrics(y, p):
    e = p - y
    return {"n": len(y), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)), "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    d, X = load()
    y, p = d.YS_MPa.values, d.physics.values
    tr = (d.holdout == "train").values
    models = ["physics", "physics_lin", "ml", "hybrid"]
    rows, preds = [], []
    for split, groups in (("grouped", d.key.values), ("system", d.system.values)):
        idx = np.where(tr)[0]
        for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(idx, groups=groups[idx])):
            itr, ite = idx[a], idx[b]
            for m in models:
                pr = fit_predict(m, X[itr], y[itr], p[itr], X[ite], p[ite])
                preds += [{"split": split, "model": m, "row": int(i), "pred": float(v)} for i, v in zip(ite, pr)]
            lo, hi = interval(X[itr], y[itr], p[itr], X[ite], p[ite], d.key.values[itr])
            preds += [{"split": split, "model": "pi90", "row": int(i), "lo": float(l), "hi": float(h)}
                      for i, l, h in zip(ite, lo, hi)]
    itr, ite = np.where(tr)[0], np.where(~tr)[0]
    for m in models:
        pr = fit_predict(m, X[itr], y[itr], p[itr], X[ite], p[ite])
        preds += [{"split": "future", "model": m, "row": int(i), "pred": float(v)} for i, v in zip(ite, pr)]
    lo, hi = interval(X[itr], y[itr], p[itr], X[ite], p[ite], d.key.values[itr])
    preds += [{"split": "future", "model": "pi90", "row": int(i), "lo": float(l), "hi": float(h)}
              for i, l, h in zip(ite, lo, hi)]

    P = pd.DataFrame(preds)
    for split in ("grouped", "system", "future"):
        for m in models:
            q = P[(P.split == split) & (P.model == m)]
            rows.append({"split": split, "model": m, **metrics(y[q.row.values], q.pred.values)})
        q = P[(P.split == split) & (P.model == "pi90")]
        cov = float(np.mean((y[q.row] >= q.lo.values) & (y[q.row] <= q.hi.values)))
        rows.append({"split": split, "model": "hybrid_PI90_coverage", "n": len(q), "MAE": np.nan,
                     "RMSE": np.nan, "bias": np.nan, "R2": cov})
    R = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    R.to_csv(os.path.join("results", "strength_models.csv"), index=False)
    P.merge(d[["key", "T_C", "YS_MPa", "test", "holdout", "system", "physics"]], left_on="row",
            right_index=True).to_csv(os.path.join("results", "strength_predictions.csv"), index=False)
    pd.set_option("display.width", 140)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
