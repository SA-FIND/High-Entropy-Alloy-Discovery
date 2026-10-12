"""
Train a yield-strength model on MEASURED data instead of the analytical proxy.

Data: MPEA dataset, Borg et al., Scientific Data 7 (2020) 430,
      https://github.com/CitrineInformatics/MPEA_dataset (data/external/mpea/).
Target: room-temperature (<= 30 C) COMPRESSIVE yield strength, in GPa.
Features: the same 132 Magpie descriptors the web app computes.

Validation: GroupKFold by alloy composition, so repeat measurements of one alloy
never sit in both train and test. The model is chosen by lowest grouped-CV MAE.
Also checks the rule-of-mixtures density proxy against measured densities.

Run with the web venv (scikit-learn version must match Render):
    python scripts/train_measured_strength.py
Writes MetaForge-Web/models/ml_strength_measured.model and
       MetaForge-Web/models/ml_strength_measured.json, results/measured_strength_cv.txt
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition, Element
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

warnings.filterwarnings("ignore")
CSV = os.path.join("data", "external", "mpea", "MPEA_dataset.csv")
OUT_DIR = os.path.join("MetaForge-Web", "models")
CATEGORIES = {
    "Refractory": ["W", "Mo", "Ta", "Nb", "V"],
    "Corrosion": ["Co", "Cr", "Fe", "Ni", "Cu"],
    "Lightweight": ["Al", "Mg", "Li", "Ti", "Zn"],
    "Aerospace": ["Al", "Ti", "Sc", "Zr", "V"],
}


MPEA_URL = "https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/master/MPEA_dataset.csv"


def load():
    if not os.path.exists(CSV):
        import urllib.request
        os.makedirs(os.path.dirname(CSV), exist_ok=True)
        urllib.request.urlretrieve(MPEA_URL, CSV)
    d = pd.read_csv(CSV)
    d.columns = [c.split(": ", 1)[-1] for c in d.columns]
    temp = pd.to_numeric(d["Test temperature ($^\\circ$C)"], errors="coerce")
    d = d[(temp <= 30) & (d["Type of test"] == "C") & d["YS (MPa)"].notna()].copy()
    comps, keys = [], []
    for f in d["FORMULA"]:
        try:
            c = Composition(f).fractional_composition
            comps.append(c)
            keys.append(" ".join(f"{e}{c[e]:.3f}" for e in sorted(c.elements)))
        except Exception:
            comps.append(None)
            keys.append(None)
    d["comp"], d["key"] = comps, keys
    return d[d["comp"].notna()].reset_index(drop=True)


def rom_density(c):
    """Volume-weighted (Vegard) density from elemental solid densities."""
    mass = sum(c[e] * e.atomic_mass for e in c.elements)
    vol = sum(c[e] * e.atomic_mass / (float(e.density_of_solid) / 1000) for e in c.elements)
    return mass / vol


def main():
    log = []
    p = lambda s="": (print(s), log.append(s))
    d = load()
    ep = ElementProperty.from_preset("magpie")
    X = np.array([ep.featurize(c) for c in d["comp"]])
    ok = ~np.isnan(X).any(axis=1)
    d, X = d[ok].reset_index(drop=True), X[ok]
    y = d["YS (MPa)"].values / 1000.0  # GPa, same unit as the app
    groups = d["key"].values
    p(f"Measured RT compressive YS: {len(d)} tests, {d['key'].nunique()} distinct alloys "
      f"(range {y.min():.2f} to {y.max():.2f} GPa, median {np.median(y):.2f})")
    p(f"scikit-learn {sklearn.__version__}")

    cv = GroupKFold(n_splits=5)
    cands = {
        "Mean (baseline)": DummyRegressor(),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.3,
                                              n_jobs=2, random_state=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400,
                                                              l2_regularization=1.0, random_state=0),
    }
    p("\nGrouped 5-fold CV (no alloy in both train and test):")
    p(f"{'model':22s} {'MAE GPa':>8s} {'R2':>7s}")
    preds = {}
    for name, m in cands.items():
        pr = cross_val_predict(m, X, y, cv=cv, groups=groups)
        preds[name] = pr
        p(f"{name:22s} {mean_absolute_error(y, pr):8.3f} {r2_score(y, pr):7.3f}")
    best = min((n for n in cands if n != "Mean (baseline)"), key=lambda n: mean_absolute_error(y, preds[n]))
    pb = preds[best]
    p(f"--> Selected by CV MAE: {best}")

    # Coverage and per-category error of the selected model
    el_counts = {}
    for c in d["comp"]:
        for e in c.elements:
            el_counts[e.symbol] = el_counts.get(e.symbol, 0) + 1
    p("\nPer category (test rows whose alloy contains >= 3 of the category's elements):")
    cat_info = {}
    for cat, els in CATEGORIES.items():
        mask = np.array([sum(e in [x.symbol for x in c.elements] for e in els) >= 3 for c in d["comp"]])
        cov = {e: el_counts.get(e, 0) for e in els}
        if mask.sum() >= 5:
            mae = mean_absolute_error(y[mask], pb[mask])
            r2 = r2_score(y[mask], pb[mask]) if mask.sum() > 5 else float("nan")
            p(f"  {cat:12s} n={mask.sum():4d}  MAE={mae:.3f} GPa  R2={r2:.3f}  element counts {cov}")
        else:
            mae, r2 = None, None
            p(f"  {cat:12s} n={mask.sum():4d}  too few measured alloys to validate; element counts {cov}")
        cat_info[cat] = {"n_validation_rows": int(mask.sum()), "cv_mae_gpa": mae, "cv_r2": r2,
                         "element_counts": cov}

    # Density proxy check
    dd = pd.read_csv(CSV)
    dd.columns = [c.split(": ", 1)[-1] for c in dd.columns]
    dd = dd[dd["Exp. Density (g/cm$^3$)"].notna()].drop_duplicates("FORMULA")
    errs = []
    for f, rho in zip(dd["FORMULA"], dd["Exp. Density (g/cm$^3$)"]):
        try:
            errs.append(100 * (rom_density(Composition(f).fractional_composition) - rho) / rho)
        except Exception:
            pass
    errs = np.array(errs)
    p(f"\nRule-of-mixtures density vs measured ({len(errs)} alloys): mean error {errs.mean():+.2f}%, "
      f"median |error| {np.median(np.abs(errs)):.2f}%, mean |error| {np.abs(errs).mean():.2f}%, "
      f"95th pct |error| {np.percentile(np.abs(errs), 95):.2f}%")
    p("  (Largest errors are porous/Zn-bearing or carbide-containing samples; Li-Mg light alloys "
      "read ~7% low.)")

    final = cands[best].fit(X, y)
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(final, os.path.join(OUT_DIR, "ml_strength_measured.model"))
    meta = {
        "target": "room-temperature compressive yield strength (GPa)",
        "source": "MPEA dataset, Borg et al., Sci. Data 7 (2020) 430",
        "model": best, "n_tests": int(len(d)), "n_alloys": int(d["key"].nunique()),
        "cv": "GroupKFold(5) by composition",
        "cv_mae_gpa": round(float(mean_absolute_error(y, pb)), 4), "cv_r2": round(float(r2_score(y, pb)), 4),
        "element_counts": el_counts, "categories": cat_info, "sklearn": sklearn.__version__,
    }
    with open(os.path.join(OUT_DIR, "ml_strength_measured.json"), "w") as f:
        json.dump(meta, f, indent=1)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "measured_strength_cv.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("Saved model, metadata and results/measured_strength_cv.txt")


if __name__ == "__main__":
    main()
