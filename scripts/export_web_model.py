"""
Export the refractory hybrid strength model to the web app (MetaForge-Web).

Render deploys MetaForge-Web/ on its own (rootDir), so the app cannot import the metaforge
package. This script
  1. fits the hybrid model exactly as in scripts/design_nsga2.py (all 72 single-phase BCC
     alloys, conformal margin from grouped out-of-fold scores, domain-of-applicability data)
     and saves its parts as plain objects to MetaForge-Web/models/refractory_hybrid.joblib;
  2. copies metaforge/strength_edge.py and metaforge/phase.py to MetaForge-Web/metaforge_core/
     (tests/test_web_copies.py fails if the copies drift from the originals).

Run (web venv): python scripts/export_web_model.py
"""
import json
import os
import shutil
import sys

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import strength_models as sm  # noqa: E402
from design_nsga2 import HybridPredictor  # noqa: E402
from metaforge.phase import label_alloys, pair_caps  # noqa: E402

WEB = os.path.join(ROOT, "MetaForge-Web")
COPIED = ["strength_edge.py", "phase.py"]


def main():
    d, X = sm.load()
    m = HybridPredictor(d, X)
    caps = pair_caps(label_alloys(os.path.join(ROOT, "data", "external", "mpea", "MPEA_dataset.csv")))
    bundle = {"res": m.res, "q_lo": m.q_lo, "q_hi": m.q_hi, "qhat": m.qhat, "scaler": m.scaler,
              "Z": m.Z, "Z_hot": m.Z_hot, "doa_threshold": m.doa_threshold, "pair_caps": caps,
              "ys_max_hot_MPa": float(d.YS_MPa[d.T_C >= 900].max()), "elements": sm.ELEMENTS}
    meta = {"n_alloys": int(d.key.nunique()), "n_measurements": int(len(d)),
            "conformal_margin_MPa": round(m.qhat, 1), "doa_threshold": round(m.doa_threshold, 2),
            "ys_max_hot_MPa": bundle["ys_max_hot_MPa"],
            "validation": "grouped-by-alloy CV R2 0.666, by-system 0.638, 2019 holdout 0.649 "
                          "(results/strength_models.csv)"}
    os.makedirs(os.path.join(WEB, "models"), exist_ok=True)
    joblib.dump(bundle, os.path.join(WEB, "models", "refractory_hybrid.joblib"))
    with open(os.path.join(WEB, "models", "refractory_hybrid.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    core = os.path.join(WEB, "metaforge_core")
    os.makedirs(core, exist_ok=True)
    with open(os.path.join(core, "__init__.py"), "w", encoding="utf-8") as f:
        f.write('"""Copies of metaforge modules made by scripts/export_web_model.py. Do not edit here."""\n')
    for name in COPIED:
        shutil.copyfile(os.path.join(ROOT, "metaforge", name), os.path.join(core, name))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
