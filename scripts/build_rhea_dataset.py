"""
Build the refractory yield-strength dataset used for the v2 study.

Source: MPEA dataset (Borg et al., Sci. Data 7 (2020) 430), downloaded by
scripts/train_measured_strength.py to data/external/mpea/MPEA_dataset.csv.

Keeps alloys made only of Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W (+ Al, Re flagged as
outside the physics domain), with a measured yield strength. Adds:
  - atomic fractions per element and a composition key (rounded fractions)
  - single_phase_bcc: microstructure reported as "BCC"
  - physics_domain: single-phase BCC and only the nine elements above
  - holdout: "future" if the alloy's FIRST appearance in the dataset is 2019,
    else "train". Holdout alloys are never used for training or model selection.

Run:  python scripts/build_rhea_dataset.py
Writes data/processed/rhea_strength.csv
"""
import os

import numpy as np
import pandas as pd
from pymatgen.core import Composition

SRC = os.path.join("data", "external", "mpea", "MPEA_dataset.csv")
OUT = os.path.join("data", "processed", "rhea_strength.csv")
PHYS = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
EXTRA = ["Al", "Re"]
HOLDOUT_YEAR = 2019


def main():
    d = pd.read_csv(SRC)
    d.columns = [c.split(": ", 1)[-1] for c in d.columns]
    d = d[d["YS (MPa)"].notna()].copy()
    comps = d["FORMULA"].map(lambda f: Composition(f).fractional_composition)
    d["els"] = comps.map(lambda c: {e.symbol for e in c.elements})
    d = d[d["els"].map(lambda s: s <= set(PHYS + EXTRA))].copy()
    comps = comps[d.index]
    for el in PHYS + EXTRA:
        d[el] = comps.map(lambda c, el=el: round(float(c[el]), 4))
    d["key"] = comps.map(lambda c: " ".join(f"{e}{c[e]:.3f}" for e in sorted(c.elements, key=str)))
    d["n_elements"] = d["els"].map(len)
    d["T_C"] = pd.to_numeric(d["Test temperature ($^\\circ$C)"], errors="coerce")
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["single_phase_bcc"] = d["Microstructure"].eq("BCC")
    d["physics_domain"] = d["single_phase_bcc"] & d["els"].map(lambda s: s <= set(PHYS))
    first_year = d.groupby("key")["year"].transform("min")
    d["holdout"] = np.where(first_year >= HOLDOUT_YEAR, "future", "train")

    out = d.rename(columns={"YS (MPa)": "YS_MPa", "Type of test": "test", "Microstructure": "microstructure",
                            "Processing method": "processing", "FORMULA": "formula"})
    cols = ["key", "formula", "n_elements"] + PHYS + EXTRA + ["T_C", "YS_MPa", "test", "microstructure",
            "processing", "single_phase_bcc", "physics_domain", "year", "holdout", "doi"]
    out = out[cols].drop_duplicates(subset=["key", "T_C", "YS_MPa", "test", "doi"])
    out = out[out["T_C"].notna()].sort_values(["key", "T_C"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    def summary(df, name):
        print(f"{name:32s} rows={len(df):4d} alloys={df['key'].nunique():3d} "
              f"RT={int((df['T_C'] <= 30).sum()):4d} >=600C={int((df['T_C'] >= 600).sum()):4d}")
    summary(out, "all refractory (incl. Al, Re)")
    summary(out[out.physics_domain], "physics domain (1-phase BCC)")
    summary(out[out.physics_domain & (out.holdout == "train")], "  of which train")
    summary(out[out.physics_domain & (out.holdout == "future")], "  of which future holdout")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
