"""
Level 2 of the potential benchmark: alloy-level inputs from SQS supercells.

For a selection of single-phase BCC training alloys (most measurements per alloy system,
plus MoNbTaW and MoNbTaVW whose lattice parameters are known from Senkov et al. 2010),
build a 54-atom BCC SQS, relax atoms and cell, and compute relaxed-ion cubic elastic
constants. Elemental rule-of-mixtures C_ij cannot be used for Ti/Zr/Hf-rich alloys when a
potential predicts unstable bcc elements (C11 < C12); the alloy itself is usually stable.

Run (mlip venv):  python scripts/mlip_alloys.py mace [chgnet ...]  [--n 15]
Writes / appends results/mlip_alloys.csv (one row per alloy and potential)
"""
import argparse
import csv
import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from benchmark_chgnet import build_sqs  # noqa: E402
from metaforge.composition import vegard_lattice_parameter  # noqa: E402
from metaforge.mlip import cubic_elastic_constants_energy, get_calculator, relax_cell, sqs_to_atoms  # noqa: E402

EL = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
OUT = os.path.join("results", "mlip_alloys.csv")
FIXED = ["Mo0.250 Nb0.250 Ta0.250 W0.250", "Mo0.200 Nb0.200 Ta0.200 V0.200 W0.200"]


def select(n):
    d = pd.read_csv(os.path.join("data", "processed", "rhea_strength.csv"))
    d = d[d.physics_domain & (d.holdout == "train")]
    d["system"] = d.apply(lambda r: "-".join(sorted(e for e in EL if r[e] > 0)), axis=1)
    counts = d.groupby(["system", "key"]).size().reset_index(name="n").sort_values("n", ascending=False)
    best = counts.drop_duplicates("system")
    keys = list(dict.fromkeys(FIXED + best.key.tolist()))[:n]
    first = d.drop_duplicates("key").set_index("key")
    return [(k, {e: float(first.loc[k, e]) for e in EL if first.loc[k, e] > 0}) for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("potentials", nargs="+")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {(r["potential"], r["key"]) for r in csv.DictReader(f)}
    alloys = select(args.n)
    for pot in args.potentials:
        calc = get_calculator(pot)
        for key, f in alloys:
            if (pot, key) in done:
                continue
            t0 = time.time()
            sqs, _ = build_sqs(f, "bcc", vegard_lattice_parameter(f, "BCC"), seed=0)
            atoms, conv = relax_cell(sqs_to_atoms(sqs), calc, fmax=0.02, steps=400)
            a = (atoms.get_volume() / len(atoms) * 2) ** (1 / 3)
            c11, c12, c44 = cubic_elastic_constants_energy(atoms, calc, relax_ions=True)
            row = {"potential": pot, "key": key, "a": round(a, 4), "C11": round(c11, 1), "C12": round(c12, 1),
                   "C44": round(c44, 1), "relax_converged": conv, "seconds": round(time.time() - t0)}
            new = not os.path.exists(OUT)
            os.makedirs("results", exist_ok=True)
            with open(OUT, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
