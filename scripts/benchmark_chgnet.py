"""
Benchmark CHGNet relaxation of SQS supercells against measured HEA data.

Why: the repo's earlier "relaxed" CIFs were written by a fallback path when
CHGNet was not installed, so they were identical to the blueprints. This script
makes the relaxation step explicit, reproducible and checkable:

  1. Builds an SQS-like supercell for each alloy by Monte Carlo minimisation of
     first- and second-shell Warren-Cowley parameters (fixed seeds).
  2. Relaxes atoms AND cell with CHGNet (StructOptimizer, relax_cell=True) until
     forces < fmax, for several seeds per alloy.
  3. Compares the relaxed lattice parameter and density with experiment and with
     two rules of mixtures (atomic-fraction average of elemental lattice
     parameters, and volume-weighted Vegard density).
  4. Writes relaxed CIFs and results/chgnet_benchmark.csv.

Run (needs chgnet and pymatgen; a CPU laptop takes a few minutes per relaxation):
    python scripts/benchmark_chgnet.py --seeds 3 --fmax 0.05

Experimental references: see REFERENCE notes below. Verify every value against
the cited paper before quoting results anywhere.
"""
import argparse
import csv
import os
import random

import numpy as np
from pymatgen.core import Lattice, Structure, Element

# Elemental room-temperature lattice parameters (Angstrom) and structures.
ELEMENT_A = {
    "W": ("bcc", 3.165), "Mo": ("bcc", 3.147), "Ta": ("bcc", 3.301), "Nb": ("bcc", 3.300),
    "V": ("bcc", 3.024), "Cr": ("bcc", 2.885), "Fe": ("bcc", 2.867),
    "Co": ("fcc", 3.545), "Ni": ("fcc", 3.524), "Mn": ("fcc", 3.63), "Cu": ("fcc", 3.615),
}

# Alloys with measured values. REFERENCE notes:
#  NbMoTaW, NbMoTaWV: Senkov et al., Intermetallics 18 (2010) 1758 (a and density).
#  CoCrFeNi, CoCrFeMnNi: typical XRD values reported in the literature (~3.57, ~3.60 A).
# Check each number in the source before publishing.
BENCHMARKS = [
    {"name": "NbMoTaW",    "comp": {"Nb": .25, "Mo": .25, "Ta": .25, "W": .25},             "lattice": "bcc", "a_exp": 3.2134, "rho_exp": 13.75},
    {"name": "NbMoTaWV",   "comp": {"Nb": .2, "Mo": .2, "Ta": .2, "W": .2, "V": .2},        "lattice": "bcc", "a_exp": 3.1832, "rho_exp": 12.36},
    {"name": "CoCrFeNi",   "comp": {"Co": .25, "Cr": .25, "Fe": .25, "Ni": .25},            "lattice": "fcc", "a_exp": 3.57,   "rho_exp": None},
    {"name": "CoCrFeMnNi", "comp": {"Co": .2, "Cr": .2, "Fe": .2, "Mn": .2, "Ni": .2},      "lattice": "fcc", "a_exp": 3.60,   "rho_exp": None},
]

SUPERCELL = {"bcc": ([[0, 0, 0], [.5, .5, .5]], [3, 3, 3]),             # 54 atoms
             "fcc": ([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], [2, 2, 3])}  # 48 atoms


def rom_lattice(comp):
    """Atomic-fraction average of elemental lattice parameters, converted to the
    alloy's lattice via atomic volume (keeps bcc and fcc elements comparable)."""
    vols = []
    for el, x in comp.items():
        kind, a = ELEMENT_A[el]
        vols.append(x * (a ** 3 / (2 if kind == "bcc" else 4)))
    return sum(vols)  # mean atomic volume, A^3/atom


def build_sqs(comp, lattice_kind, a0, seed, steps=3000):
    basis, sc = SUPERCELL[lattice_kind]
    s = Structure(Lattice.cubic(a0), ["X"] * len(basis), basis)
    s.make_supercell(sc)
    n = len(s)
    counts = {el: int(round(x * n)) for el, x in comp.items()}
    counts[max(counts, key=counts.get)] += n - sum(counts.values())
    species = [el for el, c in counts.items() for _ in range(c)]
    rng = random.Random(seed)
    rng.shuffle(species)
    els = list(counts)
    idx = np.array([els.index(e) for e in species])
    x = np.array([counts[e] / n for e in els])
    # neighbour shells from distances in the ideal lattice
    d = s.distance_matrix
    shell1 = np.isclose(d, d[d > 0.1].min(), atol=0.05)
    rest = d[(d > 0.1) & ~shell1]
    shell2 = np.isclose(d, rest.min(), atol=0.05)

    def loss(ix):
        tot = 0.0
        for k in range(len(els)):
            if x[k] == 0:
                continue
            a_sites = ix == k
            for w, sh in ((1.0, shell1), (0.5, shell2)):
                nb = sh[a_sites]
                same = (nb & (ix == k)[None, :]).sum() / max(nb.sum(), 1)
                alpha = 1 - same / x[k]
                tot += w * alpha ** 2
        return tot

    cur = loss(idx)
    best, best_l, T = idx.copy(), cur, 0.1
    for _ in range(steps):
        i, j = rng.sample(range(n), 2)
        if idx[i] == idx[j]:
            continue
        idx[i], idx[j] = idx[j], idx[i]
        new = loss(idx)
        if new < cur or rng.random() < np.exp(-(new - cur) / max(T, 1e-9)):
            cur = new
            if cur < best_l:
                best, best_l = idx.copy(), cur
        else:
            idx[i], idx[j] = idx[j], idx[i]
        T *= 0.998
    for i in range(n):
        s.replace(i, els[best[i]])
    return s, best_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="build SQS only, no CHGNet")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("data", "structures", "benchmark"), exist_ok=True)
    relaxer = None
    if not args.dry_run:
        from chgnet.model.dynamics import StructOptimizer  # fails loudly if missing
        relaxer = StructOptimizer()

    rows = []
    for b in BENCHMARKS:
        v_rom = rom_lattice(b["comp"])
        per_cell = 2 if b["lattice"] == "bcc" else 4
        a_rom = (v_rom * per_cell) ** (1 / 3)
        for seed in range(args.seeds):
            sqs, sro_loss = build_sqs(b["comp"], b["lattice"], a_rom, seed)
            row = {"alloy": b["name"], "seed": seed, "sro_loss": round(sro_loss, 4),
                   "a_rom": round(a_rom, 4), "a_exp": b["a_exp"], "rho_exp": b["rho_exp"],
                   "rho_rom_vegard": round(sqs.density, 3)}
            if relaxer:
                res = relaxer.relax(sqs, fmax=args.fmax, steps=args.steps, relax_cell=True, verbose=False)
                fs = res["final_structure"]
                n_traj = len(res["trajectory"].energies)
                a_relax = (fs.volume / len(fs) * per_cell) ** (1 / 3)
                row.update({
                    "converged_steps": n_traj,
                    "a_chgnet": round(a_relax, 4),
                    "a_err_pct_vs_exp": round(100 * (a_relax - b["a_exp"]) / b["a_exp"], 2),
                    "rho_chgnet": round(fs.density, 3),
                    "rho_err_pct_vs_exp": round(100 * (fs.density - b["rho_exp"]) / b["rho_exp"], 2) if b["rho_exp"] else "",
                    "vol_change_pct_vs_rom": round(100 * (fs.volume - sqs.volume) / sqs.volume, 2),
                    "energy_eV_per_atom": round(res["trajectory"].energies[-1] / len(fs), 4),
                })
                fs.to(filename=os.path.join("data", "structures", "benchmark", f"{b['name']}_seed{seed}_relaxed.cif"))
            sqs.to(filename=os.path.join("data", "structures", "benchmark", f"{b['name']}_seed{seed}_sqs.cif"))
            rows.append(row)
            print(row)

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    with open(os.path.join("results", "chgnet_benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("Wrote results/chgnet_benchmark.csv")


if __name__ == "__main__":
    main()
