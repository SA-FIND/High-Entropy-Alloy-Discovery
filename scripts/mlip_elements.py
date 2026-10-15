"""
Level 1 of the potential benchmark: bcc elements.

For each potential, compute the bcc lattice parameter (E-V fit) and cubic elastic
constants of Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W, and compare with the elemental inputs of
Lee et al. (2021) (experimental; Ti, Zr, Hf from high-temperature bcc measurements).

Run (mlip venv):  python scripts/mlip_elements.py chgnet mace sevennet mattersim
Writes results/mlip_elements.csv
"""
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metaforge.mlip import bcc_cell, cubic_elastic_constants, eos_lattice_parameter, get_calculator  # noqa: E402
from metaforge.strength_edge import ELEMENTS_LEE2021  # noqa: E402

ELEMENTS = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
OUT = os.path.join("results", "mlip_elements.csv")


def main(potentials):
    rows = []
    if os.path.exists(OUT):
        with open(OUT) as f:
            rows = [r for r in csv.DictReader(f) if r["potential"] not in potentials]
    for pot in potentials:
        t0 = time.time()
        try:
            calc = get_calculator(pot)
        except Exception as e:  # potential not installable on this machine
            print(f"[{pot}] unavailable: {e}")
            continue
        for el in ELEMENTS:
            v_ref, c11r, c12r, c44r = ELEMENTS_LEE2021[el]
            a_ref = (2 * v_ref) ** (1 / 3)
            a = eos_lattice_parameter(el, calc, a_ref)
            c11, c12, c44 = cubic_elastic_constants(bcc_cell(el, a), calc)
            row = {"potential": pot, "element": el, "a": round(a, 4), "a_ref": round(a_ref, 4),
                   "V": round(a ** 3 / 2, 3), "V_ref": v_ref,
                   "C11": round(c11, 1), "C12": round(c12, 1), "C44": round(c44, 1),
                   "C11_ref": c11r, "C12_ref": c12r, "C44_ref": c44r}
            rows.append(row)
            print(pot, row)
        print(f"[{pot}] {time.time() - t0:.0f} s")
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("Wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or ["chgnet", "mace", "sevennet", "mattersim"])
