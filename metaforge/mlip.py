"""Universal ML interatomic potentials: lattice parameters and cubic elastic constants.

All potentials are used through ASE calculators on CPU:
  chgnet    CHGNet 0.3.0 (Deng et al., Nat. Mach. Intell. 5 (2023) 1031)
  mace      MACE-MP-0 "medium" (Batatia et al., arXiv:2401.00096)
  sevennet  SevenNet-0 "7net-0" (Park et al., J. Chem. Theory Comput. 20 (2024) 4857)
  mattersim MatterSim v1 1M (Yang et al., arXiv:2405.04967)
"""
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

EV_A3_TO_GPA = 160.21766208


def get_calculator(name):
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device="cpu")
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", default_dtype="float64", device="cpu")
    if name == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator(model="7net-0", device="cpu")
    if name == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device="cpu")
    raise ValueError(name)


def bcc_cell(symbol, a):
    return bulk(symbol, "bcc", a=a, cubic=True)


def eos_lattice_parameter(symbol, calc, a_guess, span=0.06, n=9):
    """Equilibrium bcc lattice parameter from a quadratic fit of E(V) near the minimum."""
    a_vals = a_guess * np.linspace(1 - span, 1 + span, n)
    e_vals = []
    for a in a_vals:
        at = bcc_cell(symbol, a); at.calc = calc
        e_vals.append(at.get_potential_energy())
    v = a_vals ** 3
    i = int(np.argmin(e_vals))
    lo, hi = max(0, i - 2), min(n, i + 3)
    if hi - lo < 4 or i in (0, n - 1):  # minimum at the edge: widen and retry
        return eos_lattice_parameter(symbol, calc, a_vals[i], span * 1.5, n)
    p = np.polyfit(v[lo:hi], np.array(e_vals)[lo:hi], 2)
    return float((-p[1] / (2 * p[0])) ** (1 / 3))


def relax_cell(atoms, calc, fmax=0.02, steps=500):
    """Relax positions and cell; returns (atoms, converged)."""
    atoms = atoms.copy(); atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)
    return atoms, bool(converged)


def _stress_voigt(atoms):
    # ASE: stress = (1/V) dE/d(strain), positive in tension, Voigt xx yy zz yz xz xy, eV/A^3
    return atoms.get_stress(voigt=True)


def cubic_elastic_constants(atoms, calc, delta=0.005, relax_ions=False, fmax=0.01, steps=300):
    """Cubic C11, C12, C44 (GPa) by central finite differences of stress.

    For disordered supercells set relax_ions=True (relaxed-ion constants) and the three
    directions are averaged to the cubic values.
    """
    cell0 = atoms.get_cell().array.copy()
    c = np.zeros((6, 6))
    for j in range(6):
        s = []
        for sign in (+1, -1):
            eps = np.zeros((3, 3))
            if j < 3:
                eps[j, j] = sign * delta
            else:
                k, l = [(1, 2), (0, 2), (0, 1)][j - 3]
                eps[k, l] = eps[l, k] = sign * delta / 2  # engineering shear strain = delta
            at = atoms.copy(); at.calc = calc
            at.set_cell(cell0 @ (np.eye(3) + eps), scale_atoms=True)
            if relax_ions:
                FIRE(at, logfile=None).run(fmax=fmax, steps=steps)
            s.append(_stress_voigt(at))
        c[:, j] = (s[0] - s[1]) / (2 * delta) * EV_A3_TO_GPA
    c11 = np.mean([c[0, 0], c[1, 1], c[2, 2]])
    c12 = np.mean([c[0, 1], c[0, 2], c[1, 2], c[1, 0], c[2, 0], c[2, 1]])
    c44 = np.mean([c[3, 3], c[4, 4], c[5, 5]])
    return float(c11), float(c12), float(c44)


def sqs_to_atoms(structure):
    """pymatgen Structure -> ASE Atoms."""
    return Atoms(symbols=[str(s.specie) for s in structure], positions=structure.cart_coords,
                 cell=structure.lattice.matrix, pbc=True)
