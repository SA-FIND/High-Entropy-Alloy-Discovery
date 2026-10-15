"""Composition handling: exact bounds, rounding, VEC and Vegard volumes."""
import numpy as np

# Room-temperature atomic volume per atom (A^3) from experimental lattice constants
# of the stable elemental phase (bcc/fcc: a^3/n; hcp: sqrt(3)/2 a^2 c / 2).
ATOMIC_VOLUME = {
    "W": 15.855, "Mo": 15.583, "Ta": 17.990, "Nb": 17.975, "V": 13.827, "Cr": 12.004,
    "Ti": 17.65, "Zr": 23.28, "Hf": 22.31, "Fe": 11.777, "Co": 11.08, "Ni": 10.94,
    "Cu": 11.81, "Mn": 12.21, "Al": 16.60, "Mg": 23.24, "Li": 21.62, "Zn": 15.21, "Sc": 25.0,
}

# Valence electron count (s + d electrons) used in Guo's VEC phase rule.
VEC = {
    "Li": 1, "Mg": 2, "Al": 3, "Sc": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5,
    "Cr": 6, "Mo": 6, "W": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
}


def project_to_bounded_simplex(y, lo=0.05, hi=0.35, tol=1e-12):
    """Euclidean projection of y onto {x : sum(x) = 1, lo <= x_i <= hi}.

    Clip-then-renormalise does not respect the bounds (renormalising can push a
    component above hi). Here x_i = clip(y_i - tau, lo, hi) with tau found by
    bisection so that sum(x) = 1, which satisfies both constraints exactly.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if not (n * lo <= 1.0 + tol and n * hi >= 1.0 - tol):
        raise ValueError(f"infeasible bounds for {n} components: [{lo}, {hi}]")
    left, right = y.min() - hi - 1.0, y.max() - lo + 1.0
    for _ in range(200):
        tau = 0.5 * (left + right)
        s = np.clip(y - tau, lo, hi).sum()
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            left = tau
        else:
            right = tau
    return np.clip(y - tau, lo, hi)


def round_on_bounded_simplex(x, decimals=3, lo=0.05, hi=0.35):
    """Round to `decimals` while keeping sum == 1 and the bounds."""
    step = 10.0 ** -decimals
    r = np.round(np.asarray(x, dtype=float), decimals)
    diff = round((1.0 - r.sum()) / step)
    order = np.argsort(-(np.asarray(x) - r)) if diff > 0 else np.argsort(np.asarray(x) - r)
    for i in order:
        if diff == 0:
            break
        new = r[i] + np.sign(diff) * step
        if lo - 1e-12 <= new <= hi + 1e-12:
            r[i] = new
            diff -= int(np.sign(diff))
    return np.round(r, decimals)


def vec(comp):
    """Composition-weighted valence electron concentration."""
    return sum(x * VEC[el] for el, x in comp.items())


def crystal_type(comp):
    """Guo et al. (2011): VEC < 6.87 -> BCC, otherwise FCC (duplex region treated as FCC)."""
    return "BCC" if vec(comp) < 6.87 else "FCC"


def vegard_volume(comp):
    """Rule-of-mixtures mean atomic volume (A^3/atom)."""
    return sum(x * ATOMIC_VOLUME[el] for el, x in comp.items())


def vegard_lattice_parameter(comp, kind=None):
    """Cubic lattice parameter (A) from the Vegard mean atomic volume."""
    kind = kind or crystal_type(comp)
    atoms_per_cell = 2 if kind == "BCC" else 4
    return (vegard_volume(comp) * atoms_per_cell) ** (1.0 / 3.0)
