"""Refractory yield strength from the hybrid model of the v2 study.

Prediction = Maresca-Curtin edge-dislocation theory (no fitted parameters) + a gradient-boosted
correction learned from 72 measured single-phase BCC alloys, with a 90% conformal interval.
The bundle is written by scripts/export_web_model.py. Valid only for Ti, Zr, Hf, V, Nb, Ta,
Cr, Mo, W; the caller falls back to other models otherwise.
"""
import numpy as np
import joblib
from pymatgen.core import Element

from metaforge_core.phase import violating_pairs
from metaforge_core.strength_edge import ELEMENTS_LEE2021, predict_rom

AMU_PER_A3_TO_G_CM3 = 1.66053906660


class RefractoryHybrid:
    def __init__(self, path):
        b = joblib.load(path)
        self.__dict__.update(b)
        self.elements = set(b["elements"])

    def covers(self, fractions):
        return all(el in self.elements for el, c in fractions.items() if c > 0)

    def _distance(self, mag, Z):
        z = self.scaler.transform(np.asarray(mag)[None, :])
        return float(np.sqrt(((Z - z) ** 2).sum(1)).min())

    def predict(self, fractions, mag, T_C):
        """(point, lower, upper) yield strength in MPa for compression at T_C (deg C)."""
        f = {el: c for el, c in fractions.items() if c > 0}
        x = np.concatenate([mag, [T_C, 0.0]])[None, :]
        phys = 1000.0 * float(predict_rom(f, T_C + 273.15))
        point = phys + float(self.res.predict(x)[0])
        lo = phys + float(self.q_lo.predict(x)[0]) - self.qhat
        hi = phys + float(self.q_hi.predict(x)[0]) + self.qhat
        return point, max(lo, 0.0), hi

    def density(self, fractions):
        """Rule-of-mixtures density in g/cm3, the same calculation as scripts/design_nsga2.py."""
        f = {el: c for el, c in fractions.items() if c > 0}
        m = sum(c * Element(el).atomic_mass for el, c in f.items())
        v = sum(c * ELEMENTS_LEE2021[el][0] for el, c in f.items())
        return float(m / v * AMU_PER_A3_TO_G_CM3)

    def warnings(self, fractions, mag):
        """Plain-language reasons to distrust a prediction (empty list if none)."""
        f = {el: c for el, c in fractions.items() if c > 0}
        out = []
        pairs = violating_pairs(f, self.pair_caps)
        if pairs:
            out.append("No single-phase BCC alloy has been reported with this much " + ", ".join(pairs)
                       + ", so expect Laves or other second phases")
        if self._distance(mag, self.Z) > self.doa_threshold:
            out.append("This composition is far from every measured alloy, so treat the numbers as a rough guess")
        if self._distance(mag, self.Z_hot) > self.doa_threshold:
            out.append("No similar alloy has been tested at 900 °C or above, so the 1000 °C value is an extrapolation")
        return out
