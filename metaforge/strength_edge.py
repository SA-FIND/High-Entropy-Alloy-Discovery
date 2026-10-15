"""Reduced edge-dislocation strengthening model for random BCC alloys.

F. Maresca and W. A. Curtin, "Mechanistic origin of high strength in refractory
BCC high entropy alloys up to 1900K", Acta Mater. 182 (2020) 235-249,
doi:10.1016/j.actamat.2019.10.015 (arXiv:1901.02100), Section 5:

    tau_y0 = 0.040 alpha^(-1/3) mu [(1+nu)/(1-nu)]^(4/3) [sum_n c_n dV_n^2 / b^6]^(2/3)
    dE_b   = 2.00  alpha^(1/3)  mu b^3 [(1+nu)/(1-nu)]^(2/3) [sum_n c_n dV_n^2 / b^6]^(1/3)

    tau_y(T) = tau_y0 [1 - (kT/dE_b ln(eps0/eps))^(2/3)]          if tau_y/tau_y0 >= 0.5
    tau_y(T) = tau_y0 exp[-(1/0.55) (kT/dE_b) ln(eps0/eps)]        otherwise

with alpha = 1/12, eps0 = 1e4 1/s, sigma_y = M tau_y, M = 3.067, b = sqrt(3)/2 a,
isotropic mu = sqrt(C44 (C11 - C12) / 2) and nu from the bulk modulus
K = (C11 + 2 C12) / 3. The same equations, coefficients and the elemental inputs in
ELEMENTS_LEE2021 are given in Lee, Maresca, Curtin et al., Nat. Commun. 12 (2021) 5474
(arXiv:2008.11671, Eqs. 3-4 and Supp. Table 2).

Note: the published two-regime form is not continuous; at the switch the strength steps
up by ~5% (0.500 tau0 -> exp(-0.3536/0.55) = 0.526 tau0). We keep it as published.

Baruffi, Maresca & Curtin, MRS Commun. 12 (2022) 1111 later used a single ad hoc form,
sigma_y = 3.06 tau_y0 exp[-(1/0.55) (kT/dE_b ln(eps0/eps))^0.91] with alpha = 1/6
(`form="bmc2022"` below). Their Eq. 9 was corrected in 2024 (doi:10.1557/s43579-023-00507-2)
without the corrected values being stated in the notice, so it is an option, not the default.
"""
import numpy as np

# Single-element BCC inputs from Lee et al. (2021), Supp. Table 2:
# atomic volume (A^3), C11, C12, C44 (GPa). Ti, Zr, Hf: high-T BCC values.
ELEMENTS_LEE2021 = {
    "Al": (14.075, 105.6, 63.9, 28.53),
    "Cr": (12.321, 339.8, 58.6, 99.0),
    "Hf": (22.528, 131.0, 103.0, 45.0),
    "Mo": (15.524, 450.02, 172.92, 125.03),
    "Nb": (17.952, 252.7, 133.2, 30.97),
    "Ta": (17.985, 266.32, 158.16, 87.36),
    "Ti": (17.387, 134.0, 110.0, 36.0),
    "V": (14.020, 232.4, 119.36, 45.95),
    "W": (15.807, 532.55, 204.95, 163.13),
    "Zr": (23.02, 104.0, 93.0, 38.0),
}

K_B_EV = 8.617333262e-5          # eV/K
GPA_A3_TO_EV = 1.0 / 160.21766   # 1 GPa * A^3 = 6.2415e-3 eV
ALPHA = 1.0 / 12.0
EPS0 = 1.0e4
TAYLOR_M = 3.067


def isotropic_moduli(c11, c12, c44):
    """(mu, nu) from cubic elastic constants in GPa, as in Maresca & Curtin (2020)."""
    mu = np.sqrt(0.5 * c44 * (c11 - c12))
    k = (c11 + 2.0 * c12) / 3.0
    nu = (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
    return mu, nu


def misfit_parameter(fractions, misfit_volumes):
    """sum_n c_n dV_n^2 in A^6. Inputs are dicts keyed by element."""
    return sum(c * misfit_volumes[el] ** 2 for el, c in fractions.items() if c > 0)


def zero_temperature(a, c11, c12, c44, fractions, misfit_volumes, alpha=ALPHA):
    """Return (tau_y0 in GPa, dE_b in eV) from the reduced model."""
    b = np.sqrt(3.0) / 2.0 * a
    mu, nu = isotropic_moduli(c11, c12, c44)
    q = misfit_parameter(fractions, misfit_volumes) / b ** 6
    f = (1.0 + nu) / (1.0 - nu)
    tau0 = 0.040 * alpha ** (-1.0 / 3.0) * mu * f ** (4.0 / 3.0) * q ** (2.0 / 3.0)
    de_b = 2.00 * alpha ** (1.0 / 3.0) * mu * b ** 3 * f ** (2.0 / 3.0) * q ** (1.0 / 3.0) * GPA_A3_TO_EV
    return tau0, de_b


def yield_stress(tau0, de_b, T_kelvin, strain_rate=1e-3, eps0=EPS0, taylor=TAYLOR_M, form="mc2020"):
    """Uniaxial yield stress sigma_y (GPa) at temperature T (K) and strain rate (1/s)."""
    T = np.asarray(T_kelvin, dtype=float)
    x = np.clip(K_B_EV * T / de_b * np.log(eps0 / strain_rate), 0.0, None)
    if form == "bmc2022":
        return 3.06 * tau0 * np.exp(-(1.0 / 0.55) * x ** 0.91)
    high = tau0 * (1.0 - x ** (2.0 / 3.0))
    low = tau0 * np.exp(-x / 0.55)
    tau = np.where(high / tau0 >= 0.5, high, low)
    return taylor * tau


def rule_of_mixtures_inputs(fractions, elements=ELEMENTS_LEE2021):
    """Alloy inputs from elemental values: (a, C11, C12, C44, misfit volumes).

    Vegard's law for the mean volume, dV_n = V_n - V_mean, rule of mixtures for C_ij
    (Lee et al. 2021). Fractions are normalised internally.
    """
    tot = sum(fractions.values())
    f = {el: c / tot for el, c in fractions.items() if c > 0}
    v_mean = sum(c * elements[el][0] for el, c in f.items())
    c11, c12, c44 = (sum(c * elements[el][k] for el, c in f.items()) for k in (1, 2, 3))
    a = (2.0 * v_mean) ** (1.0 / 3.0)
    dv = {el: elements[el][0] - v_mean for el in f}
    return a, c11, c12, c44, dv


def predict_rom(fractions, T_kelvin, strain_rate=1e-3, alpha=ALPHA, form="mc2020",
                elements=ELEMENTS_LEE2021):
    """sigma_y (GPa) from elemental rule-of-mixtures inputs (the Lee et al. 2021 screening route)."""
    tot = sum(fractions.values())
    f = {el: c / tot for el, c in fractions.items() if c > 0}
    a, c11, c12, c44, dv = rule_of_mixtures_inputs(f, elements)
    tau0, de_b = zero_temperature(a, c11, c12, c44, f, dv, alpha=alpha)
    return yield_stress(tau0, de_b, T_kelvin, strain_rate, form=form)
