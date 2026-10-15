"""Reproduce published numbers for the reduced edge-dislocation model."""
import numpy as np

from metaforge.strength_edge import (
    isotropic_moduli, predict_rom, rule_of_mixtures_inputs, yield_stress, zero_temperature,
)

MONBTAW = {"Mo": .25, "Nb": .25, "Ta": .25, "W": .25}
MONBTAVW = {"Mo": .2, "Nb": .2, "Ta": .2, "V": .2, "W": .2}


def test_energy_barriers_match_maresca_curtin_2020_full_model():
    # Maresca & Curtin (2020): full theory with DFT inputs (their Table 2) gives
    # dE_b = 2.2 eV (MoNbTaW) and 2.5 eV (MoNbTaVW); the reduced model is within ~10%.
    _, eb1 = zero_temperature(3.237, 374, 163, 64, MONBTAW,
                              {"Mo": -1.251, "Nb": 1.153, "Ta": 1.132, "W": -1.034})
    _, eb2 = zero_temperature(3.192, 338, 164, 51, MONBTAVW,
                              {"Mo": -0.824, "Nb": 1.848, "Ta": 1.882, "V": -2.380, "W": -0.526})
    assert abs(eb1 - 2.2) / 2.2 < 0.05
    assert abs(eb2 - 2.5) / 2.5 < 0.10


def test_nbtativ_moduli_match_lee_2021():
    # Lee et al. (2021): rule-of-mixtures NbTaTiV gives mu = 47.8 GPa and nu = 0.365.
    _, c11, c12, c44, _ = rule_of_mixtures_inputs({"Nb": 1, "Ta": 1, "Ti": 1, "V": 1})
    mu, nu = isotropic_moduli(c11, c12, c44)
    assert abs(mu - 47.8) < 0.2
    assert abs(nu - 0.365) < 0.003


def test_vegard_misfits_match_maresca_curtin_2020_table2():
    a, c11, c12, c44, dv = rule_of_mixtures_inputs(MONBTAW)
    assert abs(a - 3.228) < 0.002
    for el, ref in {"Mo": -1.293, "Nb": 1.135, "Ta": 1.168, "W": -1.010}.items():
        assert abs(dv[el] - ref) < 0.01
    assert abs(c11 - 375.5) < 0.5 and abs(c12 - 167.3) < 0.5 and abs(c44 - 101.6) < 0.5


def test_strength_decreases_with_temperature_except_known_switch_step():
    # The published two-regime form is not continuous: at x^(2/3) = 0.5 the upper branch
    # gives 0.500 tau0 and the lower branch exp(-0.3536/0.55) = 0.526 tau0, a ~5% upward
    # step. Everywhere else strength must fall with temperature.
    T = np.linspace(77, 2000, 2000)
    s = predict_rom(MONBTAW, T)
    d = np.diff(s)
    rises = np.where(d > 1e-12)[0]
    assert len(rises) <= 1
    if len(rises):
        assert d[rises[0]] / s[rises[0]] < 0.06


def test_regime_switch_is_at_half_tau0():
    tau0, eb = 0.5, 2.0
    s = yield_stress(tau0, eb, np.array([10.0, 3000.0]))
    assert s[0] / 3.067 > 0.5 * tau0 and s[1] / 3.067 < 0.5 * tau0
