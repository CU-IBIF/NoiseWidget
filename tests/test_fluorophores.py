import pytest

from noisewidget.physics.fluorophores import FLUOROPHORES, wavelength_to_rgb


def test_egfp_cross_section():
    # 3.82e-21 x 55,900 = 2.14e-16 cm^2
    assert FLUOROPHORES["EGFP"].cross_section_cm2 == pytest.approx(2.14e-16, rel=0.01)


def test_emission_is_linear_at_widefield_irradiance_and_saturates_far_above():
    f = FLUOROPHORES["EGFP"]
    assert f.emission_rate(20) == pytest.approx(10 * f.emission_rate(2), rel=1e-3)
    # Ground-state depletion: the rate approaches QY / tau.
    assert f.emission_rate(1e8) < f.quantum_yield / (f.lifetime_ns * 1e-9)
    assert f.emission_rate(1e8) > 0.5 * f.quantum_yield / (f.lifetime_ns * 1e-9)


def test_wavelength_colours_are_sensible():
    assert wavelength_to_rgb(520)[1] == 1.0  # green
    assert wavelength_to_rgb(680)[0] == 1.0  # red
