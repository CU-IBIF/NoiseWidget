"""The measurement is honest: unbiased where it should be, spread by noise."""

import numpy as np
import pytest

from noisewidget.physics.acquisition import Levers, Simulator
from noisewidget.physics.quantitation import FWHM_PER_SIGMA, QuantSettings, fit_gaussians, quantify


def test_fit_recovers_a_known_gaussian():
    x = np.linspace(-1, 1, 41)
    y = 7.0 * np.exp(-0.5 * ((x - 0.1) / 0.2) ** 2) + 2.0
    params, ok = fit_gaussians(x, y[None, :])
    assert ok[0]
    assert params[0] == pytest.approx([7.0, 0.1, 0.2, 2.0], rel=1e-4, abs=1e-6)


def test_fit_flags_a_profile_that_has_no_peak():
    x = np.linspace(-1, 1, 21)
    rng = np.random.default_rng(0)
    _, ok = fit_gaussians(x, rng.normal(0, 1, (50, 21)))
    assert ok.mean() < 1.0


@pytest.fixture(scope="module")
def simulator():
    return Simulator()


def test_noise_free_width_matches_the_prediction(simulator):
    q = quantify(simulator.run(Levers()), QuantSettings(realizations=20))
    assert q.ideal.fwhm_um == pytest.approx(q.predicted_fwhm_um, rel=0.03)
    # And it is the PSF, nowhere near the 25 nm microtubule.
    assert q.ideal.fwhm_um > 6 * q.true_diameter_nm / 1000


def test_counting_is_unbiased_at_high_signal(simulator):
    q = quantify(simulator.run(Levers(exposure_ms=1000.0)), QuantSettings(realizations=50))
    assert q.count.mean == pytest.approx(q.labels_present, rel=0.05)
    assert q.summed.mean == pytest.approx(q.labels_present, rel=0.05)


def test_beads_are_counted_too(simulator):
    q = quantify(simulator.run(Levers(specimen="Beads, 100 nm")), QuantSettings(realizations=30))
    assert q.count.mean == pytest.approx(q.labels_present, rel=0.05)
    assert q.ideal.fwhm_um == pytest.approx(q.predicted_fwhm_um, rel=0.04)


def test_less_light_spreads_the_measurement(simulator):
    bright = quantify(simulator.run(Levers()), QuantSettings(realizations=80))
    dim = quantify(simulator.run(Levers(irradiance_w_cm2=1.0, exposure_ms=20.0)), QuantSettings(realizations=80))
    assert dim.fwhm.sd > 3 * bright.fwhm.sd
    assert dim.count.cv > 3 * bright.count.cv
    assert dim.centre.sd > 3 * bright.centre.sd


def test_clipping_biases_width_and_count(simulator):
    q = quantify(
        simulator.run(Levers(fluorophore="Alexa Fluor 488", irradiance_w_cm2=200.0, exposure_ms=1500.0)),
        QuantSettings(realizations=10),
    )
    assert q.fwhm.mean > 1.1 * q.ideal.fwhm_um
    assert q.count.mean < 0.9 * q.labels_present


def test_dark_subtraction_rescues_a_hot_pixel_field(simulator):
    warm = dict(camera="CMOS, uncooled (3.45 um)", temperature_c=40.0, exposure_ms=2000.0,
                irradiance_w_cm2=0.5, magnification=60.0)
    raw = quantify(simulator.run(Levers(**warm)), QuantSettings(realizations=20))
    fixed = quantify(simulator.run(Levers(dark_subtract=True, **warm)), QuantSettings(realizations=20))
    assert abs(fixed.count.mean / fixed.labels_present - 1) < abs(raw.count.mean / raw.labels_present - 1)


def test_fwhm_constant():
    assert FWHM_PER_SIGMA == pytest.approx(2.3548, rel=1e-4)
