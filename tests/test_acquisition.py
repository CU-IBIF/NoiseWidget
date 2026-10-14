"""The simulation agrees with the analytic budget it is explained by.

These are the tests that matter most: the graph is drawn from :func:`budget`, the
images from :func:`acquire`, and if the two disagree the widget teaches something
false. Each case records a flat field many times and compares the measured pixel
variance with the predicted one.
"""

import dataclasses

import numpy as np
import pytest

from noisewidget.physics.acquisition import (
    Levers,
    NoiseSources,
    Simulator,
    acquire,
    budget,
    budget_sweep,
    photometry,
)
from noisewidget.physics.camera import CAMERAS, fixed_pattern

EMCCD = "EMCCD, back-illuminated (13 um)"
SCMOS = "sCMOS, front-illuminated (6.5 um)"
CCD = "Interline CCD (6.45 um)"


def measured_variance(levers: Levers, signal: float, dark: float = 0.0, cic: float = 0.0, shape=(200, 200)):
    """Variance across pixels of a flat field, referred to input electrons."""
    pattern = fixed_pattern(levers.camera_model, shape, levers.binning, levers.gain,
                            enabled=levers.noise.fixed_pattern)
    rng = np.random.default_rng(5)
    light = np.full(shape, signal) * pattern.gain
    thermal = dark * pattern.dark_scale + cic
    adu, _ = acquire(light, thermal, levers, pattern, rng, max(levers.frames, 1))
    cam = levers.camera_model
    e = (adu - cam.offset_adu) * cam.adc_gain_e_per_adu / levers.gain
    return float(e.var()), float(e.mean())


def quiet(**changes) -> NoiseSources:
    """Noise sources with the fixed pattern off, so pixel variance is temporal."""
    return dataclasses.replace(NoiseSources(fixed_pattern=False), **changes)


@pytest.mark.parametrize("signal", [2.0, 20.0, 500.0])
def test_scmos_variance_matches_the_budget(signal):
    lv = Levers(camera=SCMOS, noise=quiet())
    var, mean = measured_variance(lv, signal)
    b = budget(lv, signal, 0.0, 0.0, 0.0)
    assert mean == pytest.approx(signal, rel=0.02, abs=0.05)
    assert var == pytest.approx(b.total, rel=0.05)


def test_emccd_doubles_shot_variance_and_divides_read_noise():
    lv = Levers(camera=EMCCD, em_gain=300.0, noise=quiet())
    var, mean = measured_variance(lv, 10.0)
    b = budget(lv, 10.0, 0.0, 0.0, 0.0)
    assert b.photon_shot == pytest.approx(20.0)  # F^2 = 2
    assert b.read == pytest.approx((50.0 / 300.0) ** 2)
    assert var == pytest.approx(b.total, rel=0.06)
    assert mean == pytest.approx(10.0, rel=0.02)


def test_emccd_without_gain_uses_the_conventional_amplifier():
    lv = Levers(camera=EMCCD, em_gain=1.0, noise=quiet())
    b = budget(lv, 10.0, 0.0, 0.0, 0.0)
    assert b.photon_shot == pytest.approx(10.0)
    assert b.read == pytest.approx(36.0)


def test_digital_binning_adds_read_noise_charge_binning_does_not():
    cmos = budget(Levers(camera=SCMOS, binning=2, noise=quiet()), 100.0, 0, 0, 0)
    ccd = budget(Levers(camera=CCD, binning=2, noise=quiet()), 100.0, 0, 0, 0)
    assert cmos.read == pytest.approx(4 * 1.4**2)
    assert ccd.read == pytest.approx(6.5**2)
    var, _ = measured_variance(Levers(camera=SCMOS, binning=2, noise=quiet()), 100.0, shape=(150, 150))
    assert var == pytest.approx(cmos.total, rel=0.06)


def test_averaging_reduces_random_noise_but_not_fixed_pattern():
    one = budget(Levers(frames=1), 1000.0, 0, 0, 0)
    sixteen = budget(Levers(frames=16), 1000.0, 0, 0, 0)
    assert sixteen.photon_shot == pytest.approx(one.photon_shot / 16)
    assert sixteen.read == pytest.approx(one.read / 16)
    assert sixteen.prnu == pytest.approx(one.prnu)
    # And in the simulation: 16 frames of a 1000 e- flat field leave PRNU behind.
    lv = Levers(camera="CMOS, uncooled (3.45 um)", frames=16)
    var, _ = measured_variance(lv, 1000.0, shape=(120, 120))
    prnu_var = (0.01 * 1000.0) ** 2
    assert var > prnu_var * 0.8


def test_flat_field_removes_prnu_from_the_budget():
    assert budget(Levers(flat_field=True), 1000.0, 0, 0, 0).prnu == 0.0


def test_dark_current_doubles_every_doubling_interval():
    cam = CAMERAS[SCMOS]
    assert cam.dark_current(cam.dark_ref_c + cam.dark_doubling_c) == pytest.approx(2 * cam.dark_e_per_s)
    assert cam.dark_current(cam.dark_ref_c - 2 * cam.dark_doubling_c) == pytest.approx(cam.dark_e_per_s / 4)


def test_thermal_charge_is_shot_noise():
    lv = Levers(noise=quiet(read=False, quantization=False))
    var, mean = measured_variance(lv, 0.0, dark=25.0)
    assert mean == pytest.approx(25.0, rel=0.02)
    assert var == pytest.approx(25.0, rel=0.06)


def test_switching_every_source_off_records_the_expectation():
    off = NoiseSources(**{f.name: False for f in dataclasses.fields(NoiseSources)})
    sim = Simulator().run(Levers(noise=off))
    assert np.allclose(sim.recorded_e, sim.expected_e)


def test_saturation_clips_at_the_full_well():
    sim = Simulator().run(Levers(fluorophore="Alexa Fluor 488", irradiance_w_cm2=200.0, exposure_ms=1500.0))
    assert sim.saturated.any()
    assert sim.recorded_e.max() <= CAMERAS[SCMOS].full_well_e * 1.01


def test_bleaching_survival():
    lv = Levers(irradiance_w_cm2=20.0, exposure_ms=100.0)
    assert photometry(lv).bleach_survival > 0.99
    harsh = photometry(Levers(irradiance_w_cm2=1000.0, exposure_ms=1000.0, frames=4))
    assert harsh.bleach_survival < 0.3


def test_signal_scales_with_exposure_irradiance_and_na():
    base = photometry(Levers()).detected_e
    assert photometry(Levers(exposure_ms=200.0)).detected_e == pytest.approx(2 * base, rel=0.01)
    assert photometry(Levers(irradiance_w_cm2=40.0)).detected_e == pytest.approx(2 * base, rel=0.01)
    assert photometry(Levers(na=0.75, medium="air")).detected_e < 0.6 * base


def test_magnification_moves_light_between_pixels_not_its_total():
    sim = Simulator()
    a = sim.run(Levers(magnification=100.0)).signal_e
    b = sim.run(Levers(magnification=50.0)).signal_e
    fa = a.shape[0] * 0.065
    fb = b.shape[0] * 0.13
    assert a.sum() / fa**2 == pytest.approx(b.sum() / fb**2, rel=0.03)
    assert b.max() > 3 * a.max()


def test_sweeps_pass_through_the_operating_point():
    sim = Simulator().run(Levers())
    from noisewidget.physics.acquisition import peak_signal

    for mode, here in (("exposure", 100.0), ("excitation", 20.0)):
        (b,) = budget_sweep(sim, mode, [here])
        assert b.signal_e == pytest.approx(peak_signal(sim))
    long = budget_sweep(sim, "exposure", [1000.0])[0]
    assert long.snr > budget_sweep(sim, "exposure", [10.0])[0].snr


def test_optics_cache_is_reused_when_only_the_camera_changes():
    sim = Simulator()
    sim.run(Levers())
    blurred = sim._blurred
    sim.run(Levers(exposure_ms=5.0, em_gain=2.0))
    assert sim._blurred is blurred
    sim.run(Levers(na=1.2))
    assert sim._blurred is not blurred
