"""The point-scanning confocal: optics, scanning, detectors, and the budget."""

import dataclasses
import math

import numpy as np
import pytest

from noisewidget.physics import optics
from noisewidget.physics.confocal import (
    DETECTORS,
    ScanLevers,
    ScanNoise,
    ScanSimulator,
    acquire_scan,
    adu_to_electrons,
    confocal_kernel,
    haze_fraction,
    peak_signal,
    scan_budget,
    scan_sample,
    simulation_budget,
)
from noisewidget.physics.presets import CONFOCAL_PRESETS
from noisewidget.physics.quantitation import QuantSettings, quantify

N, DX = 512, 0.025


@pytest.fixture(scope="module")
def sim():
    return ScanSimulator()


# --- optics -------------------------------------------------------------------


def test_one_airy_unit_passes_the_airy_disc():
    # 1 AU is the Airy disc's diameter, which holds ~84% of the light.
    k = confocal_kernel(N, DX, ScanLevers(pinhole_au=1.0))
    assert k.pinhole_fraction == pytest.approx(0.84, abs=0.02)


def test_peak_irradiance_of_a_focused_spot():
    # For an aberration-free focus I0 = P pi NA^2 / lambda^2.
    lv = ScanLevers(laser_power_uw=10.0)
    k = confocal_kernel(N, 0.0125, lv)
    analytic = 10e-6 * math.pi * lv.na**2 / (lv.excitation_um * 1e-4) ** 2
    assert k.peak_irradiance_w_cm2 == pytest.approx(analytic, rel=0.05)


def test_a_closed_pinhole_sharpens_and_an_open_one_does_not():
    widefield = optics.fwhm_um(0.507, 1.4)
    closed = confocal_kernel(N, DX, ScanLevers(pinhole_au=0.2, laser_power_uw=0.1)).fwhm_um
    open_ = confocal_kernel(N, DX, ScanLevers(pinhole_au=10.0, laser_power_uw=0.1)).fwhm_um
    assert closed < widefield / 1.25
    # Wide open, only the excitation spot (at 488 nm) is left.
    assert open_ == pytest.approx(optics.fwhm_um(0.488, 1.4), rel=0.05)


def test_saturation_flattens_the_return_on_power_and_widens_the_spot():
    low = confocal_kernel(N, DX, ScanLevers(laser_power_uw=1.0))
    high = confocal_kernel(N, DX, ScanLevers(laser_power_uw=300.0))
    gain = high.rate.max() / low.rate.max()
    assert gain < 0.6 * 300
    assert high.fwhm_um > 1.1 * low.fwhm_um
    assert high.saturation > 0.5 > low.saturation


def test_haze_share_grows_with_the_pinhole():
    assert haze_fraction(0.5) < haze_fraction(1.0) < haze_fraction(5.0) < 1.0


# --- scanning -----------------------------------------------------------------


def test_scan_sampling_averages_in_x_and_point_samples_in_y():
    fine = np.zeros((N, N))
    fine[:, 100] = 1.0  # one bright column
    xs = np.array([100.5 * DX])
    out = scan_sample(fine, DX, xs, np.array([3.0]), px=4 * DX)
    assert out[0, 0] == pytest.approx(0.25)  # the column is a quarter of the sweep
    rowline = np.zeros((N, N))
    rowline[200, :] = 1.0
    hit = scan_sample(rowline, DX, np.array([5.0]), np.array([200.5 * DX]), px=DX)
    miss = scan_sample(rowline, DX, np.array([5.0]), np.array([203.5 * DX]), px=DX)
    assert hit[0, 0] == pytest.approx(1.0) and miss[0, 0] == 0.0


def test_signal_per_pixel_follows_dwell_not_zoom(sim):
    a = peak_signal(sim.run(ScanLevers(zoom=8.0)))
    assert peak_signal(sim.run(ScanLevers(zoom=8.0, dwell_us=6.4))) == pytest.approx(2 * a, rel=0.02)
    # A scanner's pixel is a dwell, not an area: oversampling does not dim it.
    assert peak_signal(sim.run(ScanLevers(zoom=20.0))) == pytest.approx(a, rel=0.1)


def test_finer_sampling_raises_the_bleaching_dose(sim):
    coarse = sim.run(ScanLevers(zoom=8.0)).photometry.emitted_per_frame
    fine = sim.run(ScanLevers(zoom=16.0)).photometry.emitted_per_frame
    assert fine == pytest.approx(4 * coarse, rel=0.01)


def test_x_and_y_pixels_are_independent():
    lv = ScanLevers(pixels_x=512, pixels_y=128)
    assert lv.pixel_y_um == pytest.approx(4 * lv.pixel_x_um)
    assert lv.frame_time_s() == pytest.approx(ScanLevers().frame_time_s() / 4)


def test_frame_time():
    lv = ScanLevers(pixels_x=512, pixels_y=512, dwell_us=2.0, averages=2)
    assert lv.frame_time_s() == pytest.approx(512 * 512 * 2e-6 / 0.7 * 2)


def test_too_coarse_a_scan_is_refused(sim):
    with pytest.raises(ValueError):
        sim.run(ScanLevers(zoom=0.6, pixels_x=64, pixels_y=64, magnification=10.0))


# --- detectors and the budget ------------------------------------------------


def flat_variance(lv: ScanLevers, light: float, dark: float = 0.0, shape=(300, 300)):
    adu, high, black = acquire_scan(np.full(shape, light), dark, lv, np.random.default_rng(3))
    e = adu_to_electrons(adu, lv)
    return float(e.var()), float(e.mean()), high, black


def quiet(**changes) -> ScanNoise:
    return dataclasses.replace(ScanNoise(laser_noise=False, afterpulsing=False), **changes)


@pytest.mark.parametrize("detector", [d for d, m in DETECTORS.items() if "analog" in m.modes])
def test_analog_variance_matches_the_budget(detector):
    lv = ScanLevers(detector=detector, mode="analog", gain_adu_per_e=2.0, offset_adu=50, noise=quiet())
    var, mean, _, _ = flat_variance(lv, 20.0)
    b = scan_budget(lv, 20.0, 0.0, 0.0)
    assert mean == pytest.approx(20.0, rel=0.02)
    assert var == pytest.approx(b.total, rel=0.06)
    assert b.photon_shot == pytest.approx(DETECTORS[detector].excess_noise_f2 * 20.0)


def test_low_gain_lets_electronics_dominate():
    lv = ScanLevers(gain_adu_per_e=0.15, noise=quiet())
    b = scan_budget(lv, 5.0, 0.0, 0.0)
    assert b.read + b.quantization > b.photon_shot
    var, _, _, _ = flat_variance(lv, 5.0)
    assert var == pytest.approx(b.total, rel=0.08)


def test_counting_is_poisson_and_has_no_electronics():
    lv = ScanLevers(detector="Hybrid detector (GaAsP HPD)", mode="counting", noise=quiet(dead_time=False))
    var, mean, _, _ = flat_variance(lv, 10.0)
    b = scan_budget(lv, 10.0, 0.0, 0.0)
    assert b.read == b.quantization == 0.0
    assert var == pytest.approx(10.0, rel=0.05) and mean == pytest.approx(10.0, rel=0.02)


def test_dead_time_caps_counts_and_correction_restores_them():
    lv = ScanLevers(detector="SPAD / APD", mode="counting", dwell_us=1.0, noise=quiet(photon_shot=False))
    _, mean, _, _ = flat_variance(lv, 40.0, shape=(50, 50))
    assert mean == pytest.approx(40.0 / (1 + 40.0 * 22e-9 / 1e-6), rel=0.02)
    _, corrected, _, _ = flat_variance(lv.with_(dead_time_correction=True), 40.0, shape=(50, 50))
    assert corrected == pytest.approx(40.0, rel=0.02)


def test_a_negative_offset_clips_black_and_biases_the_mean():
    lv = ScanLevers(offset_adu=-20.0, gain_adu_per_e=8.0)
    _, mean, _, black = flat_variance(lv, 1.0)
    assert black.mean() > 0.5
    assert mean > 1.0 + 1.0  # clipping the negative tail pushes the mean up


def test_high_gain_clips_at_full_scale():
    lv = ScanLevers(gain_adu_per_e=60.0, bit_depth=12)
    _, mean, high, _ = flat_variance(lv, 200.0, shape=(50, 50))  # 3x the 68 e- full scale
    assert high.all()
    assert mean == pytest.approx((4095 - lv.offset_adu) / 60.0, rel=0.01)


def test_laser_noise_is_multiplicative():
    lv = ScanLevers(laser_rin=0.02, gain_adu_per_e=2.0, bit_depth=16, noise=ScanNoise(photon_shot=False, amplifier=False, quantization=False,
                                                    gain_noise=False, afterpulsing=False))
    var, _, _, _ = flat_variance(lv, 1000.0)
    assert math.sqrt(var) == pytest.approx(20.0, rel=0.05)


def test_dark_counts_double_with_temperature():
    d = DETECTORS["GaAsP PMT (cooled)"]
    assert d.dark_rate(d.dark_ref_c + d.dark_doubling_c) == pytest.approx(2 * d.dark_cps)


def test_simulation_budget_and_report_agree_on_snr(sim):
    s = sim.run(ScanLevers())
    b = simulation_budget(s)
    assert 2.0 < b.snr < 10.0
    assert b.dominant() == "photon_shot"


# --- quantitation on a scanner -------------------------------------------------


def test_counting_is_unbiased_at_nyquist(sim):
    q = quantify(sim.run(ScanLevers(dwell_us=12.8)), QuantSettings(realizations=40))
    assert q.count.mean == pytest.approx(q.labels_present, rel=0.06)
    assert q.ideal.fwhm_um == pytest.approx(q.predicted_fwhm_um, rel=0.04)


def test_skipping_lines_ruins_the_measurement(sim):
    good = quantify(sim.run(ScanLevers()), QuantSettings(realizations=60))
    sparse = quantify(sim.run(ScanLevers(pixels_y=128)), QuantSettings(realizations=60))
    assert sparse.count.cv > 3 * good.count.cv


def test_dead_time_biases_the_count(sim):
    q = quantify(sim.run(CONFOCAL_PRESETS["Counting too fast: dead time"].levers), QuantSettings(realizations=20))
    assert q.count.mean < 0.8 * q.labels_present


@pytest.mark.parametrize("name", list(CONFOCAL_PRESETS))
def test_every_confocal_preset_runs(sim, name):
    from noisewidget.physics.scan_report import build_scan_report

    s = sim.run(CONFOCAL_PRESETS[name].levers)
    q = quantify(s, QuantSettings(realizations=10))
    assert all(rep.rows for rep in build_scan_report(s, q).values())
