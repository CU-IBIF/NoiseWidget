"""NA and wavelength: the closed forms the rest of the model leans on."""

import math

import numpy as np
import pytest

from noisewidget.physics import optics


def test_otf_is_one_at_dc_and_zero_past_cutoff():
    o = optics.otf(256, 0.025, 0.5, 1.4)
    assert o[0, 0] == pytest.approx(1.0)
    f = optics.frequency_grid(256, 0.025)
    assert np.all(o[f > optics.cutoff_cycles_per_um(0.5, 1.4) * 1.001] == 0.0)


def test_psf_fwhm_matches_the_airy_disc():
    # Measured off the rendered PSF, not assumed: the analytic OTF must produce the
    # textbook 0.514 lambda / NA at any NA on the same grid -- including low NA,
    # which a sampled pupil would get wrong.
    for na in (0.3, 0.75, 1.4):
        dx = 0.01 if na > 0.5 else 0.025
        h = optics.psf(512, dx, 0.52, na)
        row = h[256, :]
        half = row.max() / 2
        above = np.nonzero(row >= half)[0]
        # Linear interpolation at both half-maximum crossings.
        def cross(i, j):
            return i + (half - row[i]) / (row[j] - row[i]) * (j - i)
        left = cross(above[0] - 1, above[0])
        right = cross(above[-1], above[-1] + 1)
        fwhm = (right - left) * dx
        assert fwhm == pytest.approx(optics.fwhm_um(0.52, na), rel=0.03)


def test_blur_conserves_photons():
    rng = np.random.default_rng(0)
    img = rng.random((128, 128))
    assert optics.blur(img, 0.025, 0.5, 1.2).sum() == pytest.approx(img.sum(), rel=1e-6)


def test_line_spread_sums_to_one():
    assert optics.line_spread(512, 0.025, 0.5, 1.4).sum() == pytest.approx(1.0)


def test_collection_efficiency_values():
    # (1 - cos(asin(NA/n))) / 2
    assert optics.collection_efficiency(1.4, 1.518) == pytest.approx(0.307, abs=0.002)
    assert optics.collection_efficiency(0.75, 1.0) == pytest.approx(0.169, abs=0.002)
    assert optics.collection_efficiency(0.3, 1.0) == pytest.approx(0.0230, abs=0.0005)


def test_na_above_the_immersion_index_is_refused():
    with pytest.raises(ValueError):
        optics.collection_efficiency(1.2, 1.0)


def test_box_integrate_conserves_and_handles_fractional_pixels():
    rng = np.random.default_rng(1)
    img = rng.random((100, 100))
    # 0.065 um pixels on a 0.025 um grid: 2.6 fine samples per pixel.
    out = optics.box_integrate(img, 0.025, 0.065)
    n = out.shape[0]
    assert n == math.floor(100 * 0.025 / 0.065)
    covered = n * 0.065 / 0.025
    whole = int(covered)
    expected = img[:whole, :whole].sum()
    assert out.sum() == pytest.approx(expected, rel=0.03)
    # Integer ratio: plain block sums.
    blocks = optics.box_integrate(img, 0.025, 0.1)
    assert blocks[0, 0] == pytest.approx(img[:4, :4].sum())


def test_nyquist_and_resolution():
    assert optics.nyquist_pixel_um(0.5, 1.25) == pytest.approx(0.1)
    assert optics.rayleigh_um(0.5, 1.0) == pytest.approx(0.305)
    assert optics.abbe_um(0.5, 1.0) == pytest.approx(0.25)
