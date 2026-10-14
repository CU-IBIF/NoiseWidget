"""Numerical aperture and wavelength: resolution, sampling and light collection.

NA and wavelength do two separate jobs in a fluorescence image, and the widget is
built to show both:

* **They set the blur.** The incoherent optical transfer function of a circular
  pupil cuts off at ``2 NA / lambda``; nothing finer than that reaches the camera.
* **NA sets the brightness.** The objective collects the fraction of an isotropic
  emitter's light that falls inside its acceptance cone, ``(1 - cos theta) / 2``
  with ``sin theta = NA / n``. Going from NA 0.75 to 1.4 nearly doubles the photons,
  and photons are what shot noise is counted in.

The OTF is evaluated analytically rather than by Fourier-transforming a sampled
pupil. A sampled pupil at NA 0.1 would be two pixels across on any grid that also
resolves NA 1.49, and the resulting PSF would be wrong; the closed form is right at
every NA on the same grid.

This is the scalar, paraxial widefield model. Above NA ~1.2 vectorial effects
broaden the real PSF by a few percent; that is noted where it matters rather than
modelled. All lengths are microns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Refractive indices at ~550 nm. Immersion oil is the ISO 8036 value.
IMMERSION_MEDIA: dict[str, float] = {
    "air": 1.000,
    "water": 1.333,
    "silicone oil": 1.406,
    "glycerol": 1.474,
    "oil": 1.518,
}


@dataclass(frozen=True)
class Objective:
    """A generic objective class, not a vendor product."""

    label: str
    magnification: float
    na: float
    medium: str

    @property
    def n(self) -> float:
        return IMMERSION_MEDIA[self.medium]


# Generic, commonly sold magnification/NA pairings. No vendor datasheet is implied.
OBJECTIVES: list[Objective] = [
    Objective("10x / 0.30 air", 10, 0.30, "air"),
    Objective("10x / 0.45 air", 10, 0.45, "air"),
    Objective("20x / 0.75 air", 20, 0.75, "air"),
    Objective("40x / 0.95 air", 40, 0.95, "air"),
    Objective("40x / 1.30 oil", 40, 1.30, "oil"),
    Objective("60x / 1.20 water", 60, 1.20, "water"),
    Objective("60x / 1.40 oil", 60, 1.40, "oil"),
    Objective("100x / 1.40 oil", 100, 1.40, "oil"),
    Objective("100x / 1.49 oil (TIRF)", 100, 1.49, "oil"),
]


def check_na(na: float, n: float) -> None:
    if na <= 0:
        raise ValueError("NA must be positive")
    if na > n:
        raise ValueError(f"NA {na:.2f} exceeds the immersion index {n:.3f}: NA = n sin(theta) <= n")


def rayleigh_um(wavelength_um: float, na: float) -> float:
    """Rayleigh two-point resolution, ``0.61 lambda / NA``: the Airy first zero."""
    return 0.61 * wavelength_um / na


def abbe_um(wavelength_um: float, na: float) -> float:
    """Abbe limit for incoherent (fluorescence) imaging, ``lambda / 2NA``.

    The period of the finest grating the OTF still passes.
    """
    return wavelength_um / (2.0 * na)


def fwhm_um(wavelength_um: float, na: float) -> float:
    """FWHM of the Airy disc, ``0.514 lambda / NA``."""
    return 0.5145 * wavelength_um / na


def gaussian_sigma_um(wavelength_um: float, na: float) -> float:
    """Best-fit Gaussian sigma of the widefield PSF, ``0.21 lambda / NA``.

    Zhang, Zerubia & Olivo-Marin, Appl. Opt. 46, 1819 (2007), paraxial widefield.
    It is what a Gaussian fit to an image of a point source recovers, so it is the
    yardstick for the widths the quantitation pane measures.
    """
    return 0.21 * wavelength_um / na


def cutoff_cycles_per_um(wavelength_um: float, na: float) -> float:
    """Incoherent OTF cutoff ``2 NA / lambda``."""
    return 2.0 * na / wavelength_um


def nyquist_pixel_um(wavelength_um: float, na: float) -> float:
    """Largest pixel (referred to the sample) that samples the OTF cutoff.

    Two samples per period of the cutoff frequency: ``lambda / 4NA``. The common
    bench rule "Rayleigh / 2.3" is about 15% looser, and says so in the report.
    """
    return wavelength_um / (4.0 * na)


def depth_of_field_um(wavelength_um: float, na: float, n: float) -> float:
    """Axial wave-optical depth of field, ``lambda n / NA^2``."""
    return wavelength_um * n / na**2


def collection_efficiency(na: float, n: float) -> float:
    """Fraction of an isotropic emitter's photons inside the acceptance cone.

    Solid angle of a cone of half-angle theta over 4 pi: ``(1 - cos theta) / 2``.
    Dipole orientation and interface effects near a coverslip shift this by tens of
    percent; for a freely rotating dye in the focal plane it is the right number.
    """
    check_na(na, n)
    theta = math.asin(min(na / n, 1.0))
    return (1.0 - math.cos(theta)) / 2.0


def frequency_grid(n: int, dx_um: float) -> np.ndarray:
    """Radial spatial frequency, cycles/um, on the unshifted FFT grid."""
    f = np.fft.fftfreq(n, d=dx_um)
    fx, fy = np.meshgrid(f, f, indexing="xy")
    return np.hypot(fx, fy)


def otf(n: int, dx_um: float, wavelength_um: float, na: float) -> np.ndarray:
    """Incoherent OTF of an aberration-free circular pupil, on the FFT grid.

    ``(2/pi) [acos(s) - s sqrt(1 - s^2)]`` with ``s = f / (2 NA / lambda)``: the
    autocorrelation of the pupil, real and non-negative, zero beyond the cutoff.
    """
    s = frequency_grid(n, dx_um) / cutoff_cycles_per_um(wavelength_um, na)
    s = np.clip(s, 0.0, 1.0)
    return (2.0 / math.pi) * (np.arccos(s) - s * np.sqrt(1.0 - s * s))


def psf(n: int, dx_um: float, wavelength_um: float, na: float) -> np.ndarray:
    """Intensity PSF, centred on the grid and normalized to unit sum."""
    h = np.real(np.fft.ifft2(otf(n, dx_um, wavelength_um, na)))
    h = np.fft.fftshift(np.clip(h, 0.0, None))
    return h / h.sum()


def line_spread(n: int, dx_um: float, wavelength_um: float, na: float) -> np.ndarray:
    """Line spread function: the image of an infinitely thin line, unit sum.

    It is the 1D inverse transform of the OTF along one frequency axis, and also
    the marginal of the 2D PSF -- so it is the profile across a thin filament *and*
    the summed profile through a point source, which is why one calibration serves
    both specimens.
    """
    f = np.abs(np.fft.fftfreq(n, d=dx_um))
    s = np.clip(f / cutoff_cycles_per_um(wavelength_um, na), 0.0, 1.0)
    transfer = (2.0 / math.pi) * (np.arccos(s) - s * np.sqrt(1.0 - s * s))
    lsf = np.clip(np.real(np.fft.ifft(transfer)), 0.0, None)
    return lsf / lsf.sum()  # centred on sample 0, wrapping


def blur(image: np.ndarray, dx_um: float, wavelength_um: float, na: float) -> np.ndarray:
    """Convolve with the widefield PSF. Conserves the total, which is photons."""
    if image.shape[0] != image.shape[1]:
        raise ValueError("the object grid must be square")
    transfer = otf(image.shape[0], dx_um, wavelength_um, na)
    out = np.real(np.fft.ifft2(np.fft.fft2(image) * transfer))
    return np.clip(out, 0.0, None)


def box_integrate(image: np.ndarray, dx_um: float, pixel_um: float) -> np.ndarray:
    """Integrate a finely sampled image over camera pixels of an arbitrary size.

    A camera pixel counts every photon that lands on it, so this is an area
    integral, not a resample. The cumulative sum of a piecewise-constant image is
    piecewise linear, which makes linear interpolation of it exact at pixel edges
    that fall between fine samples. Pixels that would overhang the field are
    dropped, so the field is a whole number of camera pixels.
    """
    if pixel_um <= 0:
        raise ValueError("pixel size must be positive")
    n = image.shape[0]
    count = int(math.floor(n * dx_um / pixel_um + 1e-9))
    if count < 1:
        raise ValueError(f"a {pixel_um:.3g} um pixel is larger than the {n * dx_um:.3g} um field")
    edges = np.arange(count + 1) * (pixel_um / dx_um)
    fine = np.arange(n + 1)

    def along(a: np.ndarray, axis: int) -> np.ndarray:
        c = np.cumsum(a, axis=axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        c = np.moveaxis(c, axis, -1)
        flat = c.reshape(-1, n + 1)
        at = np.stack([np.interp(edges, fine, row) for row in flat])
        out = np.diff(at, axis=-1).reshape(c.shape[:-1] + (count,))
        return np.moveaxis(out, -1, axis)

    return along(along(image, 1), 0)
