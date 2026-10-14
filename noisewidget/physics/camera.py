"""Cameras: quantum efficiency, read noise, dark current, gain and fixed pattern.

The presets are **generic sensor classes**, not products. Their numbers are typical
of what each class is sold with and are recorded from secondary knowledge; they are
not datasheet values for any camera, and the widget says so. What matters for the
lesson is the *ratio* between them -- an EMCCD's 50 e- read noise divided by an EM
gain of 300 against an sCMOS's 1.5 e- -- and those ratios are representative.

Dark current is thermally generated charge. It roughly doubles every ~6-7 C, so it
is modelled as ``D(T) = D_ref * 2 ** ((T - T_ref) / T_double)``. Its noise is shot
noise on thermally generated electrons -- the "thermal shot noise" -- and it adds
to the variance exactly as photon shot noise does.

Fixed-pattern noise is the part that does *not* change from frame to frame:

* **PRNU** (photo-response non-uniformity): pixel-to-pixel gain variation, so its
  noise grows in proportion to the signal and averaging frames never removes it.
* **DSNU** (dark-signal non-uniformity) and **hot pixels**: pixel-to-pixel dark
  current variation.
* **Pixel-dependent read noise** in sCMOS: every pixel has its own amplifier, so
  read noise is a distribution with a long tail, not one number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Camera:
    name: str
    pixel_um: float
    quantum_efficiency: float
    read_noise_e: float  # rms, median pixel
    full_well_e: float
    bit_depth: int
    adc_gain_e_per_adu: float  # at EM gain 1
    offset_adu: float
    dark_e_per_s: float  # at dark_ref_c
    dark_ref_c: float
    default_temperature_c: float
    min_temperature_c: float
    dark_doubling_c: float = 6.5
    prnu: float = 0.005  # fractional rms
    dsnu_e_per_s: float = 0.0  # rms pixel-to-pixel dark current at dark_ref_c
    read_noise_spread: float = 0.0  # lognormal sigma of per-pixel read noise
    hot_pixel_fraction: float = 0.0
    hot_pixel_factor: float = 100.0
    has_em_gain: bool = False
    max_em_gain: float = 1.0
    em_read_noise_e: float = 0.0  # read noise of the EM amplifier, before gain
    cic_e: float = 0.0  # clock-induced charge per pixel per frame
    register_well_e: float = 0.0  # gain-register capacity, output electrons (0: none)
    charge_binning: bool = False  # CCD bins charge before readout; CMOS bins digits
    kind: str = "sCMOS"

    def dark_current(self, temperature_c: float) -> float:
        """Mean dark current at a sensor temperature, e-/pixel/s."""
        return self.dark_e_per_s * 2.0 ** ((temperature_c - self.dark_ref_c) / self.dark_doubling_c)

    def read_noise(self, em_gain: float) -> float:
        """Read noise of the amplifier in use (before any EM gain), e- rms."""
        return self.em_read_noise_e if self.has_em_gain and em_gain > 1.0 else self.read_noise_e

    @property
    def max_adu(self) -> int:
        return 2**self.bit_depth - 1

    def with_overrides(self, **changes) -> "Camera":
        return replace(self, **changes)


CAMERAS: dict[str, Camera] = {
    c.name: c
    for c in (
        Camera(
            "sCMOS, front-illuminated (6.5 um)", 6.5, 0.82, 1.4, 30_000, 16, 0.46, 100,
            dark_e_per_s=0.6, dark_ref_c=0.0, default_temperature_c=0.0, min_temperature_c=-30.0,
            prnu=0.003, dsnu_e_per_s=0.2, read_noise_spread=0.35, hot_pixel_fraction=2e-4,
            kind="sCMOS",
        ),
        Camera(
            "sCMOS, back-illuminated (11 um)", 11.0, 0.95, 1.6, 80_000, 16, 1.25, 100,
            dark_e_per_s=0.5, dark_ref_c=-10.0, default_temperature_c=-10.0, min_temperature_c=-30.0,
            prnu=0.005, dsnu_e_per_s=0.15, read_noise_spread=0.3, hot_pixel_fraction=2e-4,
            kind="sCMOS",
        ),
        Camera(
            "EMCCD, back-illuminated (13 um)", 13.0, 0.95, 6.0, 80_000, 16, 3.5, 100,
            dark_e_per_s=0.0003, dark_ref_c=-80.0, default_temperature_c=-80.0, min_temperature_c=-95.0,
            prnu=0.01, dsnu_e_per_s=0.0001, hot_pixel_fraction=5e-5,
            has_em_gain=True, max_em_gain=1000.0, em_read_noise_e=50.0, cic_e=0.005,
            register_well_e=800_000,
            charge_binning=True, kind="EMCCD",
        ),
        Camera(
            "Interline CCD (6.45 um)", 6.45, 0.65, 6.5, 18_000, 14, 1.1, 100,
            dark_e_per_s=0.05, dark_ref_c=-30.0, default_temperature_c=-30.0, min_temperature_c=-40.0,
            prnu=0.01, dsnu_e_per_s=0.02, hot_pixel_fraction=1e-4,
            charge_binning=True, kind="CCD",
        ),
        Camera(
            "CMOS, uncooled (3.45 um)", 3.45, 0.70, 2.3, 10_500, 12, 2.6, 16,
            dark_e_per_s=3.0, dark_ref_c=25.0, default_temperature_c=35.0, min_temperature_c=20.0,
            prnu=0.01, dsnu_e_per_s=1.0, read_noise_spread=0.4, hot_pixel_fraction=1e-3,
            kind="CMOS",
        ),
    )
}

DEFAULT_CAMERA = "sCMOS, front-illuminated (6.5 um)"


def excess_noise_factor_squared(camera: Camera, em_gain: float) -> float:
    """``F^2`` of the gain register: ~2 for any real EM gain, 1 without it.

    A multiplication register amplifies each electron by a random amount, which
    doubles the variance of whatever enters it. That is the price of beating read
    noise, and it is the same as halving the quantum efficiency -- the reason an
    EMCCD loses to an sCMOS once there are more than a few photons per pixel.
    """
    return 2.0 if camera.has_em_gain and em_gain > 1.0 else 1.0


@dataclass(frozen=True)
class FixedPattern:
    """The per-pixel maps that are the same in every frame, at the binned grid."""

    gain: np.ndarray  # PRNU multiplier, mean 1
    dark_scale: np.ndarray  # dark current multiplier, mean ~1 (hot pixels >> 1)
    read_noise_e: np.ndarray  # rms read noise per (super)pixel, e-


def fixed_pattern(
    camera: Camera, shape: tuple[int, int], binning: int, em_gain: float, seed: int = 4242,
    enabled: bool = True,
) -> FixedPattern:
    """Build the fixed-pattern maps for a sensor region, at binned resolution.

    Seeded per camera, so the pattern is a property of *this* sensor: it stays put
    while noise is re-rolled, which is how a user discovers that averaging frames
    cannot remove it.

    Binning is handled by generating every native pixel and aggregating, so the
    arithmetic is the camera's: a CCD sums charge and reads once (one read noise per
    superpixel); a CMOS reads every pixel and sums the digits (``b^2`` read noises
    in quadrature).
    """
    b = max(int(binning), 1)
    native = (shape[0] * b, shape[1] * b)
    rng = np.random.default_rng(seed + sum(map(ord, camera.name)))

    def bin_mean(a: np.ndarray) -> np.ndarray:
        return a.reshape(shape[0], b, shape[1], b).mean(axis=(1, 3))

    read = camera.read_noise(em_gain)
    if enabled:
        gain = 1.0 + camera.prnu * rng.standard_normal(native)
        dsnu = camera.dsnu_e_per_s / camera.dark_e_per_s if camera.dark_e_per_s > 0 else 0.0
        dark = np.clip(1.0 + dsnu * rng.standard_normal(native), 0.0, None)
        hot = rng.random(native) < camera.hot_pixel_fraction
        dark[hot] *= camera.hot_pixel_factor
        if camera.read_noise_spread > 0:
            # Lognormal with the stated median: most pixels are quiet, a few are not.
            per_pixel = read * np.exp(camera.read_noise_spread * rng.standard_normal(native))
        else:
            per_pixel = np.full(native, read)
    else:
        gain = np.ones(native)
        dark = np.ones(native)
        per_pixel = np.full(native, read)

    if camera.charge_binning:
        # One read of the summed charge. The read noise of a CCD is a property of
        # its output amplifier, so it is the same for a superpixel as for a pixel.
        read_map = np.full(shape, read) if b > 1 else per_pixel
    else:
        read_map = np.sqrt((per_pixel**2).reshape(shape[0], b, shape[1], b).sum(axis=(1, 3)))
    return FixedPattern(bin_mean(gain), bin_mean(dark), read_map)


def effective_full_well(camera: Camera, binning: int) -> float:
    """Charge a (super)pixel can hold before it clips.

    CMOS bins after readout, so each native pixel clips on its own and the sum can
    hold ``b^2`` wells. A CCD sums charge into the serial register, which is
    typically about twice a pixel's well; past that a binned superpixel blooms.
    """
    b2 = max(int(binning), 1) ** 2
    if camera.charge_binning:
        return camera.full_well_e * min(b2, 2) if b2 > 1 else camera.full_well_e
    return camera.full_well_e * b2


def describe_temperature(camera: Camera, temperature_c: float) -> str:
    rate = camera.dark_current(temperature_c)
    return (
        f"D = {camera.dark_e_per_s:g} x 2^(({temperature_c:g} - {camera.dark_ref_c:g}) / "
        f"{camera.dark_doubling_c:g}) = {rate:.3g} e-/px/s"
    )


def clamp_temperature(camera: Camera, temperature_c: float) -> float:
    return max(camera.min_temperature_c, min(temperature_c, 45.0))


def quantization_noise_e(camera: Camera, em_gain: float) -> float:
    """Rounding to whole ADU adds ``1/sqrt(12)`` ADU rms, referred to the input."""
    return camera.adc_gain_e_per_adu / max(em_gain, 1.0) / math.sqrt(12.0)
