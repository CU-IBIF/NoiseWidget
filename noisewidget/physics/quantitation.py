"""What noise does to a measurement.

The reference structure at the centre of the field is measured the way it would be
at the bench: a line profile across it, averaged along a stretch of filament (or
summed over a bead), fitted with a Gaussian on a flat background. Three things are
read off the fit:

* **width** (FWHM) -- for a 25 nm microtubule this is ~230 nm at NA 1.4: the PSF,
  not the microtubule. Subtracting the PSF in quadrature gives an estimate of the
  structure's own width, and noise is what makes that estimate fall apart.
* **intensity** -- the area under the profile, divided by the photoelectrons one
  fluorophore yields, is a count of labels. Compared with the labels actually
  placed, it says how well the image can count molecules.
* **position** -- the fitted centre, whose scatter is the localization precision.

One frame gives one number. The noise is what makes that number differ from the
next frame's, so the same noise-free image is re-recorded ``realizations`` times
(only the region being measured, so it stays interactive) and the spread of the
results is reported alongside the single-frame answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from typing import Protocol

from .specimens import Specimen

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


class Measurable(Protocol):
    """What a simulation must offer to be measured -- camera or point scanner.

    Rows run across the reference structure (``pixel_y_um`` apart) and columns
    along it (``pixel_x_um`` apart). A camera's pixels are square; a scanner's
    need not be, and its rows are lines the beam visits rather than areas it
    integrates, which is why sampling is behind methods rather than assumed.
    """

    specimen: Specimen
    signal_e: np.ndarray
    expected_e: np.ndarray
    recorded_e: np.ndarray

    @property
    def pixel_x_um(self) -> float: ...

    @property
    def row_centres_um(self) -> np.ndarray: ...

    @property
    def col_centres_um(self) -> np.ndarray: ...

    @property
    def pixel_y_um(self) -> float: ...

    @property
    def psf_sigma_um(self) -> float: ...

    @property
    def detected_per_fluor(self) -> float: ...

    @property
    def noise_seed(self) -> int: ...

    def rerecord(self, crop: tuple[slice, slice], realizations: int, rng: np.random.Generator) -> np.ndarray: ...

    def line_response(self, r0: int, r1: int) -> np.ndarray: ...


@dataclass(frozen=True)
class QuantSettings:
    profile_length_um: float = 2.0  # length of filament averaged into the profile
    realizations: int = 100
    seed: int = 99


def half_width_um(sim: Measurable) -> float:
    """Half-height of the window across the reference, in um.

    Wide enough to hold the profile's tails and some background on either side,
    but inside the clear box the specimen keeps around the reference.
    """
    sigma = math.hypot(sim.psf_sigma_um, sim.specimen.kind.label_sigma_nm / 1000)
    return min(max(6.0 * sigma, 5.0 * sim.pixel_y_um), 1.7)


def reference_window(sim: Measurable, settings: QuantSettings | None = None) -> tuple[int, int, int, int]:
    """Pixel bounds ``(row0, row1, col0, col1)`` of the measured region."""
    settings = settings or QuantSettings()
    rows, cols = sim.signal_e.shape
    cx, cy = sim.specimen.reference.centre_um
    hw = half_width_um(sim)
    centres_r = sim.row_centres_um
    sel_r = np.nonzero(np.abs(centres_r - cy) <= hw)[0]
    if sel_r.size < 5:  # a fit needs points; take the nearest five
        k = int(np.argmin(np.abs(centres_r - cy)))
        sel_r = np.arange(max(k - 2, 0), min(k + 3, rows))
    if sim.specimen.kind.kind == "filament":
        half_len = settings.profile_length_um / 2.0
    else:
        half_len = hw
    centres_c = sim.col_centres_um
    sel_c = np.nonzero(np.abs(centres_c - cx) <= half_len)[0]
    if sel_c.size == 0:
        k = int(np.argmin(np.abs(centres_c - cx)))
        sel_c = np.array([k])
    return int(sel_r[0]), int(sel_r[-1]) + 1, int(sel_c[0]), int(sel_c[-1]) + 1


def profile(image: np.ndarray, window: tuple[int, int, int, int], kind: str) -> np.ndarray:
    """Profile across the reference: mean over columns (filament), sum (bead).

    Leading axes are realizations, so a stack of frames gives a stack of profiles.
    """
    r0, r1, c0, c1 = window
    region = image[..., r0:r1, c0:c1]
    return region.mean(axis=-1) if kind == "filament" else region.sum(axis=-1)


# --- fitting ---------------------------------------------------------------------


def fit_gaussians(x: np.ndarray, y: np.ndarray, iterations: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt fit of ``A exp(-(x-mu)^2 / 2 s^2) + B`` to many profiles.

    ``y`` is ``(R, m)``. Batched across realizations so a hundred fits cost one
    small set of array operations, not a hundred Python loops. Returns parameters
    ``(R, 4)`` as ``A, mu, sigma, B`` and a success mask: a fit that wandered off the
    window, went negative, or did not converge is a failure, and failures are
    reported rather than hidden -- at low light, "could not measure it" is the
    result.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    R, m = y.shape
    span = x[-1] - x[0]
    step = (x[1] - x[0]) if m > 1 else 1.0

    # Moment-based starting guesses.
    b0 = np.minimum(y[:, 0], y[:, -1])
    yy = np.clip(y - b0[:, None], 0.0, None)
    w = yy.sum(axis=1)
    w_safe = np.where(w > 0, w, 1.0)
    mu0 = np.where(w > 0, (yy * x).sum(axis=1) / w_safe, x[m // 2])
    var0 = np.where(w > 0, (yy * (x - mu0[:, None]) ** 2).sum(axis=1) / w_safe, (span / 6) ** 2)
    s0 = np.clip(np.sqrt(np.abs(var0)), step / 2, span)
    a0 = np.maximum(y.max(axis=1) - b0, 1e-9)
    p = np.stack([a0, mu0, s0, b0], axis=1)
    lam = np.full(R, 1e-2)

    def model(p):
        a, mu, s, b = (p[:, i : i + 1] for i in range(4))
        g = np.exp(-0.5 * ((x - mu) / s) ** 2)
        return a * g + b, g

    def cost(p):
        f, _ = model(p)
        return ((y - f) ** 2).sum(axis=1)

    c = cost(p)
    for _ in range(iterations):
        f, g = model(p)
        a, mu, s = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        d = (x - mu) / s
        J = np.stack([g, a * g * d / s, a * g * d * d / s, np.ones_like(g)], axis=2)
        r = y - f
        JTJ = np.einsum("rmi,rmj->rij", J, J)
        JTr = np.einsum("rmi,rm->ri", J, r)
        damp = JTJ + lam[:, None, None] * np.einsum("rii->ri", JTJ)[:, :, None] * np.eye(4)
        try:
            delta = np.linalg.solve(damp + 1e-12 * np.eye(4), JTr[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        trial = p + delta
        trial[:, 2] = np.abs(trial[:, 2]) + 1e-12
        ct = cost(trial)
        better = ct < c
        p = np.where(better[:, None], trial, p)
        c = np.where(better, ct, c)
        lam = np.where(better, lam * 0.3, lam * 10.0)
        lam = np.clip(lam, 1e-9, 1e9)

    a, mu, s, _ = p.T
    ok = (
        np.isfinite(p).all(axis=1)
        & (a > 0)
        & (s > step * 0.15)
        & (s < span)
        & (mu > x[0])
        & (mu < x[-1])
    )
    return p, ok


@dataclass(frozen=True)
class Instrument:
    """What this system does to an infinitely thin line of unit brightness."""

    sigma_um: float  # the Gaussian sigma a fit recovers
    fit_recovery: float  # fraction of the line's light the fitted Gaussian contains
    sum_recovery: float  # fraction a background-subtracted sum over the window recovers


def instrument_response(sim: Measurable, window: tuple[int, int, int, int]) -> Instrument:
    """Fit this system's own image of an infinitely thin line.

    The calibration a careful user makes with a sub-resolution bead: the model's
    own line spread function, positioned where the reference is, integrated over
    the same camera pixels and fitted with the same Gaussian. It folds in the PSF,
    its non-Gaussian tails and the pixel size, so subtracting it in quadrature
    removes exactly what the instrument added -- and nothing else.

    The same fit says how much of the light a measurement recovers. The Airy
    pattern's rings fall outside any Gaussian, and the tails beyond the window are
    taken for background, so both a fit and a sum under-count a little; dividing by
    the recovery is the bead calibration that corrects it.
    """
    _, cy = sim.specimen.reference.centre_um
    r0, r1, _, _ = window
    py = sim.pixel_y_um
    response = sim.line_response(r0, r1)
    x = sim.row_centres_um[r0:r1] - cy
    params, _ = fit_gaussians(x, response[None, :])
    a, _, sigma, b = (float(v) for v in params[0])
    fit_area = a * sigma * math.sqrt(2 * math.pi) / py
    summed = float((response - b).sum())
    return Instrument(sigma, fit_area, summed)


# --- the measurement -------------------------------------------------------------


@dataclass
class Fit:
    amplitude: float
    centre_um: float
    sigma_um: float
    background: float
    ok: bool

    @property
    def fwhm_um(self) -> float:
        return FWHM_PER_SIGMA * self.sigma_um

    def curve(self, x: np.ndarray) -> np.ndarray:
        return self.amplitude * np.exp(-0.5 * ((x - self.centre_um) / self.sigma_um) ** 2) + self.background


@dataclass
class Spread:
    """Statistics of one measured quantity over the noise realizations."""

    values: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values.size else math.nan

    @property
    def sd(self) -> float:
        return float(np.std(self.values, ddof=1)) if self.values.size > 1 else math.nan

    @property
    def cv(self) -> float:
        m = self.mean
        return self.sd / abs(m) if m else math.nan


@dataclass
class Quantitation:
    kind: str
    x_um: np.ndarray  # profile positions, relative to the true centre
    measured: np.ndarray  # this frame, e-
    expected: np.ndarray  # noise-free, e-
    fit: Fit  # of this frame
    ideal: Fit  # of the noise-free profile
    object_x_um: np.ndarray  # fine-sampled label distribution across the structure
    object_profile: np.ndarray
    fwhm: Spread  # um, successful fits
    centre: Spread  # um
    count: Spread  # fluorophores per um (filament) or per bead, from the fit
    summed: Spread  # the same, by summing the background-subtracted profile
    corrected_width: Spread  # PSF subtracted in quadrature, um (real roots only)
    peak_snr: Spread
    success: float  # fraction of realizations that fitted
    realizations: int
    # ground truth
    true_diameter_nm: float
    label_diameter_nm: float
    label_sigma_nm: float
    nominal_density: float
    labels_present: float  # actually in the window, per um or per bead
    density_units: str
    psf_sigma_um: float  # 0.21 lambda / NA, the textbook approximation
    instrument_sigma_um: float  # fitted to this system's own line image
    fit_recovery: float
    sum_recovery: float
    pixel_um: float  # pitch across the structure (rows)
    pixel_x_um: float  # pitch along it (columns)
    predicted_fwhm_um: float
    detected_per_fluor: float
    window_length_um: float

    @property
    def single_count(self) -> float:
        """This frame's fitted count, calibrated like the realizations."""
        raw = _area_to_count(
            self.fit.amplitude, self.fit.sigma_um, self.pixel_um, self.pixel_x_um, self.kind, self.detected_per_fluor
        )
        return raw / self.fit_recovery


def _area_to_count(a, s, py, px, kind, per_fluor):
    """Fluorophores from a fitted profile.

    The profile is sampled once per row, so its integral in electrons is
    ``A s sqrt(2 pi) / py``. For a filament the profile is a per-column mean, so
    that is electrons per column, and dividing by the column pitch ``px`` makes it
    per um.
    """
    area = a * s * math.sqrt(2 * math.pi) / py if np.isscalar(a) else a * s * np.sqrt(2 * np.pi) / py
    if kind == "filament":
        area = area / px
    return area / per_fluor if per_fluor > 0 else area * math.nan


def quantify(sim: Measurable, settings: QuantSettings | None = None) -> Quantitation:
    settings = settings or QuantSettings()
    kind = sim.specimen.kind
    window = reference_window(sim, settings)
    r0, r1, c0, c1 = window
    py, px = sim.pixel_y_um, sim.pixel_x_um
    cx, cy = sim.specimen.reference.centre_um
    x = sim.row_centres_um[r0:r1] - cy

    measured = profile(sim.recorded_e, window, kind.kind)
    expected = profile(sim.expected_e, window, kind.kind)

    # Re-record just this region, many times, with fresh noise each time.
    crop = (slice(r0, r1), slice(c0, c1))
    R = max(int(settings.realizations), 2)
    rng = np.random.default_rng(settings.seed + sim.noise_seed * 7)
    frames_e = sim.rerecord(crop, R, rng)
    profiles = profile(frames_e, (0, r1 - r0, 0, c1 - c0), kind.kind)

    all_profiles = np.vstack([measured[None, :], expected[None, :], profiles])
    params, ok = fit_gaussians(x, all_profiles)

    def as_fit(i):
        a, mu, s, b = params[i]
        return Fit(float(a), float(mu), float(s), float(b), bool(ok[i]))

    fit, ideal = as_fit(0), as_fit(1)
    mc, mc_ok = params[2:], ok[2:]
    good = mc[mc_ok]

    psf_sigma = sim.psf_sigma_um
    response = instrument_response(sim, window)
    instrument = response.sigma_um
    widths = good[:, 2] if good.size else np.empty(0)
    corr_var = widths**2 - instrument**2
    corrected = FWHM_PER_SIGMA * np.sqrt(corr_var[corr_var > 0])

    # Peak SNR of each realization: fitted amplitude over the background scatter.
    noise = profiles - expected[None, :]
    noise_sd = noise.std(axis=1, ddof=1)
    amp = good[:, 0] if good.size else np.empty(0)
    snr = amp / np.maximum(noise_sd[mc_ok], 1e-12) if good.size else np.empty(0)

    # Ground truth: count the labels that were actually placed in the window.
    ref = sim.specimen.reference
    x0, x1 = sim.col_centres_um[c0] - px / 2, sim.col_centres_um[c1 - 1] + px / 2
    y0, y1 = sim.row_centres_um[r0] - py / 2, sim.row_centres_um[r1 - 1] + py / 2
    if kind.kind == "filament":
        length = x1 - x0
        present = ref.count_within(x0, x1, y0, y1) / length
    else:
        length = 0.0
        present = float(ref.weight.sum())

    label_sigma = kind.label_sigma_nm / 1000.0
    predicted = FWHM_PER_SIGMA * math.sqrt(instrument**2 + label_sigma**2)

    # The structure as it really is, across its axis: a fine histogram of labels.
    ox = np.linspace(-0.15, 0.15, 31)  # 10 nm bins
    near = (np.abs(ref.x - cx) <= max((x1 - x0) / 2, 0.15)) if kind.kind == "filament" else np.ones(ref.x.size, bool)
    hist, edges = np.histogram(ref.y[near] - cy, bins=ox, weights=ref.weight[near])
    centres = 0.5 * (edges[:-1] + edges[1:])

    per_fluor = sim.detected_per_fluor
    counts = _area_to_count(good[:, 0], good[:, 2], py, px, kind.kind, per_fluor) if good.size else np.empty(0)
    counts = counts / response.fit_recovery
    # Aperture photometry: no model, so no model bias -- but every pixel's noise
    # goes into the sum, so it is noisier than the fit.
    summed = (profiles[mc_ok] - good[:, 3:4]).sum(axis=1) if good.size else np.empty(0)
    if kind.kind == "filament":
        summed = summed / px
    summed = summed / per_fluor if per_fluor > 0 else summed * math.nan
    summed = summed / response.sum_recovery

    return Quantitation(
        kind=kind.kind,
        x_um=x,
        measured=measured,
        expected=expected,
        fit=fit,
        ideal=ideal,
        object_x_um=centres,
        object_profile=hist,
        fwhm=Spread(FWHM_PER_SIGMA * widths),
        centre=Spread(good[:, 1] if good.size else np.empty(0)),
        count=Spread(counts),
        summed=Spread(summed),
        corrected_width=Spread(corrected),
        peak_snr=Spread(snr),
        success=float(mc_ok.mean()),
        realizations=R,
        true_diameter_nm=kind.true_diameter_nm,
        label_diameter_nm=kind.label_diameter_nm,
        label_sigma_nm=kind.label_sigma_nm,
        nominal_density=sim.specimen.density_value,
        labels_present=present,
        density_units=kind.density_units,
        psf_sigma_um=psf_sigma,
        instrument_sigma_um=instrument,
        fit_recovery=response.fit_recovery,
        sum_recovery=response.sum_recovery,
        pixel_um=py,
        pixel_x_um=px,
        predicted_fwhm_um=predicted,
        detected_per_fluor=per_fluor,
        window_length_um=length,
    )
