"""Point-scanning confocal: one beam, one pinhole, one detector, one pixel at a time.

A camera collects every pixel at once for the whole exposure. A confocal visits each
pixel for a **dwell time** of microseconds, so the photon budget per pixel is set by
dwell, laser power and how finely the field is sampled -- and the noise is the
detector's, not a sensor array's.

The image model::

    excitation   I(r) = I0 h_ex(r) / h_ex(0),  I0 = P h_ex(0) / dx^2       the focused spot
    emission     E(r) = QY k(r) / (1 + k(r) tau),  k = sigma I / h nu       saturates at the focus
    detection    D(r) = (h_em (x) pinhole)(r)                               pinhole in Airy units
    kernel       K(r) = E(r) D(r)                                           photons/s per fluorophore
    image        rate(r0) = CE T QE  sum_i K(r_i - r0)                      object (x) K

Scanning samples that rate along lines: the beam sweeps continuously across each
pixel in x (a box average over ``px``) and steps between lines in y (a point sample
every ``py``). With x and y pixel counts set independently, the two directions can
be sampled differently -- and an under-sampled y simply misses what lies between
lines.

Field of view is ``FN / (M zoom)``. The specimen is a 12.8 um patch, so at low
zoom only the scanned pixels that land on it are simulated; the frame time is
still that of the whole frame.

Detectors are **generic classes** (multialkali PMT, GaAsP PMT, hybrid, SPAD), with
typical-of-class figures from secondary sources -- not any vendor's datasheet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import optics
from .fluorophores import FLUOROPHORES
from .specimens import DEFAULT_SPECIMEN, SPECIMENS, Specimen, make_specimen

FIELD_NUMBER_MM = 18.0  # scan field at zoom 1 is FN / M


# --- detectors -------------------------------------------------------------------


@dataclass(frozen=True)
class Detector:
    name: str
    kind: str
    quantum_efficiency: float
    excess_noise_f2: float  # F^2 of the gain process in analog mode
    dark_cps: float  # dark counts per second at dark_ref_c
    dark_ref_c: float
    default_temperature_c: float
    min_temperature_c: float
    dead_time_ns: float
    afterpulse: float  # probability per detected photon
    amplifier_noise_adu: float  # electronics noise after the gain, analog mode
    modes: tuple[str, ...]
    dark_doubling_c: float = 6.0

    def dark_rate(self, temperature_c: float) -> float:
        """Thermionic dark counts per second: roughly doubles every ~6 C."""
        return self.dark_cps * 2.0 ** ((temperature_c - self.dark_ref_c) / self.dark_doubling_c)


DETECTORS: dict[str, Detector] = {
    d.name: d
    for d in (
        Detector("PMT, multialkali", "PMT", 0.20, 1.25, 150.0, 20.0, 20.0, 20.0,
                 20.0, 0.01, 1.5, ("analog",)),
        Detector("GaAsP PMT (cooled)", "GaAsP", 0.40, 1.5, 2000.0, 25.0, 5.0, 0.0,
                 20.0, 0.01, 1.5, ("analog", "counting")),
        Detector("Hybrid detector (GaAsP HPD)", "hybrid", 0.45, 1.05, 300.0, 25.0, 25.0, 15.0,
                 2.0, 0.001, 1.0, ("counting", "analog")),
        Detector("SPAD / APD", "SPAD", 0.55, 1.0, 100.0, 20.0, 20.0, 0.0,
                 22.0, 0.005, 0.0, ("counting",)),
    )
}
DEFAULT_DETECTOR = "GaAsP PMT (cooled)"


@dataclass(frozen=True)
class ScanNoise:
    """Which noise sources are simulated. All on is a real instrument."""

    labeling: bool = True
    photon_shot: bool = True
    background: bool = True
    dark_counts: bool = True
    gain_noise: bool = True  # dynode / avalanche excess noise
    amplifier: bool = True
    quantization: bool = True
    laser_noise: bool = True
    dead_time: bool = True
    afterpulsing: bool = True
    clipping: bool = True

    LABELS = {
        "labeling": "labelling stochasticity",
        "photon_shot": "photon shot noise",
        "background": "out-of-focus haze (and its shot noise)",
        "dark_counts": "dark counts (thermal shot noise)",
        "gain_noise": "gain (excess) noise, F^2",
        "amplifier": "amplifier noise (analog)",
        "quantization": "ADC quantization (analog)",
        "laser_noise": "laser intensity noise",
        "dead_time": "dead time (counting)",
        "afterpulsing": "afterpulsing",
        "clipping": "ADC clipping (high and black)",
    }


@dataclass(frozen=True)
class ScanLevers:
    """Everything a user can turn on a point-scanning confocal."""

    # specimen
    specimen: str = DEFAULT_SPECIMEN
    label_density: float | None = None
    fluorophore: str = "EGFP"
    specimen_seed: int = 1
    # optics
    na: float = 1.40
    medium: str = "oil"
    magnification: float = 63.0
    wavelength_nm: float | None = None  # emission; None: the fluorophore's peak
    transmission: float = 0.30  # detection path: objective, scan optics, dichroic, filter
    pinhole_au: float = 1.0
    # illumination
    laser_power_uw: float = 15.0  # at the sample
    background_per_um2: float = 40.0  # out-of-focus haze, fluorophore-equivalents per um^2
    laser_rin: float = 0.005  # laser intensity noise, fractional rms per pixel
    # scan
    zoom: float = 8.0
    pixels_x: int = 512
    pixels_y: int = 512
    dwell_us: float = 3.2
    averages: int = 1  # line averages
    bidirectional: bool = False
    # detector
    detector: str = DEFAULT_DETECTOR
    mode: str = "analog"  # or "counting"
    gain_adu_per_e: float = 8.0  # analog: ADU per photoelectron (set by the HV)
    offset_adu: float = 10.0
    bit_depth: int = 12
    temperature_c: float | None = None
    dead_time_correction: bool = False
    # noise
    noise: ScanNoise = field(default_factory=ScanNoise)
    noise_seed: int = 0

    def with_(self, **changes) -> "ScanLevers":
        return replace(self, **changes)

    @property
    def detector_model(self) -> Detector:
        return DETECTORS[self.detector]

    @property
    def fluor(self):
        return FLUOROPHORES[self.fluorophore]

    @property
    def wavelength_um(self) -> float:
        nm = self.wavelength_nm if self.wavelength_nm is not None else self.fluor.emission_nm
        return nm / 1000.0

    @property
    def excitation_um(self) -> float:
        return self.fluor.excitation_nm / 1000.0

    @property
    def n_medium(self) -> float:
        return optics.IMMERSION_MEDIA[self.medium]

    @property
    def density(self) -> float:
        return SPECIMENS[self.specimen].default_density if self.label_density is None else self.label_density

    @property
    def acquisition_mode(self) -> str:
        """The mode in use: a detector that cannot do the one asked for uses its own."""
        modes = self.detector_model.modes
        return self.mode if self.mode in modes else modes[0]

    @property
    def sensor_temperature_c(self) -> float:
        d = self.detector_model
        t = d.default_temperature_c if self.temperature_c is None else self.temperature_c
        return max(d.min_temperature_c, t)

    @property
    def field_um(self) -> float:
        """Scan field edge at the sample: ``FN / (M zoom)``."""
        return FIELD_NUMBER_MM * 1000.0 / (self.magnification * max(self.zoom, 1e-6))

    @property
    def pixel_x_um(self) -> float:
        return self.field_um / max(int(self.pixels_x), 1)

    @property
    def pixel_y_um(self) -> float:
        return self.field_um / max(int(self.pixels_y), 1)

    @property
    def dwell_s(self) -> float:
        return self.dwell_us * 1e-6

    @property
    def max_adu(self) -> int:
        return 2 ** int(self.bit_depth) - 1

    def frame_time_s(self) -> float:
        """Whole frame, every average: the line runs at a duty cycle below one.

        A galvo spends its flyback doing nothing (~70% duty unidirectional);
        bidirectional scanning uses both strokes (~90%).
        """
        duty = 0.9 if self.bidirectional else 0.7
        line = self.pixels_x * self.dwell_s / duty
        return line * self.pixels_y * max(self.averages, 1)


# --- optics ----------------------------------------------------------------------


@dataclass(frozen=True)
class Kernel:
    """The confocal response to one fluorophore, on the fine grid."""

    rate: np.ndarray  # detected-before-CE photons/s: E(r) D(r), centred
    emission: np.ndarray  # E(r): emitted photons/s per fluorophore at offset r
    pinhole_fraction: float  # D(0): an on-axis emitter's light through the pinhole
    peak_irradiance_w_cm2: float
    saturation: float  # k tau at the focus
    sigma_um: float  # Gaussian sigma of the kernel, measured
    fwhm_um: float


def _pinhole_disk(n: int, dx: float, radius_um: float) -> np.ndarray:
    """Disk of the pinhole's back-projected radius, area-weighted at its edge."""
    ax = (np.arange(n) - n // 2) * dx
    sub = 4  # supersample, so a sub-pixel pinhole still has the right area
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    acc = np.zeros((n, n))
    for oy in offs:
        for ox in offs:
            xx, yy = np.meshgrid(ax + ox * dx, ax + oy * dx, indexing="xy")
            acc += (np.hypot(xx, yy) <= radius_um)
    disk = acc / sub**2
    if disk.sum() == 0:
        disk[n // 2, n // 2] = 1.0
    return disk


def _centre_fwhm(k: np.ndarray, dx: float) -> float:
    row = k[k.shape[0] // 2]
    half = row.max() / 2
    above = np.nonzero(row >= half)[0]
    i0, i1 = above[0], above[-1]
    left = i0 - 1 + (half - row[i0 - 1]) / (row[i0] - row[i0 - 1])
    right = i1 + (row[i1] - half) / (row[i1] - row[i1 + 1])
    return float((right - left) * dx)


def confocal_kernel(n: int, dx: float, levers: ScanLevers) -> Kernel:
    lv = levers
    fl = lv.fluor
    h_ex = optics.psf(n, dx, lv.excitation_um, lv.na)
    # The focused spot: a unit-sum PSF is the fraction of the power per fine pixel,
    # so its peak over the pixel area is the peak irradiance, exactly.
    i0 = lv.laser_power_uw * 1e-6 * h_ex.max() / (dx * 1e-4) ** 2
    irradiance = i0 * h_ex / h_ex.max()
    emission = np.asarray(fl.emission_rate(irradiance), dtype=float)
    k0 = float(fl.absorption_rate(i0))

    h_em = optics.psf(n, dx, lv.wavelength_um, lv.na)
    radius = lv.pinhole_au * 0.61 * lv.wavelength_um / lv.na
    disk = _pinhole_disk(n, dx, radius)
    detect = np.real(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(h_em)) * np.fft.fft2(np.fft.ifftshift(disk))))
    detect = np.clip(np.fft.fftshift(detect), 0.0, 1.0)
    rate = emission * detect
    fwhm = _centre_fwhm(rate, dx)
    return Kernel(
        rate, emission, float(detect[n // 2, n // 2]), i0, k0 * fl.lifetime_ns * 1e-9,
        fwhm / (2 * math.sqrt(2 * math.log(2))), fwhm,
    )


def haze_fraction(pinhole_au: float) -> float:
    """Share of out-of-focus haze that reaches the detector, relative to widefield.

    Out-of-focus light arrives at the pinhole plane as a broad blur, so what gets
    through grows with the pinhole's area; wide open it approaches widefield. This
    ``AU^2 / (1 + AU^2)`` is a pedagogical stand-in for that scaling -- it gives
    the right trend (sectioning is lost as the pinhole opens), not a 3D model.
    """
    a2 = pinhole_au**2
    return a2 / (1.0 + a2)


def scan_sample(fine: np.ndarray, dx: float, xs: np.ndarray, ys: np.ndarray, px: float) -> np.ndarray:
    """Sample a fine map as a scanner does: lines at ``ys``, sweeping ``px`` in x.

    In y the beam sits on each line, so this is a point sample (interpolated
    between fine rows). In x the beam moves continuously through each pixel's
    dwell, so it is the average over ``[x - px/2, x + px/2]`` -- exact for the
    piecewise-constant fine map via its cumulative sum.
    """
    n = fine.shape[0]
    # Interpolate every column at the line positions at once.
    iy = np.clip((ys - 0.5 * dx) / dx, 0, n - 1)
    lo = np.floor(iy).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (iy - lo)[:, None]
    rows = fine[lo] * (1 - w) + fine[hi] * w
    cum = np.concatenate([np.zeros((rows.shape[0], 1)), np.cumsum(rows, axis=1)], axis=1)
    edges_lo = np.clip((xs - px / 2) / dx, 0, n)
    edges_hi = np.clip((xs + px / 2) / dx, 0, n)
    fine_edges = np.arange(n + 1)
    a = np.stack([np.interp(edges_lo, fine_edges, r) for r in cum])
    b = np.stack([np.interp(edges_hi, fine_edges, r) for r in cum])
    width = np.maximum(edges_hi - edges_lo, 1e-12)
    return (b - a) / width


# --- simulation ------------------------------------------------------------------


@dataclass(frozen=True)
class ScanPhotometry:
    detection: float  # CE x T x QE
    collection: float
    qe: float
    emitted_per_frame: float  # photons one fluorophore emits per frame (all pixels)
    survival: float  # mean unbleached fraction over the averages
    detected_per_fluor: float  # e- one fluorophore puts into the whole image, per frame


@dataclass
class ScanSimulation:
    levers: ScanLevers
    specimen: Specimen
    kernel: Kernel
    photometry: ScanPhotometry
    xs: np.ndarray  # pixel centres along x (um), the simulated part of the frame
    ys: np.ndarray  # line positions (um)
    signal_e: np.ndarray  # expected photoelectrons per pixel per average
    background_e: float
    dark_e: float
    expected_e: np.ndarray
    recorded_adu: np.ndarray
    recorded_e: np.ndarray
    saturated: np.ndarray  # clipped high (ADC full scale) in any average
    black_clipped: np.ndarray  # clipped at zero ADU in any average
    dead_time_loss: float  # fraction of counts lost to dead time at the peak
    notes: list[str] = field(default_factory=list)

    # --- quantitation.Measurable --------------------------------------------

    @property
    def pixel_x_um(self) -> float:
        return self.levers.pixel_x_um

    @property
    def pixel_y_um(self) -> float:
        return self.levers.pixel_y_um

    @property
    def pixel_um(self) -> float:
        return self.levers.pixel_y_um

    @property
    def row_centres_um(self) -> np.ndarray:
        return self.ys

    @property
    def col_centres_um(self) -> np.ndarray:
        return self.xs

    @property
    def psf_sigma_um(self) -> float:
        return self.kernel.sigma_um

    @property
    def detected_per_fluor(self) -> float:
        return self.photometry.detected_per_fluor

    @property
    def noise_seed(self) -> int:
        return self.levers.noise_seed

    def rerecord(self, crop, realizations: int, rng: np.random.Generator) -> np.ndarray:
        light = (self.signal_e + self.background_e)[crop]
        shape = (realizations,) + light.shape
        adu, _, _ = acquire_scan(np.broadcast_to(light, shape), self.dark_e, self.levers, rng)
        return adu_to_electrons(adu, self.levers)

    def line_response(self, r0: int, r1: int) -> np.ndarray:
        """Rows of the noise-free image of a thin line of unit brightness.

        The kernel's marginal along x is its line spread function; the scanner
        point-samples it at the line positions, so each row holds the line's
        density there times the line pitch -- a set that sums to ~1.
        """
        spec = self.specimen
        dx = spec.dx_um
        lsf = self.kernel.rate.sum(axis=1)
        lsf = lsf / lsf.sum()
        n = lsf.size
        _, cy = spec.reference.centre_um
        offsets = (np.arange(n) - n // 2) * dx
        return np.interp(self.ys[r0:r1] - cy, offsets, lsf) / dx * self.pixel_y_um

    @property
    def origin_um(self) -> tuple[float, float]:
        return float(self.xs[0] - self.pixel_x_um / 2), float(self.ys[0] - self.pixel_y_um / 2)

    @property
    def extent_um(self) -> tuple[float, float]:
        return self.xs.size * self.pixel_x_um, self.ys.size * self.pixel_y_um


def scan_positions(levers: ScanLevers, field_um: float, margin: float = 0.3):
    """Pixel centres of the frame that fall on the simulated specimen patch."""
    c = field_um / 2

    def axis(count: int, pitch: float) -> np.ndarray:
        k = np.arange(count)
        pos = c + (k - (count - 1) / 2) * pitch
        keep = (pos >= margin) & (pos <= field_um - margin)
        return pos[keep]

    xs = axis(int(levers.pixels_x), levers.pixel_x_um)
    ys = axis(int(levers.pixels_y), levers.pixel_y_um)
    if xs.size < 3 or ys.size < 3:
        raise ValueError(
            f"only {xs.size} x {ys.size} pixels land on the specimen: the pixels "
            f"({levers.pixel_x_um:.2f} x {levers.pixel_y_um:.2f} um) are coarser than the "
            "field can show -- zoom in or use more pixels"
        )
    return xs, ys


def acquire_scan(light_e: np.ndarray, dark_e: float, levers: ScanLevers, rng: np.random.Generator):
    """Record the averages of a scan and average them in ADU (or counts).

    Returns ``(adu, high_clipped, black_clipped)``.
    """
    lv = levers
    det = lv.detector_model
    src = lv.noise
    mode = lv.acquisition_mode
    lam = np.clip(light_e, 0.0, None)
    total = np.zeros(lam.shape)
    high = np.zeros(lam.shape, dtype=bool)
    black = np.zeros(lam.shape, dtype=bool)
    tau = det.dead_time_ns * 1e-9
    for _ in range(max(int(lv.averages), 1)):
        light = lam
        if src.laser_noise and lv.laser_rin > 0:
            light = np.clip(light * (1.0 + lv.laser_rin * rng.standard_normal(lam.shape)), 0.0, None)
        n = rng.poisson(light).astype(float) if src.photon_shot else light.copy()
        if dark_e > 0:
            n = n + rng.poisson(dark_e, lam.shape)
        if src.afterpulsing and det.afterpulse > 0:
            n = n + rng.poisson(det.afterpulse * n)
        if mode == "counting":
            if src.dead_time and tau > 0:
                # Non-paralyzable: each count blinds the detector for tau.
                n = n / (1.0 + n * tau / lv.dwell_s)
                if lv.dead_time_correction:
                    n = n / np.maximum(1.0 - n * tau / lv.dwell_s, 1e-6)
            adu = np.round(n)
        else:
            f2 = det.excess_noise_f2
            if src.gain_noise and f2 > 1.0:
                # Gain cascade: mean n, variance (F^2 - 1) n on top of Poisson.
                out = np.zeros_like(n)
                pos = n > 0
                out[pos] = rng.gamma(n[pos] / (f2 - 1.0), f2 - 1.0)
            else:
                out = n
            adu = out * lv.gain_adu_per_e + lv.offset_adu
            if src.amplifier and det.amplifier_noise_adu > 0:
                adu = adu + det.amplifier_noise_adu * rng.standard_normal(lam.shape)
            if src.quantization:
                adu = np.round(adu)
        if src.clipping:
            high |= adu >= lv.max_adu
            if mode == "analog":
                # Zero counts is a measurement; zero ADU in analog is a clip.
                black |= adu <= 0
            adu = np.clip(adu, 0, lv.max_adu)
        total += adu
    return total / max(int(lv.averages), 1), high, black


def adu_to_electrons(adu: np.ndarray, levers: ScanLevers) -> np.ndarray:
    if levers.acquisition_mode == "counting":
        return np.asarray(adu, dtype=float)
    return (adu - levers.offset_adu) / levers.gain_adu_per_e


class ScanSimulator:
    """Runs :class:`ScanLevers` through the chain, caching specimen and optics."""

    def __init__(self) -> None:
        self._spec_key = None
        self._spec: Specimen | None = None
        self._kernel_key = None
        self._kernel: Kernel | None = None
        self._rate_key = None
        self._rate: np.ndarray | None = None

    def specimen(self, lv: ScanLevers) -> Specimen:
        key = (lv.specimen, lv.density, lv.noise.labeling, lv.specimen_seed)
        if key != self._spec_key:
            self._spec = make_specimen(lv.specimen, lv.density, lv.noise.labeling, lv.specimen_seed)
            self._spec_key = key
            self._rate_key = None
        return self._spec

    def kernel(self, lv: ScanLevers) -> Kernel:
        spec = self.specimen(lv)
        key = (lv.na, lv.excitation_um, lv.wavelength_um, lv.pinhole_au, lv.laser_power_uw, lv.fluorophore)
        if key != self._kernel_key:
            self._kernel = confocal_kernel(spec.density.shape[0], spec.dx_um, lv)
            self._kernel_key = key
            self._rate_key = None
        return self._kernel

    def rate_map(self, lv: ScanLevers) -> np.ndarray:
        """Object (x) kernel: photons/s reaching the detector optics, per position."""
        k = self.kernel(lv)
        if self._rate_key != self._kernel_key or self._rate is None:
            spec = self._spec
            transfer = np.fft.fft2(np.fft.ifftshift(k.rate))
            self._rate = np.clip(np.real(np.fft.ifft2(np.fft.fft2(spec.density) * transfer)), 0.0, None)
            self._rate_key = self._kernel_key
        return self._rate

    def run(self, lv: ScanLevers) -> ScanSimulation:
        optics.check_na(lv.na, lv.n_medium)
        det = lv.detector_model
        spec = self.specimen(lv)
        kern = self.kernel(lv)
        rate = self.rate_map(lv)
        xs, ys = scan_positions(lv, spec.field_um)

        ce = optics.collection_efficiency(lv.na, lv.n_medium)
        detection = ce * lv.transmission * det.quantum_efficiency
        dx2 = spec.dx_um**2
        # Each fluorophore is lit by every pixel the spot passes near: its dose per
        # frame is dwell x (sum of E over pixel positions) ~ dwell x int E dA / pixel
        # area. Finer sampling (higher zoom, more pixels) bleaches faster.
        per_area = lv.dwell_s / (lv.pixel_x_um * lv.pixel_y_um)
        emitted = float(kern.emission.sum() * dx2 * per_area)
        kb = emitted / lv.fluor.photon_budget
        n_avg = max(int(lv.averages), 1)
        if kb > 1e-12:
            per = (1 - math.exp(-kb)) / kb
            survival = per * (1 - math.exp(-kb * n_avg)) / (n_avg * (1 - math.exp(-kb)))
        else:
            survival = 1.0
        detected_per_fluor = float(kern.rate.sum() * dx2 * per_area * detection * survival)
        photometry = ScanPhotometry(detection, ce, det.quantum_efficiency, emitted, survival, detected_per_fluor)

        signal = scan_sample(rate, spec.dx_um, xs, ys, lv.pixel_x_um) * detection * lv.dwell_s * survival
        background = 0.0
        if lv.noise.background:
            # Haze: a uniform sheet of emitters, cut down by the pinhole.
            sheet = float(kern.emission.sum() * dx2) * lv.background_per_um2
            background = sheet * detection * lv.dwell_s * survival * haze_fraction(lv.pinhole_au)
        dark = det.dark_rate(lv.sensor_temperature_c) * lv.dwell_s if lv.noise.dark_counts else 0.0

        expected = signal + background + dark
        rng = np.random.default_rng(lv.noise_seed)
        adu, high, black = acquire_scan(signal + background, dark, lv, rng)
        recorded = adu_to_electrons(adu, lv)

        loss = 0.0
        if lv.acquisition_mode == "counting" and lv.noise.dead_time:
            peak = float(expected.max())
            loss = 1.0 - 1.0 / (1.0 + peak * det.dead_time_ns * 1e-9 / lv.dwell_s)

        notes = []
        if high.any():
            notes.append(f"{high.mean():.1%} of pixels at full scale")
        if black.mean() > 0.01:
            notes.append(f"{black.mean():.0%} of pixels clipped at zero (offset too low)")
        if survival < 0.9:
            notes.append(f"photobleaching: {1 - survival:.0%} of the signal lost on average")
        if kern.saturation > 0.1:
            notes.append(f"excitation saturated at the focus (k tau = {kern.saturation:.2f})")
        return ScanSimulation(
            lv, spec, kern, photometry, xs, ys, signal, background, dark, expected, adu, recorded,
            high, black, loss, notes,
        )


# --- the analytic budget ---------------------------------------------------------


@dataclass(frozen=True)
class ScanBudget:
    """Variance per pixel in photoelectrons^2, after averaging -- the graph's terms."""

    signal_e: float
    photon_shot: float
    background_shot: float
    dark_shot: float
    read: float  # amplifier noise, referred to the input
    quantization: float
    laser: float
    frames: int

    TERMS = ("photon_shot", "background_shot", "dark_shot", "read", "quantization", "laser")
    LABELS = {
        "photon_shot": "photon shot",
        "background_shot": "haze shot",
        "dark_shot": "dark-count (thermal) shot",
        "read": "amplifier",
        "quantization": "quantization",
        "laser": "laser intensity",
    }

    @property
    def total(self) -> float:
        return sum(getattr(self, t) for t in self.TERMS)

    @property
    def noise_e(self) -> float:
        return math.sqrt(self.total)

    @property
    def snr(self) -> float:
        return self.signal_e / self.noise_e if self.total > 0 else math.inf

    @property
    def shot_limited_snr(self) -> float:
        return math.sqrt(max(self.signal_e, 0.0) * self.frames)

    def fractions(self) -> dict[str, float]:
        total = self.total
        return {t: (getattr(self, t) / total if total else 0.0) for t in self.TERMS}

    def dominant(self) -> str:
        return max(self.TERMS, key=lambda t: getattr(self, t))


def scan_budget(lv: ScanLevers, signal_e: float, background_e: float, dark_e: float) -> ScanBudget:
    det = lv.detector_model
    src = lv.noise
    n = max(int(lv.averages), 1)
    counting = lv.acquisition_mode == "counting"
    f2 = 1.0 if counting or not src.gain_noise else det.excess_noise_f2
    f2 *= 1.0 + (2 * det.afterpulse if src.afterpulsing else 0.0)
    shot = 1.0 if src.photon_shot else 0.0
    g = lv.gain_adu_per_e
    amp = 0.0 if counting or not src.amplifier else (det.amplifier_noise_adu / g) ** 2
    quant = 0.0 if counting or not src.quantization else (1.0 / g) ** 2 / 12.0
    laser = (lv.laser_rin * (signal_e + background_e)) ** 2 if src.laser_noise else 0.0
    return ScanBudget(
        signal_e,
        f2 * signal_e * shot / n,
        f2 * background_e * shot / n,
        f2 * dark_e / n,
        amp / n,
        quant / n,
        laser / n,
        n,
    )


def peak_signal(sim: ScanSimulation) -> float:
    from .quantitation import reference_window

    r0, r1, c0, c1 = reference_window(sim)
    region = sim.signal_e[r0:r1, c0:c1]
    if region.size == 0:
        return float(sim.signal_e.max())
    return float(region.mean(axis=1).max()) if sim.specimen.kind.kind == "filament" else float(region.max())


def simulation_budget(sim: ScanSimulation, signal_e: float | None = None) -> ScanBudget:
    s = peak_signal(sim) if signal_e is None else signal_e
    return scan_budget(sim.levers, s, sim.background_e, sim.dark_e)


SWEEPS = {
    "signal": "signal per pixel (e-)",
    "dwell": "pixel dwell (µs)",
    "laser": "laser power (µW)",
}


def budget_sweep(sim: ScanSimulation, mode: str, values) -> list[ScanBudget]:
    """Budgets as one lever is swept, others held.

    ``dwell`` scales signal, haze and dark counts together. ``laser`` scales
    signal and haze by how much more the fluorophore emits at the focus -- which
    flattens as excitation saturates -- and leaves dark counts alone.
    """
    lv = sim.levers
    s0 = peak_signal(sim)
    out = []
    fl = lv.fluor
    i0 = sim.kernel.peak_irradiance_w_cm2
    e0 = float(fl.emission_rate(i0)) or 1e-30
    for v in values:
        v = float(v)
        if mode == "signal":
            out.append(scan_budget(lv, v, sim.background_e, sim.dark_e))
        elif mode == "dwell":
            k = v / lv.dwell_us
            out.append(scan_budget(lv, s0 * k, sim.background_e * k, sim.dark_e * k))
        elif mode == "laser":
            k = float(fl.emission_rate(i0 * v / lv.laser_power_uw)) / e0
            out.append(scan_budget(lv, s0 * k, sim.background_e * k, sim.dark_e))
        else:
            raise ValueError(f"unknown sweep {mode!r}")
    return out


def operating_value(sim: ScanSimulation, mode: str) -> float:
    return {"signal": peak_signal(sim), "dwell": sim.levers.dwell_us, "laser": sim.levers.laser_power_uw}[mode]


def sweep_axis(mode: str) -> tuple[np.ndarray, str, str]:
    if mode == "signal":
        return np.logspace(-2, 4, 160), "signal per pixel (e-)", "e-"
    if mode == "dwell":
        return np.logspace(-1.3, 3, 160), "pixel dwell (µs)", "µs"
    return np.logspace(-2, 3.3, 160), "laser power (µW)", "µW"
