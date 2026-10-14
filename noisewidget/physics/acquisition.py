"""From levers to a recorded image, and the noise budget that explains it.

The chain, in the order light meets it::

    fluorophores --(excitation, exposure)--> photons emitted
                 --(NA, n: collection)----> photons collected
                 --(transmission)---------> photons at the sensor
                 --(lambda, NA: the PSF)--> blurred
                 --(pixel / M, binning)---> integrated over camera pixels
                 --(QE)-------------------> photoelectrons, expected
    + background, dark current, clock-induced charge
                 --(Poisson)--------------> shot noise: photon and thermal
                 --(EM register)----------> excess noise
                 --(full well)------------> clipping
                 --(amplifier)------------> read noise
                 --(ADC)------------------> quantization, offset, bit depth

Every random stage can be switched off on its own (:class:`NoiseSources`), so a
user can remove one noise source at a time and see which one was hurting.

Two clocks, as in MicroscopeVuilder's workspace: the optics (an FFT of the 512^2
object) are cached and recomputed only when NA, wavelength, magnification, pixel
size or binning change; everything after the PSF is cheap and reruns on every lever.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from . import optics
from .camera import (
    CAMERAS,
    DEFAULT_CAMERA,
    Camera,
    FixedPattern,
    effective_full_well,
    excess_noise_factor_squared,
    fixed_pattern,
    quantization_noise_e,
)
from .fluorophores import FLUOROPHORES
from .specimens import DEFAULT_SPECIMEN, SPECIMENS, Specimen, make_specimen


@dataclass(frozen=True)
class NoiseSources:
    """Which noise sources are simulated. All on is a real camera."""

    labeling: bool = True  # stochastic number and placement of labels
    photon_shot: bool = True
    background: bool = True  # the background light itself, and its shot noise
    dark_shot: bool = True  # thermal: dark current and its shot noise
    em_excess: bool = True
    cic: bool = True
    read: bool = True
    fixed_pattern: bool = True  # PRNU, DSNU, hot pixels, per-pixel read noise
    quantization: bool = True
    saturation: bool = True

    LABELS = {
        "labeling": "labelling stochasticity",
        "photon_shot": "photon shot noise",
        "background": "background light (and its shot noise)",
        "dark_shot": "dark current (thermal shot noise)",
        "em_excess": "EM excess noise (F^2 = 2)",
        "cic": "clock-induced charge",
        "read": "read noise",
        "fixed_pattern": "fixed pattern (PRNU, DSNU, hot pixels)",
        "quantization": "ADC quantization",
        "saturation": "full-well / ADC clipping",
    }


@dataclass(frozen=True)
class Levers:
    """Everything a user can turn. Defaults are a sensible widefield setup."""

    # specimen
    specimen: str = DEFAULT_SPECIMEN
    label_density: float | None = None  # None: the specimen's default
    fluorophore: str = "EGFP"
    specimen_seed: int = 1
    # optics
    na: float = 1.40
    medium: str = "oil"
    magnification: float = 100.0
    wavelength_nm: float | None = None  # None: the fluorophore's emission peak
    transmission: float = 0.40  # objective, filters, tube lens, mirrors
    # illumination
    irradiance_w_cm2: float = 20.0
    background_per_um2: float = 40.0  # fluorophore-equivalents per um^2 (autofluorescence, out of focus)
    # camera and acquisition
    camera: str = DEFAULT_CAMERA
    exposure_ms: float = 100.0
    binning: int = 1
    temperature_c: float | None = None  # None: the camera's default
    em_gain: float = 1.0
    frames: int = 1  # frames averaged
    dark_subtract: bool = False  # subtract a master dark frame
    flat_field: bool = False  # divide by a master flat (removes PRNU)
    # noise
    noise: NoiseSources = field(default_factory=NoiseSources)
    noise_seed: int = 0

    # --- resolved values ----------------------------------------------------

    @property
    def camera_model(self) -> Camera:
        return CAMERAS[self.camera]

    @property
    def fluor(self):
        return FLUOROPHORES[self.fluorophore]

    @property
    def wavelength_um(self) -> float:
        nm = self.wavelength_nm if self.wavelength_nm is not None else self.fluor.emission_nm
        return nm / 1000.0

    @property
    def n_medium(self) -> float:
        return optics.IMMERSION_MEDIA[self.medium]

    @property
    def density(self) -> float:
        return SPECIMENS[self.specimen].default_density if self.label_density is None else self.label_density

    @property
    def sensor_temperature_c(self) -> float:
        cam = self.camera_model
        t = cam.default_temperature_c if self.temperature_c is None else self.temperature_c
        return max(cam.min_temperature_c, t)

    @property
    def gain(self) -> float:
        cam = self.camera_model
        return min(max(self.em_gain, 1.0), cam.max_em_gain) if cam.has_em_gain else 1.0

    @property
    def pixel_um(self) -> float:
        """Camera (super)pixel referred to the sample."""
        return self.camera_model.pixel_um * max(self.binning, 1) / self.magnification

    def with_(self, **changes) -> "Levers":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


@dataclass(frozen=True)
class Photometry:
    """Photon accounting for one fluorophore in one frame."""

    emission_rate: float  # photons/s
    collection: float
    transmission: float
    qe: float
    exposure_s: float
    bleach_survival: float  # mean surviving fraction over the acquisition

    @property
    def emitted(self) -> float:
        return self.emission_rate * self.exposure_s

    @property
    def detected_e(self) -> float:
        """Photoelectrons per fluorophore per frame, averaged over bleaching."""
        return self.emitted * self.collection * self.transmission * self.qe * self.bleach_survival


def photometry(levers: Levers) -> Photometry:
    fl = levers.fluor
    rate = fl.emission_rate(levers.irradiance_w_cm2)
    t = levers.exposure_ms / 1000.0
    # Survival after k frames is exp(-k_b t k); the averaged image sees the mean.
    kb = fl.bleach_rate(levers.irradiance_w_cm2) * t
    frames = max(levers.frames, 1)
    if kb > 0:
        # Mean over frames of the fraction alive through each exposure, exactly:
        # each frame integrates exp(-k_b t) over its own exposure window.
        per_frame = (1 - math.exp(-kb)) / kb
        survival = per_frame * (1 - math.exp(-kb * frames)) / (frames * (1 - math.exp(-kb)))
    else:
        survival = 1.0
    return Photometry(
        rate,
        optics.collection_efficiency(levers.na, levers.n_medium),
        levers.transmission,
        levers.camera_model.quantum_efficiency,
        t,
        survival,
    )


@dataclass
class Simulation:
    levers: Levers
    specimen: Specimen
    photometry: Photometry
    pixel_um: float
    blurred_fine: np.ndarray  # fluorophores per fine px, after the PSF
    signal_e: np.ndarray  # expected photoelectrons per camera px from the specimen
    background_e: float
    dark_e: float  # mean, per px per frame
    cic_e: float
    pattern: FixedPattern
    expected_e: np.ndarray  # signal + background + dark + cic, fixed pattern, after calibration
    recorded_adu: np.ndarray  # mean of the averaged frames, in ADU
    recorded_e: np.ndarray  # recorded image referred back to input electrons
    saturated: np.ndarray  # bool mask: any frame clipped at this pixel
    notes: list[str] = field(default_factory=list)

    @property
    def camera(self) -> Camera:
        return self.levers.camera_model


class Simulator:
    """Runs :func:`Levers` through the chain, caching the expensive optics."""

    def __init__(self) -> None:
        self._specimen_key = None
        self._specimen: Specimen | None = None
        self._optics_key = None
        self._blurred: np.ndarray | None = None
        self._binned_key = None
        self._binned: np.ndarray | None = None
        self._pattern_key = None
        self._pattern: FixedPattern | None = None

    def specimen(self, levers: Levers) -> Specimen:
        key = (levers.specimen, levers.density, levers.noise.labeling, levers.specimen_seed)
        if key != self._specimen_key:
            self._specimen = make_specimen(
                levers.specimen, levers.density, levers.noise.labeling, levers.specimen_seed
            )
            self._specimen_key = key
            self._optics_key = None
        return self._specimen

    def blurred(self, levers: Levers) -> np.ndarray:
        spec = self.specimen(levers)
        key = (levers.na, levers.wavelength_um)
        if key != self._optics_key or self._blurred is None:
            self._blurred = optics.blur(spec.density, spec.dx_um, levers.wavelength_um, levers.na)
            self._optics_key = key
            self._binned_key = None
        return self._blurred

    def binned(self, levers: Levers) -> np.ndarray:
        blurred = self.blurred(levers)
        key = (levers.pixel_um,)
        if key != self._binned_key or self._binned is None:
            self._binned = optics.box_integrate(blurred, self._specimen.dx_um, levers.pixel_um)
            self._binned_key = key
        return self._binned

    def pattern(self, levers: Levers, shape) -> FixedPattern:
        key = (levers.camera, shape, levers.binning, levers.gain > 1.0, levers.noise.fixed_pattern)
        if key != self._pattern_key or self._pattern is None:
            self._pattern = fixed_pattern(
                levers.camera_model, shape, levers.binning, levers.gain,
                enabled=levers.noise.fixed_pattern,
            )
            self._pattern_key = key
        return self._pattern

    def run(self, levers: Levers) -> Simulation:
        optics.check_na(levers.na, levers.n_medium)
        cam = levers.camera_model
        ph = photometry(levers)
        fluor_per_px = self.binned(levers)
        spec = self._specimen
        pattern = self.pattern(levers, fluor_per_px.shape)

        signal = fluor_per_px * ph.detected_e
        p2 = levers.pixel_um**2
        background = levers.background_per_um2 * p2 * ph.detected_e if levers.noise.background else 0.0
        b2 = max(levers.binning, 1) ** 2
        dark = cam.dark_current(levers.sensor_temperature_c) * ph.exposure_s * b2 if levers.noise.dark_shot else 0.0
        cic = (cam.cic_e * (1 if cam.charge_binning else b2)) if (cam.has_em_gain and levers.noise.cic) else 0.0

        light = (signal + background) * pattern.gain
        thermal = dark * pattern.dark_scale + cic
        expected = light + thermal
        rng = np.random.default_rng(levers.noise_seed)
        adu, sat = acquire(light, thermal, levers, pattern, rng, max(levers.frames, 1))
        recorded_e = calibrate(adu_to_electrons(adu, levers), levers, pattern, thermal)
        expected = calibrate(expected, levers, pattern, thermal)

        notes = []
        if sat.any():
            notes.append(f"{sat.mean():.1%} of pixels clipped (full well or ADC)")
        if ph.bleach_survival < 0.9:
            notes.append(f"photobleaching: {1 - ph.bleach_survival:.0%} of the signal lost on average")
        return Simulation(
            levers, spec, ph, levers.pixel_um, self._blurred, signal, float(background),
            float(dark), float(cic), pattern, expected, adu, recorded_e, sat, notes,
        )


def acquire(
    light_e: np.ndarray, thermal_e: np.ndarray | float, levers: Levers, pattern: FixedPattern,
    rng: np.random.Generator, frames: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Record ``frames`` frames and average them in ADU.

    ``light_e`` is the expected photoelectrons from light (specimen plus
    background), ``thermal_e`` those generated in the dark (dark current plus
    clock-induced charge). They are separate so photon shot noise can be switched
    off on its own while thermal charge stays honest Poisson.

    Works on any array shape whose fixed-pattern maps match, which is what lets
    the quantitation run hundreds of realizations of just the region it measures.
    Returns the averaged ADU image and a mask of pixels that clipped in any frame.
    """
    cam = levers.camera_model
    src = levers.noise
    gain = levers.gain
    well = effective_full_well(cam, levers.binning)
    lam = np.clip(light_e, 0.0, None)
    therm = np.clip(np.broadcast_to(thermal_e, lam.shape), 0.0, None)
    total = np.zeros(lam.shape)
    clipped = np.zeros(lam.shape, dtype=bool)

    for _ in range(frames):
        n = rng.poisson(lam).astype(float) if src.photon_shot else lam.copy()
        n = n + rng.poisson(therm)
        if src.saturation:
            clipped |= n >= well
            n = np.minimum(n, well)
        if gain > 1.0:
            if src.em_excess:
                # Each electron through the register comes out as a geometric-ish
                # cascade; the sum of n of them is Gamma(n, G). n = 0 stays 0.
                out = np.zeros_like(n)
                pos = n > 0
                out[pos] = rng.gamma(n[pos], gain)
            else:
                out = n * gain
            if src.saturation and cam.register_well_e > 0:
                clipped |= out >= cam.register_well_e
                out = np.minimum(out, cam.register_well_e)
        else:
            out = n
        if src.read:
            out = out + rng.standard_normal(lam.shape) * pattern.read_noise_e
        adu = out / cam.adc_gain_e_per_adu + cam.offset_adu
        if src.quantization:
            adu = np.round(adu)
        if src.saturation:
            clipped |= adu >= cam.max_adu
            adu = np.clip(adu, 0, cam.max_adu)
        total += adu
    return total / frames, clipped


def adu_to_electrons(adu: np.ndarray, levers: Levers) -> np.ndarray:
    """Undo offset, ADC gain and EM gain: the image as photoelectrons at the input."""
    cam = levers.camera_model
    return (adu - cam.offset_adu) * cam.adc_gain_e_per_adu / levers.gain


def calibrate(
    electrons: np.ndarray, levers: Levers, pattern: FixedPattern, thermal_e: np.ndarray | float,
) -> np.ndarray:
    """Apply the corrections a careful user would: dark subtraction, flat field.

    The master frames are taken as noise-free (averages of many frames), so a dark
    subtraction removes the *mean* of dark current, DSNU and hot pixels but none
    of their shot noise; a flat field removes PRNU. Neither touches random noise,
    which is the lesson: calibration fixes patterns, only photons fix noise.
    """
    out = electrons
    if levers.dark_subtract:
        out = out - thermal_e
    if levers.flat_field:
        out = out / pattern.gain
    return out


# --- the analytic budget ---------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """Variance per pixel, in input electrons squared, for one signal level.

    These are the terms the live graph plots. Random terms divide by the number of
    frames averaged; fixed-pattern terms do not, which is the whole point of
    showing them separately.
    """

    signal_e: float
    photon_shot: float
    background_shot: float
    dark_shot: float
    cic: float
    read: float
    quantization: float
    prnu: float
    frames: int

    TERMS = ("photon_shot", "background_shot", "dark_shot", "cic", "read", "quantization", "prnu")
    LABELS = {
        "photon_shot": "photon shot",
        "background_shot": "background shot",
        "dark_shot": "dark (thermal) shot",
        "cic": "clock-induced charge",
        "read": "read",
        "quantization": "quantization",
        "prnu": "PRNU (fixed pattern)",
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
        """The best any detector could do: ``sqrt(S)`` per frame, ``sqrt(N S)`` averaged."""
        return math.sqrt(max(self.signal_e, 0.0) * self.frames)

    def fractions(self) -> dict[str, float]:
        total = self.total
        return {t: (getattr(self, t) / total if total else 0.0) for t in self.TERMS}

    def dominant(self) -> str:
        return max(self.TERMS, key=lambda t: getattr(self, t))


def budget(levers: Levers, signal_e: float, background_e: float, dark_e: float, cic_e: float) -> Budget:
    """Noise budget for a pixel holding ``signal_e`` expected photoelectrons."""
    cam = levers.camera_model
    src = levers.noise
    f2 = excess_noise_factor_squared(cam, levers.gain) if src.em_excess else 1.0
    frames = max(levers.frames, 1)
    b2 = max(levers.binning, 1) ** 2
    read = cam.read_noise(levers.gain)
    if not cam.charge_binning:
        read *= math.sqrt(b2)
    read_var = (read / levers.gain) ** 2 if src.read else 0.0
    q = quantization_noise_e(cam, levers.gain) ** 2 if src.quantization else 0.0
    prnu = (
        (cam.prnu / math.sqrt(b2)) ** 2 * (signal_e + background_e) ** 2
        if src.fixed_pattern and not levers.flat_field else 0.0
    )
    shot = 1.0 if src.photon_shot else 0.0
    return Budget(
        signal_e,
        f2 * signal_e * shot / frames,
        f2 * background_e * shot / frames,
        f2 * dark_e / frames if src.dark_shot else 0.0,
        f2 * cic_e / frames,
        read_var / frames,
        q / frames,
        prnu,
        frames,
    )


def simulation_budget(sim: Simulation, signal_e: float | None = None) -> Budget:
    s = peak_signal(sim) if signal_e is None else signal_e
    return budget(sim.levers, s, sim.background_e, sim.dark_e, sim.cic_e)


def peak_signal(sim: Simulation) -> float:
    """The specimen signal at the brightest point of the reference structure.

    Taken from the noise-free image so the operating point on the graph does not
    jitter with each noise realization.
    """
    from .quantitation import reference_window

    r0, r1, c0, c1 = reference_window(sim)
    region = sim.signal_e[r0:r1, c0:c1]
    if region.size == 0:
        return float(sim.signal_e.max())
    return float(region.mean(axis=1).max()) if sim.specimen.kind.kind == "filament" else float(region.max())


SWEEPS = {
    "signal": "signal per pixel (e-)",
    "exposure": "exposure (ms)",
    "excitation": "excitation (W/cm²)",
}


def budget_sweep(sim: Simulation, mode: str, values) -> list[Budget]:
    """Noise budgets as one lever is swept, everything else held where it is.

    ``signal`` sweeps the signal alone, holding background and dark fixed -- the
    classic photon-transfer view. ``exposure`` scales signal, background and dark
    current together, because all three integrate over time. ``excitation`` scales
    signal and background but not dark current, which does not care about light.
    Bleaching is held at the operating point's value, so far from it the curves
    are the optimistic ones.
    """
    lv = sim.levers
    s0 = peak_signal(sim)
    out = []
    for v in values:
        if mode == "signal":
            out.append(budget(lv, float(v), sim.background_e, sim.dark_e, sim.cic_e))
            continue
        if mode == "exposure":
            k = float(v) / lv.exposure_ms
            out.append(budget(lv, s0 * k, sim.background_e * k, sim.dark_e * k, sim.cic_e))
        elif mode == "excitation":
            k = float(v) / max(lv.irradiance_w_cm2, 1e-12)
            out.append(budget(lv, s0 * k, sim.background_e * k, sim.dark_e, sim.cic_e))
        else:
            raise ValueError(f"unknown sweep {mode!r}")
    return out


def operating_value(sim: Simulation, mode: str) -> float:
    """Where the current settings sit on a sweep's x axis."""
    return {
        "signal": peak_signal(sim),
        "exposure": sim.levers.exposure_ms,
        "excitation": sim.levers.irradiance_w_cm2,
    }[mode]
