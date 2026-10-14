"""The explained numbers: every derived quantity, with its equation and your values.

This is the widget's scorecard. Each row is something a demonstrator would point
at -- "you are under-sampling", "you are read-noise limited", "you have clipped" --
and each carries the arithmetic so the user can redo it on paper.
"""

from __future__ import annotations

import math

from . import optics
from .acquisition import Simulation, peak_signal, simulation_budget
from .base import Report, Row, Status
from .camera import describe_temperature, effective_full_well, excess_noise_factor_squared
from .quantitation import Quantitation


def _fmt(v: float, unit: str = "", digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    return f"{v:.{digits}g}{(' ' + unit) if unit else ''}"


def optics_rows(sim: Simulation) -> list[Row]:
    lv = sim.levers
    lam, na, n = lv.wavelength_um, lv.na, lv.n_medium
    rows = []
    rows.append(Row(
        "lateral resolution (Rayleigh)",
        f"{optics.rayleigh_um(lam, na) * 1000:.0f} nm",
        f"d = 0.61 lambda / NA = 0.61 x {lam * 1000:.0f} nm / {na:.2f}",
    ))
    rows.append(Row(
        "PSF FWHM",
        f"{optics.fwhm_um(lam, na) * 1000:.0f} nm   (Gaussian sigma {optics.gaussian_sigma_um(lam, na) * 1000:.0f} nm)",
        "FWHM = 0.514 lambda / NA;  sigma = 0.21 lambda / NA (Zhang et al. 2007)"
        + ("; scalar model -- a real NA > 1.2 PSF is a few % wider" if na > 1.2 else ""),
    ))
    nyq = optics.nyquist_pixel_um(lam, na)
    p = sim.pixel_um
    ratio = p / nyq
    if ratio > 1.05:
        status, verdict = Status.WARN if ratio < 2 else Status.FAIL, "under-sampled: the camera discards detail the optics resolved"
        remedy = "raise the magnification, use smaller pixels or bin less"
    elif ratio < 0.5:
        status, verdict = Status.WARN, "over-sampled: photons are spread over more pixels, so each one carries more read noise"
        remedy = "bin, or use a lower magnification -- you gain signal per pixel and lose no resolution"
    else:
        status, verdict, remedy = Status.PASS, "sampled at Nyquist", ""
    rows.append(Row(
        "sampling (Nyquist)",
        f"pixel {p * 1000:.0f} nm at the sample against {nyq * 1000:.0f} nm needed -- {verdict}",
        f"p = {lv.camera_model.pixel_um:g} um x {lv.binning} bin / {lv.magnification:g}x;  "
        f"Nyquist p <= lambda / 4NA = {nyq * 1000:.0f} nm (bench rule Rayleigh/2.3 = "
        f"{optics.rayleigh_um(lam, na) / 2.3 * 1000:.0f} nm)",
        status, remedy,
    ))
    ce = sim.photometry.collection
    rows.append(Row(
        "light collection",
        f"{ce:.1%} of emitted photons enter the objective",
        f"(1 - cos theta) / 2,  theta = asin(NA / n) = asin({na:.2f} / {n:.3f}) = "
        f"{math.degrees(math.asin(min(na / n, 1))):.1f} deg",
    ))
    rows.append(Row(
        "depth of field",
        f"{optics.depth_of_field_um(lam, na, n):.2f} um",
        f"lambda n / NA^2 = {lam * 1000:.0f} nm x {n:.3f} / {na:.2f}^2 (not imaged: the model is one focal plane)",
    ))
    return rows


def photon_rows(sim: Simulation) -> list[Row]:
    lv = sim.levers
    ph = sim.photometry
    fl = lv.fluor
    rows = [Row(
        "emission per fluorophore",
        f"{ph.emission_rate:.3g} photons/s  ({ph.emitted:.3g} per {lv.exposure_ms:g} ms frame)",
        f"QY sigma I / h nu = {fl.quantum_yield} x {fl.cross_section_cm2:.2e} cm^2 x "
        f"{lv.irradiance_w_cm2:g} W/cm^2 / {fl.photon_energy_j():.2e} J  ({fl.name}, {fl.excitation_nm:g} nm)",
    )]
    rows.append(Row(
        "detected per fluorophore",
        f"{ph.detected_e:.3g} e- per frame",
        f"{ph.emitted:.3g} emitted x {ph.collection:.3f} collected x {ph.transmission:.2f} "
        f"transmitted x {ph.qe:.2f} QE" + (f" x {ph.bleach_survival:.2f} unbleached" if ph.bleach_survival < 0.999 else ""),
    ))
    lost = 1 - ph.bleach_survival
    rows.append(Row(
        "photobleaching",
        f"{lost:.1%} of the signal lost, averaged over {lv.frames} frame(s)",
        f"k_b = k_em / budget = {ph.emission_rate:.3g} / {fl.photon_budget:.0e} per s "
        "(budgets are order-of-magnitude)",
        Status.WARN if lost > 0.2 else Status.PASS if lost < 0.05 else Status.INFO,
        "bleaching tracks the photons emitted: lower the irradiance, shorten the exposure "
        "or average fewer frames, and spend the photons where they count",
    ))
    return rows


def noise_rows(sim: Simulation) -> list[Row]:
    lv = sim.levers
    cam = lv.camera_model
    S = peak_signal(sim)
    b = simulation_budget(sim, S)
    f2 = excess_noise_factor_squared(cam, lv.gain) if lv.noise.em_excess else 1.0
    rows = [Row(
        "signal at the structure",
        f"{S:.3g} e- peak, on {sim.background_e:.3g} e- background, {sim.dark_e:.3g} e- dark",
        f"background = {lv.background_per_um2:g} /um^2 x ({sim.pixel_um:.3f} um)^2 x {sim.photometry.detected_e:.3g} e-;  "
        + describe_temperature(cam, lv.sensor_temperature_c) + f" x {lv.exposure_ms:g} ms",
    )]
    read = cam.read_noise(lv.gain)
    read_in = math.sqrt(b.read * b.frames)
    rows.append(Row(
        "read noise, referred to the input",
        f"{read_in:.3g} e- rms per (super)pixel per frame",
        f"{read:g} e- / EM gain {lv.gain:g}"
        + ("" if cam.charge_binning or lv.binning == 1 else f" x sqrt({lv.binning}^2) (digital binning reads every pixel)"),
    ))
    shot_var = b.photon_shot + b.background_shot + b.dark_shot + b.cic
    if shot_var > 3 * b.read:
        regime, status = "shot-noise limited: only more photons help", Status.PASS
        remedy = ""
    elif b.read > 3 * shot_var:
        regime, status = "read-noise limited: the camera, not the light, sets the noise", Status.WARN
        remedy = "bin, raise exposure or irradiance, or use EM gain / a lower-read-noise camera"
    else:
        regime, status, remedy = "between regimes: read and shot noise are comparable", Status.INFO, ""
    rows.append(Row(
        "noise regime",
        regime,
        f"shot variance {shot_var:.3g} e-^2 (F^2 = {f2:g}) against read variance {b.read:.3g} e-^2"
        f"; crossover at S = sigma_read^2 / F^2 = {read_in ** 2 / f2:.3g} e-",
        status, remedy,
    ))
    snr = b.snr
    rows.append(Row(
        "SNR at the structure",
        f"{snr:.3g}  (a perfect detector: {b.shot_limited_snr:.3g})",
        f"S / sqrt(F^2 (S + B + D + CIC) + sigma_r^2 + sigma_q^2 + (PRNU S)^2), averaged over {b.frames} frame(s);  "
        f"dominant term: {b.LABELS[b.dominant()]}",
        Status.FAIL if snr < 3 else Status.WARN if snr < 5 else Status.PASS,
        "below the Rose criterion (SNR 5) a feature cannot be reliably detected",
    ))
    well = effective_full_well(cam, lv.binning)
    peak_e = float(sim.expected_e.max())
    sat_frac = float(sim.saturated.mean())
    adc_e = (cam.max_adu - cam.offset_adu) * cam.adc_gain_e_per_adu / lv.gain
    limit = min(well, adc_e)
    rows.append(Row(
        "headroom",
        (f"{sat_frac:.2%} of pixels clipped" if sat_frac > 0 else f"brightest pixel at {peak_e / limit:.0%} of the clip level"),
        f"clip at min(full well {well:.3g} e-, ADC {cam.bit_depth}-bit = {adc_e:.3g} e- at the input)",
        Status.FAIL if sat_frac > 0.001 else Status.WARN if sat_frac > 0 else Status.PASS,
        "clipped pixels destroy intensity measurements: shorten the exposure, lower EM gain or irradiance",
    ))
    rows.append(Row(
        "dynamic range",
        f"{limit / max(read_in, 1e-9):.3g} : 1  ({math.log2(max(limit / max(read_in, 1e-9), 1)):.1f} stops)",
        "clip level / read noise",
    ))
    return rows


def quant_rows(q: Quantitation) -> list[Row]:
    units = "per um" if q.kind == "filament" else "per bead"
    rows = []
    rows.append(Row(
        "true size",
        f"{q.true_diameter_nm:g} nm structure; labels span {q.label_diameter_nm:g} nm",
        f"label positions across the axis have sigma {q.label_sigma_nm:.1f} nm",
    ))
    rows.append(Row(
        "expected apparent width",
        f"{q.predicted_fwhm_um * 1000:.0f} nm FWHM",
        f"2.355 sqrt(sigma_instrument^2 + sigma_label^2) = 2.355 sqrt({q.instrument_sigma_um * 1000:.1f}^2 + "
        f"{q.label_sigma_nm:.1f}^2) nm;  sigma_instrument is this system's fit to a thin line "
        f"(PSF + sampling; the PSF alone fits sigma ~ {q.psf_sigma_um * 1000:.0f} nm)",
    ))
    ok = q.success > 0
    bias = abs(q.fwhm.mean / q.ideal.fwhm_um - 1) if ok and q.ideal.ok else 0.0
    spread = q.fwhm.cv if ok else math.inf
    if not ok or spread >= 0.15 or bias >= 0.10:
        width_status = Status.FAIL
    elif spread >= 0.05 or bias >= 0.03:
        width_status = Status.WARN
    else:
        width_status = Status.PASS
    rows.append(Row(
        "measured width (FWHM)",
        (f"{q.fit.fwhm_um * 1000:.0f} nm this frame;  {q.fwhm.mean * 1000:.0f} +/- {q.fwhm.sd * 1000:.0f} nm "
         f"over {q.realizations} frames" if ok else "no frame could be fitted"),
        f"noise-free fit {q.ideal.fwhm_um * 1000:.0f} nm: the optics' bias; the +/- is the noise's"
        + (f"; the mean is {bias:.0%} off the noise-free value" if bias >= 0.03 else ""),
        width_status,
        "too uncertain, or biased (clipping, hot pixels), to compare structures: collect more "
        "photons, average, avoid saturation, or calibrate",
    ))
    n_real = q.corrected_width.values.size
    rows.append(Row(
        "structure width, PSF subtracted",
        (f"{q.corrected_width.mean * 1000:.0f} +/- {q.corrected_width.sd * 1000:.0f} nm "
         f"(real in {n_real}/{q.realizations} frames)" if n_real > 1 else "undefined: measured width below the PSF"),
        f"2.355 sqrt(sigma_fit^2 - sigma_instrument^2) against a true label width of "
        f"~{2.355 * q.label_sigma_nm:.0f} nm FWHM: below the diffraction limit, noise swamps it",
        Status.WARN,
    ))
    rows.append(Row(
        f"label count ({units})",
        (f"fit {q.single_count:.0f} this frame;  {q.count.mean:.0f} +/- {q.count.sd:.0f} (CV {q.count.cv:.0%});  "
         f"sum {q.summed.mean:.0f} +/- {q.summed.sd:.0f}" if ok else "--"),
        f"{q.labels_present:.0f} actually present ({q.nominal_density:g} nominal);  "
        f"count = area / ({q.detected_per_fluor:.3g} e- per fluorophore x recovery);  "
        f"recovery {q.fit_recovery:.2f} (fit), {q.sum_recovery:.2f} (sum): the Airy rings a "
        "Gaussian misses, calibrated out",
        Status.PASS if ok and abs(q.count.mean / max(q.labels_present, 1e-9) - 1) < 0.1 and q.count.cv < 0.1
        else Status.WARN,
        "a count this far off, or this uncertain, cannot report stoichiometry: more photons, "
        "fewer clipped pixels, or calibrate the fixed pattern",
    ))
    # Photoelectrons from the structure inside the measured window.
    span = q.window_length_um if q.kind == "filament" else 1.0
    photons = max(q.labels_present * span * q.detected_per_fluor, 1e-9)
    floor = math.hypot(q.psf_sigma_um, q.pixel_um / math.sqrt(12)) / math.sqrt(photons)
    rows.append(Row(
        "position precision",
        f"{q.centre.sd * 1000:.1f} nm rms" if q.centre.values.size > 1 else "--",
        f"scatter of the fitted centre; shot-noise floor sqrt(sigma^2 + p^2/12) / sqrt(N) = "
        f"{floor * 1000:.1f} nm for N = {photons:.3g} e-; background and read noise add to it",
    ))
    rows.append(Row(
        "peak SNR of the profile",
        f"{q.peak_snr.mean:.3g}" if q.peak_snr.values.size else "--",
        "fitted amplitude / rms deviation from the noise-free profile",
    ))
    return rows


def build_report(sim: Simulation, q: Quantitation | None = None) -> dict[str, Report]:
    sections = {
        "optics": Report(optics_rows(sim)),
        "photons": Report(photon_rows(sim)),
        "noise": Report(noise_rows(sim)),
    }
    if q is not None:
        sections["quantitation"] = Report(quant_rows(q))
    return sections


def format_report(sections: dict[str, Report]) -> str:
    out = []
    for name, rep in sections.items():
        out.append(f"-- {name} --")
        out.append(rep.format())
        out.append("")
    return "\n".join(out).rstrip()
