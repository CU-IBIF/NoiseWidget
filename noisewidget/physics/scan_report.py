"""The explained numbers for the point-scanning confocal.

Same contract as :mod:`report`: every row carries its equation with the user's
numbers in it. The quantitation rows are shared, because a measurement is a
measurement whichever instrument made the image.
"""

from __future__ import annotations

import math

from . import optics
from .base import Report, Row, Status
from .confocal import ScanSimulation, haze_fraction, peak_signal, simulation_budget
from .quantitation import Quantitation
from .report import quant_rows


def confocal_nyquist_um(lv) -> tuple[float, float]:
    """(strict, open-pinhole) Nyquist pixel sizes for a confocal.

    With a small pinhole the confocal OTF reaches ``2NA/lambda_ex + 2NA/lambda_em``,
    so two samples per period of that is ``1 / (2 (2NA/lex + 2NA/lem))`` -- the
    ~lambda_ex / 8NA the usual calculators quote. Wide open, detection no longer
    sharpens anything and excitation alone sets ``lambda_ex / 4NA``.
    """
    strict = 1.0 / (2.0 * (2 * lv.na / lv.excitation_um + 2 * lv.na / lv.wavelength_um))
    return strict, lv.excitation_um / (4.0 * lv.na)


def scan_optics_rows(sim: ScanSimulation) -> list[Row]:
    lv = sim.levers
    k = sim.kernel
    lam, na = lv.wavelength_um, lv.na
    rows = [Row(
        "confocal PSF (measured from the model)",
        f"{k.fwhm_um * 1000:.0f} nm FWHM against {optics.fwhm_um(lam, na) * 1000:.0f} nm widefield",
        f"K = E(r) x (h_em (x) pinhole)(r);  excitation {lv.excitation_um * 1000:.0f} nm, emission "
        f"{lam * 1000:.0f} nm, NA {na:.2f}; ideal small-pinhole limit ~ widefield / 1.4",
    )]
    radius = lv.pinhole_au * 0.61 * lam / na
    rows.append(Row(
        "pinhole",
        f"{lv.pinhole_au:g} AU = {2 * radius * 1000:.0f} nm at the sample; passes {k.pinhole_fraction:.0%} "
        f"of an in-focus emitter's light and ~{haze_fraction(lv.pinhole_au):.0%} of the haze",
        "1 AU = 1.22 lambda_em / NA;  haze share AU^2 / (1 + AU^2) (pedagogical)",
        Status.WARN if lv.pinhole_au > 2 else Status.INFO,
        "a pinhole this open gives up optical sectioning: the haze it admits brings shot noise and no signal",
    ))
    strict, relaxed = confocal_nyquist_um(lv)
    for axis, p, count in (("x", lv.pixel_x_um, lv.pixels_x), ("y", lv.pixel_y_um, lv.pixels_y)):
        if p > relaxed * 1.05:
            status = Status.FAIL if p > 2 * relaxed else Status.WARN
            verdict = "under-sampled: detail between pixels is lost" + (
                ", and between lines it is not even averaged -- it is skipped" if axis == "y" else "")
            remedy = "zoom in or add pixels along this axis"
        elif p < strict / 2:
            status, verdict = Status.WARN, "over-sampled: each pixel gets fewer photons, the frame is slower and bleaching faster"
            remedy = "zoom out or use fewer pixels: no resolution is gained below Nyquist"
        else:
            status, verdict, remedy = Status.PASS, "sampled at Nyquist", ""
        rows.append(Row(
            f"sampling in {axis}",
            f"{p * 1000:.0f} nm per pixel -- {verdict}",
            f"p = FN / (M zoom N) = {lv.field_um:.1f} um / {count};  Nyquist {strict * 1000:.0f} nm "
            f"(small pinhole) to {relaxed * 1000:.0f} nm (open)",
            status, remedy,
        ))
    rows.append(Row(
        "scan field and frame time",
        f"{lv.field_um:.1f} um field, {lv.pixels_x} x {lv.pixels_y} px, {lv.frame_time_s():.2f} s per frame"
        f" ({1 / max(lv.frame_time_s(), 1e-9):.2g} fps)",
        f"FN {18:g} mm / ({lv.magnification:g}x x zoom {lv.zoom:g});  t = Ny x Nx x dwell / duty x averages "
        f"= {lv.pixels_y} x {lv.pixels_x} x {lv.dwell_us:g} us / {0.9 if lv.bidirectional else 0.7} x {lv.averages}",
    ))
    return rows


def scan_photon_rows(sim: ScanSimulation) -> list[Row]:
    lv = sim.levers
    k = sim.kernel
    ph = sim.photometry
    fl = lv.fluor
    rows = [Row(
        "irradiance at the focus",
        f"{k.peak_irradiance_w_cm2 / 1000:.3g} kW/cm² from {lv.laser_power_uw:g} µW",
        "I0 = P x h_ex(0) / dx^2 -- a diffraction-limited spot concentrates microwatts into kW/cm^2",
    )]
    rows.append(Row(
        "excitation saturation",
        f"k tau = {k.saturation:.3f} at the focus: the fluorophore emits {1 / (1 + k.saturation):.0%} of what linear "
        "excitation would give",
        f"k = sigma I0 / h nu, tau = {fl.lifetime_ns} ns ({fl.name}); emission ~ k / (1 + k tau)",
        Status.FAIL if k.saturation > 0.5 else Status.WARN if k.saturation > 0.1 else Status.PASS,
        "more power now buys little signal, widens the PSF and bleaches faster: lengthen the dwell or average instead",
    ))
    rows.append(Row(
        "detected per fluorophore, per frame",
        f"{ph.detected_per_fluor:.3g} e- summed over every pixel that sees it",
        f"dwell / (px py) x int K dA x CE {ph.collection:.3f} x T {lv.transmission:.2f} x QE {ph.qe:.2f}",
    ))
    lost = 1 - ph.survival
    rows.append(Row(
        "photobleaching",
        f"{ph.emitted_per_frame:.3g} photons emitted per fluorophore per frame; {lost:.1%} of the signal lost "
        f"over {lv.averages} average(s)",
        f"dose = dwell / (px py) x int E dA: halving the pixel size quadruples it; budget {fl.photon_budget:.0e} "
        "(order of magnitude)",
        Status.WARN if lost > 0.2 else Status.PASS if lost < 0.05 else Status.INFO,
        "bleaching follows the dose: lower the power, the dwell, the zoom or the averages",
    ))
    return rows


def scan_noise_rows(sim: ScanSimulation) -> list[Row]:
    lv = sim.levers
    det = lv.detector_model
    S = peak_signal(sim)
    b = simulation_budget(sim, S)
    counting = lv.acquisition_mode == "counting"
    rows = [Row(
        "signal at the structure",
        f"{S:.3g} e- peak per pixel per pass, on {sim.background_e:.3g} e- haze and {sim.dark_e:.2g} e- dark",
        f"dark = {det.dark_rate(lv.sensor_temperature_c):.3g} counts/s at {lv.sensor_temperature_c:g} C x "
        f"{lv.dwell_us:g} us dwell -- a dwell this short makes thermal noise almost irrelevant",
    )]
    f2 = 1.0 if counting else det.excess_noise_f2
    rows.append(Row(
        "detector",
        f"{det.name}, {lv.acquisition_mode}: QE {det.quantum_efficiency:.0%}, F² = {f2:g}"
        + ("" if counting else f", {lv.gain_adu_per_e:g} ADU per photoelectron"),
        "photon counting removes gain noise and the amplifier entirely, at the price of dead time"
        if counting else "analog: gain cascade (F^2), then amplifier noise and an ADC",
    ))
    if not counting:
        g = lv.gain_adu_per_e
        status = Status.WARN if g < 1.5 else Status.PASS
        rows.append(Row(
            "detector gain (HV)",
            f"one photoelectron = {g:g} ADU; amplifier noise {det.amplifier_noise_adu:g} ADU = "
            f"{det.amplifier_noise_adu / g:.2g} e-; full scale = {(lv.max_adu - lv.offset_adu) / g:.0f} e-",
            "too low and amplifier and quantization noise swamp single photons; too high and bright pixels clip",
            status, "raise the gain until one photon is a few ADU, then check nothing clips",
        ))
    shot = b.photon_shot + b.background_shot + b.dark_shot
    other = b.read + b.quantization
    if shot > 3 * other:
        regime, status, remedy = "shot-noise limited: only more photons help", Status.PASS, ""
    elif other > 3 * shot:
        regime, status, remedy = ("electronics limited: amplifier and ADC noise dominate", Status.WARN,
                                  "raise the detector gain, or switch to photon counting")
    else:
        regime, status, remedy = "between regimes", Status.INFO, ""
    rows.append(Row(
        "noise regime", regime,
        f"shot variance {shot:.3g} e-^2 against electronics {other:.3g} e-^2 and laser {b.laser:.3g} e-^2",
        status, remedy,
    ))
    rows.append(Row(
        "SNR at the structure",
        f"{b.snr:.3g}  (a perfect detector with QE 1: {math.sqrt(S / det.quantum_efficiency * b.frames):.3g}; "
        f"this QE with no excess noise: {b.shot_limited_snr:.3g})",
        f"S / sqrt(F^2 (S + B + D) + sigma_amp^2/g^2 + 1/(12 g^2) + (RIN (S + B))^2), {b.frames} average(s);  "
        f"dominant: {b.LABELS[b.dominant()]}",
        Status.FAIL if b.snr < 3 else Status.WARN if b.snr < 5 else Status.PASS,
        "below the Rose criterion (SNR 5): dwell longer, average, open the pinhole slightly, or raise power "
        "if not saturated",
    ))
    high = float(sim.saturated.mean())
    black = float(sim.black_clipped.mean())
    if counting:
        rows.append(Row(
            "dead time",
            f"{sim.dead_time_loss:.1%} of counts lost at the brightest pixel"
            + (" (corrected)" if lv.dead_time_correction else ""),
            f"measured = true / (1 + true tau / dwell), tau = {det.dead_time_ns:g} ns, "
            f"max {lv.dwell_s / (det.dead_time_ns * 1e-9):.0f} counts per {lv.dwell_us:g} us dwell",
            Status.FAIL if sim.dead_time_loss > 0.1 else Status.WARN if sim.dead_time_loss > 0.02 else Status.PASS,
            "counting saturates: lower the power or enable dead-time correction",
        ))
    rows.append(Row(
        "clipping",
        f"{high:.2%} at full scale ({lv.bit_depth}-bit = {lv.max_adu} ADU)"
        + ("" if counting else f";  {black:.1%} clipped at zero (offset {lv.offset_adu:g} ADU)"),
        "clipped pixels bias every intensity measurement; a black-clipped background reads too dark "
        "and makes faint structures look brighter relative to it",
        Status.FAIL if high > 0.001 or black > 0.05 else Status.WARN if high > 0 or black > 0.005 else Status.PASS,
        "high: lower the gain or power;  black: raise the offset until the background sits above zero",
    ))
    return rows


def build_scan_report(sim: ScanSimulation, q: Quantitation | None = None) -> dict[str, Report]:
    sections = {
        "optics and scanning": Report(scan_optics_rows(sim)),
        "photons": Report(scan_photon_rows(sim)),
        "detector and noise": Report(scan_noise_rows(sim)),
    }
    if q is not None:
        sections["quantitation"] = Report(quant_rows(q))
    return sections
