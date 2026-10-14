"""Starting points that each make one point.

A preset is a set of levers plus the sentence it is there to demonstrate. They come
in pairs where that helps: the read-noise-limited image, then the same image rescued
by binning, then by EM gain, so the comparison is one click rather than ten.
"""

from __future__ import annotations

from dataclasses import dataclass

from .acquisition import Levers

EMCCD = "EMCCD, back-illuminated (13 um)"
UNCOOLED = "CMOS, uncooled (3.45 um)"


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    levers: Levers


_DIM = dict(irradiance_w_cm2=2.0, exposure_ms=20.0)

PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset(
            "Widefield, 100x/1.40 oil, sCMOS",
            "A well-set-up widefield fluorescence image: Nyquist-sampled and shot-noise limited.",
            Levers(),
        ),
        Preset(
            "Read-noise limited (dim, short exposure)",
            "A handful of photons per pixel: read noise, not the light, sets the noise floor.",
            Levers(**_DIM),
        ),
        Preset(
            "... rescued by 2x2 binning",
            "The same light, four times the signal per pixel. Read noise grows only 2x on sCMOS; "
            "resolution is what you pay.",
            Levers(binning=2, **_DIM),
        ),
        Preset(
            "... rescued by an EMCCD (gain 300)",
            "EM gain divides read noise by 300 -- at the cost of an excess noise factor F^2 = 2. "
            "(Its 13 um pixels also gather 4x the light per pixel at 100x.)",
            Levers(camera=EMCCD, em_gain=300.0, magnification=100.0, **_DIM),
        ),
        Preset(
            "EMCCD in bright light",
            "With plenty of photons, EM gain buys nothing and F^2 = 2 halves the effective QE.",
            Levers(camera=EMCCD, em_gain=100.0, irradiance_w_cm2=10.0, exposure_ms=20.0),
        ),
        Preset(
            "Low NA: 20x/0.75 air",
            "Fewer photons collected and a wider PSF -- and 325 nm pixels under-sample it.",
            Levers(na=0.75, medium="air", magnification=20.0),
        ),
        Preset(
            "60x/1.40 oil, Nyquist",
            "Same NA as 100x but 108 nm pixels: close to Nyquist, and 2.8x the photons per pixel.",
            Levers(magnification=60.0),
        ),
        Preset(
            "Far red: Alexa Fluor 647",
            "Longer wavelength, wider PSF: 665 nm light resolves ~30% less than 507 nm.",
            Levers(fluorophore="Alexa Fluor 647", specimen="Microtubules, immunostained (25 nm, ~60 nm with IgG)"),
        ),
        Preset(
            "Warm uncooled CMOS, long exposure",
            "Thermal charge: dark current doubles every ~6.5 C, and its shot noise is real noise.",
            Levers(camera=UNCOOLED, temperature_c=40.0, exposure_ms=2000.0, irradiance_w_cm2=0.5,
                   magnification=60.0),
        ),
        Preset(
            "... with dark-frame subtraction",
            "A master dark removes the mean dark signal, DSNU and hot pixels -- but not their shot noise.",
            Levers(camera=UNCOOLED, temperature_c=40.0, exposure_ms=2000.0, irradiance_w_cm2=0.5,
                   magnification=60.0, dark_subtract=True),
        ),
        Preset(
            "Averaging cannot remove fixed pattern",
            "16 frames average away shot and read noise; PRNU, DSNU and hot pixels stay.",
            Levers(camera=UNCOOLED, temperature_c=40.0, exposure_ms=500.0, frames=16,
                   magnification=60.0, irradiance_w_cm2=5.0),
        ),
        Preset(
            "Saturated",
            "Too much light on a photostable dye: clipped pixels flatten the profile and wreck "
            "intensity measurements. (EGFP would bleach before it clipped.)",
            Levers(fluorophore="Alexa Fluor 488", irradiance_w_cm2=200.0, exposure_ms=1500.0),
        ),
        Preset(
            "Sub-resolution beads, 100 nm",
            "A bead smaller than the PSF images as the PSF: its measured width is the optics'.",
            Levers(specimen="Beads, 100 nm", irradiance_w_cm2=5.0),
        ),
    )
}


# --- point-scanning confocal -----------------------------------------------------

from .confocal import ScanLevers  # noqa: E402

HPD = "Hybrid detector (GaAsP HPD)"

CONFOCAL_PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset(
            "Confocal, 63x/1.40 oil, GaAsP analog",
            "A typical setup: 1 AU pinhole, 70 nm pixels (Nyquist), 3.2 us dwell, a few tens of photons.",
            ScanLevers(),
        ),
        Preset(
            "Zoom 1: under-sampled",
            "The whole 286 um field in 512 pixels: 558 nm pixels cannot carry a 180 nm PSF.",
            ScanLevers(zoom=1.0),
        ),
        Preset(
            "Zoom 30: over-sampled",
            "19 nm pixels: the same photons per pixel (a scanner's pixel is a dwell, not an area), no new "
            "detail, a 3.7x smaller field, and 14x the light dose per area -- so 14x the bleaching.",
            ScanLevers(zoom=30.0, averages=4),
        ),
        Preset(
            "512 x 128: y under-sampled",
            "Fewer lines than pixels: x is at Nyquist, y skips 280 nm between lines -- a fast scan that misses things.",
            ScanLevers(pixels_y=128),
        ),
        Preset(
            "Short dwell (0.4 us)",
            "Resonant-scanner speeds: a few photons per pixel, and the image is shot-noise mush.",
            ScanLevers(dwell_us=0.4),
        ),
        Preset(
            "... rescued by 8 line averages",
            "The same fast scan averaged 8x: the photons of a 3.2 us dwell, spread over time.",
            ScanLevers(dwell_us=0.4, averages=8),
        ),
        Preset(
            "Saturating the fluorophore (200 uW)",
            "k tau ~ 0.7: 13x the power buys only ~9x the signal, a wider PSF and faster bleaching.",
            ScanLevers(laser_power_uw=200.0),
        ),
        Preset(
            "Gain too low",
            "0.15 ADU per photoelectron: the 1.5 ADU amplifier noise is now 10 photoelectrons, and the ADC "
            "step is 7 -- single photons vanish.",
            ScanLevers(gain_adu_per_e=0.15),
        ),
        Preset(
            "Offset too low (black clipping)",
            "A negative offset sends the background below zero ADU: it clips, the mean is wrong, counts halve.",
            ScanLevers(offset_adu=-20.0),
        ),
        Preset(
            "Gain too high (clipping)",
            "60 ADU per photoelectron on a 12-bit ADC: full scale is 68 photons, and the filament clips.",
            ScanLevers(gain_adu_per_e=60.0, laser_power_uw=40.0),
        ),
        Preset(
            "Photon counting, hybrid detector",
            "No gain noise (F^2 ~ 1), no amplifier, no ADC step: each photon is a count.",
            ScanLevers(detector=HPD, mode="counting"),
        ),
        Preset(
            "Counting too fast: dead time",
            "A 20 ns dead time caps a 1 us dwell at 50 counts: the bright pixels lose counts and the profile "
            "flattens -- unless dead-time correction is on.",
            ScanLevers(mode="counting", laser_power_uw=100.0, dwell_us=1.0),
        ),
        Preset(
            "Pinhole wide open (5 AU)",
            "Little lateral gain from a 5 AU pinhole, and the haze it admits brings shot noise without signal.",
            ScanLevers(pinhole_au=5.0, background_per_um2=300.0),
        ),
        Preset(
            "Multialkali PMT",
            "QE 20% instead of 40%: half the photoelectrons, so sqrt(2) less SNR for the same light.",
            ScanLevers(detector="PMT, multialkali"),
        ),
    )
}
