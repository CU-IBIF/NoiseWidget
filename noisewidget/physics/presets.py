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
