"""Fluorophores and excitation: how many photons one label emits.

Photon emission per fluorophore is the first lever on the signal::

    k_abs = sigma * I / (h c / lambda_ex)            absorptions per second
    k_em  = QY * k_abs / (1 + k_abs * tau)           photons per second

``sigma`` is the absorption cross-section, ``3.82e-21 * epsilon`` cm^2 for a molar
extinction coefficient in M^-1 cm^-1. The ``1 + k_abs tau`` term is ground-state
depletion: it is negligible at widefield irradiances and only bites near
kW/cm^2, which is where the lever runs out.

Spectral values are peak figures from the literature (FPbase and vendor tables) and
treat the laser line as sitting on the absorption peak. Photobleaching budgets --
how many photons a fluorophore emits before it dies -- vary by an order of magnitude
with environment and are quoted here as orders of magnitude. All of this is
pedagogical: the point is the scaling, and every number is shown where it is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PLANCK = 6.62607015e-34
LIGHT_SPEED = 2.99792458e8
CROSS_SECTION_PER_EPSILON = 1000.0 * math.log(10) / 6.02214076e23  # cm^2 per (M^-1 cm^-1)


@dataclass(frozen=True)
class Fluorophore:
    name: str
    excitation_nm: float  # laser line used to excite it
    emission_nm: float  # emission peak: this sets the PSF
    epsilon: float  # M^-1 cm^-1 at the absorption peak
    quantum_yield: float
    lifetime_ns: float
    photon_budget: float  # photons emitted before bleaching, order of magnitude

    @property
    def cross_section_cm2(self) -> float:
        return CROSS_SECTION_PER_EPSILON * self.epsilon

    def photon_energy_j(self) -> float:
        return PLANCK * LIGHT_SPEED / (self.excitation_nm * 1e-9)

    def absorption_rate(self, irradiance_w_cm2: float) -> float:
        """Absorptions per second at an irradiance in W/cm^2."""
        # np.maximum, not max: confocal passes a whole irradiance map at once.
        return self.cross_section_cm2 * np.maximum(irradiance_w_cm2, 0.0) / self.photon_energy_j()

    def emission_rate(self, irradiance_w_cm2: float) -> float:
        """Emitted photons per second, with ground-state depletion."""
        k = self.absorption_rate(irradiance_w_cm2)
        return self.quantum_yield * k / (1.0 + k * self.lifetime_ns * 1e-9)

    def bleach_rate(self, irradiance_w_cm2: float) -> float:
        """Probability per second of photobleaching, ``k_em / budget``."""
        return self.emission_rate(irradiance_w_cm2) / self.photon_budget


FLUOROPHORES: dict[str, Fluorophore] = {
    f.name: f
    for f in (
        Fluorophore("EGFP", 488, 507, 55_900, 0.60, 2.6, 1e5),
        Fluorophore("Alexa Fluor 488", 488, 519, 71_000, 0.92, 4.1, 1e6),
        Fluorophore("mCherry", 561, 610, 72_000, 0.22, 1.5, 4e4),
        Fluorophore("Cy3", 561, 570, 150_000, 0.15, 0.3, 3e5),
        Fluorophore("Alexa Fluor 647", 640, 665, 239_000, 0.33, 1.0, 1e6),
    )
}


def wavelength_to_rgb(nm: float) -> tuple[float, float, float]:
    """Approximate display colour of a wavelength, for the emission LUT.

    Piecewise-linear spectral approximation (after Dan Bruton). Only used to tint
    the display; it has no bearing on any number.
    """
    if nm < 440:
        r, g, b = (440 - nm) / 60, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50, 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, (510 - nm) / 20
    elif nm < 580:
        r, g, b = (nm - 510) / 70, 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, (645 - nm) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    return clamp(r), clamp(g), clamp(b)
