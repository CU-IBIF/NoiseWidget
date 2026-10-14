"""Fluorescent test objects with known sizes.

Each specimen is a set of fluorophore positions, splatted onto a fine grid (25 nm by
default) that is well below any resolution the optics can reach. One structure sits
at the centre of the field and is the **reference**: the quantitation pane measures
it, and because it was generated here, the true answer is known.

Sizes are the textbook values, and the label geometry is modelled because it is part
of what a measurement sees:

* **Microtubule**: 25 nm outer diameter, 13 protofilaments of 8 nm alpha/beta-tubulin
  dimers, so 13 / 0.008 = 1625 dimers per um. A tubulin-GFP fusion puts the label on
  the outer wall (radius ~12.5 nm). Primary plus secondary IgG displaces the dye by
  ~17.5 nm, so an immunostained microtubule is ~60 nm across in its labels -- the
  width super-resolution measures, not the 25 nm of the polymer.
* **Actin filament**: ~7 nm diameter, one subunit per 2.7 nm (~370 per um); a
  phalloidin conjugate binds subunits directly.
* **Beads**: dye distributed through the volume of a sphere of the stated diameter.

Fluorophores on a filament sit on its surface, so across the filament their
positions are ``r cos(phi)`` with phi uniform: the projection of a cylinder wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_GRID = 512
DEFAULT_DX_UM = 0.025


@dataclass(frozen=True)
class SpecimenType:
    name: str
    kind: str  # "filament" or "bead"
    true_diameter_nm: float  # the structure itself
    label_radius_nm: float  # where the fluorophores sit, from the structure's axis
    default_density: float  # fluorophores per um (filament) or per bead
    density_units: str
    derivation: str  # how the default density follows from the biology
    others: int  # structures in the field besides the reference
    persistence_um: float = 20.0  # display curvature of the other filaments

    @property
    def label_diameter_nm(self) -> float:
        return 2.0 * self.label_radius_nm

    @property
    def label_sigma_nm(self) -> float:
        """Standard deviation of the label positions across the structure.

        A projected cylinder wall (``r cos phi``) has variance ``r^2 / 2``. A
        projected uniform ball has variance ``r^2 / 5`` along any one axis.
        """
        r = self.label_radius_nm
        return r / math.sqrt(2.0) if self.kind == "filament" else r / math.sqrt(5.0)


SPECIMENS: dict[str, SpecimenType] = {
    s.name: s
    for s in (
        SpecimenType(
            "Microtubules, tubulin-GFP (25 nm)", "filament", 25.0, 12.5, 160.0,
            "fluorophores/um",
            "1625 dimers/um (13 protofilaments / 8 nm) x ~10% labelled",
            others=7, persistence_um=25.0,
        ),
        SpecimenType(
            "Microtubules, immunostained (25 nm, ~60 nm with IgG)", "filament", 25.0, 30.0,
            400.0, "fluorophores/um",
            "~130 IgG/um on 1625 epitopes/um x ~3 dyes per secondary",
            others=7, persistence_um=25.0,
        ),
        SpecimenType(
            "Actin filaments, phalloidin (7 nm)", "filament", 7.0, 3.5, 120.0,
            "fluorophores/um",
            "~370 subunits/um (2.7 nm rise) x ~1/3 occupied",
            others=14, persistence_um=4.0,
        ),
        SpecimenType(
            "Beads, 100 nm", "bead", 100.0, 50.0, 2000.0, "fluorophores/bead",
            "order of magnitude for a 100 nm dyed polystyrene bead",
            others=24,
        ),
        SpecimenType(
            "Beads, 200 nm", "bead", 200.0, 100.0, 12000.0, "fluorophores/bead",
            "volume scaling from the 100 nm bead (x8), slightly under-dyed",
            others=18,
        ),
    )
}

DEFAULT_SPECIMEN = "Microtubules, tubulin-GFP (25 nm)"


@dataclass(frozen=True)
class Reference:
    """The measured structure and its ground truth.

    ``points`` are the fluorophore positions (um) and weights belonging to the
    reference alone, so the quantitation can count exactly how many labels lie in
    whatever window it measures.
    """

    kind: str
    centre_um: tuple[float, float]  # (x, y)
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray

    def count_within(self, x0: float, x1: float, y0: float, y1: float) -> float:
        inside = (self.x >= x0) & (self.x < x1) & (self.y >= y0) & (self.y < y1)
        return float(self.weight[inside].sum())


@dataclass(frozen=True)
class Specimen:
    kind: SpecimenType
    density: np.ndarray  # fluorophores per fine pixel, indexed [y, x]
    dx_um: float
    density_value: float
    stochastic: bool
    reference: Reference

    @property
    def field_um(self) -> float:
        return self.density.shape[0] * self.dx_um

    @property
    def total(self) -> float:
        return float(self.density.sum())


def splat(n: int, dx_um: float, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Bilinear deposit of weighted points onto a grid whose pixel centres are at
    ``(i + 0.5) dx``. Conserves the total weight of points inside the field."""
    gx = x / dx_um - 0.5
    gy = y / dx_um - 0.5
    ix = np.floor(gx).astype(int)
    iy = np.floor(gy).astype(int)
    fx = gx - ix
    fy = gy - iy
    out = np.zeros(n * n)
    for ox, oy, ww in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        cx = ix + ox
        cy = iy + oy
        ok = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)
        out += np.bincount(cy[ok] * n + cx[ok], weights=(w * ww)[ok], minlength=n * n)
    return out.reshape(n, n)


def _label_filament(
    path_x: np.ndarray, path_y: np.ndarray, density: float, radius_um: float,
    rng: np.random.Generator, stochastic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place fluorophores along a polyline at ``density`` per um, on the wall."""
    seg = np.hypot(np.diff(path_x), np.diff(path_y))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    length = arc[-1]
    mean = density * length
    if stochastic:
        count = rng.poisson(mean)
        weight = np.ones(count)
        s = rng.uniform(0.0, length, count)
        phi = rng.uniform(0.0, 2 * math.pi, count)
    else:
        # The mean labelling with none of its randomness: evenly spaced, evenly
        # spread around the wall, total exactly density x length.
        count = max(int(math.ceil(length / 0.002)), 1) * 8
        weight = np.full(count, mean / count)
        s = (np.arange(count) + 0.5) * length / count
        phi = (np.arange(count) % 8) * (2 * math.pi / 8) + 0.3
    x = np.interp(s, arc, path_x)
    y = np.interp(s, arc, path_y)
    # Unit normal of the local segment.
    k = np.clip(np.searchsorted(arc, s) - 1, 0, len(seg) - 1)
    tx = np.diff(path_x)[k] / np.maximum(seg[k], 1e-12)
    ty = np.diff(path_y)[k] / np.maximum(seg[k], 1e-12)
    offset = radius_um * np.cos(phi)
    return x - ty * offset, y + tx * offset, weight


def _label_bead(
    cx: float, cy: float, count_mean: float, radius_um: float,
    rng: np.random.Generator, stochastic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stochastic:
        count = rng.poisson(count_mean)
        weight = np.ones(count)
    else:
        count = 400
        weight = np.full(count, count_mean / count)
        rng = np.random.default_rng(12345)  # a fixed, smooth stand-in for "uniform"
    v = rng.normal(size=(count, 3))
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    r = radius_um * rng.uniform(0.0, 1.0, count) ** (1.0 / 3.0)
    return cx + v[:, 0] * r, cy + v[:, 1] * r, weight


def _random_filament(field: float, rng: np.random.Generator, persistence_um: float):
    """A worm-like chain: heading diffuses with variance ``ds / Lp`` per step."""
    step = 0.02
    length = rng.uniform(0.5, 1.2) * field
    steps = int(length / step)
    heading = rng.uniform(-math.pi, math.pi) + np.cumsum(
        rng.normal(0.0, math.sqrt(step / persistence_um), steps)
    )
    x0, y0 = rng.uniform(0.0, field, 2)
    x = x0 + np.concatenate([[0.0], np.cumsum(step * np.cos(heading))])
    y = y0 + np.concatenate([[0.0], np.cumsum(step * np.sin(heading))])
    return x, y


def make_specimen(
    name: str = DEFAULT_SPECIMEN,
    density: float | None = None,
    stochastic: bool = True,
    seed: int = 1,
    n: int = DEFAULT_GRID,
    dx_um: float = DEFAULT_DX_UM,
) -> Specimen:
    """Generate a specimen. Same arguments, same specimen: it is seeded."""
    kind = SPECIMENS[name]
    density = kind.default_density if density is None else max(float(density), 0.0)
    field = n * dx_um
    cx = cy = field / 2.0
    radius = kind.label_radius_nm / 1000.0
    rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng(seed + 7919)

    xs, ys, ws = [], [], []
    if kind.kind == "filament":
        margin = 0.4
        px = np.linspace(margin, field - margin, 400)
        py = np.full_like(px, cy)
        rx, ry, rw = _label_filament(px, py, density, radius, label_rng, stochastic)
        # Keep the other filaments out of a box around the measured segment, so the
        # line profile sees one microtubule and not a crossing.
        keep_x, keep_y = 3.8, 1.8
        for _ in range(kind.others):
            fx, fy = _random_filament(field, rng, kind.persistence_um)
            x, y, w = _label_filament(fx, fy, density, radius, label_rng, stochastic)
            clear = ~((np.abs(x - cx) < keep_x) & (np.abs(y - cy) < keep_y))
            xs.append(x[clear]); ys.append(y[clear]); ws.append(w[clear])  # noqa: E702
    else:
        rx, ry, rw = _label_bead(cx, cy, density, radius, label_rng, stochastic)
        placed: list[tuple[float, float]] = [(cx, cy)]
        tries = 0
        while len(placed) < kind.others + 1 and tries < 5000:
            tries += 1
            bx, by = rng.uniform(0.6, field - 0.6, 2)
            if math.hypot(bx - cx, by - cy) < 2.0:
                continue
            if any(math.hypot(bx - qx, by - qy) < 0.8 for qx, qy in placed):
                continue
            placed.append((bx, by))
            x, y, w = _label_bead(bx, by, density, radius, label_rng, stochastic)
            xs.append(x); ys.append(y); ws.append(w)  # noqa: E702

    xs.append(rx); ys.append(ry); ws.append(rw)  # noqa: E702
    grid = splat(n, dx_um, np.concatenate(xs), np.concatenate(ys), np.concatenate(ws))
    reference = Reference(kind.kind, (cx, cy), rx, ry, rw)
    return Specimen(kind, grid, dx_um, density, stochastic, reference)
