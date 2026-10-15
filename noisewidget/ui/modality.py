"""The two frames: what differs between a camera and a point scanner, in one place.

The widget, graph, image panes and quantitation pane are shared. Everything that
is instrument-specific -- its levers, its simulator, its sweeps, its report, how
its image is laid out -- is gathered into a :class:`Modality`, so the shared
widgets never ask which instrument they are drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..physics import acquisition as cam
from ..physics import confocal as scan
from ..physics.presets import CONFOCAL_PRESETS, PRESETS
from ..physics.report import build_report
from ..physics.scan_report import build_scan_report


@dataclass(frozen=True)
class ImageInfo:
    """How a simulation's image sits on the specimen, and how to caption it."""

    origin_um: tuple[float, float]
    extent_um: tuple[float, float]  # (x, y)
    wavelength_nm: float
    full_scale: tuple[float, float]  # the ADC's range, in input electrons
    optical_caption: str
    recorded_caption: str


@dataclass(frozen=True)
class Modality:
    key: str
    title: str
    default_levers: Callable[[], object]
    make_simulator: Callable[[], object]
    make_panel: Callable[[object], object]
    presets: dict
    sweeps: dict
    sweep_axis: Callable[[str], tuple]
    budget_sweep: Callable
    operating_value: Callable
    simulation_budget: Callable
    peak_signal: Callable
    build_report: Callable
    image_info: Callable[[object], ImageInfo]
    electronics_terms: tuple[str, ...]  # what counts as "read noise" for the shaded band
    electronics_label: str
    noise_title: str
    recorded_title: str


# --- camera ----------------------------------------------------------------------


def _camera_axis(mode: str):
    if mode == "signal":
        return np.logspace(-1, 5, 160), "signal per pixel (e-)", "e-"
    if mode == "exposure":
        return np.logspace(-1, 4, 160), "exposure (ms)", "ms"
    return np.logspace(-2, 3.3, 160), "excitation (W/cm²)", "W/cm²"


def _camera_info(sim) -> ImageInfo:
    lv = sim.levers
    c = lv.camera_model
    n = sim.signal_e.shape[0]
    field = n * sim.pixel_um
    return ImageInfo(
        (0.0, 0.0), (field, field), lv.wavelength_um * 1000,
        (0.0, (c.max_adu - c.offset_adu) * c.adc_gain_e_per_adu / lv.gain),
        f"{sim.pixel_um * 1000:.0f} nm px · {n}² px",
        f"{lv.frames} frame(s) · {lv.exposure_ms:g} ms",
    )


def _camera_panel(levers):
    from .controls import LeverPanel

    return LeverPanel(levers)


CAMERA = Modality(
    key="camera",
    title="Camera (widefield)",
    default_levers=cam.Levers,
    make_simulator=cam.Simulator,
    make_panel=_camera_panel,
    presets=PRESETS,
    sweeps=cam.SWEEPS,
    sweep_axis=_camera_axis,
    budget_sweep=cam.budget_sweep,
    operating_value=cam.operating_value,
    simulation_budget=cam.simulation_budget,
    peak_signal=cam.peak_signal,
    build_report=build_report,
    image_info=_camera_info,
    electronics_terms=("read",),
    electronics_label="read-noise limited",
    noise_title="noise per pixel, referred to the sensor input",
    recorded_title="recorded image",
)


# --- point-scanning confocal -----------------------------------------------------


def _scan_info(sim) -> ImageInfo:
    lv = sim.levers
    if lv.acquisition_mode == "counting":
        full = (0.0, float(lv.max_adu))
    else:
        full = (0.0, (lv.max_adu - lv.offset_adu) / lv.gain_adu_per_e)
    ny, nx = sim.signal_e.shape
    shown = "" if (nx, ny) == (lv.pixels_x, lv.pixels_y) else f" (of {lv.pixels_x}×{lv.pixels_y})"
    return ImageInfo(
        sim.origin_um, sim.extent_um, lv.wavelength_um * 1000, full,
        f"{lv.pixel_x_um * 1000:.0f}×{lv.pixel_y_um * 1000:.0f} nm px · {nx}×{ny}{shown}",
        f"{lv.dwell_us:g} µs dwell · {lv.averages} avg · {lv.frame_time_s():.2g} s/frame",
    )


def _scan_panel(levers):
    from .scan_controls import ScanLeverPanel

    return ScanLeverPanel(levers)


CONFOCAL = Modality(
    key="confocal",
    title="Point-scanning confocal",
    default_levers=scan.ScanLevers,
    make_simulator=scan.ScanSimulator,
    make_panel=_scan_panel,
    presets=CONFOCAL_PRESETS,
    sweeps=scan.SWEEPS,
    sweep_axis=scan.sweep_axis,
    budget_sweep=scan.budget_sweep,
    operating_value=scan.operating_value,
    simulation_budget=scan.simulation_budget,
    peak_signal=scan.peak_signal,
    build_report=build_scan_report,
    image_info=_scan_info,
    electronics_terms=("read", "quantization"),
    electronics_label="electronics limited",
    noise_title="noise per pixel, referred to the detector input (photoelectrons)",
    recorded_title="recorded scan",
)

MODALITIES = {m.key: m for m in (CAMERA, CONFOCAL)}
