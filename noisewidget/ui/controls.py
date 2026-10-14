"""The levers: every control a user has on a real widefield microscope.

Grouped the way they are met at the bench -- specimen, light, objective, camera --
with the noise sources last, as switches. Turning one noise source off and back on
is the fastest way to learn which one was spoiling the image.

The panel owns a :class:`Levers` value and emits a new one on every change. It
never computes anything itself.
"""

from __future__ import annotations

import dataclasses
import math

from PySide6 import QtCore, QtWidgets

from ..physics.acquisition import Levers, NoiseSources
from ..physics.camera import CAMERAS
from ..physics.fluorophores import FLUOROPHORES
from ..physics.optics import IMMERSION_MEDIA, OBJECTIVES
from ..physics.specimens import SPECIMENS
from . import style


class LeverSlider(QtWidgets.QWidget):
    """A slider and a spin box bound together, optionally on a log scale.

    The slider is for feel -- dragging exposure across three decades and watching
    the image change is the point -- and the spin box is for typing an exact value.
    """

    valueChanged = QtCore.Signal(float)
    STEPS = 1000

    def __init__(self, lo: float, hi: float, value: float, log: bool = False, decimals: int = 2,
                 suffix: str = "", parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.log = lo, hi, log
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(92)
        if log:
            self.spin.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        else:
            self.spin.setSingleStep((hi - lo) / 100)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.set_value(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _to_pos(self, v: float) -> int:
        v = min(max(v, self.lo), self.hi)
        if self.log:
            f = (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        else:
            f = (v - self.lo) / (self.hi - self.lo)
        return int(round(f * self.STEPS))

    def _from_pos(self, pos: int) -> float:
        f = pos / self.STEPS
        if self.log:
            return 10 ** (math.log10(self.lo) + f * (math.log10(self.hi) - math.log10(self.lo)))
        return self.lo + f * (self.hi - self.lo)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, v: float) -> None:
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(self._to_pos(v))
        for w in (self.slider, self.spin):
            w.blockSignals(False)

    def set_range(self, lo: float, hi: float) -> None:
        self.lo, self.hi = lo, hi
        v = min(max(self.value(), lo), hi)
        self.spin.blockSignals(True)
        self.spin.setRange(lo, hi)
        self.spin.blockSignals(False)
        self.set_value(v)

    def _from_slider(self, pos: int) -> None:
        v = self._from_pos(pos)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


def _combo(items, current) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox()
    box.addItems(list(items))
    box.setCurrentText(current)
    box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(12)
    return box


class LeverPanel(QtWidgets.QWidget):
    leversChanged = QtCore.Signal(object)

    def __init__(self, levers: Levers | None = None, parent=None):
        super().__init__(parent)
        self.levers = levers or Levers()
        self._updating = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # --- specimen -------------------------------------------------------
        box, form = self._group("specimen")
        self.specimen = _combo(SPECIMENS, self.levers.specimen)
        self.specimen.setToolTip("structures of known size; the one at the centre is measured")
        form.addRow(self.specimen)
        self.density = LeverSlider(1, 50_000, self.levers.density, log=True, decimals=0)
        self.density_label = QtWidgets.QLabel("label density")
        form.addRow(self.density_label, self.density)
        self.derivation = QtWidgets.QLabel("")
        self.derivation.setWordWrap(True)
        self.derivation.setStyleSheet(f"color: {style.MUTED}; font-size: 11px;")
        form.addRow(self.derivation)
        self.reseed = QtWidgets.QPushButton("new specimen")
        self.reseed.setToolTip("a different field of the same specimen: new filaments, new labels")
        form.addRow(self.reseed)
        outer.addWidget(box)

        # --- light ------------------------------------------------------------
        box, form = self._group("fluorophore and illumination")
        self.fluorophore = _combo(FLUOROPHORES, self.levers.fluorophore)
        form.addRow("fluorophore", self.fluorophore)
        self.irradiance = LeverSlider(0.01, 2000, self.levers.irradiance_w_cm2, log=True, decimals=2, suffix=" W/cm²")
        self.irradiance.setToolTip("excitation irradiance at the sample: more photons, faster bleaching")
        form.addRow("excitation", self.irradiance)
        self.background = LeverSlider(0, 2000, self.levers.background_per_um2, decimals=0, suffix=" /µm²")
        self.background.setToolTip(
            "background light, as fluorophore-equivalents per um^2: autofluorescence, "
            "out-of-focus blur, unbound dye. It carries shot noise but no information."
        )
        form.addRow("background", self.background)
        outer.addWidget(box)

        # --- objective --------------------------------------------------------
        box, form = self._group("objective and optics")
        self.objective = QtWidgets.QComboBox()
        self.objective.addItem("custom")
        for o in OBJECTIVES:
            self.objective.addItem(o.label)
        form.addRow("preset", self.objective)
        self.magnification = LeverSlider(2, 200, self.levers.magnification, log=True, decimals=1, suffix=" x")
        form.addRow("magnification", self.magnification)
        self.na = LeverSlider(0.05, 1.65, self.levers.na, decimals=2)
        self.na.setToolTip("NA = n sin(theta): sets resolution (0.61 lambda/NA) and light collection")
        form.addRow("NA", self.na)
        self.medium = _combo(IMMERSION_MEDIA, self.levers.medium)
        form.addRow("immersion", self.medium)
        self.follow = QtWidgets.QCheckBox("emission peak of the fluorophore")
        self.follow.setChecked(self.levers.wavelength_nm is None)
        form.addRow("wavelength", self.follow)
        self.wavelength = LeverSlider(400, 800, self.levers.wavelength_um * 1000, decimals=0, suffix=" nm")
        self.wavelength.setEnabled(self.levers.wavelength_nm is not None)
        form.addRow("", self.wavelength)
        self.transmission = LeverSlider(1, 100, self.levers.transmission * 100, decimals=0, suffix=" %")
        self.transmission.setToolTip("objective, dichroic, emission filter, tube lens: what survives to the camera")
        form.addRow("transmission", self.transmission)
        outer.addWidget(box)

        # --- camera -----------------------------------------------------------
        box, form = self._group("camera and acquisition")
        self.camera = _combo(CAMERAS, self.levers.camera)
        form.addRow(self.camera)
        self.camera_info = QtWidgets.QLabel("")
        self.camera_info.setWordWrap(True)
        self.camera_info.setStyleSheet(f"color: {style.MUTED}; font-size: 11px;")
        form.addRow(self.camera_info)
        self.exposure = LeverSlider(0.1, 10_000, self.levers.exposure_ms, log=True, decimals=1, suffix=" ms")
        form.addRow("exposure", self.exposure)
        self.binning = QtWidgets.QComboBox()
        self.binning.addItems(["1x1", "2x2", "3x3", "4x4"])
        self.binning.setCurrentIndex(self.levers.binning - 1)
        form.addRow("binning", self.binning)
        self.temperature = LeverSlider(-100, 45, self.levers.sensor_temperature_c, decimals=0, suffix=" °C")
        self.temperature.setToolTip("dark current roughly doubles every ~6.5 C")
        form.addRow("sensor temp.", self.temperature)
        self.em_gain = LeverSlider(1, 1000, self.levers.gain, log=True, decimals=0)
        self.em_gain.setToolTip("EMCCD only: divides read noise by the gain, at the cost of F^2 = 2")
        form.addRow("EM gain", self.em_gain)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, 256)
        self.frames.setValue(self.levers.frames)
        self.frames.setToolTip("frames averaged: random noise falls as 1/sqrt(N); fixed pattern does not")
        form.addRow("frames averaged", self.frames)
        self.dark_subtract = QtWidgets.QCheckBox("subtract a master dark")
        self.flat_field = QtWidgets.QCheckBox("flat-field correct (PRNU)")
        form.addRow("calibration", self.dark_subtract)
        form.addRow("", self.flat_field)
        outer.addWidget(box)

        # --- noise sources ----------------------------------------------------
        box, form = self._group("noise sources (untick to remove one)")
        self.sources: dict[str, QtWidgets.QCheckBox] = {}
        term_of = {
            "photon_shot": "photon_shot", "background": "background_shot", "dark_shot": "dark_shot",
            "cic": "cic", "read": "read", "quantization": "quantization", "fixed_pattern": "prnu",
        }
        for f in dataclasses.fields(NoiseSources):
            check = QtWidgets.QCheckBox(NoiseSources.LABELS[f.name])
            check.setChecked(getattr(self.levers.noise, f.name))
            color = style.TERM_COLORS.get(term_of.get(f.name, ""), None)
            if color:
                check.setStyleSheet(f"QCheckBox::indicator:checked {{ background: {color}; border: 1px solid {color}; }}")
            self.sources[f.name] = check
            form.addRow(check)
        outer.addWidget(box)
        outer.addStretch(1)

        self._connect()
        self._sync_dependent()

    def _group(self, title: str):
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(6, 6, 6, 6)
        return box, form

    def _connect(self) -> None:
        self.specimen.currentTextChanged.connect(self._specimen_changed)
        self.reseed.clicked.connect(lambda _checked=False: self._set(specimen_seed=self.levers.specimen_seed + 1))
        self.density.valueChanged.connect(lambda v: self._set(label_density=v))
        self.fluorophore.currentTextChanged.connect(lambda t: self._set(fluorophore=t))
        self.irradiance.valueChanged.connect(lambda v: self._set(irradiance_w_cm2=v))
        self.background.valueChanged.connect(lambda v: self._set(background_per_um2=v))
        self.objective.currentIndexChanged.connect(self._objective_chosen)
        self.magnification.valueChanged.connect(lambda v: self._set(magnification=v, _custom=True))
        self.na.valueChanged.connect(lambda v: self._set(na=v, _custom=True))
        self.medium.currentTextChanged.connect(lambda t: self._set(medium=t, _custom=True))
        self.follow.toggled.connect(self._follow_toggled)
        self.wavelength.valueChanged.connect(lambda v: self._set(wavelength_nm=v))
        self.transmission.valueChanged.connect(lambda v: self._set(transmission=v / 100.0))
        self.camera.currentTextChanged.connect(self._camera_changed)
        self.exposure.valueChanged.connect(lambda v: self._set(exposure_ms=v))
        self.binning.currentIndexChanged.connect(lambda i: self._set(binning=i + 1))
        self.temperature.valueChanged.connect(lambda v: self._set(temperature_c=v))
        self.em_gain.valueChanged.connect(lambda v: self._set(em_gain=v))
        self.frames.valueChanged.connect(lambda v: self._set(frames=int(v)))
        self.dark_subtract.toggled.connect(lambda on: self._set(dark_subtract=on))
        self.flat_field.toggled.connect(lambda on: self._set(flat_field=on))
        for name, check in self.sources.items():
            check.toggled.connect(lambda on, name=name: self._set_noise(name, on))

    # --- changes ------------------------------------------------------------

    def _set(self, _custom: bool = False, **changes) -> None:
        if self._updating:
            return
        self.levers = self.levers.with_(**changes)
        if _custom and self.objective.currentIndex() != 0:
            self.objective.blockSignals(True)
            self.objective.setCurrentIndex(0)
            self.objective.blockSignals(False)
        self._sync_dependent()
        self.leversChanged.emit(self.levers)

    def _set_noise(self, name: str, on: bool) -> None:
        self._set(noise=dataclasses.replace(self.levers.noise, **{name: on}))

    def _specimen_changed(self, name: str) -> None:
        self._updating = True
        self.density.set_value(SPECIMENS[name].default_density)
        self._updating = False
        self._set(specimen=name, label_density=None)

    def _objective_chosen(self, index: int) -> None:
        if index == 0:
            return
        o = OBJECTIVES[index - 1]
        self._updating = True
        self.medium.setCurrentText(o.medium)
        self.magnification.set_value(o.magnification)
        self.na.set_value(o.na)
        self._updating = False
        self._set(magnification=float(o.magnification), na=o.na, medium=o.medium)

    def _follow_toggled(self, on: bool) -> None:
        self.wavelength.setEnabled(not on)
        self._set(wavelength_nm=None if on else self.wavelength.value())

    def _camera_changed(self, name: str) -> None:
        cam = CAMERAS[name]
        self._updating = True
        self.temperature.set_value(cam.default_temperature_c)
        self._updating = False
        self._set(camera=name, temperature_c=None)

    def _sync_dependent(self) -> None:
        """Keep the controls that depend on other controls honest."""
        lv = self.levers
        spec = SPECIMENS[lv.specimen]
        self.density_label.setText("labels/µm" if spec.kind == "filament" else "labels/bead")
        self.derivation.setText(f"default {spec.default_density:g}: {spec.derivation}")
        cam = lv.camera_model
        self.em_gain.setEnabled(cam.has_em_gain)
        self.temperature.set_range(cam.min_temperature_c, 45.0)
        read = f"{cam.read_noise_e:g} e-" + (f" ({cam.em_read_noise_e:g} e- EM amp)" if cam.has_em_gain else "")
        self.camera_info.setText(
            f"QE {cam.quantum_efficiency:.0%}, read {read}, {cam.pixel_um:g} µm pixels, "
            f"well {cam.full_well_e:,.0f} e-, {cam.bit_depth}-bit, "
            + ("charge binning" if cam.charge_binning else "digital binning")
        )
        n = IMMERSION_MEDIA[lv.medium]
        if lv.na > n:
            # NA = n sin(theta) cannot exceed n; pull the NA back rather than refuse.
            self._updating = True
            self.na.set_value(round(n * 0.99, 2))
            self._updating = False
            self.levers = self.levers.with_(na=round(n * 0.99, 2))
        if self.follow.isChecked():
            self._updating = True
            self.wavelength.set_value(lv.fluor.emission_nm)
            self._updating = False

    def set_levers(self, levers: Levers) -> None:
        """Load a whole set of levers (a preset) into the controls."""
        self._updating = True
        self.levers = levers
        self.specimen.setCurrentText(levers.specimen)
        self.density.set_value(levers.density)
        self.fluorophore.setCurrentText(levers.fluorophore)
        self.irradiance.set_value(levers.irradiance_w_cm2)
        self.background.set_value(levers.background_per_um2)
        self.objective.setCurrentIndex(0)
        for i, o in enumerate(OBJECTIVES, start=1):
            if (o.magnification, o.na, o.medium) == (levers.magnification, levers.na, levers.medium):
                self.objective.setCurrentIndex(i)
                break
        self.magnification.set_value(levers.magnification)
        self.na.set_value(levers.na)
        self.medium.setCurrentText(levers.medium)
        self.follow.setChecked(levers.wavelength_nm is None)
        self.wavelength.setEnabled(levers.wavelength_nm is not None)
        self.wavelength.set_value(levers.wavelength_um * 1000)
        self.transmission.set_value(levers.transmission * 100)
        self.camera.setCurrentText(levers.camera)
        self.exposure.set_value(levers.exposure_ms)
        self.binning.setCurrentIndex(levers.binning - 1)
        self.temperature.set_range(levers.camera_model.min_temperature_c, 45.0)
        self.temperature.set_value(levers.sensor_temperature_c)
        self.em_gain.set_value(levers.gain)
        self.frames.setValue(levers.frames)
        self.dark_subtract.setChecked(levers.dark_subtract)
        self.flat_field.setChecked(levers.flat_field)
        for name, check in self.sources.items():
            check.setChecked(getattr(levers.noise, name))
        self._updating = False
        self._sync_dependent()
        self.leversChanged.emit(self.levers)
