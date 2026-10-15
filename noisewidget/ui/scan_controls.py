"""The confocal's levers: laser, pinhole, scan and detector.

Laid out as at a confocal's acquisition panel: what is excited, what the pinhole
lets through, how the field is scanned (zoom, pixel format, dwell, averaging) and
how the detector turns photons into numbers (gain, offset, bit depth, counting).
Like :class:`~noisewidget.ui.controls.LeverPanel` it owns a value and emits a new
one on every change; it computes nothing.
"""

from __future__ import annotations

import dataclasses

from PySide6 import QtCore, QtWidgets

from ..physics.confocal import DETECTORS, FIELD_NUMBER_MM, ScanLevers, ScanNoise
from ..physics.fluorophores import FLUOROPHORES
from ..physics.optics import IMMERSION_MEDIA, OBJECTIVES
from ..physics.specimens import SPECIMENS
from . import style
from .controls import LeverSlider, _combo

FORMATS = (64, 128, 256, 512, 1024, 2048)

# Which noise toggle each graph term belongs to, so their colours match.
TERM_OF = {
    "photon_shot": "photon_shot", "background": "background_shot", "dark_counts": "dark_shot",
    "amplifier": "read", "quantization": "quantization", "laser_noise": "laser",
}


class ScanLeverPanel(QtWidgets.QWidget):
    leversChanged = QtCore.Signal(object)

    def __init__(self, levers: ScanLevers | None = None, parent=None):
        super().__init__(parent)
        self.levers = levers or ScanLevers()
        self._updating = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        lv = self.levers

        # --- specimen -------------------------------------------------------
        box, form = self._group("specimen")
        self.specimen = _combo(SPECIMENS, lv.specimen)
        form.addRow(self.specimen)
        self.density = LeverSlider(1, 50_000, lv.density, log=True, decimals=0)
        self.density_label = QtWidgets.QLabel("labels/µm")
        form.addRow(self.density_label, self.density)
        self.reseed = QtWidgets.QPushButton("new specimen")
        form.addRow(self.reseed)
        outer.addWidget(box)

        # --- excitation -------------------------------------------------------
        box, form = self._group("fluorophore and laser")
        self.fluorophore = _combo(FLUOROPHORES, lv.fluorophore)
        form.addRow("fluorophore", self.fluorophore)
        self.power = LeverSlider(0.01, 2000, lv.laser_power_uw, log=True, decimals=2, suffix=" µW")
        self.power.setToolTip("laser power at the sample; focused, microwatts become kW/cm² and can saturate the dye")
        form.addRow("laser power", self.power)
        self.rin = LeverSlider(0, 5, lv.laser_rin * 100, decimals=2, suffix=" %")
        self.rin.setToolTip("laser intensity noise, rms per pixel: multiplies the signal, so it grows with it")
        form.addRow("laser noise", self.rin)
        self.haze = LeverSlider(0, 2000, lv.background_per_um2, decimals=0, suffix=" /µm²")
        self.haze.setToolTip("out-of-focus haze and autofluorescence; the pinhole rejects most of it")
        form.addRow("haze", self.haze)
        outer.addWidget(box)

        # --- optics -----------------------------------------------------------
        box, form = self._group("objective and pinhole")
        self.objective = QtWidgets.QComboBox()
        self.objective.addItem("custom")
        for o in OBJECTIVES:
            self.objective.addItem(o.label)
        form.addRow("preset", self.objective)
        self.magnification = LeverSlider(2, 200, lv.magnification, log=True, decimals=1, suffix=" x")
        form.addRow("magnification", self.magnification)
        self.na = LeverSlider(0.05, 1.65, lv.na, decimals=2)
        form.addRow("NA", self.na)
        self.medium = _combo(IMMERSION_MEDIA, lv.medium)
        form.addRow("immersion", self.medium)
        self.follow = QtWidgets.QCheckBox("emission peak of the fluorophore")
        self.follow.setChecked(lv.wavelength_nm is None)
        form.addRow("emission λ", self.follow)
        self.wavelength = LeverSlider(400, 800, lv.wavelength_um * 1000, decimals=0, suffix=" nm")
        self.wavelength.setEnabled(lv.wavelength_nm is not None)
        form.addRow("", self.wavelength)
        self.pinhole = LeverSlider(0.1, 10, lv.pinhole_au, log=True, decimals=2, suffix=" AU")
        self.pinhole.setToolTip("1 Airy unit = 1.22 λ/NA: smaller sections and sharpens a little, at the cost of light")
        form.addRow("pinhole", self.pinhole)
        self.transmission = LeverSlider(1, 100, lv.transmission * 100, decimals=0, suffix=" %")
        form.addRow("transmission", self.transmission)
        outer.addWidget(box)

        # --- scan -------------------------------------------------------------
        box, form = self._group("scan")
        self.zoom = LeverSlider(0.6, 60, lv.zoom, log=True, decimals=1, suffix=" x")
        self.zoom.setToolTip(f"field = FN {FIELD_NUMBER_MM:g} mm / (M zoom): zoom sets the pixel size, not the resolution")
        form.addRow("zoom", self.zoom)
        self.pixels_x = QtWidgets.QComboBox()
        self.pixels_y = QtWidgets.QComboBox()
        for box_ in (self.pixels_x, self.pixels_y):
            box_.addItems([str(f) for f in FORMATS])
        self.pixels_x.setCurrentText(str(lv.pixels_x))
        self.pixels_y.setCurrentText(str(lv.pixels_y))
        self.pixels_x.setToolTip("pixels per line: x sampling")
        self.pixels_y.setToolTip("lines per frame: y sampling. Fewer lines is faster -- and skips what lies between them")
        fmt = QtWidgets.QHBoxLayout()
        fmt.addWidget(self.pixels_x)
        fmt.addWidget(QtWidgets.QLabel("×"))
        fmt.addWidget(self.pixels_y)
        form.addRow("format (x × y)", fmt)
        self.sampling = QtWidgets.QLabel("")
        self.sampling.setStyleSheet(f"color: {style.MUTED}; font-size: 11px;")
        self.sampling.setWordWrap(True)
        form.addRow(self.sampling)
        self.dwell = LeverSlider(0.1, 200, lv.dwell_us, log=True, decimals=2, suffix=" µs")
        self.dwell.setToolTip("time the beam spends per pixel: the confocal's exposure")
        form.addRow("pixel dwell", self.dwell)
        self.averages = QtWidgets.QSpinBox()
        self.averages.setRange(1, 64)
        self.averages.setValue(lv.averages)
        self.averages.setToolTip("line averages: random noise falls as 1/sqrt(N); bleaching rises as N")
        form.addRow("line averages", self.averages)
        self.bidirectional = QtWidgets.QCheckBox("bidirectional scan (faster frames)")
        self.bidirectional.setChecked(lv.bidirectional)
        form.addRow("", self.bidirectional)
        outer.addWidget(box)

        # --- detector -----------------------------------------------------------
        box, form = self._group("detector")
        self.detector = _combo(DETECTORS, lv.detector)
        form.addRow(self.detector)
        self.detector_info = QtWidgets.QLabel("")
        self.detector_info.setWordWrap(True)
        self.detector_info.setStyleSheet(f"color: {style.MUTED}; font-size: 11px;")
        form.addRow(self.detector_info)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["analog", "counting"])
        self.mode.setCurrentText(lv.acquisition_mode)
        form.addRow("mode", self.mode)
        self.gain = LeverSlider(0.05, 500, lv.gain_adu_per_e, log=True, decimals=2, suffix=" ADU/e⁻")
        self.gain.setToolTip("detector gain (HV), as ADU per photoelectron: too low loses photons in the "
                             "electronics, too high clips")
        form.addRow("gain (HV)", self.gain)
        self.offset = LeverSlider(-100, 500, lv.offset_adu, decimals=0, suffix=" ADU")
        self.offset.setToolTip("below zero, the background clips black and every mean is biased")
        form.addRow("offset", self.offset)
        self.bit_depth = QtWidgets.QComboBox()
        self.bit_depth.addItems(["8", "12", "16"])
        self.bit_depth.setCurrentText(str(lv.bit_depth))
        form.addRow("bit depth", self.bit_depth)
        self.temperature = LeverSlider(-30, 45, lv.sensor_temperature_c, decimals=0, suffix=" °C")
        form.addRow("detector temp.", self.temperature)
        self.dead_time = QtWidgets.QCheckBox("dead-time correction")
        self.dead_time.setChecked(lv.dead_time_correction)
        form.addRow("counting", self.dead_time)
        outer.addWidget(box)

        # --- noise sources ----------------------------------------------------
        box, form = self._group("noise sources (untick to remove one)")
        self.sources: dict[str, QtWidgets.QCheckBox] = {}
        for f in dataclasses.fields(ScanNoise):
            check = QtWidgets.QCheckBox(ScanNoise.LABELS[f.name])
            check.setChecked(getattr(lv.noise, f.name))
            color = style.TERM_COLORS.get(TERM_OF.get(f.name, ""))
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
        self.reseed.clicked.connect(lambda _c=False: self._set(specimen_seed=self.levers.specimen_seed + 1))
        self.density.valueChanged.connect(lambda v: self._set(label_density=v))
        self.fluorophore.currentTextChanged.connect(lambda t: self._set(fluorophore=t))
        self.power.valueChanged.connect(lambda v: self._set(laser_power_uw=v))
        self.rin.valueChanged.connect(lambda v: self._set(laser_rin=v / 100.0))
        self.haze.valueChanged.connect(lambda v: self._set(background_per_um2=v))
        self.objective.currentIndexChanged.connect(self._objective_chosen)
        self.magnification.valueChanged.connect(lambda v: self._set(magnification=v, _custom=True))
        self.na.valueChanged.connect(lambda v: self._set(na=v, _custom=True))
        self.medium.currentTextChanged.connect(lambda t: self._set(medium=t, _custom=True))
        self.follow.toggled.connect(self._follow_toggled)
        self.wavelength.valueChanged.connect(lambda v: self._set(wavelength_nm=v))
        self.pinhole.valueChanged.connect(lambda v: self._set(pinhole_au=v))
        self.transmission.valueChanged.connect(lambda v: self._set(transmission=v / 100.0))
        self.zoom.valueChanged.connect(lambda v: self._set(zoom=v))
        self.pixels_x.currentTextChanged.connect(lambda t: self._set(pixels_x=int(t)))
        self.pixels_y.currentTextChanged.connect(lambda t: self._set(pixels_y=int(t)))
        self.dwell.valueChanged.connect(lambda v: self._set(dwell_us=v))
        self.averages.valueChanged.connect(lambda v: self._set(averages=int(v)))
        self.bidirectional.toggled.connect(lambda on: self._set(bidirectional=on))
        self.detector.currentTextChanged.connect(self._detector_changed)
        self.mode.currentTextChanged.connect(lambda t: self._set(mode=t))
        self.gain.valueChanged.connect(lambda v: self._set(gain_adu_per_e=v))
        self.offset.valueChanged.connect(lambda v: self._set(offset_adu=v))
        self.bit_depth.currentTextChanged.connect(lambda t: self._set(bit_depth=int(t)))
        self.temperature.valueChanged.connect(lambda v: self._set(temperature_c=v))
        self.dead_time.toggled.connect(lambda on: self._set(dead_time_correction=on))
        for name, check in self.sources.items():
            check.toggled.connect(lambda on, name=name: self._set(noise=dataclasses.replace(self.levers.noise, **{name: on})))

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

    def _detector_changed(self, name: str) -> None:
        det = DETECTORS[name]
        self._updating = True
        self.temperature.set_value(det.default_temperature_c)
        self._updating = False
        self._set(detector=name, temperature_c=None)

    def _sync_dependent(self) -> None:
        lv = self.levers
        spec = SPECIMENS[lv.specimen]
        self.density_label.setText("labels/µm" if spec.kind == "filament" else "labels/bead")
        det = lv.detector_model
        self.detector_info.setText(
            f"QE {det.quantum_efficiency:.0%}, F² {det.excess_noise_f2:g} (analog), dark "
            f"{det.dark_cps:g} cps at {det.dark_ref_c:g} °C, dead time {det.dead_time_ns:g} ns; "
            f"modes: {', '.join(det.modes)}"
        )
        # A detector offers only the modes it has; show the one actually in use.
        model = self.mode.model()
        for i in range(self.mode.count()):
            item = model.item(i)
            item.setEnabled(self.mode.itemText(i) in det.modes)
        if self.mode.currentText() != lv.acquisition_mode:
            self.mode.blockSignals(True)
            self.mode.setCurrentText(lv.acquisition_mode)
            self.mode.blockSignals(False)
        analog = lv.acquisition_mode == "analog"
        self.gain.setEnabled(analog)
        self.offset.setEnabled(analog)
        self.dead_time.setEnabled(not analog)
        self.temperature.set_range(det.min_temperature_c, 45.0)
        self.sampling.setText(
            f"field {lv.field_um:.1f} µm; pixels {lv.pixel_x_um * 1000:.0f} (x) × "
            f"{lv.pixel_y_um * 1000:.0f} (y) nm; frame {lv.frame_time_s():.2g} s"
        )
        n = IMMERSION_MEDIA[lv.medium]
        if lv.na > n:
            self._updating = True
            self.na.set_value(round(n * 0.99, 2))
            self._updating = False
            self.levers = self.levers.with_(na=round(n * 0.99, 2))
        if self.follow.isChecked():
            self._updating = True
            self.wavelength.set_value(lv.fluor.emission_nm)
            self._updating = False

    def set_levers(self, levers: ScanLevers) -> None:
        self._updating = True
        self.levers = levers
        self.specimen.setCurrentText(levers.specimen)
        self.density.set_value(levers.density)
        self.fluorophore.setCurrentText(levers.fluorophore)
        self.power.set_value(levers.laser_power_uw)
        self.rin.set_value(levers.laser_rin * 100)
        self.haze.set_value(levers.background_per_um2)
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
        self.pinhole.set_value(levers.pinhole_au)
        self.transmission.set_value(levers.transmission * 100)
        self.zoom.set_value(levers.zoom)
        self.pixels_x.setCurrentText(str(levers.pixels_x))
        self.pixels_y.setCurrentText(str(levers.pixels_y))
        self.dwell.set_value(levers.dwell_us)
        self.averages.setValue(levers.averages)
        self.bidirectional.setChecked(levers.bidirectional)
        self.detector.setCurrentText(levers.detector)
        self.mode.setCurrentText(levers.acquisition_mode)
        self.gain.set_value(levers.gain_adu_per_e)
        self.offset.set_value(levers.offset_adu)
        self.bit_depth.setCurrentText(str(levers.bit_depth))
        self.temperature.set_range(levers.detector_model.min_temperature_c, 45.0)
        self.temperature.set_value(levers.sensor_temperature_c)
        self.dead_time.setChecked(levers.dead_time_correction)
        for name, check in self.sources.items():
            check.setChecked(getattr(levers.noise, name))
        self._updating = False
        self._sync_dependent()
        self.leversChanged.emit(self.levers)
