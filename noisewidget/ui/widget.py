"""The noise widget: levers, images, the live noise graph and the quantitation.

:class:`NoiseWidget` is a plain ``QWidget`` so MicroscopeWidget can drop it into a
tab or a dock as-is; :class:`NoiseWindow` wraps it with the toolbar and status bar
for standalone use, in MicroscopeVuilder's workspace style.

Two clocks, as in the Vuilder workspace:

* **Lever changes** are debounced (40 ms) and recompute only what they touch. The
  object-to-image FFT -- the only expensive step, ~0.1 s -- is cached on NA,
  wavelength and the specimen; exposure, gain, camera and noise switches rerun in
  a few milliseconds, so the sliders are live.
* **Live mode** re-rolls the noise on a timer, as a camera streaming frames would,
  so the user sees noise as something that moves rather than a texture.

Everything is computed on the GUI thread. At these costs a worker thread would add
more latency (and lifetime hazards) than it removes.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..physics.base import Report, Status
from ..physics.quantitation import QuantSettings, Quantitation, quantify, reference_window
from . import style
from .modality import CAMERA, MODALITIES, Modality
from .panes import LUTS, ImageGrid, to_qimage
from .plots import Band, BudgetBar, Marker, PlotCanvas, PlotSpec, Series

CONTRAST_MODES = ("shared with the noise-free image", "auto per pane", "full ADC range")


# --- the live noise graph -------------------------------------------------------


class NoiseGraph(QtWidgets.QWidget):
    """Noise terms and SNR against signal, exposure or excitation.

    The top plot is the photon-transfer view: each noise term in electrons rms,
    so the read-noise floor, the sqrt(S) shot-noise slope and the linear PRNU
    slope are all visible at once, with the operating point marked. The bottom
    plot is what those terms mean for the image: SNR, against what a perfect
    detector would achieve with the same photons.
    """

    def __init__(self, modality: Modality = CAMERA, parent=None):
        super().__init__(parent)
        self.modality = modality
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("x axis:"))
        self.mode = QtWidgets.QComboBox()
        for key, label in modality.sweeps.items():
            self.mode.addItem(label, key)
        row.addWidget(self.mode)
        row.addStretch(1)
        self.headline = QtWidgets.QLabel("")
        self.headline.setStyleSheet("font-weight: 600;")
        row.addWidget(self.headline)
        layout.addLayout(row)
        self.noise_plot = PlotCanvas()
        self.snr_plot = PlotCanvas()
        self.bar = BudgetBar()
        layout.addWidget(self.noise_plot, 3)
        layout.addWidget(self.snr_plot, 2)
        layout.addWidget(self.bar)
        self.mode.currentIndexChanged.connect(lambda _i: self.redraw())
        self._sim = None

    def show_simulation(self, sim) -> None:
        self._sim = sim
        self.redraw()

    def redraw(self) -> None:
        sim = self._sim
        if sim is None:
            return
        m = self.modality
        mode = self.mode.currentData()
        here = m.operating_value(sim, mode)
        xs, xlabel, unit = m.sweep_axis(mode)
        budgets = m.budget_sweep(sim, mode, xs)
        terms, labels = budgets[0].TERMS, budgets[0].LABELS
        series = []
        for t in terms:
            y = np.array([math.sqrt(getattr(b, t)) for b in budgets])
            if not (y > 0).any():
                continue
            series.append(Series(xs, y, style.TERM_COLORS[t], labels[t], 2.0,
                                 dashed=(t == "quantization"), direct_label=False))
        total = np.array([b.noise_e for b in budgets])
        series.append(Series(xs, total, style.TERM_COLORS["total"], "total noise", 3.0, direct_label=True))
        if mode != "signal":
            sig = np.array([b.signal_e for b in budgets])
            series.insert(0, Series(xs, sig, "#5c6370", "signal", 1.5, dashed=True, direct_label=True))

        # Shade where read noise outweighs every shot term: the read-noise limit.
        shot_terms = ("photon_shot", "background_shot", "dark_shot", "cic")
        shot = np.array([sum(getattr(b, t, 0.0) for t in shot_terms) for b in budgets])
        read = np.array([sum(getattr(b, t) for t in m.electronics_terms) for b in budgets])
        bands = []
        limited = read > shot
        if limited.any():
            idx = np.nonzero(limited)[0]
            bands.append(Band(xs[idx[0]], xs[idx[-1]], style.TERM_COLORS["read"], m.electronics_label))
        markers = [Marker(here, "#e6edf3", "you are here")]
        if mode == "signal" and sim.background_e > 0:
            markers.append(Marker(sim.background_e, style.TERM_COLORS["background_shot"], "background"))

        ymin = max(min((s.y[s.y > 0].min() for s in series if (s.y > 0).any()), default=1e-2), 1e-2)
        ymax = max(s.y.max() for s in series)
        self.noise_plot.set_spec(PlotSpec(
            title=m.noise_title,
            xlabel=xlabel, ylabel="noise (e- rms)", xlog=True, ylog=True,
            xrange=(xs[0], xs[-1]), yrange=(ymin * 0.8, ymax * 1.3),
            series=series, markers=markers, bands=bands, x_unit=unit, y_unit="e-", legend_left=True,
        ))

        snr = np.array([b.snr for b in budgets])
        ideal = np.array([b.shot_limited_snr for b in budgets])
        self.snr_plot.set_spec(PlotSpec(
            title="signal-to-noise at the structure",
            xlabel=xlabel, ylabel="SNR", xlog=True, ylog=True,
            xrange=(xs[0], xs[-1]), yrange=(0.05, max(ideal.max(), snr.max()) * 1.5),
            series=[
                Series(xs, ideal, "#5c6370", "perfect detector (sqrt N)", 2.0, dashed=True),
                Series(xs, snr, style.TERM_COLORS["total"], "this system", 3.0),
            ],
            markers=[Marker(here, "#e6edf3", ""), Marker(5.0, style.STATUS_COLORS[Status.WARN], "Rose criterion, SNR 5", vertical=False)],
            bands=bands, x_unit=unit, legend_left=True,
        ))

        b = m.simulation_budget(sim)
        self.bar.set_budget(b.fractions(), b.LABELS, f"variance at the structure: S = {b.signal_e:.3g} e-, noise {b.noise_e:.3g} e-")
        self.headline.setText(f"SNR {b.snr:.3g}  ·  dominant: {b.LABELS[b.dominant()]}")


# --- quantitation ----------------------------------------------------------------


class QuantPanel(QtWidgets.QWidget):
    """The measurement of the reference structure, and how noise spreads it."""

    settingsChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("repeat frames:"))
        self.realizations = QtWidgets.QSpinBox()
        self.realizations.setRange(10, 2000)
        self.realizations.setValue(100)
        self.realizations.setToolTip("how many times the same field is re-recorded to measure the spread")
        row.addWidget(self.realizations)
        row.addWidget(QtWidgets.QLabel("  profile length:"))
        self.length = QtWidgets.QDoubleSpinBox()
        self.length.setRange(0.1, 6.0)
        self.length.setValue(2.0)
        self.length.setSingleStep(0.5)
        self.length.setSuffix(" µm")
        self.length.setToolTip("how much filament is averaged into the profile: longer is less noisy")
        row.addWidget(self.length)
        row.addStretch(1)
        layout.addLayout(row)

        self.profile_plot = PlotCanvas()
        self.hist_plot = PlotCanvas()
        self.count_plot = PlotCanvas()
        plots = QtWidgets.QHBoxLayout()
        plots.addWidget(self.hist_plot)
        plots.addWidget(self.count_plot)
        layout.addWidget(self.profile_plot, 3)
        layout.addLayout(plots, 2)

        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["measurement", "value"])
        self.table.setColumnWidth(0, 190)
        self.table.setRootIsDecorated(True)
        layout.addWidget(self.table, 3)
        self.realizations.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.length.valueChanged.connect(lambda _v: self.settingsChanged.emit())

    def settings(self, noise_seed: int = 0) -> QuantSettings:
        return QuantSettings(self.length.value(), self.realizations.value())

    def show_quantitation(self, q: Quantitation, report: Report) -> None:
        nm = 1000.0
        x = q.x_um * nm
        # The object, scaled to the noise-free peak so its narrowness reads at a glance.
        obj = q.object_profile
        scale = (q.expected.max() - q.ideal.background) / obj.max() if obj.max() > 0 else 0.0
        obj_y = obj * scale + q.ideal.background
        fine = np.linspace(x[0], x[-1], 300)
        series = [
            Series(q.object_x_um * nm, obj_y, "#5c6370", "object (labels, scaled)", 1.5, kind="step"),
            Series(x, q.expected, style.TERM_COLORS["photon_shot"], "noise-free image", 2.0),
            Series(x, q.measured, "#e6edf3", "this frame", 3.0, kind="points"),
        ]
        if q.fit.ok:
            series.append(Series(fine, q.fit.curve(fine / nm), style.TERM_COLORS["read"],
                                 f"Gaussian fit, FWHM {q.fit.fwhm_um * nm:.0f} nm", 2.0))
        across = "across the filament" if q.kind == "filament" else "through the bead (summed)"
        self.profile_plot.set_spec(PlotSpec(
            title=f"line profile {across}",
            xlabel="position (nm)", ylabel="e- per pixel" if q.kind == "filament" else "e- per row",
            series=series, x_unit="nm", y_unit="e-",
        ))

        self._histogram(
            self.hist_plot, q.fwhm.values * nm, f"width over frames (true {q.true_diameter_nm:g} nm)",
            "FWHM (nm)",
            [Marker(q.ideal.fwhm_um * nm, style.TERM_COLORS["photon_shot"], "noise-free")],
            "nm",
        )
        self._histogram(
            self.count_plot, q.count.values, "label count over repeat frames",
            "labels per µm" if q.kind == "filament" else "labels per bead",
            [Marker(q.labels_present, "#e6edf3", "true")],
            "",
        )

        self.table.clear()
        for r in report.rows:
            node = QtWidgets.QTreeWidgetItem([r.name, r.summary])
            node.setForeground(0, QtGui.QColor(style.STATUS_COLORS[r.status]))
            node.setToolTip(1, r.summary)
            if r.equation:
                child = QtWidgets.QTreeWidgetItem(["", r.equation])
                child.setToolTip(1, r.equation)
                node.addChild(child)
            if r.remedy and r.status in (Status.WARN, Status.FAIL):
                remedy = QtWidgets.QTreeWidgetItem(["", f"remedy: {r.remedy}"])
                remedy.setForeground(1, QtGui.QColor(style.STATUS_COLORS[Status.WARN]))
                node.addChild(remedy)
            self.table.addTopLevelItem(node)

    @staticmethod
    def _histogram(plot: PlotCanvas, values: np.ndarray, title: str, xlabel: str, markers, unit: str) -> None:
        values = values[np.isfinite(values)]
        if values.size < 2:
            plot.set_spec(PlotSpec(title=title + " -- no successful fits", xlabel=xlabel))
            return
        lo = min(values.min(), *(m.value for m in markers))
        hi = max(values.max(), *(m.value for m in markers))
        pad = (hi - lo) * 0.08 or abs(hi) * 0.05 or 1.0
        bins = np.linspace(lo - pad, hi + pad, 31)
        counts, edges = np.histogram(values, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        plot.set_spec(PlotSpec(
            title=title, xlabel=xlabel, ylabel="frames",
            xrange=(edges[0], edges[-1]),
            series=[Series(centres, counts.astype(float), style.TERM_COLORS["cic"], "", kind="bars")],
            markers=markers, legend=False, x_unit=unit,
        ))


class ReportPanel(QtWidgets.QTreeWidget):
    """Every derived number, with its equation: the Vuilder scorecard, for noise."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["quantity", "value"])
        self.setColumnWidth(0, 210)

    def show_report(self, sections: dict[str, Report]) -> None:
        self.clear()
        for name, report in sections.items():
            head = QtWidgets.QTreeWidgetItem([name, ""])
            font = head.font(0)
            font.setBold(True)
            head.setFont(0, font)
            self.addTopLevelItem(head)
            for r in report.rows:
                node = QtWidgets.QTreeWidgetItem([r.name, r.summary])
                node.setForeground(0, QtGui.QColor(style.STATUS_COLORS[r.status]))
                node.setToolTip(1, r.summary)
                if r.equation:
                    child = QtWidgets.QTreeWidgetItem(["", r.equation])
                    child.setToolTip(1, r.equation)
                    node.addChild(child)
                if r.remedy and r.status in (Status.WARN, Status.FAIL):
                    remedy = QtWidgets.QTreeWidgetItem(["", f"remedy: {r.remedy}"])
                    remedy.setForeground(1, QtGui.QColor(style.STATUS_COLORS[Status.WARN]))
                    node.addChild(remedy)
                head.addChild(node)
                node.setExpanded(r.status in (Status.WARN, Status.FAIL))
            head.setExpanded(True)


# --- the widget ------------------------------------------------------------------


class NoiseWidget(QtWidgets.QWidget):
    """Embeddable: levers on the left, images over the noise graph, measurements right."""

    simulated = QtCore.Signal(object)  # the Simulation, after every recompute
    summaryChanged = QtCore.Signal(str)

    def __init__(self, levers=None, parent=None, modality: Modality = CAMERA):
        super().__init__(parent)
        self.modality = modality
        self.simulator = modality.make_simulator()
        self.sim = None
        self.quant: Quantitation | None = None
        self.noise_seed = 0
        self.lut = "grey"
        self.contrast = CONTRAST_MODES[0]
        self.show_window = True
        self.show_saturation = True

        self.levers_panel = modality.make_panel(levers)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.levers_panel)
        scroll.setMinimumWidth(330)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.images = ImageGrid()
        self.images.recorded.title = modality.recorded_title
        self.graph = NoiseGraph(modality)
        self.quant_panel = QuantPanel()
        self.report_panel = ReportPanel()
        self.problem = QtWidgets.QLabel("")
        self.problem.setStyleSheet(style.BANNER)
        self.problem.setWordWrap(True)
        self.problem.setVisible(False)

        centre = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.problem)
        top_layout.addWidget(self.images, 1)
        centre.addWidget(top)
        centre.addWidget(self.graph)
        centre.setSizes([280, 560])

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.quant_panel, "quantitation")
        self.tabs.addTab(self.report_panel, "the numbers")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(centre)
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 760, 480])
        splitter.setStretchFactor(1, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self.refresh)
        self.live_timer = QtCore.QTimer(self)
        self.live_timer.setInterval(200)
        self.live_timer.timeout.connect(self.new_noise)

        self.levers_panel.leversChanged.connect(lambda _lv: self._debounce.start())
        self.quant_panel.settingsChanged.connect(self._debounce.start)
        self.refresh()

    # --- state --------------------------------------------------------------

    @property
    def levers(self):
        return self.levers_panel.levers.with_(noise_seed=self.noise_seed)

    def set_levers(self, levers) -> None:
        self.levers_panel.set_levers(levers)
        self.refresh()

    def load_preset(self, name: str) -> None:
        self.set_levers(self.modality.presets[name].levers)

    def new_noise(self) -> None:
        """Record another frame of the same scene: new noise, same everything else."""
        self.noise_seed += 1
        self.refresh()

    def set_live(self, on: bool) -> None:
        if on:
            self.live_timer.start()
        else:
            self.live_timer.stop()

    def set_display(self, lut: str | None = None, contrast: str | None = None,
                    show_window: bool | None = None, show_saturation: bool | None = None) -> None:
        if lut is not None:
            self.lut = lut
        if contrast is not None:
            self.contrast = contrast
        if show_window is not None:
            self.show_window = show_window
        if show_saturation is not None:
            self.show_saturation = show_saturation
        if self.sim is not None:
            self._draw_images(self.sim, self.quant)

    # --- the loop -----------------------------------------------------------

    def refresh(self) -> None:
        levers = self.levers
        try:
            sim = self.simulator.run(levers)
        except ValueError as exc:
            self.problem.setText(f"cannot image this: {exc}")
            self.problem.setVisible(True)
            return
        self.problem.setVisible(False)
        q = quantify(sim, self.quant_panel.settings())
        self.sim, self.quant = sim, q
        self._draw_images(sim, q)
        self.graph.show_simulation(sim)
        sections = self.modality.build_report(sim, q)
        self.quant_panel.show_quantitation(q, sections["quantitation"])
        self.report_panel.show_report(sections)
        self.simulated.emit(sim)
        b = self.modality.simulation_budget(sim)
        px, py = sim.pixel_x_um * 1000, sim.pixel_y_um * 1000
        bits = [
            f"pixel {px:.0f} nm" if abs(px - py) < 0.5 else f"pixel {px:.0f} x {py:.0f} nm",
            f"{self.modality.peak_signal(sim):.3g} e- at the structure",
            f"SNR {b.snr:.3g}",
            f"width {q.fit.fwhm_um * 1000:.0f} nm (true {q.true_diameter_nm:g})" if q.fit.ok else "width: no fit",
        ]
        bits += sim.notes
        problems = [r.name for rep in sections.values() for r in rep.rows if r.status is Status.FAIL]
        if problems:
            bits.append("check: " + ", ".join(problems))
        self.summaryChanged.emit("   |   ".join(bits))

    def _draw_images(self, sim, q) -> None:
        info = self.modality.image_info(sim)
        nm = info.wavelength_nm
        fx, fy = info.extent_um
        origin = info.origin_um

        # The object: block-summed to <= 256 px so 25 nm filaments survive display.
        obj = sim.specimen.density
        k = max(obj.shape[0] // 256, 1)
        if k > 1:
            m = obj.shape[0] // k
            obj = obj[: m * k, : m * k].reshape(m, k, m, k).sum(axis=(1, 3))
        # Sparse: most pixels are empty, so scale to the labelled ones.
        labelled = obj[obj > 0]
        hi = float(np.percentile(labelled, 90)) if labelled.size else 1.0
        self.images.object.show_image(
            to_qimage(obj, 0.0, hi, self.lut, nm), sim.specimen.field_um,
            f"{sim.specimen.dx_um * k * 1000:.0f} nm px · {sim.specimen.total:,.0f} labels",
        )

        expected, recorded = sim.expected_e, sim.recorded_e
        if self.contrast == CONTRAST_MODES[2]:
            lo_e, hi_e = info.full_scale
            lo_r, hi_r = lo_e, hi_e
        elif self.contrast == CONTRAST_MODES[1]:
            lo_e, hi_e = (float(v) for v in np.percentile(expected, (0.1, 99.5)))
            lo_r, hi_r = (float(v) for v in np.percentile(recorded, (0.1, 99.5)))
        else:
            lo_e, hi_e = (float(v) for v in np.percentile(expected, (0.1, 99.5)))
            lo_r, hi_r = lo_e, hi_e
        sat = sim.saturated if self.show_saturation else None
        self.images.optical.show_image(
            to_qimage(expected, lo_e, hi_e, self.lut, nm), fx, info.optical_caption, fy, origin,
        )
        self.images.recorded.show_image(
            to_qimage(recorded, lo_r, hi_r, self.lut, nm, saturated=sat), fx,
            info.recorded_caption + (" · red = clipped" if sat is not None and sat.any() else ""), fy, origin,
        )
        noise = recorded - expected
        sd = float(noise.std()) or 1.0
        self.images.noise.show_image(
            to_qimage(noise, -3 * sd, 3 * sd, diverging=True), fx,
            f"rms {sd:.3g} e- · blue below, orange above", fy, origin,
        )

        rect = None
        if self.show_window:
            r0, r1, c0, c1 = reference_window(sim, self.quant_panel.settings())
            px, py = sim.pixel_x_um, sim.pixel_y_um
            x0 = sim.col_centres_um[c0] - px / 2
            y0 = sim.row_centres_um[r0] - py / 2
            rect = QtCore.QRectF(x0, y0, (c1 - c0) * px, (r1 - r0) * py)
        for pane in self.images.panes():
            pane.set_window(rect)
        # On the object, outline what was imaged when it is less than the specimen.
        spec_field = sim.specimen.field_um
        imaged = QtCore.QRectF(origin[0], origin[1], fx, fy)
        covers = imaged.width() >= spec_field * 0.95 and imaged.height() >= spec_field * 0.95
        self.images.object.region_um = None if covers else imaged
        self.images.object.update()


# --- the window ------------------------------------------------------------------


class NoiseWindow(QtWidgets.QMainWindow):
    """Two frames, one per instrument, sharing a toolbar: camera and confocal."""

    def __init__(self, levers=None, parent=None, frame: str = "camera"):
        super().__init__(parent)
        self.setWindowTitle("NoiseWidget -- where the noise in a micrograph comes from")
        self.frames = QtWidgets.QTabWidget()
        self.frames.setDocumentMode(True)
        self.widgets: dict[str, NoiseWidget] = {}
        for key, modality in MODALITIES.items():
            start = levers if (levers is not None and isinstance(levers, type(modality.default_levers()))) else None
            w = NoiseWidget(start, modality=modality)
            w.summaryChanged.connect(lambda text, w=w: self._summary(w, text))
            self.widgets[key] = w
            self.frames.addTab(w, modality.title)
        self.setCentralWidget(self.frames)
        self._build_toolbar()
        self.frames.currentChanged.connect(self._frame_changed)
        if levers is not None:
            frame = next(k for k, m in MODALITIES.items() if isinstance(levers, type(m.default_levers())))
        self.frames.setCurrentWidget(self.widgets[frame])
        self._frame_changed()
        self._warn_about_presets()
        self.resize(1600, 960)

    @property
    def widget(self) -> NoiseWidget:
        """The frame in front."""
        return self.frames.currentWidget()

    def _summary(self, w: NoiseWidget, text: str) -> None:
        if w is self.widget:
            self.statusBar().showMessage(text)

    def _frame_changed(self, _index: int = 0) -> None:
        for w in self.widgets.values():
            w.set_live(w is self.widget and self.live.isChecked())
        self.presets.blockSignals(True)
        self.presets.clear()
        self.presets.addItem("--")
        for name in self.widget.modality.presets:
            self.presets.addItem(name)
        self.presets.blockSignals(False)
        self.widget.refresh()

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("main")
        bar.setMovable(False)

        new = QtGui.QAction("New frame", self)
        new.setShortcut("Ctrl+R")
        new.setToolTip("record the same scene again: same light, same specimen, new noise")
        # QAction.triggered carries a `checked` bool; swallow it.
        new.triggered.connect(lambda _checked=False: self.widget.new_noise())  # the frame in front
        bar.addAction(new)

        self.live = QtGui.QAction("Live", self, checkable=True, checked=False)
        self.live.setShortcut("Space")
        self.live.setToolTip("stream frames, as the camera would: noise moves, signal stays")
        self.live.toggled.connect(lambda on: self.widget.set_live(on))
        bar.addAction(self.live)

        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel("  preset: "))
        self.presets = QtWidgets.QComboBox()
        self.presets.addItem("--")
        self.presets.setToolTip("starting points that each demonstrate one thing")
        self.presets.currentTextChanged.connect(self._preset_chosen)
        bar.addWidget(self.presets)

        reset = QtGui.QAction("Reset", self)
        reset.triggered.connect(lambda _checked=False: self.widget.set_levers(self.widget.modality.default_levers()))
        bar.addAction(reset)

        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel("  LUT: "))
        self.lut = QtWidgets.QComboBox()
        self.lut.addItems(list(LUTS))
        self.lut.currentTextChanged.connect(lambda t: self._display(lut=t))
        bar.addWidget(self.lut)
        bar.addWidget(QtWidgets.QLabel("  contrast: "))
        self.contrast = QtWidgets.QComboBox()
        self.contrast.addItems(list(CONTRAST_MODES))
        self.contrast.setToolTip(
            "shared: the recording uses the noise-free image's display range, so noise "
            "looks as large as it is. Auto rescales each pane, which flatters a dim image."
        )
        self.contrast.currentTextChanged.connect(lambda t: self._display(contrast=t))
        bar.addWidget(self.contrast)
        window = QtGui.QAction("Measured region", self, checkable=True, checked=True)
        window.toggled.connect(lambda on: self._display(show_window=on))
        bar.addAction(window)
        sat = QtGui.QAction("Show clipping", self, checkable=True, checked=True)
        sat.toggled.connect(lambda on: self._display(show_saturation=on))
        bar.addAction(sat)

        bar.addSeparator()
        help_action = QtGui.QAction("How this works", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(lambda _checked=False: self.show_help())
        bar.addAction(help_action)

    def _display(self, **settings) -> None:
        """Display settings apply to both frames, so switching keeps the look."""
        for w in self.widgets.values():
            w.set_display(**settings)

    def _preset_chosen(self, name: str) -> None:
        presets = self.widget.modality.presets
        if name in presets:
            self.widget.load_preset(name)
            self.statusBar().showMessage(presets[name].description, 8000)

    def _warn_about_presets(self) -> None:
        """Say where the camera and dye numbers come from, where it will be seen."""
        banner = QtWidgets.QLabel(
            "Camera, detector and dye figures are typical-of-class, not vendor datasheets; "
            "bleaching budgets are orders of magnitude."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(style.BANNER)
        self.statusBar().addPermanentWidget(banner, 0)

    def show_help(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("How the noise widget works")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText("<b>Turn a lever, watch the image, the graph and the measurement follow.</b>")
        box.setInformativeText(
            "<p><b>Two frames</b>: a camera on a widefield microscope, and a point-scanning "
            "confocal. The camera exposes every pixel at once; the confocal visits each pixel "
            "for a dwell time, through a pinhole, onto a single detector — so its levers are "
            "dwell, zoom, pixel format, pinhole, laser power and detector gain.</p>"
            "<p><b>Images</b>, left to right: the specimen as it really is (fluorophore "
            "positions on a 25 nm grid); the image the optics form, sampled by the camera "
            "pixels or the scan, with no noise; what is recorded; and the difference — the "
            "noise itself.</p>"
            "<p><b>The graph</b> plots every noise source in electrons against signal (or "
            "exposure, or excitation). Read noise is flat, shot noise rises as √S, fixed "
            "pattern rises as S. Where the white total meets the read-noise floor, you are "
            "read-noise limited (shaded). The bar shows where the variance goes at your "
            "settings.</p>"
            "<p><b>Quantitation</b> measures the structure at the centre (dashed box): width "
            "and label count from a line profile, repeated over many frames so the spread "
            "the noise causes is visible next to the truth.</p>"
            "<p><b>Noise sources</b> can be switched off one at a time. <b>New frame</b> "
            "(Ctrl+R) re-records the scene; <b>Live</b> (Space) streams frames.</p>"
            "<p><b>The numbers</b> tab shows every derived quantity with its equation and "
            "your values in it.</p>"
        )
        box.exec()

    def closeEvent(self, event) -> None:
        self.widget.set_live(False)
        super().closeEvent(event)


def launch(levers=None, frame: str = "camera") -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = NoiseWindow(levers, frame=frame)
    window.show()
    return app.exec()
