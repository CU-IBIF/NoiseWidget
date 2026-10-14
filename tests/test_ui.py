"""The widget builds, follows its levers, and shuts down cleanly -- headless."""

import pytest

pytest.importorskip("PySide6", reason="the widget needs the 'ui' extra")

from PySide6 import QtWidgets  # noqa: E402

from noisewidget.physics.acquisition import Levers  # noqa: E402
from noisewidget.physics.presets import PRESETS  # noqa: E402
from noisewidget.ui.controls import LeverPanel, LeverSlider  # noqa: E402
from noisewidget.ui.widget import CONTRAST_MODES, NoiseWidget, NoiseWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    windows = []

    def make(levers=None):
        w = NoiseWindow(levers)
        windows.append(w)
        return w

    yield make
    for w in windows:
        w.close()
        w.deleteLater()
    QtWidgets.QApplication.processEvents()


def test_window_builds_and_renders(window):
    w = window()
    w.show()
    QtWidgets.QApplication.processEvents()
    widget = w.widget
    assert widget.sim is not None and widget.quant is not None
    for pane in widget.images.panes():
        assert pane.image is not None
    assert w.grab().width() > 0
    assert "SNR" in w.statusBar().currentMessage()


def test_the_widget_embeds_without_a_window(app):
    # MicroscopeWidget will host it as a plain QWidget.
    widget = NoiseWidget()
    parent = QtWidgets.QTabWidget()
    parent.addTab(widget, "noise")
    assert widget.sim is not None
    parent.deleteLater()


def test_a_lever_reaches_the_simulation(window):
    w = window()
    widget = w.widget
    before = widget.sim.photometry.detected_e
    widget.levers_panel.exposure.set_value(10.0)
    widget.levers_panel.exposure.valueChanged.emit(10.0)
    widget.refresh()
    assert widget.sim.levers.exposure_ms == pytest.approx(10.0)
    assert widget.sim.photometry.detected_e == pytest.approx(before / 10, rel=0.02)


def test_objective_preset_sets_three_levers(app):
    panel = LeverPanel()
    seen = []
    panel.leversChanged.connect(seen.append)
    panel.objective.setCurrentText("20x / 0.75 air")
    lv = seen[-1]
    assert (lv.magnification, lv.na, lv.medium) == (20.0, 0.75, "air")


def test_na_is_pulled_below_the_immersion_index(app):
    panel = LeverPanel()
    panel.na.set_value(1.4)
    panel.medium.setCurrentText("air")
    assert panel.levers.na < 1.0


def test_untick_a_noise_source(app):
    panel = LeverPanel()
    panel.sources["read"].setChecked(False)
    assert panel.levers.noise.read is False


def test_presets_round_trip_through_the_panel(window):
    w = window()
    for name, preset in PRESETS.items():
        w.widget.load_preset(name)
        lv = w.widget.sim.levers
        assert lv.camera == preset.levers.camera, name
        assert lv.exposure_ms == pytest.approx(preset.levers.exposure_ms), name
        assert lv.magnification == pytest.approx(preset.levers.magnification), name
        assert lv.frames == preset.levers.frames, name


def test_new_frame_changes_noise_not_signal(window):
    w = window()
    widget = w.widget
    first = widget.sim.recorded_e.copy()
    expected = widget.sim.expected_e.copy()
    widget.new_noise()
    assert (widget.sim.recorded_e != first).any()
    assert (widget.sim.expected_e == expected).all()


def test_display_modes(window):
    w = window()
    for mode in CONTRAST_MODES:
        w.widget.set_display(contrast=mode, lut="fire")
    w.widget.set_display(show_window=False, show_saturation=False)
    assert all(p.window_um is None for p in w.widget.images.panes())


def test_graph_axes(window):
    w = window()
    graph = w.widget.graph
    for i in range(graph.mode.count()):
        graph.mode.setCurrentIndex(i)
        assert graph.noise_plot.spec.series


def test_live_mode_starts_and_stops(window):
    w = window()
    w.live.setChecked(True)
    assert w.widget.live_timer.isActive()
    w.live.setChecked(False)
    assert not w.widget.live_timer.isActive()


def test_log_slider_round_trips(app):
    s = LeverSlider(0.1, 10_000, 100.0, log=True)
    s.slider.setValue(s.slider.maximum())
    assert s.value() == pytest.approx(10_000)
    s.set_value(1.0)
    assert s.slider.value() == pytest.approx(s.STEPS * 0.2, abs=1)


def test_an_impossible_setting_shows_a_problem_not_a_crash(window):
    w = window()
    widget = w.widget
    widget.levers_panel.levers = Levers(na=1.45, medium="water")
    widget.refresh()
    assert not widget.problem.isHidden()
