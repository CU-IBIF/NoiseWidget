import pytest

from noisewidget.__main__ import main, parse_assignment
from noisewidget.physics.acquisition import Levers, Simulator
from noisewidget.physics.presets import PRESETS
from noisewidget.physics.quantitation import QuantSettings, quantify
from noisewidget.physics.report import build_report


def test_default_report_runs(capsys):
    assert main(["--realizations", "10"]) == 0
    out = capsys.readouterr().out
    for heading in ("-- optics --", "-- photons --", "-- noise --", "-- quantitation --"):
        assert heading in out


def test_set_parses_types_and_noise_switches():
    lv = parse_assignment(Levers(), "na=0.75")
    assert lv.na == 0.75
    lv = parse_assignment(lv, "medium=air")
    assert lv.medium == "air"
    lv = parse_assignment(lv, "binning=2")
    assert lv.binning == 2 and isinstance(lv.binning, int)
    lv = parse_assignment(lv, "read=off")
    assert lv.noise.read is False
    lv = parse_assignment(lv, "wavelength_nm=600")
    assert lv.wavelength_um == pytest.approx(0.6)


def test_unknown_lever_is_an_error():
    with pytest.raises(SystemExit):
        parse_assignment(Levers(), "nonsense=1")


def test_impossible_na_is_reported_not_raised(capsys):
    assert main(["--set", "medium=air", "--set", "na=1.4"]) == 2
    assert "exceeds" in capsys.readouterr().out


def test_unknown_preset(capsys):
    assert main(["--preset", "nope"]) == 2


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_runs_and_reports(name):
    sim = Simulator().run(PRESETS[name].levers)
    q = quantify(sim, QuantSettings(realizations=10))
    sections = build_report(sim, q)
    assert all(rep.rows for rep in sections.values())


def test_the_read_noise_preset_is_read_noise_affected_and_binning_helps():
    from noisewidget.physics.acquisition import simulation_budget

    dim = simulation_budget(Simulator().run(PRESETS["Read-noise limited (dim, short exposure)"].levers))
    binned = simulation_budget(Simulator().run(PRESETS["... rescued by 2x2 binning"].levers))
    emccd = simulation_budget(Simulator().run(PRESETS["... rescued by an EMCCD (gain 300)"].levers))
    assert dim.fractions()["read"] > 0.25
    assert binned.snr > dim.snr
    assert emccd.fractions()["read"] < 0.01


def test_confocal_frame_report(capsys):
    assert main(["--frame", "confocal", "--realizations", "10", "--set", "mode=counting"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[confocal]")
    for heading in ("-- optics and scanning --", "-- detector and noise --", "-- quantitation --"):
        assert heading in out


def test_a_confocal_preset_selects_its_frame(capsys):
    assert main(["--preset", "Gain too low", "--realizations", "10"]) == 0
    assert "[confocal]" in capsys.readouterr().out


def test_set_works_on_confocal_levers():
    from noisewidget.physics.confocal import ScanLevers

    lv = parse_assignment(ScanLevers(), "pixels_y=128")
    assert lv.pixels_y == 128
    lv = parse_assignment(lv, "dead_time=off")
    assert lv.noise.dead_time is False
