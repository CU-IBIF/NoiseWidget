"""The things that break only once the app is frozen."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_entry_point_defaults_to_opening_the_widget():
    # A double-clicked executable has no arguments, so it must land somewhere
    # useful rather than printing a report to a console nobody can see.
    source = (REPO / "packaging" / "entry.py").read_text(encoding="utf-8")
    assert "--ui" in source
    assert "freeze_support" in source


def test_the_spec_builds_one_file_named_noisewidget():
    spec = (REPO / "packaging" / "noisewidget.spec").read_text(encoding="utf-8")
    assert 'name="NoiseWidget"' in spec
    assert "COLLECT(" not in spec, "COLLECT produces a one-folder build"
    assert "noisewidget.ui.widget" in spec  # imported lazily by the CLI


def test_release_smoke_test_uses_this_apps_commands():
    import yaml

    text = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    spec = yaml.safe_load(text)
    assert set(spec["jobs"]) == {"build", "publish"}
    assert "--round" not in text  # a MicroscopeVuilder flag; this CLI has none
    assert "--list-presets" in text
