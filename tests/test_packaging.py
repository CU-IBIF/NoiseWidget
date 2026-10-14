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


def test_release_builds_all_four_platforms_on_runners_that_exist():
    # Same matrix as MicroscopeWidget. macos-13 was retired and never schedules, so
    # a release job pinned to it waits forever and nothing publishes.
    import yaml

    spec = yaml.safe_load((REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))
    include = spec["jobs"]["build"]["strategy"]["matrix"]["include"]
    assert {m["label"] for m in include} == {"linux-x86_64", "windows-x86_64", "macos-x86_64", "macos-arm64"}
    assert "macos-13" not in {m["os"] for m in include}
    assert all(m["asset"].startswith("NoiseWidget-") for m in include)


def test_pyside_is_not_a_release_that_aborts_on_setpen():
    """Guard the PySide6 exclusion, so a bump cannot silently re-break CI.

    PySide6 6.12.0 over-decrefs None inside setPen; on Python 3.11 enough redraws
    drive None's refcount to zero and the process dies with an abort (exit 134)
    that names no test. Every pane here calls setPen on every paint, and live mode
    paints five times a second. If this fails, verify the bug is fixed in the
    installed version before widening the pin in pyproject.toml.
    """
    import pytest

    pyside6 = pytest.importorskip("PySide6")
    assert pyside6.__version__ not in {"6.12.0"}, (
        f"PySide6 {pyside6.__version__} is excluded in pyproject.toml because setPen aborts the interpreter"
    )
