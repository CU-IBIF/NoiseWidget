"""Ground truth: the specimens contain what they say they contain."""

import numpy as np
import pytest

from noisewidget.physics.specimens import SPECIMENS, make_specimen, splat


def test_splat_conserves_weight():
    x = np.array([0.1, 0.5126, 1.0])
    y = np.array([0.3, 0.7, 1.2])
    grid = splat(64, 0.025, x, y, np.array([1.0, 2.0, 3.0]))
    assert grid.sum() == pytest.approx(6.0)


def test_same_seed_same_specimen():
    a = make_specimen(seed=3)
    b = make_specimen(seed=3)
    assert np.array_equal(a.density, b.density)
    assert not np.array_equal(a.density, make_specimen(seed=4).density)


def test_reference_filament_carries_the_stated_density():
    s = make_specimen(density=200.0, stochastic=False)
    cx, cy = s.reference.centre_um
    count = s.reference.count_within(cx - 1, cx + 1, cy - 0.5, cy + 0.5)
    assert count / 2.0 == pytest.approx(200.0, rel=0.01)


def test_stochastic_labelling_is_poisson_around_the_density():
    counts = []
    for seed in range(30):
        s = make_specimen(density=100.0, seed=seed)
        cx, cy = s.reference.centre_um
        counts.append(s.reference.count_within(cx - 2, cx + 2, cy - 0.5, cy + 0.5))
    counts = np.array(counts)
    assert counts.mean() == pytest.approx(400, rel=0.05)
    assert counts.std() == pytest.approx(20, rel=0.4)  # sqrt(400)


def test_label_positions_sit_on_the_wall():
    s = make_specimen("Microtubules, immunostained (25 nm, ~60 nm with IgG)")
    cy = s.reference.centre_um[1]
    offsets = np.abs(s.reference.y - cy)
    assert offsets.max() <= 0.030 + 1e-9
    assert np.std(s.reference.y - cy) * 1000 == pytest.approx(
        SPECIMENS["Microtubules, immunostained (25 nm, ~60 nm with IgG)"].label_sigma_nm, rel=0.1
    )


def test_the_reference_is_isolated():
    # Nothing else may sit in the clear box around the measured segment, or the
    # line profile would measure a crossing instead of one microtubule.
    s = make_specimen()
    cx, cy = s.reference.centre_um
    n = s.density.shape[0]
    centres = (np.arange(n) + 0.5) * s.dx_um
    rows = np.abs(centres - cy) < 1.7
    cols = np.abs(centres - cx) < 3.7
    box = s.density[np.ix_(rows, cols)].sum()
    ref_in_box = s.reference.count_within(cx - 3.7, cx + 3.7, cy - 1.7, cy + 1.7)
    assert box == pytest.approx(ref_in_box, rel=0.02)


@pytest.mark.parametrize("name", list(SPECIMENS))
def test_every_specimen_builds(name):
    s = make_specimen(name)
    assert s.total > 0
    assert s.reference.weight.sum() > 0
