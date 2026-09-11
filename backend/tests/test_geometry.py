from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from surgicalvision.geometry import (
    clip_score,
    direction_reversals,
    interpolate_nans,
    path_efficiency,
    path_length,
    sequence_edit_distance,
    velocities,
)


def test_path_length_and_efficiency() -> None:
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    assert path_length(xy) == pytest.approx(7.0)
    assert path_efficiency(xy) == pytest.approx(5.0 / 7.0)


def test_interpolate_nans_fills_interior_only() -> None:
    xy = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 2.0], [np.nan, np.nan]])
    filled = interpolate_nans(xy)
    assert filled[1, 0] == pytest.approx(1.0)
    assert np.isnan(filled[3, 0])


def test_velocities_scale_with_fps() -> None:
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    speed = velocities(xy, fps=10.0)
    assert speed[1] == pytest.approx(100.0)


def test_direction_reversals_counts_backtracks() -> None:
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    assert direction_reversals(xy, min_step_px=1.0) >= 1


def test_edit_distance_and_clip() -> None:
    assert sequence_edit_distance(["a", "b"], ["a", "b"]) == 0
    assert sequence_edit_distance(["a", "x", "b"], ["a", "b"]) == 1
    assert clip_score(140) == 100.0
    assert clip_score(-4) == 0.0
