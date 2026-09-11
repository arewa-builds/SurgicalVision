from __future__ import annotations

import numpy as np

from surgicalvision.pipeline.gestures import _window_label, observed_sequence
from surgicalvision.schemas import GestureEvent


def test_window_labels_reach_and_idle() -> None:
    # Straight fast path → reach
    reach = np.stack([np.linspace(0, 200, 20), np.linspace(200, 40, 20)], axis=1)
    label, _ = _window_label(reach, fps=20, diag=1000)
    assert label == "reach"

    idle = np.tile(np.array([[50.0, 50.0]]), (20, 1))
    label, _ = _window_label(idle, fps=20, diag=1000)
    assert label == "idle"


def test_observed_sequence_collapses_repeats() -> None:
    events = [
        GestureEvent(label="idle", start_s=0, end_s=0.4, instrument="left", confidence=0.8),
        GestureEvent(label="reach", start_s=0.4, end_s=1.0, instrument="left", confidence=0.8),
        GestureEvent(label="reach", start_s=1.0, end_s=1.4, instrument="left", confidence=0.8),
        GestureEvent(label="suture", start_s=1.4, end_s=2.2, instrument="left", confidence=0.8),
    ]
    assert observed_sequence(events) == ["reach", "suture"]
