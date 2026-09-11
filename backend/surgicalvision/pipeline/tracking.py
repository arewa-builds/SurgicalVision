from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from surgicalvision.constants import LEFT, RIGHT
from surgicalvision.geometry import interpolate_nans
from surgicalvision.pipeline.detection import Detection


@dataclass
class Track:
    label: str
    tips: np.ndarray
    boxes: list[tuple[int, int, int, int] | None] = field(default_factory=list)
    conf: np.ndarray | None = None


def track_instruments(frame_detections: list[list[Detection]], frame_count: int) -> dict[str, Track]:
    """Associate per-frame detections into persistent left/right tracks."""
    tips = {LEFT: np.full((frame_count, 2), np.nan), RIGHT: np.full((frame_count, 2), np.nan)}
    boxes: dict[str, list[tuple[int, int, int, int] | None]] = {
        LEFT: [None] * frame_count,
        RIGHT: [None] * frame_count,
    }
    confs = {LEFT: np.zeros(frame_count), RIGHT: np.zeros(frame_count)}

    for i, dets in enumerate(frame_detections):
        used: set[str] = set()
        for det in sorted(dets, key=lambda d: d.conf, reverse=True):
            label = det.label if det.label in (LEFT, RIGHT) else (LEFT if det.tip[0] < 1e9 else RIGHT)
            if label in used:
                # Fall back to the other side if this label is taken.
                other = RIGHT if label == LEFT else LEFT
                if other not in used:
                    label = other
                else:
                    continue
            used.add(label)
            tips[label][i] = det.tip
            boxes[label][i] = det.bbox
            confs[label][i] = det.conf

        # If unlabeled extras remain, assign by x position.
        remaining = [d for d in dets if d.label not in used]
        for det in remaining:
            label = LEFT if det.tip[0] <= np.nanmean([det.tip[0]]) else RIGHT
            # choose empty side
            for candidate in (LEFT, RIGHT):
                if candidate not in used:
                    label = candidate
                    break
            if label in used:
                continue
            used.add(label)
            tips[label][i] = det.tip
            boxes[label][i] = det.bbox
            confs[label][i] = det.conf

    tracks: dict[str, Track] = {}
    for label in (LEFT, RIGHT):
        filled = interpolate_nans(tips[label])
        if not np.isfinite(filled).any():
            continue
        tracks[label] = Track(label=label, tips=filled, boxes=boxes[label], conf=confs[label])
    return tracks
