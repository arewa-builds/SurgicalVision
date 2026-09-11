from __future__ import annotations

import numpy as np

from surgicalvision.constants import EXPECTED_SUTURE_SEQUENCE
from surgicalvision.geometry import path_length, turning_angles, velocities
from surgicalvision.pipeline.tracking import Track
from surgicalvision.schemas import GestureEvent


TASK_GESTURES = set(EXPECTED_SUTURE_SEQUENCE) | {"manipulate", "cut", "reposition"}


def _window_label(xy: np.ndarray, fps: float, diag: float) -> tuple[str, float]:
    if len(xy) < 4 or diag <= 0:
        return "idle", 0.2
    speed = velocities(xy, fps) / diag
    mean_speed = float(np.nanmean(speed))
    straight = path_length(xy)
    disp = float(np.linalg.norm(xy[-1] - xy[0]))
    straightness = float(disp / straight) if straight > 1e-6 else 1.0
    turns = turning_angles(xy)
    mean_turn = float(np.mean(turns)) if len(turns) else 0.0
    # jerk proxy
    accel = np.diff(speed)
    jerk = float(np.mean(np.abs(np.diff(accel)))) if len(accel) > 2 else 0.0

    if mean_speed < 0.018:
        return "grasp" if 0.006 < mean_speed < 0.018 and straightness < 0.55 else "idle", 0.72

    if mean_speed >= 0.11 and straightness >= 0.72:
        return "reach", 0.8
    if 0.04 <= mean_speed < 0.11 and straightness >= 0.68 and mean_turn < 0.45:
        return "position", 0.74
    if mean_turn >= 0.85 and disp / diag < 0.08:
        return "knot", 0.7
    if mean_turn >= 0.55 and 0.08 <= disp / diag < 0.28:
        return "suture", 0.68
    if mean_speed >= 0.09 and straightness < 0.45:
        return "reposition", 0.66
    if jerk > 0.08 and straightness > 0.7:
        return "cut", 0.55
    if mean_speed >= 0.05:
        return "manipulate", 0.6
    if mean_speed >= 0.03 and straightness < 0.5:
        return "reposition", 0.55
    return "position", 0.45


def recognize_gestures(tracks: dict[str, Track], fps: float, frame_size: tuple[int, int]) -> list[GestureEvent]:
    w, h = frame_size
    diag = float(np.hypot(w, h))
    window = max(6, int(round(0.45 * fps)))
    hop = max(2, int(round(0.12 * fps)))
    events: list[GestureEvent] = []

    for label, track in tracks.items():
        xy = track.tips
        n = len(xy)
        i = 0
        raw: list[GestureEvent] = []
        while i + window <= n:
            sl = xy[i : i + window]
            if not np.isfinite(sl).all():
                i += hop
                continue
            name, conf = _window_label(sl, fps, diag)
            raw.append(
                GestureEvent(
                    label=name,
                    start_s=i / fps,
                    end_s=(i + window) / fps,
                    instrument=label,
                    confidence=round(conf, 3),
                )
            )
            i += hop
        events.extend(_merge_windows(raw))

    events.sort(key=lambda e: (e.start_s, e.instrument))
    return _tag_releases(events, tracks, fps, diag)


def _merge_windows(windows: list[GestureEvent]) -> list[GestureEvent]:
    if not windows:
        return []
    merged: list[GestureEvent] = [windows[0]]
    for ev in windows[1:]:
        last = merged[-1]
        if ev.label == last.label and ev.instrument == last.instrument and ev.start_s <= last.end_s + 0.05:
            last.end_s = max(last.end_s, ev.end_s)
            last.confidence = max(last.confidence, ev.confidence)
        else:
            merged.append(ev)
    return [e for e in merged if (e.end_s - e.start_s) >= 0.18]


def _tag_releases(
    events: list[GestureEvent],
    tracks: dict[str, Track],
    fps: float,
    diag: float,
) -> list[GestureEvent]:
    """Relabel late outward motion as release."""
    out: list[GestureEvent] = []
    for ev in events:
        if ev.label in {"reach", "manipulate"} and ev.start_s > 0.7:
            track = tracks.get(ev.instrument)
            if track is not None:
                i0 = int(ev.start_s * fps)
                i1 = min(len(track.tips) - 1, int(ev.end_s * fps))
                a, b = track.tips[i0], track.tips[i1]
                if np.isfinite(a).all() and np.isfinite(b).all():
                    # Moving toward the image border counts as release near the end.
                    if b[1] > a[1] + 0.02 * diag:
                        ev = ev.model_copy(update={"label": "release"})
        out.append(ev)
    return out


def observed_sequence(events: list[GestureEvent], primary: str = "left") -> list[str]:
    ordered = [e for e in events if e.instrument == primary and e.label in TASK_GESTURES]
    if not ordered:
        ordered = [e for e in events if e.label in TASK_GESTURES]
    seq: list[str] = []
    for e in sorted(ordered, key=lambda x: x.start_s):
        if not seq or seq[-1] != e.label:
            seq.append(e.label)
    return seq
