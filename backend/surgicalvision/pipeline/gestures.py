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
    traveled = path_length(xy)
    disp = float(np.linalg.norm(xy[-1] - xy[0]))
    straightness = float(disp / traveled) if traveled > 1e-6 else 1.0
    loopiness = 1.0 - straightness
    turns = turning_angles(xy)
    mean_turn = float(np.mean(turns)) if len(turns) else 0.0
    path_n = traveled / diag

    if mean_speed < 0.016:
        if 0.004 < mean_speed < 0.016:
            return "grasp", 0.7
        return "idle", 0.75

    if loopiness > 0.42 and mean_turn > 0.32:
        if path_n < 0.075:
            return "knot", 0.72
        return "suture", 0.74

    if mean_speed >= 0.10 and straightness >= 0.62:
        return "reach", 0.8
    if 0.028 <= mean_speed < 0.10 and straightness >= 0.58 and mean_turn < 0.5:
        return "position", 0.74
    if straightness < 0.38 and mean_speed >= 0.055:
        return "reposition", 0.66
    if mean_speed >= 0.04:
        return "manipulate", 0.55
    return "position", 0.4


def recognize_gestures(tracks: dict[str, Track], fps: float, frame_size: tuple[int, int]) -> list[GestureEvent]:
    w, h = frame_size
    diag = float(np.hypot(w, h))
    window = max(8, int(round(0.55 * fps)))
    hop = max(2, int(round(0.16 * fps)))
    events: list[GestureEvent] = []
    duration = max((len(track.tips) / fps for track in tracks.values()), default=0.0)

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
    return _tag_releases(events, tracks, fps, diag, duration)


def _merge_windows(windows: list[GestureEvent]) -> list[GestureEvent]:
    if not windows:
        return []
    merged: list[GestureEvent] = [windows[0]]
    for ev in windows[1:]:
        last = merged[-1]
        if ev.label == last.label and ev.instrument == last.instrument and ev.start_s <= last.end_s + 0.08:
            last.end_s = max(last.end_s, ev.end_s)
            last.confidence = max(last.confidence, ev.confidence)
        else:
            merged.append(ev)
    return [e for e in merged if (e.end_s - e.start_s) >= 0.2]


def _tag_releases(
    events: list[GestureEvent],
    tracks: dict[str, Track],
    fps: float,
    diag: float,
    duration: float,
) -> list[GestureEvent]:
    out: list[GestureEvent] = []
    gate = duration * 0.70
    for ev in events:
        if ev.label in {"reach", "manipulate", "position"} and ev.start_s >= gate:
            track = tracks.get(ev.instrument)
            if track is not None:
                i0 = int(ev.start_s * fps)
                i1 = min(len(track.tips) - 1, int(ev.end_s * fps))
                a, b = track.tips[i0], track.tips[i1]
                if np.isfinite(a).all() and np.isfinite(b).all() and b[1] > a[1] + 0.015 * diag:
                    ev = ev.model_copy(update={"label": "release"})
        out.append(ev)
    return out


def observed_sequence(events: list[GestureEvent], primary: str = "left") -> list[str]:
    ordered = [e for e in events if e.instrument == primary and e.label in TASK_GESTURES]
    if not ordered:
        ordered = [e for e in events if e.label in TASK_GESTURES]
    if not ordered:
        return []
    t0 = min(e.start_s for e in ordered)
    t1 = max(e.end_s for e in ordered)
    bin_w = 0.7
    seq: list[str] = []
    t = t0
    while t < t1 - 0.05:
        t_end = t + bin_w
        scores: dict[str, float] = {}
        for event in ordered:
            overlap = min(event.end_s, t_end) - max(event.start_s, t)
            if overlap > 0:
                scores[event.label] = scores.get(event.label, 0.0) + overlap * event.confidence
        if scores:
            label = max(scores, key=scores.get)
            if not seq or seq[-1] != label:
                seq.append(label)
        t = t_end
    while seq and seq[0] in {"grasp", "knot", "manipulate"}:
        seq.pop(0)
    cleaned: list[str] = []
    for i, label in enumerate(seq):
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        prev = cleaned[-1] if cleaned else None
        if label == "manipulate" and prev in {"position", "grasp"} and nxt in {"suture", "knot"}:
            continue
        if label == prev:
            continue
        cleaned.append(label)
    return cleaned
