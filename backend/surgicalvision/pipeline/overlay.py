from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from surgicalvision.constants import LEFT
from surgicalvision.pipeline.tracking import Track
from surgicalvision.schemas import GestureEvent, TimelineEvent

GREEN = (96, 210, 120)
AMBER = (40, 180, 232)
RED = (70, 70, 232)
CYAN = (255, 220, 40)
GOLD = (20, 170, 250)
WHITE = (236, 240, 244)
MUTED = (160, 170, 180)


def _active_gesture(events: list[GestureEvent], t: float, instrument: str) -> str | None:
    hits = [e for e in events if e.instrument == instrument and e.start_s <= t <= e.end_s]
    if not hits:
        return None
    return hits[-1].label


def classify_frames(track: Track, fps: float) -> list[str]:
    """Per-frame path tag: efficient | corrective | idle."""
    xy = track.tips
    n = len(xy)
    tags = ["efficient"] * n
    if n < 3 or fps <= 0:
        return tags
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    idle_t = 0.025 * 1000
    v = np.diff(xy, axis=0)
    dots = np.ones(n)
    dots[2:] = (v[:-1] * v[1:]).sum(axis=1)
    for i in range(n):
        s = speed[i - 1] if i else (speed[0] if len(speed) else 0)
        if s < idle_t / 30:
            tags[i] = "idle"
        elif dots[i] < 0 and s > 40:
            tags[i] = "corrective"
    return tags


def build_timeline(
    tracks: dict[str, Track],
    gestures: list[GestureEvent],
    fps: float,
    duration_s: float,
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for label, track in tracks.items():
        tags = classify_frames(track, fps)
        i = 0
        n = len(tags)
        while i < n:
            if tags[i] != "corrective":
                i += 1
                continue
            j = i
            while j < n and tags[j] == "corrective":
                j += 1
            t = i / fps
            events.append(
                TimelineEvent(
                    t=round(t, 2),
                    kind="corrective",
                    title=f"{label.title()} corrective movement",
                    detail="Direction reversal on the instrument path.",
                    severity="amber",
                )
            )
            i = j

    # Extra re-grasp / reposition as review markers.
    for g in gestures:
        if g.label == "reposition":
            events.append(
                TimelineEvent(
                    t=round(g.start_s, 2),
                    kind="review",
                    title=f"{g.instrument.title()} reposition",
                    detail="Inefficient sequence step worth reviewing.",
                    severity="red",
                )
            )

    # Deduplicate nearby events.
    events.sort(key=lambda e: e.t)
    compact: list[TimelineEvent] = []
    for ev in events:
        if compact and abs(compact[-1].t - ev.t) < 0.35 and compact[-1].title == ev.title:
            continue
        compact.append(ev)

    if not compact and duration_s > 0:
        compact.append(
            TimelineEvent(
                t=round(duration_s * 0.35, 2),
                kind="review",
                title="Suture phase",
                detail="Peak task activity — compare path economy against the expected gesture order.",
                severity="green",
            )
        )
    return compact[:16]


def _color_for_tag(tag: str, instrument: str) -> tuple[int, int, int]:
    if tag == "corrective":
        return AMBER
    if tag == "idle":
        return MUTED
    return GREEN if instrument == LEFT else (90, 200, 160)


def render_overlay(
    frames: list[np.ndarray],
    tracks: dict[str, Track],
    gestures: list[GestureEvent],
    timeline: list[TimelineEvent],
    fps: float,
    dest: Path,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to render")
    h, w = frames[0].shape[:2]
    raw = dest.with_name(dest.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open overlay video writer")

    tags = {label: classify_frames(track, fps) for label, track in tracks.items()}
    review_times = [e.t for e in timeline if e.severity == "red"]

    trail = 48
    for idx, frame in enumerate(frames):
        canvas = frame.copy()
        t = idx / fps
        for label, track in tracks.items():
            color_base = CYAN if label == LEFT else GOLD
            pts = []
            for j in range(max(0, idx - trail), idx + 1):
                p = track.tips[j]
                if np.isfinite(p).all():
                    pts.append((int(p[0]), int(p[1]), tags[label][j]))
            for k in range(1, len(pts)):
                x0, y0, tag = pts[k - 1]
                x1, y1, _ = pts[k]
                cv2.line(canvas, (x0, y0), (x1, y1), _color_for_tag(tag, label), 2, cv2.LINE_AA)
            # faint full path
            full = [(int(p[0]), int(p[1])) for p in track.tips[: idx + 1] if np.isfinite(p).all()]
            if len(full) > 2:
                cv2.polylines(canvas, [np.array(full, dtype=np.int32)], False, (*color_base, ), 1, cv2.LINE_AA)
            tip = track.tips[idx]
            if np.isfinite(tip).all():
                cv2.circle(canvas, (int(tip[0]), int(tip[1])), 7, color_base, -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(tip[0]), int(tip[1])), 7, WHITE, 1, cv2.LINE_AA)
                gesture = _active_gesture(gestures, t, label)
                text = f"{label[0].upper()}  {gesture or '—'}"
                cv2.putText(
                    canvas,
                    text,
                    (int(tip[0]) + 10, int(tip[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )
            box = track.boxes[idx] if idx < len(track.boxes) else None
            if box:
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color_base, 1)

        for rt in review_times:
            if abs(rt - t) < (1.0 / max(fps, 1)):
                cv2.circle(canvas, (w // 2, 36), 8, RED, -1, cv2.LINE_AA)

        _draw_hud(canvas, t)

        writer.write(canvas)
    writer.release()
    _transcode_h264(raw, dest)
    return dest


def _draw_hud(canvas: np.ndarray, t: float) -> None:
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (10, 12, 16), -1)
    cv2.rectangle(overlay, (16, h - 36), (360, h - 10), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, f"{t:05.2f}s", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "green efficient   amber corrective   red review",
        (140, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "L cyan    R gold",
        (24, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        MUTED,
        1,
        cv2.LINE_AA,
    )


def _transcode_h264(src: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dest.exists():
        # Keep the mp4v file so analysis can still complete.
        if src.exists():
            src.replace(dest)
        return
    src.unlink(missing_ok=True)
