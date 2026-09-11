from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from surgicalvision.constants import LEFT_BGR, RIGHT_BGR


def _smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _phase_u(t: float, t0: float, t1: float) -> float:
    if t1 <= t0:
        return 1.0
    return float(_smoothstep((t - t0) / (t1 - t0)))


def _lerp(a: tuple[float, float], b: tuple[float, float], u: float) -> tuple[float, float]:
    u = float(_smoothstep(u))
    return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u


def _circle(center: tuple[float, float], radius: float, u: float, turns: float) -> tuple[float, float]:
    ang = 2 * np.pi * turns * u
    return center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)


def _tissue_background(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    noise = (
        14 * np.sin(xx * 0.035 + yy * 0.02)
        + 9 * np.sin(xx * 0.09 - yy * 0.06)
        + 6 * np.sin((xx + yy) * 0.028)
        + 5 * np.sin(xx * 0.015) * np.cos(yy * 0.02)
    )
    b = np.clip(28 + noise * 0.35, 0, 255)
    g = np.clip(52 + noise * 0.55, 0, 255)
    r = np.clip(118 + noise * 0.85, 0, 255)
    frame = np.dstack([b, g, r]).astype(np.uint8)
    # vessels
    for k in range(7):
        y0 = int(h * (0.18 + 0.1 * k))
        amp = 18 + 6 * (k % 3)
        xs = np.arange(w)
        ys = y0 + (amp * np.sin(xs * 0.02 + k)).astype(int)
        for x, y in zip(xs[::2], ys[::2]):
            if 0 <= y < h:
                cv2.circle(frame, (int(x), int(y)), 1, (18, 22, 70), -1)
    # suture target
    cv2.ellipse(frame, (int(w * 0.52), int(h * 0.44)), (28, 16), 18, 0, 360, (90, 110, 170), -1)
    cv2.ellipse(frame, (int(w * 0.52), int(h * 0.44)), (28, 16), 18, 0, 360, (40, 50, 90), 1)
    return frame


def _vignette_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    ry, rx = h * 0.52, w * 0.52
    rr = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.clip(1.15 - rr, 0, 1)
    mask = (mask * 255).astype(np.uint8)
    return mask


def _draw_instrument(
    frame: np.ndarray, tip: tuple[float, float], entry: tuple[float, float], bgr: tuple[int, int, int]
) -> None:
    h, w = frame.shape[:2]
    x0, y0 = int(entry[0]), int(entry[1])
    x1, y1 = int(np.clip(tip[0], 0, w - 1)), int(np.clip(tip[1], 0, h - 1))
    cv2.line(frame, (x0, y0), (x1, y1), bgr, 7, cv2.LINE_AA)
    # metallic highlight
    mid = ((x0 + x1) // 2, (y0 + y1) // 2)
    cv2.line(frame, (x0, y0), (x1, y1), (min(255, bgr[0] + 40), min(255, bgr[1] + 40), min(255, bgr[2] + 40)), 2, cv2.LINE_AA)
    # grasper jaws
    vx, vy = x1 - x0, y1 - y0
    nrm = max(1.0, float(np.hypot(vx, vy)))
    ux, uy = vx / nrm, vy / nrm
    px, py = -uy, ux
    jaw = 11
    j1 = (int(x1 + px * jaw - ux * 4), int(y1 + py * jaw - uy * 4))
    j2 = (int(x1 - px * jaw - ux * 4), int(y1 - py * jaw - uy * 4))
    cv2.line(frame, (x1, y1), j1, bgr, 3, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), j2, bgr, 3, cv2.LINE_AA)
    cv2.circle(frame, (x1, y1), 4, (240, 240, 240), -1, cv2.LINE_AA)
    cv2.circle(frame, mid, 2, (255, 255, 255), -1)


def _efficient_tips(t: float, duration: float) -> tuple[tuple[float, float], tuple[float, float]]:
    u = t / duration
    # Left primary, right support. Normalized coords.
    l_idle, r_idle = (0.30, 0.82), (0.70, 0.82)
    l_reach, r_reach = (0.46, 0.52), (0.60, 0.54)
    l_pos, r_pos = (0.52, 0.44), (0.58, 0.46)
    if u < 0.08:
        return l_idle, r_idle
    if u < 0.22:
        p = _phase_u(u, 0.08, 0.22)
        return _lerp(l_idle, l_reach, p), _lerp(r_idle, r_reach, p)
    if u < 0.32:
        p = _phase_u(u, 0.22, 0.32)
        return _lerp(l_reach, l_pos, p), _lerp(r_reach, r_pos, p)
    if u < 0.40:
        return l_pos, r_pos
    if u < 0.62:
        p = _phase_u(u, 0.40, 0.62)
        return _circle(l_pos, 0.045, p, 2.0), r_pos
    if u < 0.75:
        p = _phase_u(u, 0.62, 0.75)
        return _circle(l_pos, 0.016, p, 3.5), _circle(r_pos, 0.012, p, 3.0)
    if u < 0.88:
        p = _phase_u(u, 0.75, 0.88)
        l_rel, r_rel = (0.40, 0.60), (0.66, 0.62)
        return _lerp(l_pos, l_rel, p), _lerp(r_pos, r_rel, p)
    p = _phase_u(u, 0.88, 1.0)
    return _lerp((0.40, 0.60), l_idle, p), _lerp((0.66, 0.62), r_idle, p)


def _novice_tips(t: float, duration: float) -> tuple[tuple[float, float], tuple[float, float]]:
    u = t / duration
    l_idle, r_idle = (0.28, 0.84), (0.74, 0.83)
    l_over, r_lag = (0.62, 0.34), (0.68, 0.70)
    l_back, r_pos = (0.50, 0.46), (0.60, 0.50)
    tremor = 0.012 * np.sin(t * 18.0), 0.01 * np.cos(t * 15.0)
    lag = 0.018 * np.sin(t * 3.0)

    def add(p, extra=(0.0, 0.0)):
        return p[0] + tremor[0] + extra[0], p[1] + tremor[1] + extra[1]

    if u < 0.10:
        return add(l_idle), add(r_idle, (lag, 0))
    if u < 0.22:
        p = _phase_u(u, 0.10, 0.22)
        return add(_lerp(l_idle, l_over, p)), add(_lerp(r_idle, r_lag, p), (lag, 0))
    if u < 0.32:
        p = _phase_u(u, 0.22, 0.32)
        return add(_lerp(l_over, l_back, p)), add(_lerp(r_lag, r_pos, p), (lag, 0))
    if u < 0.38:
        return add(l_back), add(r_pos, (lag, 0))
    if u < 0.46:
        p = _phase_u(u, 0.38, 0.46)
        dropped = (0.42, 0.58)
        return add(_lerp(l_back, dropped, p)), add(r_pos, (lag, 0))
    if u < 0.54:
        p = _phase_u(u, 0.46, 0.54)
        return add(_lerp((0.42, 0.58), l_back, p)), add(_lerp(r_pos, (0.57, 0.47), p), (lag, 0))
    if u < 0.72:
        p = _phase_u(u, 0.54, 0.72)
        messy = _circle(l_back, 0.07, p, 2.4)
        return add(messy), add(_circle((0.57, 0.47), 0.03, p, 1.4), (lag, 0))
    if u < 0.82:
        p = _phase_u(u, 0.72, 0.82)
        return add(_circle(l_back, 0.03, p, 2.0)), add(r_pos, (lag, 0))
    if u < 0.92:
        p = _phase_u(u, 0.82, 0.92)
        return add(_lerp(l_back, (0.34, 0.70), p)), add(_lerp(r_pos, (0.72, 0.70), p), (lag, 0))
    p = _phase_u(u, 0.92, 1.0)
    return add(_lerp((0.34, 0.70), l_idle, p)), add(_lerp((0.72, 0.70), r_idle, p), (lag, 0))


def generate_synthetic_case(
    dest: Path,
    profile: str = "efficient",
    seconds: float = 8.0,
    fps: int = 24,
    size: tuple[int, int] = (960, 540),
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    n = int(round(seconds * fps))
    tissue = _tissue_background(h, w)
    mask = _vignette_mask(h, w)
    raw = dest.with_name(dest.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open synthetic video writer")

    left_entry = (w * 0.12, h * 1.05)
    right_entry = (w * 0.88, h * 1.05)
    motion = _efficient_tips if profile == "efficient" else _novice_tips

    for i in range(n):
        t = i / fps
        frame = tissue.copy()
        (lx, ly), (rx, ry) = motion(t, seconds)
        left_tip = (lx * w, ly * h)
        right_tip = (rx * w, ry * h)
        circle = np.zeros((h, w), np.uint8)
        cv2.circle(circle, (w // 2, h // 2), int(min(w, h) * 0.52), 255, -1)
        frame = cv2.bitwise_and(frame, frame, mask=circle)
        vig = cv2.merge([mask, mask, mask])
        frame = cv2.multiply(frame, vig, scale=1 / 255.0)
        _draw_instrument(frame, left_tip, left_entry, LEFT_BGR)
        _draw_instrument(frame, right_tip, right_entry, RIGHT_BGR)
        writer.write(frame)

    writer.release()

    from surgicalvision.pipeline.overlay import _transcode_h264

    _transcode_h264(raw, dest)
    return dest
