from __future__ import annotations

import math

import numpy as np


def interpolate_nans(xy: np.ndarray) -> np.ndarray:
    """Linearly fill NaN gaps in an (N, 2) trajectory. Leading/trailing NaNs stay NaN."""
    out = xy.astype(np.float64).copy()
    n = len(out)
    if n == 0:
        return out
    for dim in (0, 1):
        series = out[:, dim]
        valid = np.isfinite(series)
        if valid.sum() < 2:
            continue
        idx = np.arange(n)
        series[~valid] = np.interp(idx[~valid], idx[valid], series[valid])
        # np.interp also fills ends; restore leading/trailing NaNs.
        first, last = int(np.argmax(valid)), int(n - 1 - np.argmax(valid[::-1]))
        series[:first] = np.nan
        series[last + 1 :] = np.nan
        out[:, dim] = series
    return out


def finite_points(xy: np.ndarray) -> np.ndarray:
    mask = np.isfinite(xy).all(axis=1)
    return xy[mask]


def path_length(xy: np.ndarray) -> float:
    pts = finite_points(xy)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def displacement(xy: np.ndarray) -> float:
    pts = finite_points(xy)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts[-1] - pts[0]))


def path_efficiency(xy: np.ndarray) -> float:
    length = path_length(xy)
    if length <= 1e-6:
        return 1.0
    return float(np.clip(displacement(xy) / length, 0.0, 1.0))


def velocities(xy: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame speed in px/s, length N, first sample 0."""
    n = len(xy)
    speed = np.zeros(n, dtype=np.float64)
    if n < 2 or fps <= 0:
        return speed
    delta = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    valid = np.isfinite(delta)
    delta = np.where(valid, delta, 0.0)
    speed[1:] = delta * fps
    speed[0] = speed[1] if n > 1 else 0.0
    return speed


def derivatives(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) < 2 or fps <= 0:
        return np.zeros_like(values, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    out[1:] = np.diff(values) * fps
    out[0] = out[1]
    return out


def turning_angles(xy: np.ndarray) -> np.ndarray:
    pts = finite_points(xy)
    if len(pts) < 3:
        return np.zeros(0)
    v = np.diff(pts, axis=0)
    a = v[:-1]
    b = v[1:]
    an = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    an = np.maximum(an, 1e-9)
    cos = np.clip((a * b).sum(axis=1) / an, -1.0, 1.0)
    return np.arccos(cos)


def direction_reversals(xy: np.ndarray, min_step_px: float = 2.0) -> int:
    pts = finite_points(xy)
    if len(pts) < 3:
        return 0
    v = np.diff(pts, axis=0)
    mag = np.linalg.norm(v, axis=1)
    keep = mag >= min_step_px
    v = v[keep]
    if len(v) < 2:
        return 0
    dots = (v[:-1] * v[1:]).sum(axis=1)
    return int((dots < 0).sum())


def convex_hull_area(xy: np.ndarray) -> float:
    pts = finite_points(xy)
    if len(pts) < 3:
        return 0.0
    # Andrew's monotone chain.
    pts = np.unique(np.round(pts, 3), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def downsample_series(xy: np.ndarray, fps: float, max_points: int = 180) -> list[list[float]]:
    n = len(xy)
    if n == 0:
        return []
    step = max(1, math.ceil(n / max_points))
    rows: list[list[float]] = []
    for i in range(0, n, step):
        x, y = xy[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        rows.append([round(i / fps, 3), float(x), float(y)])
    return rows


def clip_score(value: float) -> float:
    return float(np.clip(round(value, 1), 0.0, 100.0))


def sequence_edit_distance(observed: list[str], expected: list[str]) -> int:
    a, b = observed, expected
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[n, m])
