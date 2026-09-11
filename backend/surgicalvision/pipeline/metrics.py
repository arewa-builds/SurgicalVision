from __future__ import annotations

import numpy as np

from surgicalvision.constants import LEFT, RIGHT
from surgicalvision.geometry import (
    convex_hull_area,
    derivatives,
    direction_reversals,
    displacement,
    downsample_series,
    path_efficiency,
    path_length,
    velocities,
)
from surgicalvision.pipeline.tracking import Track
from surgicalvision.schemas import BimanualMetrics, InstrumentMetrics


def compute_instrument_metrics(
    track: Track, fps: float, frame_size: tuple[int, int]
) -> InstrumentMetrics:
    w, h = frame_size
    xy = track.tips
    speed = velocities(xy, fps)
    accel = derivatives(speed, fps)
    jerk = derivatives(accel, fps)
    idle_thresh = 0.025 * float(np.hypot(w, h))
    idle_mask = speed < idle_thresh
    idle_frames = int(idle_mask.sum())
    duration = len(xy) / fps if fps else 0.0
    hull = convex_hull_area(xy)
    return InstrumentMetrics(
        label=track.label,
        path_length_px=round(path_length(xy), 2),
        displacement_px=round(displacement(xy), 2),
        path_efficiency=round(path_efficiency(xy), 4),
        mean_velocity_px_s=round(float(np.nanmean(speed)), 2),
        max_velocity_px_s=round(float(np.nanmax(speed)), 2),
        mean_acceleration_px_s2=round(float(np.nanmean(np.abs(accel))), 2),
        mean_jerk_px_s3=round(float(np.nanmean(np.abs(jerk))), 2),
        idle_time_s=round(idle_frames / fps if fps else 0.0, 3),
        idle_fraction=round(idle_frames / max(len(xy), 1), 4),
        workspace_utilization=round(hull / max(w * h, 1), 4),
        corrective_movements=direction_reversals(xy, min_step_px=max(2.0, 0.004 * w)),
        tip_series=downsample_series(xy, fps),
    )


def compute_bimanual(tracks: dict[str, Track], fps: float) -> BimanualMetrics | None:
    if LEFT not in tracks or RIGHT not in tracks:
        return None
    left, right = tracks[LEFT].tips, tracks[RIGHT].tips
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    lv = velocities(left, fps)
    rv = velocities(right, fps)
    diag = float(np.hypot(np.nanmax(left[:, 0]) - np.nanmin(left[:, 0]) + 1, 1))
    # Activity sync: both moving or both idle.
    thresh = 8.0
    both_move = (lv >= thresh) & (rv >= thresh)
    both_idle = (lv < thresh) & (rv < thresh)
    sync = float((both_move | both_idle).mean()) if n else 0.0
    if np.std(lv) < 1e-6 or np.std(rv) < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(lv, rv)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    dist = np.linalg.norm(left - right, axis=1)
    dual = float(both_move.mean()) if n else 0.0
    return BimanualMetrics(
        time_sync=round(sync, 4),
        velocity_correlation=round(corr, 4),
        mean_tip_distance_px=round(float(np.nanmean(dist)), 2),
        dual_activity_fraction=round(dual, 4),
    )


def metrics_for_tracks(
    tracks: dict[str, Track], fps: float, frame_size: tuple[int, int]
) -> tuple[list[InstrumentMetrics], BimanualMetrics | None]:
    instruments = [
        compute_instrument_metrics(tracks[label], fps, frame_size)
        for label in (LEFT, RIGHT)
        if label in tracks
    ]
    return instruments, compute_bimanual(tracks, fps)
