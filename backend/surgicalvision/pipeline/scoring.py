from __future__ import annotations

from surgicalvision.constants import DIMENSIONS, EXPECTED_SUTURE_SEQUENCE
from surgicalvision.geometry import clip_score, sequence_edit_distance
from surgicalvision.schemas import BimanualMetrics, DimensionScore, GestureEvent, InstrumentMetrics


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def score_analysis(
    instruments: list[InstrumentMetrics],
    bimanual: BimanualMetrics | None,
    observed: list[str],
    duration_s: float,
    expected: tuple[str, ...] = EXPECTED_SUTURE_SEQUENCE,
) -> tuple[float, list[DimensionScore], list[str]]:
    notes: list[str] = []
    if not instruments:
        notes.append("No instruments were tracked; scores are withheld.")
        dims = [DimensionScore(key=k, label=lab, score=0.0, detail="No tracks") for k, lab in DIMENSIONS]
        return 0.0, dims, notes

    path_eff = _mean([i.path_efficiency for i in instruments])
    jerk = _mean([i.mean_jerk_px_s3 for i in instruments])
    idle = _mean([i.idle_fraction for i in instruments])
    corrective = sum(i.corrective_movements for i in instruments)
    workspace = _mean([i.workspace_utilization for i in instruments])
    max_vel = _mean([i.max_velocity_px_s for i in instruments])

    # Motion economy: efficient paths, limited thrash.
    motion = 100 * (0.55 * path_eff + 0.25 * max(0.0, 1 - idle * 1.2) + 0.20 * max(0.0, 1 - min(jerk / 25000.0, 1)))
    if corrective > 8:
        motion -= min(18, (corrective - 8) * 1.4)

    # Instrument control: smoothness and idle discipline.
    control = 100 * (0.45 * max(0.0, 1 - min(jerk / 20000.0, 1)) + 0.35 * max(0.0, 1 - abs(idle - 0.18)) + 0.20 * path_eff)
    if idle > 0.45:
        notes.append("High idle time mid-task — possible hesitation or lost visualization.")

    # Bimanual.
    if bimanual:
        corr = (bimanual.velocity_correlation + 1) / 2  # 0..1
        bimanual_score = 100 * (0.5 * bimanual.time_sync + 0.3 * corr + 0.2 * min(1.0, bimanual.dual_activity_fraction / 0.4))
    else:
        bimanual_score = 55.0
        notes.append("Only one instrument track was available; bimanual score is limited.")

    # Procedural efficiency from sequence alignment and duration.
    edits = sequence_edit_distance(observed, list(expected))
    extra = max(0, len(observed) - len(expected))
    duration_pen = 0.0
    if duration_s > 0:
        # Efficient suturing sim is ~8s; much longer is hesitation.
        duration_pen = min(1.0, max(0.0, (duration_s - 9.0) / 12.0))
    procedural = 100 * (0.55 * max(0.0, 1 - edits / 6) + 0.25 * max(0.0, 1 - extra / 4) + 0.20 * (1 - duration_pen))
    if "reposition" in observed:
        notes.append("Reposition / re-grasp steps appeared in the gesture sequence.")
        procedural -= 6
    if observed.count("grasp") > 1:
        notes.append("Multiple grasp events — possible needle drop or re-grasp.")
        procedural -= 5

    # Tissue handling proxy: high jerk + high peak speed in a small workspace is harsher.
    tissue = 100 * (
        0.4 * max(0.0, 1 - min(jerk / 22000.0, 1))
        + 0.3 * max(0.0, 1 - min(max_vel / 900.0, 1))
        + 0.3 * min(1.0, workspace / 0.12 + 0.4)
    )

    # Error avoidance: reversals, extra grasps, idle spikes.
    error_hits = corrective + extra + observed.count("reposition") + max(0, observed.count("grasp") - 1)
    error = 100 * max(0.0, 1 - error_hits / 14)
    if idle > 0.5:
        error -= 8

    scores = {
        "motion_economy": clip_score(motion),
        "instrument_control": clip_score(control),
        "bimanual_coordination": clip_score(bimanual_score),
        "procedural_efficiency": clip_score(procedural),
        "tissue_handling": clip_score(tissue),
        "error_avoidance": clip_score(error),
    }
    details = {
        "motion_economy": f"Path efficiency {path_eff:.2f}; {corrective} direction reversals.",
        "instrument_control": f"Mean jerk {jerk:.0f} px/s³; idle {idle:.0%}.",
        "bimanual_coordination": (
            f"Time sync {bimanual.time_sync:.2f}; velocity corr {bimanual.velocity_correlation:.2f}."
            if bimanual
            else "Single-instrument case."
        ),
        "procedural_efficiency": f"Sequence edit distance {edits}; extra steps {extra}.",
        "tissue_handling": f"Peak speed {max_vel:.0f} px/s (workspace proxy, not tissue contact).",
        "error_avoidance": f"{error_hits} review-linked events from reversals and extra gestures.",
    }
    dimensions = [
        DimensionScore(key=k, label=lab, score=scores[k], detail=details[k]) for k, lab in DIMENSIONS
    ]
    weights = {
        "motion_economy": 0.2,
        "instrument_control": 0.18,
        "bimanual_coordination": 0.16,
        "procedural_efficiency": 0.18,
        "tissue_handling": 0.14,
        "error_avoidance": 0.14,
    }
    overall = clip_score(sum(scores[k] * w for k, w in weights.items()))
    return overall, dimensions, notes
