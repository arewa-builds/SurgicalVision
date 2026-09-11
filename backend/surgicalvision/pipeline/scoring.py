from __future__ import annotations

from surgicalvision.constants import DIMENSIONS, EXPECTED_SUTURE_SEQUENCE
from surgicalvision.geometry import clip_score, sequence_edit_distance
from surgicalvision.schemas import BimanualMetrics, DimensionScore, InstrumentMetrics


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _primary(instruments: list[InstrumentMetrics]) -> InstrumentMetrics:
    return next((i for i in instruments if i.label == "left"), instruments[0])


def score_analysis(
    instruments: list[InstrumentMetrics],
    bimanual: BimanualMetrics | None,
    observed: list[str],
    duration_s: float,
    expected: tuple[str, ...] = EXPECTED_SUTURE_SEQUENCE,
    expected_duration_s: float = 9.0,
) -> tuple[float, list[DimensionScore], list[str]]:
    notes: list[str] = []
    if not instruments:
        notes.append("No instruments were tracked; scores are withheld.")
        dims = [DimensionScore(key=k, label=lab, score=0.0, detail="No tracks") for k, lab in DIMENSIONS]
        return 0.0, dims, notes

    primary = _primary(instruments)
    path_eff = primary.path_efficiency
    jerk = primary.mean_jerk_px_s3
    idle = primary.idle_fraction
    corrective = primary.corrective_movements
    workspace = _mean([i.workspace_utilization for i in instruments])
    max_vel = primary.max_velocity_px_s

    jerk_pen = min(1.0, jerk / 24000.0)
    rev_pen = min(1.0, corrective / 8.0)
    idle_pen = min(1.0, max(0.0, idle - 0.08) / 0.42)

    motion = 100 * (0.40 * (1 - jerk_pen) + 0.35 * (1 - rev_pen) + 0.25 * path_eff)
    control = 100 * (0.50 * (1 - jerk_pen) + 0.30 * (1 - idle_pen) + 0.20 * path_eff)
    if idle > 0.42:
        notes.append("High idle time mid-task — possible hesitation or lost visualization.")

    if bimanual:
        corr = (bimanual.velocity_correlation + 1) / 2
        # Experts often stabilize with one hand; both-instruments-flailing is not coordination.
        complementary = max(0.0, 1.0 - abs(bimanual.dual_activity_fraction - 0.35) / 0.65)
        bimanual_score = 100 * (0.58 * complementary + 0.42 * corr)
    else:
        bimanual_score = 55.0
        notes.append("Only one instrument track was available; bimanual score is limited.")

    edits = sequence_edit_distance(observed, list(expected))
    extra = max(0, len(observed) - len(expected))
    over = max(0.0, duration_s - expected_duration_s)
    scale = max(12.0, expected_duration_s * 0.5)
    duration_pen = min(1.0, over / scale) if duration_s > 0 else 0.0
    procedural = 100 * (
        0.50 * max(0.0, 1 - edits / 6) + 0.30 * max(0.0, 1 - extra / 4) + 0.20 * (1 - duration_pen)
    )
    if "reposition" in observed:
        notes.append("Reposition / re-grasp steps appeared in the gesture sequence.")
        procedural -= 8
    if observed.count("grasp") > 1:
        notes.append("Multiple grasp events — possible needle drop or re-grasp.")
        procedural -= 6

    tissue = 100 * (
        0.45 * (1 - jerk_pen)
        + 0.30 * max(0.0, 1 - min(max_vel / 1200.0, 1))
        + 0.25 * min(1.0, workspace / 0.08 + 0.45)
    )

    error_hits = corrective + extra + observed.count("reposition") + max(0, observed.count("grasp") - 1)
    error = 100 * max(0.0, 1 - error_hits / 16)
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
        "motion_economy": f"Path smoothness {path_eff:.2f}; {corrective} direction reversals.",
        "instrument_control": f"Mean jerk {jerk:.0f} px/s³; mid-task idle {idle:.0%}.",
        "bimanual_coordination": (
            f"Complementary activity {bimanual.dual_activity_fraction:.2f}; velocity corr {bimanual.velocity_correlation:.2f}."
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
