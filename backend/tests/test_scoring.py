from __future__ import annotations

from surgicalvision.pipeline.scoring import score_analysis
from surgicalvision.schemas import BimanualMetrics, InstrumentMetrics


def _inst(label: str, **overrides) -> InstrumentMetrics:
    base = dict(
        label=label,
        path_length_px=400.0,
        displacement_px=280.0,
        path_efficiency=0.7,
        mean_velocity_px_s=80.0,
        max_velocity_px_s=160.0,
        mean_acceleration_px_s2=200.0,
        mean_jerk_px_s3=800.0,
        idle_time_s=1.2,
        idle_fraction=0.15,
        workspace_utilization=0.08,
        corrective_movements=2,
        tip_series=[],
    )
    base.update(overrides)
    return InstrumentMetrics(**base)


def test_efficient_metrics_score_higher_than_novice() -> None:
    efficient_inst = [
        _inst("left"),
        _inst("right", path_efficiency=0.65, corrective_movements=1),
    ]
    novice_inst = [
        _inst(
            "left",
            path_efficiency=0.22,
            mean_jerk_px_s3=28000,
            idle_fraction=0.42,
            corrective_movements=14,
            max_velocity_px_s=900,
        ),
        _inst("right", path_efficiency=0.2, mean_jerk_px_s3=24000, corrective_movements=9),
    ]
    good_bi = BimanualMetrics(
        time_sync=0.86, velocity_correlation=0.72, mean_tip_distance_px=90, dual_activity_fraction=0.4
    )
    poor_bi = BimanualMetrics(
        time_sync=0.4, velocity_correlation=0.05, mean_tip_distance_px=180, dual_activity_fraction=0.12
    )
    expected = ["reach", "position", "grasp", "suture", "knot", "release"]
    novice_seq = ["reach", "reposition", "grasp", "release", "grasp", "suture", "knot", "release"]
    good, _, _ = score_analysis(efficient_inst, good_bi, expected, 8.0)
    poor, _, _ = score_analysis(novice_inst, poor_bi, novice_seq, 14.0)
    assert good > poor
    assert good >= 70
    assert poor <= good - 15


def test_realistic_task_length_is_not_penalized() -> None:
    inst = [_inst("left"), _inst("right")]
    bi = BimanualMetrics(
        time_sync=0.8, velocity_correlation=0.6, mean_tip_distance_px=100, dual_activity_fraction=0.35
    )
    seq = ["reach", "position", "grasp", "suture", "knot", "release"]
    short, dims_short, _ = score_analysis(inst, bi, seq, 8.0, expected_duration_s=9.0)
    long, dims_long, _ = score_analysis(inst, bi, seq, 62.0, expected_duration_s=70.0)
    proc_short = next(d.score for d in dims_short if d.key == "procedural_efficiency")
    proc_long = next(d.score for d in dims_long if d.key == "procedural_efficiency")
    assert abs(proc_short - proc_long) < 1.0
    assert long >= short - 1.0
