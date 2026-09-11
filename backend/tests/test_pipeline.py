from __future__ import annotations

from pathlib import Path

from surgicalvision.constants import DISCLAIMER
from surgicalvision.demo.synthetic_video import generate_synthetic_case
from surgicalvision.pipeline.runner import run_pipeline
from surgicalvision.schemas import AnalysisResult
from surgicalvision.api.store import utc_now


def _analyze(tmp_path: Path, profile: str) -> AnalysisResult:
    video = tmp_path / f"{profile}.mp4"
    generate_synthetic_case(video, profile=profile, seconds=4.0, fps=15, size=(480, 270))
    result = AnalysisResult(
        id=profile,
        status="queued",
        created_at=utc_now(),
        source=f"demo_{profile}",
    )
    return run_pipeline(video, tmp_path / profile, result)


def test_pipeline_detects_two_instruments_and_scores(tmp_path: Path) -> None:
    result = _analyze(tmp_path, "efficient")
    assert result.status == "complete"
    assert len(result.instruments) == 2
    assert result.overall_score is not None and result.overall_score > 0
    assert result.disclaimer == DISCLAIMER
    assert (tmp_path / "efficient" / "overlay.mp4").exists()
    assert result.sequence_expected[0] == "reach"
    labels = {i.label for i in result.instruments}
    assert labels == {"left", "right"}
    assert any(i.path_length_px > 30 for i in result.instruments)


def test_novice_scores_lower_than_efficient(tmp_path: Path) -> None:
    efficient = _analyze(tmp_path, "efficient")
    novice = _analyze(tmp_path, "novice")
    assert efficient.overall_score is not None
    assert novice.overall_score is not None
    assert novice.overall_score < efficient.overall_score
    assert efficient.overall_score >= 55
