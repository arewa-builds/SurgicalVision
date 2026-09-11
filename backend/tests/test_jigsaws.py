from __future__ import annotations

from pathlib import Path

import cv2

from surgicalvision.demo.cases import CASES, JIGSAWS_DIR
from surgicalvision.pipeline.detection import DarkShaftDetector
from surgicalvision.pipeline.runner import run_pipeline
from surgicalvision.schemas import AnalysisResult
from surgicalvision.api.store import utc_now


def test_jigsaws_clips_are_present() -> None:
    assert JIGSAWS_DIR.is_dir()
    for case in CASES.values():
        assert case.path().is_file(), case.filename
        assert case.path().stat().st_size > 10_000


def test_dark_shaft_detector_finds_both_tools_on_suturing_frame() -> None:
    cap = cv2.VideoCapture(str(CASES["suturing"].path()))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 897)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
    dets = DarkShaftDetector().detect(frame)
    labels = {d.label for d in dets}
    assert labels == {"left", "right"}
    left = next(d for d in dets if d.label == "left")
    right = next(d for d in dets if d.label == "right")
    assert left.tip[0] < right.tip[0]


def test_short_jigsaws_pipeline_completes(tmp_path: Path) -> None:
    src = CASES["knot_tying"].path()
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip = tmp_path / "clip.avi"
    ok, frame = cap.read()
    assert ok
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    writer.write(frame)
    for _ in range(int(fps * 2) - 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()

    result = AnalysisResult(
        id="jigsaws-short",
        status="queued",
        created_at=utc_now(),
        source="demo_knot_tying",
        extras={"task": "knot_tying", "dataset": "JIGSAWS"},
    )
    out = run_pipeline(clip, tmp_path / "out", result)
    assert out.status == "complete"
    assert out.frame_count > 10
    assert (tmp_path / "out" / "overlay.mp4").exists()
    assert out.sequence_expected == list(CASES["knot_tying"].expected_sequence)
    assert out.disclaimer
    assert out.overall_score is not None
