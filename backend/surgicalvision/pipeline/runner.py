from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from surgicalvision.constants import DISCLAIMER, EXPECTED_SUTURE_SEQUENCE, PIPELINE_STEPS
from surgicalvision.demo.cases import CASES
from surgicalvision.pipeline.detection import (
    AdaptiveInstrumentDetector,
    ColorInstrumentDetector,
    MotionInstrumentDetector,
    build_detector,
)
from surgicalvision.pipeline.gestures import observed_sequence, recognize_gestures
from surgicalvision.pipeline.metrics import metrics_for_tracks
from surgicalvision.pipeline.overlay import build_timeline, render_overlay
from surgicalvision.pipeline.scoring import score_analysis
from surgicalvision.pipeline.tracking import track_instruments
from surgicalvision.schemas import AnalysisResult


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    cap.release()
    return fps if fps > 1 else 24.0


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    result: AnalysisResult,
    on_progress: Callable[[AnalysisResult], None] | None = None,
) -> AnalysisResult:
    def bump(step_index: int) -> None:
        result.status = "processing"
        result.step = PIPELINE_STEPS[step_index]
        result.progress = int(round(100 * (step_index + 1) / len(PIPELINE_STEPS)))
        if on_progress:
            on_progress(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    fps = video_fps(video_path)
    result.fps = round(fps, 3)

    bump(0)
    detector = build_detector()
    detections = []
    size: tuple[int, int] | None = None
    for frame in iter_frames(video_path):
        if size is None:
            h, w = frame.shape[:2]
            size = (w, h)
        detections.append(detector.detect(frame))
    if not detections or size is None:
        raise RuntimeError(f"Video contained no frames: {video_path}")

    result.frame_count = len(detections)
    result.width, result.height = size
    result.duration_s = round(len(detections) / fps, 3)

    hits = sum(len(d) for d in detections)
    if hits < max(8, len(detections) // 4):
        motion = MotionInstrumentDetector()
        detections = [motion.detect(frame) for frame in iter_frames(video_path)]
        result.notes.append("Primary detector found few instruments; used motion fallback.")
        result.extras["detector"] = "MotionInstrumentDetector"
    elif isinstance(detector, AdaptiveInstrumentDetector):
        result.extras["detector"] = detector.mode if detector.mode != "auto" else type(detector).__name__
    elif isinstance(detector, ColorInstrumentDetector):
        result.extras["detector"] = "ColorInstrumentDetector"
    else:
        result.extras["detector"] = type(detector).__name__

    bump(1)
    tracks = track_instruments(detections, len(detections))
    result.extras["n_tracks"] = len(tracks)

    task_id = str(result.extras.get("task") or "")
    if not task_id and result.source.startswith("demo_"):
        task_id = result.source.removeprefix("demo_")
    case = CASES.get(task_id)
    expected = case.expected_sequence if case else EXPECTED_SUTURE_SEQUENCE
    expected_duration = case.expected_duration_s if case else max(result.duration_s, 9.0)

    bump(2)
    gestures = recognize_gestures(tracks, fps, size)
    sequence = observed_sequence(gestures)
    result.gestures = gestures
    result.sequence_expected = list(expected)
    result.sequence_observed = sequence

    bump(3)
    instruments, bimanual = metrics_for_tracks(tracks, fps, size)
    result.instruments = instruments
    result.bimanual = bimanual

    bump(4)
    overall, dimensions, notes = score_analysis(
        instruments,
        bimanual,
        sequence,
        result.duration_s,
        expected=expected,
        expected_duration_s=expected_duration,
    )
    result.overall_score = overall
    result.dimensions = dimensions
    result.notes.extend(notes)
    result.disclaimer = DISCLAIMER
    result.timeline_events = build_timeline(tracks, gestures, fps, result.duration_s)

    bump(5)
    overlay_path = output_dir / "overlay.mp4"
    render_overlay(video_path, tracks, gestures, result.timeline_events, fps, overlay_path)

    result.status = "complete"
    result.progress = 100
    result.step = "Complete"
    if on_progress:
        on_progress(result)
    return result
