from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from surgicalvision.constants import DISCLAIMER, EXPECTED_SUTURE_SEQUENCE, PIPELINE_STEPS
from surgicalvision.pipeline.detection import ColorInstrumentDetector, MotionInstrumentDetector, build_detector
from surgicalvision.pipeline.gestures import observed_sequence, recognize_gestures
from surgicalvision.pipeline.metrics import metrics_for_tracks
from surgicalvision.pipeline.overlay import build_timeline, render_overlay
from surgicalvision.pipeline.scoring import score_analysis
from surgicalvision.pipeline.tracking import track_instruments
from surgicalvision.schemas import AnalysisResult


def load_frames(path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Video contained no frames: {path}")
    h, w = frames[0].shape[:2]
    return frames, fps if fps > 1 else 24.0, (w, h)


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
    frames, fps, size = load_frames(video_path)
    result.fps = round(fps, 3)
    result.frame_count = len(frames)
    result.width, result.height = size
    result.duration_s = round(len(frames) / fps, 3)

    bump(0)
    detector = build_detector()
    detections = [detector.detect(frame) for frame in frames]
    color_hits = sum(len(d) for d in detections)
    if color_hits < max(8, len(frames) // 4):
        motion = MotionInstrumentDetector()
        detections = [motion.detect(frame) for frame in frames]
        result.notes.append("Color detector found few instruments; used motion fallback.")
    if isinstance(detector, ColorInstrumentDetector) is False:
        result.extras["detector"] = type(detector).__name__
    else:
        result.extras["detector"] = "ColorInstrumentDetector"

    bump(1)
    tracks = track_instruments(detections, len(frames))
    result.extras["n_tracks"] = len(tracks)

    bump(2)
    gestures = recognize_gestures(tracks, fps, size)
    sequence = observed_sequence(gestures)
    result.gestures = gestures
    result.sequence_expected = list(EXPECTED_SUTURE_SEQUENCE)
    result.sequence_observed = sequence

    bump(3)
    instruments, bimanual = metrics_for_tracks(tracks, fps, size)
    result.instruments = instruments
    result.bimanual = bimanual

    bump(4)
    overall, dimensions, notes = score_analysis(
        instruments, bimanual, sequence, result.duration_s
    )
    result.overall_score = overall
    result.dimensions = dimensions
    result.notes.extend(notes)
    result.disclaimer = DISCLAIMER
    result.timeline_events = build_timeline(tracks, gestures, fps, result.duration_s)

    bump(5)
    overlay_path = output_dir / "overlay.mp4"
    render_overlay(frames, tracks, gestures, result.timeline_events, fps, overlay_path)

    result.status = "complete"
    result.progress = 100
    result.step = "Complete"
    if on_progress:
        on_progress(result)
    return result
