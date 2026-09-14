from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from surgicalvision.constants import LEFT, LEFT_BGR, RIGHT, RIGHT_BGR
from surgicalvision.config import YOLO_WEIGHTS


@dataclass
class Detection:
    label: str
    conf: float
    bbox: tuple[int, int, int, int]
    tip: tuple[float, float]
    centroid: tuple[float, float]


def _bgr_to_hsv(bgr: tuple[int, int, int]) -> np.ndarray:
    px = np.uint8([[bgr]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]


def _mask_hue(hsv: np.ndarray, center_bgr: tuple[int, int, int], h_pad: int = 16) -> np.ndarray:
    h, s, v = (int(x) for x in _bgr_to_hsv(center_bgr))
    lo = np.array([max(0, h - h_pad), 35, 35], dtype=np.uint8)
    hi = np.array([min(179, h + h_pad), 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _detections_from_mask(
    mask: np.ndarray, label: str, entry: tuple[int, int], min_area: float
) -> list[Detection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[Detection] = []
    h, w = mask.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh < min_area:
            continue
        pts = contour.reshape(-1, 2).astype(np.float64)
        dist = np.linalg.norm(pts - np.array(entry, dtype=np.float64), axis=1)
        tip = pts[int(np.argmax(dist))]
        m = cv2.moments(contour)
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        else:
            cx, cy = x + bw / 2, y + bh / 2
        conf = float(np.clip(area / (w * h * 0.08), 0.35, 0.99))
        found.append(
            Detection(
                label=label,
                conf=conf,
                bbox=(int(x), int(y), int(x + bw), int(y + bh)),
                tip=(float(tip[0]), float(tip[1])),
                centroid=(float(cx), float(cy)),
            )
        )
    found.sort(key=lambda d: d.conf, reverse=True)
    return found[:1]


class ColorInstrumentDetector:
    """HSV detector tuned to the synthetic laparoscopic instruments."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        min_area = max(80.0, w * h * 0.002)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        left_mask = _mask_hue(hsv, LEFT_BGR)
        right_mask = _mask_hue(hsv, RIGHT_BGR)
        left = _detections_from_mask(left_mask, LEFT, (0, h), min_area)
        right = _detections_from_mask(right_mask, RIGHT, (w, h), min_area)
        return left + right


class MotionInstrumentDetector:
    """Fallback: elongated moving blobs, labeled by image side."""

    def __init__(self) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=40, varThreshold=16, detectShadows=False
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        fg = self._subtractor.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Detection] = []
        min_area = w * h * 0.004
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if aspect < 1.6:
                continue
            pts = contour.reshape(-1, 2).astype(np.float64)
            entry = (0, h) if (x + bw / 2) < w / 2 else (w, h)
            dist = np.linalg.norm(pts - np.array(entry, dtype=np.float64), axis=1)
            tip = pts[int(np.argmax(dist))]
            label = LEFT if tip[0] < w / 2 else RIGHT
            candidates.append(
                Detection(
                    label=label,
                    conf=0.45,
                    bbox=(x, y, x + bw, y + bh),
                    tip=(float(tip[0]), float(tip[1])),
                    centroid=(x + bw / 2, y + bh / 2),
                )
            )
        by_label: dict[str, Detection] = {}
        for det in sorted(candidates, key=lambda d: d.conf, reverse=True):
            by_label.setdefault(det.label, det)
        return list(by_label.values())


class DarkShaftDetector:
    """Detect dark metallic da Vinci shafts on a bright JIGSAWS workspace.

    Color cues from the synthetic renderer do not apply here. Instruments are
    locally dark, sit on the bright pad, and enter from the left and right
    borders. Orange foam, ceiling clutter, and compact targets (cones) are
    rejected with those priors.
    """

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        mask = _workspace_dark_mask(frame)
        left_mask = mask.copy()
        left_mask[:, int(w * 0.64) :] = 0
        right_mask = mask.copy()
        right_mask[:, : int(w * 0.36)] = 0
        left = _best_shaft(left_mask, LEFT, h, w)
        right = _best_shaft(right_mask, RIGHT, h, w)
        return [d for d in (left, right) if d is not None]


def _workspace_dark_mask(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    orange = (hue <= 22) & (sat >= 85) & (val >= 45)
    bright = ((val > 95) & ~orange).astype(np.uint8) * 255
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    workspace = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k15)
    workspace = cv2.dilate(workspace, k15, iterations=2)
    blur = cv2.GaussianBlur(val, (21, 21), 0)
    locally_dark = (val.astype(np.int16) < blur.astype(np.int16) - 14) & (val < 120)
    abs_dark = val < 70
    dark = ((locally_dark | abs_dark) & ~orange).astype(np.uint8) * 255
    mask = cv2.bitwise_and(dark, workspace)
    mask[: int(0.14 * h), :] = 0
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)
    return mask


def _best_shaft(mask: np.ndarray, label: str, h: int, w: int) -> Detection | None:
    n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scored: list[tuple[float, int]] = []
    for idx in range(1, n_labels):
        x, y, bw, bh, area = (int(v) for v in stats[idx])
        if area < 250 or area > 0.32 * w * h:
            continue
        if bw > 0.72 * w:
            continue
        cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
        if cy < 0.16 * h:
            continue
        aspect = max(bw, bh) / max(1.0, float(min(bw, bh)))
        touches_left = x <= 10
        touches_right = (x + bw) >= (w - 10)
        if touches_left and touches_right:
            continue
        if label == LEFT:
            if touches_right and not touches_left:
                continue
            if cx > 0.70 * w:
                continue
            border = 4.0 if touches_left else (1.2 if cx < 0.45 * w else 0.25)
        else:
            if touches_left and not touches_right:
                continue
            if cx < 0.30 * w:
                continue
            border = 4.0 if touches_right else (1.2 if cx > 0.55 * w else 0.25)
        if aspect < 1.45 and not (touches_left or touches_right):
            continue
        shaft = 0.55 + 0.55 * min(aspect, 6.0)
        score = float(area) * border * shaft
        scored.append((score, idx))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_idx = scored[0][1]
    x, y, bw, bh, area = (int(v) for v in stats[best_idx])
    ys, xs = np.where(cc == best_idx)
    if xs.size == 0:
        return None
    if label == LEFT:
        tip_i = int(np.argmax(xs.astype(np.float64) + 0.08 * ys))
    else:
        tip_i = int(np.argmin(xs.astype(np.float64) - 0.08 * ys))
    conf = float(np.clip(0.45 + min(int(area), 9000) / 14000.0, 0.45, 0.95))
    return Detection(
        label=label,
        conf=conf,
        bbox=(x, y, x + bw, y + bh),
        tip=(float(xs[tip_i]), float(ys[tip_i])),
        centroid=(float(centroids[best_idx][0]), float(centroids[best_idx][1])),
    )


class AdaptiveInstrumentDetector:
    """Prefer the synthetic color detector; fall back to dark shafts for JIGSAWS."""

    def __init__(self) -> None:
        self.color = ColorInstrumentDetector()
        self.dark = DarkShaftDetector()
        self._mode = "auto"
        self._color_hits = 0
        self._frames = 0

    @property
    def mode(self) -> str:
        return self._mode

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._frames += 1
        if self._mode == "color":
            return self.color.detect(frame)
        if self._mode == "dark":
            return self.dark.detect(frame)
        color = self.color.detect(frame)
        if len(color) >= 2:
            self._color_hits += 1
            if self._color_hits >= 3:
                self._mode = "color"
            return color
        dark = self.dark.detect(frame)
        if self._frames >= 8 and self._color_hits == 0:
            self._mode = "dark"
        return dark


class YOLODetector:
    """Optional Ultralytics YOLO hook. Used only when weights are configured."""

    def __init__(self, weights: str) -> None:
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(weights)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        results = self.model.predict(frame, verbose=False)
        if not results:
            return []
        out: list[Detection] = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            label = LEFT if cx < w / 2 else RIGHT
            out.append(
                Detection(
                    label=label,
                    conf=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    tip=(float(cx), float(y1)),
                    centroid=(float(cx), float(cy)),
                )
            )
        return out


def build_detector():
    if YOLO_WEIGHTS:
        try:
            return YOLODetector(YOLO_WEIGHTS)
        except Exception:
            pass
    return AdaptiveInstrumentDetector()
