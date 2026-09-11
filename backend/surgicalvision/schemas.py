from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    key: str
    label: str
    score: float
    detail: str


class InstrumentMetrics(BaseModel):
    label: str
    path_length_px: float
    displacement_px: float
    path_efficiency: float
    mean_velocity_px_s: float
    max_velocity_px_s: float
    mean_acceleration_px_s2: float
    mean_jerk_px_s3: float
    idle_time_s: float
    idle_fraction: float
    workspace_utilization: float
    corrective_movements: int
    tip_series: list[list[float]] = Field(
        default_factory=list,
        description="Downsampled [t_seconds, x, y] samples.",
    )


class GestureEvent(BaseModel):
    label: str
    start_s: float
    end_s: float
    instrument: str
    confidence: float


class TimelineEvent(BaseModel):
    t: float
    kind: str
    title: str
    detail: str
    severity: Literal["green", "amber", "red"]


class BimanualMetrics(BaseModel):
    time_sync: float
    velocity_correlation: float
    mean_tip_distance_px: float
    dual_activity_fraction: float


class AnalysisResult(BaseModel):
    id: str
    status: Literal["queued", "processing", "complete", "failed"]
    progress: int = 0
    step: str = ""
    error: str | None = None
    created_at: str
    source: str
    duration_s: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    overall_score: float | None = None
    dimensions: list[DimensionScore] = Field(default_factory=list)
    instruments: list[InstrumentMetrics] = Field(default_factory=list)
    bimanual: BimanualMetrics | None = None
    gestures: list[GestureEvent] = Field(default_factory=list)
    sequence_expected: list[str] = Field(default_factory=list)
    sequence_observed: list[str] = Field(default_factory=list)
    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)


class DemoRequest(BaseModel):
    profile: Literal["efficient", "novice"] = "efficient"
