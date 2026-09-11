export type DimensionScore = {
  key: string;
  label: string;
  score: number;
  detail: string;
};

export type InstrumentMetrics = {
  label: string;
  path_length_px: number;
  displacement_px: number;
  path_efficiency: number;
  mean_velocity_px_s: number;
  max_velocity_px_s: number;
  mean_acceleration_px_s2: number;
  mean_jerk_px_s3: number;
  idle_time_s: number;
  idle_fraction: number;
  workspace_utilization: number;
  corrective_movements: number;
  tip_series: number[][];
};

export type GestureEvent = {
  label: string;
  start_s: number;
  end_s: number;
  instrument: string;
  confidence: number;
};

export type TimelineEvent = {
  t: number;
  kind: string;
  title: string;
  detail: string;
  severity: "green" | "amber" | "red";
};

export type BimanualMetrics = {
  time_sync: number;
  velocity_correlation: number;
  mean_tip_distance_px: number;
  dual_activity_fraction: number;
};

export type AnalysisResult = {
  id: string;
  status: "queued" | "processing" | "complete" | "failed";
  progress: number;
  step: string;
  error: string | null;
  created_at: string;
  source: string;
  duration_s: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  overall_score: number | null;
  dimensions: DimensionScore[];
  instruments: InstrumentMetrics[];
  bimanual: BimanualMetrics | null;
  gestures: GestureEvent[];
  sequence_expected: string[];
  sequence_observed: string[];
  timeline_events: TimelineEvent[];
  notes: string[];
  disclaimer: string;
};

export const PIPELINE_STEPS = [
  "Instrument Detection",
  "Instrument Tracking",
  "Gesture Recognition",
  "Motion Analytics",
  "Technique Scoring",
  "Visual Feedback",
];
