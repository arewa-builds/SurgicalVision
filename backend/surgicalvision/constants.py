"""Shared labels, colors, and product copy."""

from __future__ import annotations

LEFT = "left"
RIGHT = "right"

# BGR colors used by the synthetic renderer and the color detector.
LEFT_BGR = (255, 220, 40)
RIGHT_BGR = (20, 170, 250)

GESTURE_VOCABULARY = (
    "idle",
    "reach",
    "position",
    "grasp",
    "manipulate",
    "cut",
    "suture",
    "knot",
    "release",
    "reposition",
)

EXPECTED_SUTURE_SEQUENCE = (
    "reach",
    "position",
    "grasp",
    "suture",
    "knot",
    "release",
)

DIMENSIONS = (
    ("motion_economy", "Motion Economy"),
    ("instrument_control", "Instrument Control"),
    ("bimanual_coordination", "Bimanual Coordination"),
    ("procedural_efficiency", "Procedural Efficiency"),
    ("tissue_handling", "Tissue Handling"),
    ("error_avoidance", "Error Avoidance"),
)

DISCLAIMER = (
    "These are research-derived technique metrics, not clinical competency scores. "
    "Converting them into validated assessments requires expert surgeon annotations "
    "and comparison with established frameworks such as OSATS."
)

PIPELINE_STEPS = (
    "Instrument Detection",
    "Instrument Tracking",
    "Gesture Recognition",
    "Motion Analytics",
    "Technique Scoring",
    "Visual Feedback",
)
