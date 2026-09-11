from __future__ import annotations

import os
from pathlib import Path


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _workspace_root()
DATA_DIR = Path(os.environ.get("SURGICALVISION_DATA", ROOT / "data" / "analyses"))
YOLO_WEIGHTS = os.environ.get("SURGICALVISION_YOLO_WEIGHTS", "").strip()

DEMO_SECONDS = float(os.environ.get("SURGICALVISION_DEMO_SECONDS", "8"))
DEMO_FPS = int(os.environ.get("SURGICALVISION_DEMO_FPS", "24"))
DEMO_SIZE = (
    int(os.environ.get("SURGICALVISION_DEMO_WIDTH", "960")),
    int(os.environ.get("SURGICALVISION_DEMO_HEIGHT", "540")),
)
