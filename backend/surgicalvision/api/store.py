from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from surgicalvision.config import DATA_DIR
from surgicalvision.schemas import AnalysisResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnalysisStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else DATA_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._items: dict[str, AnalysisResult] = {}

    def create(self, source: str) -> AnalysisResult:
        analysis_id = uuid.uuid4().hex[:12]
        result = AnalysisResult(
            id=analysis_id,
            status="queued",
            progress=0,
            step="Queued",
            created_at=utc_now(),
            source=source,
        )
        path = self.dir_for(analysis_id)
        path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._items[analysis_id] = result
        self._write(result)
        return result

    def dir_for(self, analysis_id: str) -> Path:
        return self.root / analysis_id

    def get(self, analysis_id: str) -> AnalysisResult | None:
        with self._lock:
            cached = self._items.get(analysis_id)
        if cached:
            return cached
        payload = self.dir_for(analysis_id) / "result.json"
        if not payload.exists():
            return None
        result = AnalysisResult.model_validate_json(payload.read_text())
        with self._lock:
            self._items[analysis_id] = result
        return result

    def save(self, result: AnalysisResult) -> None:
        with self._lock:
            self._items[result.id] = result.model_copy(deep=True)
        self._write(result)

    def _write(self, result: AnalysisResult) -> None:
        path = self.dir_for(result.id) / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.model_dump(), indent=2))

    def overlay_path(self, analysis_id: str) -> Path:
        return self.dir_for(analysis_id) / "overlay.mp4"

    def original_path(self, analysis_id: str) -> Path:
        original = self.dir_for(analysis_id) / "original.mp4"
        if original.exists():
            return original
        return self.dir_for(analysis_id) / "source.mp4"
