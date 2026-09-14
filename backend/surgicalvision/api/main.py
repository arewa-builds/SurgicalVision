from __future__ import annotations

import os
import shutil
import threading
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from surgicalvision import __version__
from surgicalvision.api.store import AnalysisStore
from surgicalvision.config import DATA_DIR, PORT, STATIC_DIR
from surgicalvision.demo.cases import CASES, get_case
from surgicalvision.pipeline.runner import run_pipeline
from surgicalvision.schemas import AnalysisResult, DemoRequest

store = AnalysisStore()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    print(
        f"SurgicalVision ready. Open http://127.0.0.1:{PORT}  "
        "(not http://0.0.0.0 — that bind address times out in browsers)",
        flush=True,
    )
    yield


app = FastAPI(title="SurgicalVision", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SURGICALVISION_CORS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _run_job(analysis_id: str, video_path: Path) -> None:
    result = store.get(analysis_id)
    if result is None:
        return
    result.status = "processing"
    store.save(result)
    try:
        run_pipeline(video_path, store.dir_for(analysis_id), result, on_progress=store.save)
    except Exception as exc:  # noqa: BLE001 — surface pipeline failures to the client
        result.status = "failed"
        result.error = str(exc)
        result.step = "Failed"
        result.extras["traceback"] = traceback.format_exc()
        store.save(result)


def _spawn(analysis_id: str, video_path: Path) -> None:
    thread = threading.Thread(target=_run_job, args=(analysis_id, video_path), daemon=True)
    thread.start()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/demo-cases")
def list_demo_cases() -> list[dict]:
    return [case.as_public() for case in CASES.values()]


@app.post("/api/analyses/demo", response_model=AnalysisResult)
def start_demo(body: DemoRequest) -> AnalysisResult:
    try:
        case = get_case(body.profile)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    src = case.path()
    if not src.exists():
        raise HTTPException(500, f"Demo clip missing: {src.name}")
    result = store.create(source=f"demo_{case.id}")
    result.extras["task"] = case.id
    result.extras["dataset"] = case.dataset
    result.notes.append(case.attribution)
    store.save(result)
    dest = store.dir_for(result.id) / f"original{src.suffix.lower()}"
    shutil.copyfile(src, dest)
    _spawn(result.id, dest)
    return result


@app.post("/api/analyses", response_model=AnalysisResult)
async def upload_video(file: UploadFile = File(...)) -> AnalysisResult:
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Upload an mp4, mov, mkv, avi, or webm recording.")
    result = store.create(source="upload")
    dest = store.dir_for(result.id) / f"original{suffix}"
    dest.write_bytes(await file.read())
    _spawn(result.id, dest)
    return result


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str) -> AnalysisResult:
    result = store.get(analysis_id)
    if result is None:
        raise HTTPException(404, "Analysis not found")
    return result


@app.get("/api/analyses/{analysis_id}/overlay")
def get_overlay(analysis_id: str) -> FileResponse:
    path = store.overlay_path(analysis_id)
    if not path.exists():
        raise HTTPException(404, "Overlay not ready")
    return FileResponse(path, media_type="video/mp4", filename="overlay.mp4")


@app.get("/api/analyses/{analysis_id}/original")
def get_original(analysis_id: str) -> FileResponse:
    path = store.original_path(analysis_id)
    if not path.exists():
        matches = list(store.dir_for(analysis_id).glob("original.*"))
        if not matches:
            raise HTTPException(404, "Original video not found")
        path = matches[0]
    suffix = path.suffix.lower()
    media = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


def _mount_ui() -> None:
    if not STATIC_DIR.exists():
        return
    assets = STATIC_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return

    @app.get("/")
    def ui_root() -> FileResponse:
        return FileResponse(index)


_mount_ui()
