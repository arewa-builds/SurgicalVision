# SurgicalVision

AI surgical technique coach for recorded laparoscopic or robotic simulation.

The platform runs one pipeline:

**Video → instrument detection → tracking → gesture recognition → motion analytics → technique scoring → overlay + timeline**

Scores are **research-derived technique metrics**, not clinical competency scores. Mapping them onto validated assessments needs expert annotations and frameworks such as OSATS.

## Phase 1

- Synthetic efficient vs novice suturing cases (the loop runs without hospital video or trained weights)
- Color/motion instrument detection, plus a YOLO hook via `SURGICALVISION_YOLO_WEIGHTS`
- Left/right tip tracking and kinematic metrics (path, velocity, jerk, idle, workspace, reversals, bimanual sync)
- Sliding-window gesture labels and expected vs observed sequence
- Six-dimension technique scores plus overall
- Overlay video: green efficient path, amber corrective, red review events
- Web dashboard: upload or run a simulation, watch the overlay, seek the review timeline

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install && cd ..

# API
PYTHONPATH=backend .venv/bin/python -m uvicorn surgicalvision.api.main:app --reload --port 8000

# UI (second terminal)
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Use **Run efficient case** or **Run novice case**, then inspect scores, gestures, and the overlay.

```bash
PYTHONPATH=backend .venv/bin/pytest
```

## Layout

- `backend/surgicalvision/pipeline/` — detect, track, gestures, metrics, scoring, overlay
- `backend/surgicalvision/demo/` — synthetic laparoscopic scene
- `backend/surgicalvision/api/` — FastAPI jobs
- `frontend/` — React dashboard

## Later phases

1. Train YOLO + ByteTrack on surgical instrument datasets (EndoVis)
2. Temporal gesture model and JIGSAWS suturing/knot tasks
3. Correlate metrics with expert ratings / OSATS; keep the research disclaimer until then
