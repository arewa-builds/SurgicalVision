# SurgicalVision

AI surgical technique coach for recorded laparoscopic or robotic simulation.

The system takes a recording, follows the instruments, interprets the gesture sequence, and returns an explainable technique report — overlay video, motion metrics, six-dimension scores, and a review timeline.

```
Video → Detect → Track → Gestures → Motion metrics → Technique scores → Overlay + timeline
```

These are **research-derived technique metrics, not clinical competency scores**. Mapping them onto validated assessments needs expert annotations and a framework such as OSATS.

## Capabilities

1. **Instrument tracking & motion efficiency** — left/right tip paths, path length, velocity, acceleration, jerk, idle time, workspace use, corrective reversals
2. **Suturing technique assessment** — expected `Reach → Position → Grasp → Suture → Knot → Release` versus what was observed
3. **Skill profile** — overall score plus Motion Economy, Instrument Control, Bimanual Coordination, Procedural Efficiency, Tissue Handling, Error Avoidance
4. **Gesture recognition** — sliding-window labels on instrument kinematics (heuristic now; a temporal model can replace this stage later)

Green path = efficient, amber = corrective, red = moment worth reviewing.

## Deploy (recommended)

One CPU-only container serves the UI and API. No GPU, no PyTorch.

```bash
docker build -t surgicalvision:local .
docker run --rm -p 8000:8000 -v surgicalvision-data:/data/analyses surgicalvision:local
```

Open [http://localhost:8000](http://localhost:8000). Click **Run efficient case** or **Run novice case**, or upload a recording.

With Compose:

```bash
docker compose up --build
```

Health check: `GET /api/health`.

The image is a single process (`uvicorn`, 1 worker). Analysis artifacts live under `/data/analyses`. Stay at **one replica** unless you add shared storage and sticky sessions — jobs are in-process.

## Kubernetes

```bash
docker build -t surgicalvision:local .
kubectl apply -f deploy/k8s.yaml
kubectl port-forward svc/surgicalvision 8000:80
```

`deploy/k8s.yaml` is a Deployment + ClusterIP Service, non-root, 256Mi–1Gi memory, probes on `/api/health`.

Before a real cluster:

1. Push the image and set `spec.template.spec.containers[0].image` (for example `ghcr.io/<org>/surgicalvision:<tag>`).
2. Replace `emptyDir` with a PersistentVolumeClaim if reports should survive pod restarts.
3. Keep `replicas: 1`.

## Use the app

| Action | What you get |
|---|---|
| Efficient simulation | Deliberate bimanual suturing — higher motion-economy / coordination scores |
| Novice simulation | Overshoot, tremor, extra repositioning — lower scores, more review markers |
| Upload video | Same pipeline on your file. Prototype detection prefers high-contrast instruments |

The report includes overlay playback, overall score, six dimensions, expected vs observed gestures, left/right metrics, and a seekable timeline.

## Local development

Python 3.11+, Node 22+, ffmpeg.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cd frontend && npm install && cd ..

# API
PYTHONPATH=backend python -m uvicorn surgicalvision.api.main:app --reload --port 8000

# UI (second terminal)
cd frontend && npm run dev
```

Dev UI: [http://localhost:5173](http://localhost:5173) (proxies `/api` to port 8000).

Production-style, one origin:

```bash
cd frontend && npm run build && cd ..
PYTHONPATH=backend SURGICALVISION_STATIC=frontend/dist \
  python -m uvicorn surgicalvision.api.main:app --port 8000
```

```bash
PYTHONPATH=backend pytest
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SURGICALVISION_DATA` | `data/analyses` (`/data/analyses` in the image) | Analysis JSON, original, overlay |
| `SURGICALVISION_STATIC` | `frontend/dist` | Built UI |
| `SURGICALVISION_CORS` | `*` | Allowed origins, comma-separated |
| `SURGICALVISION_DEMO_SECONDS` | `8` | Synthetic case length |
| `SURGICALVISION_DEMO_FPS` | `24` | Synthetic case frame rate |
| `SURGICALVISION_DEMO_WIDTH` / `_HEIGHT` | `960` / `540` | Synthetic case size |
| `SURGICALVISION_YOLO_WEIGHTS` | unset | Optional detector weights; not in the base image |
| `PORT` | `8000` | Listen port inside the container |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness |
| `POST` | `/api/analyses/demo` | `{ "profile": "efficient" \| "novice" }` |
| `POST` | `/api/analyses` | Multipart `file` upload |
| `GET` | `/api/analyses/{id}` | Status and report JSON |
| `GET` | `/api/analyses/{id}/overlay` | Annotated MP4 |
| `GET` | `/api/analyses/{id}/original` | Source video |

Poll `GET /api/analyses/{id}` until `status` is `complete` or `failed`.

## Layout

```
backend/surgicalvision/
  api/          FastAPI jobs + static UI
  pipeline/     detect, track, gestures, metrics, scoring, overlay
  demo/         synthetic laparoscopic scene
frontend/       React dashboard (baked into the image)
deploy/k8s.yaml Deployment + Service
Dockerfile      Multi-stage: Node build → Python runtime
```

## What’s in this phase vs later

**Now:** color/motion detector, tip tracking, kinematic gestures, research scores, overlay dashboard, Docker/Kubernetes packaging.

**Later (keep them out of the default image):** YOLO + ByteTrack on surgical datasets (EndoVis), a video transformer for gestures, JIGSAWS suturing/knot tasks, correlation with expert ratings / OSATS. Optional weights can mount through `SURGICALVISION_YOLO_WEIGHTS` without baking PyTorch into the base container.
