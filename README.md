# SurgicalVision

AI surgical technique coach for recorded laparoscopic or robotic simulation.

The platform runs one pipeline:

**Video → instrument detection → tracking → gesture recognition → motion analytics → technique scoring → overlay + timeline**

Scores are **research-derived technique metrics**, not clinical competency scores. Mapping them onto validated assessments needs expert annotations and frameworks such as OSATS.

The runtime is intentionally light: one process, no GPU, no PyTorch, CPU OpenCV only. The UI is static files served by the API so the whole app fits in a single container.

## Run with Docker

One container serves the UI and API. No GPU, no Compose plugin required:

```bash
docker build -t surgicalvision:local .
docker run --rm -p 8000:8000 -v surgicalvision-data:/data/analyses surgicalvision:local
```

Open [http://localhost:8000](http://localhost:8000). If the Compose plugin is installed, `docker compose up --build` does the same thing.

## Kubernetes

The image is a single Deployment (one replica — jobs are in-process). Apply after loading the image into the cluster:

```bash
docker build -t surgicalvision:local .
kubectl apply -f deploy/k8s.yaml
kubectl port-forward svc/surgicalvision 8000:80
```

Point `image:` at your registry when you push (`ghcr.io/<org>/surgicalvision:tag`). Attach a PersistentVolumeClaim instead of `emptyDir` if analyses should survive pod restarts.

## Run locally (dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cd frontend && npm install && cd ..

PYTHONPATH=backend .venv/bin/python -m uvicorn surgicalvision.api.main:app --reload --port 8000
cd frontend && npm run dev
```

Dev UI: [http://localhost:5173](http://localhost:5173). Or build the UI and use the same origin as production:

```bash
cd frontend && npm run build && cd ..
PYTHONPATH=backend SURGICALVISION_STATIC=frontend/dist \
  .venv/bin/python -m uvicorn surgicalvision.api.main:app --port 8000
```

```bash
PYTHONPATH=backend .venv/bin/pytest
```

## Layout

- `backend/surgicalvision/pipeline/` — detect, track, gestures, metrics, scoring, overlay
- `backend/surgicalvision/demo/` — synthetic laparoscopic scene
- `backend/surgicalvision/api/` — FastAPI jobs + static UI
- `frontend/` — React dashboard (baked into the image)
- `Dockerfile` / `docker-compose.yml` / `deploy/k8s.yaml` — single-container deploy

## Later phases

1. Train YOLO + ByteTrack on surgical instrument datasets (EndoVis)
2. Temporal gesture model and JIGSAWS suturing/knot tasks
3. Correlate metrics with expert ratings / OSATS; keep the research disclaimer until then

Keep heavy model runtimes out of the default image. Optional weights can mount via `SURGICALVISION_YOLO_WEIGHTS` without baking PyTorch into the base container.
