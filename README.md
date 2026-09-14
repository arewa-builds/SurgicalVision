# SurgicalVision

AI surgical technique coach for recorded laparoscopic or robotic simulation.

The system takes a recording, follows the instruments, interprets the gesture sequence, and returns an explainable technique report — overlay video, motion metrics, six-dimension scores, and a review timeline.

```
Video → Detect → Track → Gestures → Motion metrics → Technique scores → Overlay + timeline
```

These are **research-derived technique metrics, not clinical competency scores**. Mapping them onto validated assessments needs expert annotations and a framework such as OSATS.

Built-in demos are real endoscopic trials from **JIGSAWS** (Johns Hopkins University / Intuitive Surgical): suturing, knot tying, and needle passing.

## Capabilities

1. **Instrument tracking & motion efficiency** — left/right tip paths, path length, velocity, acceleration, jerk, idle time, workspace use, corrective reversals
2. **Task technique assessment** — expected gesture order for the selected JIGSAWS task versus what was observed
3. **Skill profile** — overall score plus Motion Economy, Instrument Control, Bimanual Coordination, Procedural Efficiency, Tissue Handling, Error Avoidance
4. **Gesture recognition** — sliding-window labels on instrument kinematics (heuristic now; a temporal model can replace this stage later)

Green path = efficient, amber = corrective, red = moment worth reviewing.

## Deploy (recommended)

One CPU-only container serves the UI and API. No GPU, no PyTorch.

```bash
docker build -t surgicalvision:local .
docker run --rm -p 8000:8000 -v surgicalvision-data:/data/analyses surgicalvision:local
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** (same as [http://localhost:8000](http://localhost:8000)).

Uvicorn logs `http://0.0.0.0:8000` because that is where the process **binds**. It is not a URL. Opening `http://0.0.0.0:8000` in a browser typically fails with `ERR_CONNECTION_TIMED_OUT`.

Click **Suturing**, **Knot tying**, or **Needle passing** (JIGSAWS), or upload a recording.

With Compose:

```bash
docker compose up --build
```

Health check: `GET /api/health`.

The image is a single process (`uvicorn`, 1 worker). Analysis artifacts live under `/data/analyses`. Stay at **one replica** unless you add shared storage and sticky sessions — jobs are in-process.

### Browser cannot connect

| Symptom | Cause | Fix |
|---|---|---|
| `ERR_CONNECTION_TIMED_OUT` on `http://0.0.0.0:8000` | `0.0.0.0` is a bind address, not a host | Use [http://127.0.0.1:8000](http://127.0.0.1:8000) |
| `ERR_CONNECTION_REFUSED` on localhost | Container is not running or port 8000 is not published | `docker ps` and rerun with `-p 8000:8000` |
| Page never loads after `docker run` without `-p` | Port not mapped to the host | Always include `-p 8000:8000` |

Confirm the API from the host:

```bash
curl http://127.0.0.1:8000/api/health
```

You should see `{"status":"ok","version":"..."}`.

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
| Suturing | JIGSAWS subject D, trial 005 — bimanual suturing on the foam pad |
| Knot tying | Same capture, knot-tying bench |
| Needle passing | Same capture, numbered-ring needle passing |
| Upload video | Same pipeline on your file. Detection is tuned for dark metallic tools on a bright workspace |

The report includes overlay playback, overall score, six dimensions, expected vs observed gestures, left/right metrics, and a seekable timeline.

## Dataset citation

Built-in demos are endoscopic videos from the **JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS)**, collected at Johns Hopkins University with the da Vinci Surgical System (Intuitive Surgical). The bundled clips are subject D, trial 005, capture 1:

| Task | File |
|---|---|
| Suturing | `backend/surgicalvision/demo/jigsaws/suturing.avi` |
| Knot tying | `backend/surgicalvision/demo/jigsaws/knot_tying.avi` |
| Needle passing | `backend/surgicalvision/demo/jigsaws/needle_passing.avi` |

If you use these recordings, please cite:

Gao, Y., Vedula, S. S., Reiley, C. E., Ahmidi, N., Varadarajan, B., Lin, H. C., Tao, L., Zappella, L., Béjar, B., Yuh, D. D., Chen, C. C. G., Vidal, R., Khudanpur, S., & Hager, G. D. (2014). The JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS): A surgical activity dataset for human motion modeling. In *Modeling and Monitoring of Computer Assisted Interventions (M2CAI) – MICCAI Workshop*.

Ahmidi, N., Tao, L., Sefati, S., Gao, Y., Lea, C., Bejar Haro, B., Zappella, L., Khudanpur, S., Vidal, R., & Hager, G. D. (2017). A dataset and benchmarks for segmentation and recognition of gestures in robotic surgery. *IEEE Transactions on Biomedical Engineering*.

Dataset page: [CIRL / JHU JIGSAWS release](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/). Clip-level notes are also in [`backend/surgicalvision/demo/jigsaws/NOTICE.md`](backend/surgicalvision/demo/jigsaws/NOTICE.md).

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
| `SURGICALVISION_YOLO_WEIGHTS` | unset | Optional detector weights; not in the base image |
| `PORT` | `8000` | Listen port inside the container |

Synthetic OpenCV videos are still generated in unit tests (`SURGICALVISION_DEMO_*` in `config.py`). The product demos are the bundled JIGSAWS clips.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness |
| `GET` | `/api/demo-cases` | JIGSAWS demo metadata |
| `POST` | `/api/analyses/demo` | `{ "profile": "suturing" \| "knot_tying" \| "needle_passing" }` |
| `POST` | `/api/analyses` | Multipart `file` upload |
| `GET` | `/api/analyses/{id}` | Status and report JSON |
| `GET` | `/api/analyses/{id}/overlay` | Annotated MP4 |
| `GET` | `/api/analyses/{id}/original` | Source video |

Poll `GET /api/analyses/{id}` until `status` is `complete` or `failed`. A full JIGSAWS trial is 40–70 seconds; analysis streams frames so it stays within the 1Gi container limit.

## Layout

```
backend/surgicalvision/
  api/          FastAPI jobs + static UI
  pipeline/     detect, track, gestures, metrics, scoring, overlay
  demo/         JIGSAWS cases + synthetic renderer for tests
  demo/jigsaws/ bundled endoscopic clips + NOTICE
frontend/       React dashboard (baked into the image)
deploy/k8s.yaml Deployment + Service
Dockerfile      Multi-stage: Node build → Python runtime
```

## What’s in this phase vs later

**Now:** JIGSAWS demo trials, dark-shaft / color / motion detectors, tip tracking, kinematic gestures, research scores, overlay dashboard, Docker/Kubernetes packaging.

**Later (keep them out of the default image):** YOLO + ByteTrack on surgical datasets (EndoVis), a video transformer for gestures, correlation with expert ratings / OSATS. Optional weights can mount through `SURGICALVISION_YOLO_WEIGHTS` without baking PyTorch into the base container.
