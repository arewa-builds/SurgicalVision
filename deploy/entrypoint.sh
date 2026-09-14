#!/bin/sh
set -eu
echo
echo "SurgicalVision is listening."
echo "Open  http://127.0.0.1:8000   (or http://localhost:8000)"
echo "Do not open http://0.0.0.0:8000 — that is the container bind address; browsers time out on it."
echo
exec python -m uvicorn surgicalvision.api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
