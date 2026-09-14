# syntax=docker/dockerfile:1
# Single lightweight image: Vite UI baked in, FastAPI + OpenCV, no PyTorch/GPU.

FROM node:22-alpine AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    SURGICALVISION_DATA=/data/analyses \
    SURGICALVISION_STATIC=/app/frontend/dist \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

COPY backend /app/backend
COPY --from=frontend /src/dist /app/frontend/dist
COPY deploy/entrypoint.sh /app/entrypoint.sh

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/analyses \
    && chmod +x /app/entrypoint.sh \
    && chown -R appuser:appuser /app /data

USER 10001
EXPOSE 8000
VOLUME ["/data/analyses"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["/app/entrypoint.sh"]
