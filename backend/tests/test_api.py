from __future__ import annotations

from fastapi.testclient import TestClient

from surgicalvision.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unknown_analysis_404() -> None:
    response = client.get("/api/analyses/does-not-exist")
    assert response.status_code == 404
