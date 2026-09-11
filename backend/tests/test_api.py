from __future__ import annotations

from fastapi.testclient import TestClient

from surgicalvision.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_demo_cases_list_jigsaws_tasks() -> None:
    response = client.get("/api/demo-cases")
    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert ids == {"suturing", "knot_tying", "needle_passing"}


def test_unknown_analysis_404() -> None:
    response = client.get("/api/analyses/does-not-exist")
    assert response.status_code == 404
