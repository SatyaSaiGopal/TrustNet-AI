"""Tests for the TrustNet AI FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest

from trustnet.api.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_api_status(client: TestClient) -> None:
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "connected"
    assert data["version"] == "0.1.0"
    assert "min_samples" in data


def test_api_models(client: TestClient) -> None:
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "installed_models" in data
    assert "active_model" in data


def test_api_showcase_cases(client: TestClient) -> None:
    res = client.get("/api/showcase/cases")
    assert res.status_code == 200
    cases = res.json()
    assert len(cases) >= 7
    domains = [c["domain"] for c in cases]
    assert "healthcare" in domains
    assert "cybersecurity" in domains
    assert "finance" in domains
    assert "engineering" in domains


def test_api_demo_cases(client: TestClient) -> None:
    res = client.get("/api/demo/cases")
    assert res.status_code == 200
    cases = res.json()
    assert isinstance(cases, list)
    assert len(cases) > 0
    first = cases[0]
    assert "task_id" in first
    assert "question" in first
    assert "category" in first


def test_api_evaluate_synthetic_task(client: TestClient) -> None:
    cases = client.get("/api/demo/cases").json()
    case = cases[0]

    res = client.post(
        "/api/task/evaluate",
        json={
            "question": case["question"],
            "category": case["category"],
            "mode": "synthetic",
            "task_id": case["task_id"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "result" in data
    assert "decision" in data
    assert "conflicting_answers" in data
    assert "trust" in data
    assert data["decision"]["outcome"] in ("decided", "abstained")


def test_api_database_seed(client: TestClient) -> None:
    res = client.post("/api/database/seed")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "tasks_seeded" in data


def test_api_get_source_document(client: TestClient) -> None:
    res = client.get("/api/sources/cve_2024_auth_bypass.txt")
    assert res.status_code == 200
    data = res.json()
    assert "CVE-2024-8812" in data["content"]


def test_api_benchmark_endpoint(client: TestClient) -> None:
    res = client.get("/api/benchmark?seed=7&development_size=30&test_size=30")
    assert res.status_code == 200
    report = res.json()
    assert "strategies" in report
    assert "agents" in report


def test_api_providers_endpoint(client: TestClient) -> None:
    res = client.get("/api/providers")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert any(p["provider"] == "openai" for p in data["presets"])
    assert any(p["provider"] == "ollama" for p in data["presets"])


def test_api_evaluate_cloud_without_api_key_fails_cleanly(client: TestClient) -> None:
    res = client.post(
        "/api/task/evaluate",
        json={
            "question": "Is this a valid task?",
            "mode": "live",
            "provider": "openai",
            "api_key": "",
        },
    )
    assert res.status_code == 400
    assert "API key is required" in res.json()["detail"]


def test_api_serves_frontend(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "TrustNet AI" in res.text
    assert "Decision Console" in res.text


def test_api_serves_head_on_root_for_cloud_healthchecks(client: TestClient) -> None:
    res = client.head("/")
    assert res.status_code == 200


def test_api_healthz_and_health_endpoints(client: TestClient) -> None:
    res_get = client.get("/healthz")
    assert res_get.status_code == 200
    assert res_get.json()["status"] == "ok"

    res_head = client.head("/healthz")
    assert res_head.status_code == 200

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"


