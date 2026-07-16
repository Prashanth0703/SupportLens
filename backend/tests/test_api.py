from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_ticket_returns_transparent_baseline() -> None:
    response = client.post(
        "/api/v1/tickets/analyze",
        json={
            "subject": "Urgent login issue",
            "description": "I cannot access my account after changing my password.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "account_access"
    assert payload["priority"] == "high"
    assert payload["model_version"] == "keyword_baseline_v0"
    assert payload["reasons"]
