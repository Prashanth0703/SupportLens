from fastapi.testclient import TestClient

from app.main import app
from app.services.ml_predictor import IntentCandidate, IntentPrediction


class FakeIntentPredictor:
    def predict(
        self,
        text: str,
        *,
        top_k: int = 3,
    ) -> IntentPrediction:
        assert text
        assert top_k == 3
        return IntentPrediction(
            intent="recover_password",
            confidence=0.82,
            top_predictions=[
                IntentCandidate("recover_password", 0.82),
                IntentCandidate("registration_problems", 0.11),
                IntentCandidate("edit_account", 0.04),
            ],
            model_version="test-model",
        )


app.state.intent_predictor = FakeIntentPredictor()
app.state.model_error = ""


def test_health_check_reports_loaded_model() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_analyze_ticket_returns_ml_prediction() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tickets/analyze",
            json={
                "subject": "Cannot log in",
                "description": "The password reset link is not working.",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "recover_password"
    assert payload["confidence"] == 0.82
    assert len(payload["top_predictions"]) == 3
    assert payload["model_version"] == "test-model"
    assert payload["priority"] == "medium"
