from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.ticket import (
    TicketAnalysisRequest,
    TicketAnalysisResponse,
    TopPrediction,
)
from app.services.priority import PriorityRulePredictor

router = APIRouter(prefix="/tickets", tags=["tickets"])
priority_predictor = PriorityRulePredictor()


@router.post("/analyze", response_model=TicketAnalysisResponse)
def analyze_ticket(
    payload: TicketAnalysisRequest,
    request: Request,
) -> TicketAnalysisResponse:
    predictor = getattr(request.app.state, "intent_predictor", None)

    if predictor is None:
        model_error = getattr(
            request.app.state,
            "model_error",
            "The intent model is unavailable.",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=model_error,
        )

    text = f"{payload.subject}. {payload.description}"
    intent_result = predictor.predict(text, top_k=3)
    priority_result = priority_predictor.predict(text)

    top_predictions = intent_result.top_predictions

    prediction_margin = (
        top_predictions[0].probability
        - top_predictions[1].probability
    )

    warnings: list[str] = []
    if intent_result.confidence < 0.60:
        warnings.append(
            "Low-confidence prediction. Human review is recommended."
        )

    if prediction_margin < 0.15:
        warnings.append(
            "The top predictions are close. The intent may be ambiguous."
        )

    return TicketAnalysisResponse(
        intent=intent_result.intent,
        confidence=intent_result.confidence,
        top_predictions=[
            TopPrediction(
                intent=candidate.intent,
                probability=candidate.probability,
            )
            for candidate in intent_result.top_predictions
        ],
        priority=priority_result.priority,
        model_version=intent_result.model_version,
        warnings=warnings,
        priority_reasons=priority_result.reasons,
    )
