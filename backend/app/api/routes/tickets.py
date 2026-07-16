from fastapi import APIRouter

from app.schemas.ticket import TicketAnalysisRequest, TicketAnalysisResponse
from app.services.predictor import KeywordBaselinePredictor

router = APIRouter(prefix="/tickets", tags=["tickets"])
predictor = KeywordBaselinePredictor()


@router.post("/analyze", response_model=TicketAnalysisResponse)
def analyze_ticket(payload: TicketAnalysisRequest) -> TicketAnalysisResponse:
    return predictor.predict(payload)
