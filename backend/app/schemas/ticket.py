from typing import Literal

from pydantic import BaseModel, Field

TicketPriority = Literal["low", "medium", "high"]


class TicketAnalysisRequest(BaseModel):
    subject: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=5000)


class TopPrediction(BaseModel):
    intent: str
    probability: float = Field(ge=0.0, le=1.0)


class TicketAnalysisResponse(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[TopPrediction]
    priority: TicketPriority
    model_version: str
    warnings: list[str]
    priority_reasons: list[str]
