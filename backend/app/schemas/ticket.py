from typing import Literal

from pydantic import BaseModel, Field

TicketCategory = Literal[
    "billing",
    "account_access",
    "technical_issue",
    "feature_request",
    "general",
]
TicketPriority = Literal["low", "medium", "high"]


class TicketAnalysisRequest(BaseModel):
    subject: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=5000)


class TicketAnalysisResponse(BaseModel):
    category: TicketCategory
    priority: TicketPriority
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    reasons: list[str]
