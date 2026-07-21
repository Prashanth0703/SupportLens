from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Priority = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class PriorityPrediction:
    priority: Priority
    reasons: list[str]


class PriorityRulePredictor:
    high_priority_keywords = (
        "urgent", "production", "security", "data loss",
        "cannot access", "outage", "blocked", "fraud",
    )
    medium_priority_keywords = (
        "error", "not working", "failed", "unable",
        "incorrect", "declined", "damaged", "late",
    )

    def predict(self, text: str) -> PriorityPrediction:
        normalised = text.lower()

        high_hits = sorted(
            keyword for keyword in self.high_priority_keywords
            if keyword in normalised
        )
        if high_hits:
            return PriorityPrediction(
                priority="high",
                reasons=["High-priority terms found: " + ", ".join(high_hits)],
            )

        medium_hits = sorted(
            keyword for keyword in self.medium_priority_keywords
            if keyword in normalised
        )
        if medium_hits:
            return PriorityPrediction(
                priority="medium",
                reasons=[
                    "Medium-priority terms found: " + ", ".join(medium_hits)
                ],
            )

        return PriorityPrediction(
            priority="low",
            reasons=["No urgency term matched; used low priority."],
        )
