from dataclasses import dataclass
from typing import cast

from app.schemas.ticket import (
    TicketAnalysisRequest,
    TicketAnalysisResponse,
    TicketCategory,
    TicketPriority,
)


@dataclass(frozen=True)
class KeywordRule:
    keywords: tuple[str, ...]
    label: TicketCategory


class KeywordBaselinePredictor:
    """Transparent integration baseline, not a trained ML model."""

    model_version = "keyword_baseline_v0"

    category_rules: tuple[KeywordRule, ...] = (
        KeywordRule(
            ("invoice", "charged", "payment", "refund", "billing", "price"),
            "billing",
        ),
        KeywordRule(
            ("login", "password", "locked", "sign in", "account access"),
            "account_access",
        ),
        KeywordRule(
            ("error", "crash", "broken", "not working", "timeout", "bug"),
            "technical_issue",
        ),
        KeywordRule(
            ("feature", "request", "would like", "enhancement", "suggestion"),
            "feature_request",
        ),
    )

    high_priority_keywords = (
        "urgent",
        "production",
        "security",
        "data loss",
        "cannot access",
        "outage",
        "blocked",
    )
    medium_priority_keywords = (
        "error",
        "not working",
        "failed",
        "unable",
        "incorrect",
    )

    def predict(self, ticket: TicketAnalysisRequest) -> TicketAnalysisResponse:
        text = f"{ticket.subject} {ticket.description}".lower()
        category, category_hits = self._predict_category(text)
        priority, priority_hits = self._predict_priority(text)

        total_hits = len(category_hits) + len(priority_hits)
        confidence = min(0.55 + (0.08 * total_hits), 0.91)

        reasons: list[str] = []
        if category_hits:
            reasons.append(
                "Category keywords found: " + ", ".join(sorted(category_hits))
            )
        else:
            reasons.append("No category keyword matched; used the general category.")

        if priority_hits:
            reasons.append(
                "Priority keywords found: " + ", ".join(sorted(priority_hits))
            )
        else:
            reasons.append("No urgency keyword matched; used low priority.")

        return TicketAnalysisResponse(
            category=category,
            priority=priority,
            confidence=round(confidence, 2),
            model_version=self.model_version,
            reasons=reasons,
        )

    def _predict_category(
        self, text: str
    ) -> tuple[TicketCategory, set[str]]:
        best_label: TicketCategory = "general"
        best_hits: set[str] = set()

        for rule in self.category_rules:
            hits = {keyword for keyword in rule.keywords if keyword in text}
            if len(hits) > len(best_hits):
                best_hits = hits
                best_label = cast(TicketCategory, rule.label)

        return best_label, best_hits

    def _predict_priority(
        self, text: str
    ) -> tuple[TicketPriority, set[str]]:
        high_hits = {
            keyword for keyword in self.high_priority_keywords if keyword in text
        }
        if high_hits:
            return "high", high_hits

        medium_hits = {
            keyword for keyword in self.medium_priority_keywords if keyword in text
        }
        if medium_hits:
            return "medium", medium_hits

        return "low", set()
