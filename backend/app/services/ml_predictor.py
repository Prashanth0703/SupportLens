from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class IntentCandidate:
    intent: str
    probability: float


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    top_predictions: list[IntentCandidate]
    model_version: str


class IntentPredictor:
    def __init__(
        self,
        model: Any,
        *,
        model_version: str = "word-tfidf-logreg-v1",
    ) -> None:
        if not hasattr(model, "predict_proba"):
            raise TypeError("Loaded model must expose predict_proba().")
        if not hasattr(model, "classes_"):
            raise TypeError("Loaded model must expose fitted classes_.")
        self._model = model
        self._model_version = model_version

    @classmethod
    def from_joblib(cls, model_path: Path) -> "IntentPredictor":
        if not model_path.exists():
            raise FileNotFoundError(
                f"SupportLens model was not found at: {model_path}"
            )
        return cls(joblib.load(model_path))

    def predict(
        self,
        text: str,
        *,
        top_k: int = 3,
    ) -> IntentPrediction:
        probabilities = self._model.predict_proba([text])[0]
        classes = np.asarray(self._model.classes_)
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        candidates = [
            IntentCandidate(
                intent=str(classes[index]),
                probability=round(float(probabilities[index]), 4),
            )
            for index in top_indices
        ]
        winner = candidates[0]

        return IntentPrediction(
            intent=winner.intent,
            confidence=winner.probability,
            top_predictions=candidates,
            model_version=self._model_version,
        )
