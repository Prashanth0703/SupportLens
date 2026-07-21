from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    REPOSITORY_ROOT / "ml" / "artifacts" / "word-tfidf-logreg-v1.joblib"
)


def get_model_path() -> Path:
    configured_path = os.getenv("SUPPORTLENS_MODEL_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_MODEL_PATH
