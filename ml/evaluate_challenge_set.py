from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "ml" / "artifacts" / "word_char_logreg_v2.joblib"
DATA_PATH = ROOT / "data" / "challenge_tickets_v1.csv"
OUTPUT_PATH = ROOT / "ml" / "artifacts" / "challenge_metrics.json"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Challenge dataset not found at {DATA_PATH}."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = {"text", "label"}
    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Challenge dataset is missing columns: {sorted(missing)}"
        )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(data["text"])

    metrics = {
        "rows": len(data),
        "classes_present": int(data["label"].nunique()),
        "accuracy": float(
            accuracy_score(data["label"], predictions)
        ),
        "macro_f1": float(
            f1_score(
                data["label"],
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                data["label"],
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            data["label"],
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    error_rows = data.copy()
    error_rows["prediction"] = predictions
    error_rows = error_rows[
        error_rows["label"] != error_rows["prediction"]
    ]

    error_rows.to_csv(
        ROOT / "ml" / "artifacts" / "challenge_errors_word_char.csv",
        index=False,
    )

    print(json.dumps({
        "rows": metrics["rows"],
        "classes_present": metrics["classes_present"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "errors": len(error_rows),
    }, indent=2))


if __name__ == "__main__":
    main()