import pandas as pd

from supportlens_ml.data import clean_dataset_frame, normalise_text


def test_normalise_text_collapses_whitespace() -> None:
    assert normalise_text("  hello   support\nteam  ") == "hello support team"


def test_clean_dataset_removes_empty_and_duplicate_rows() -> None:
    frame = pd.DataFrame(
        {
            "instruction": [
                "Reset my password",
                "Reset my password",
                "",
                "Cancel my order",
                "Track my order",
                "Refund my order",
            ],
            "intent": [
                "password_reset",
                "password_reset",
                "password_reset",
                "cancel_order",
                "track_order",
                "refund_order",
            ],
        }
    )

    cleaned = clean_dataset_frame(frame)

    assert cleaned["text"].is_unique
    assert "" not in cleaned["text"].tolist()
