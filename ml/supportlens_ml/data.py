from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATASET_FILENAME = (
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

TEXT_COLUMN = "instruction"
LABEL_COLUMN = "intent"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalise_text(value: object) -> str:
    if value is None:
        return ""

    return _WHITESPACE_PATTERN.sub(" ", str(value)).strip()


def clean_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    cleaned = frame[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    cleaned.columns = ["text", "label"]

    cleaned["text"] = cleaned["text"].map(normalise_text)
    cleaned["label"] = cleaned["label"].map(normalise_text)

    cleaned = cleaned[
        (cleaned["text"].str.len() > 0)
        & (cleaned["label"].str.len() > 0)
    ]

    cleaned = cleaned.drop_duplicates(
        subset=["text"]
    ).reset_index(drop=True)

    class_counts = cleaned["label"].value_counts()
    rare_labels = class_counts[class_counts < 3].index

    if len(rare_labels) > 0:
        cleaned = cleaned[
            ~cleaned["label"].isin(rare_labels)
        ].reset_index(drop=True)

    return cleaned


def download_dataset() -> pd.DataFrame:
    """Download the source CSV without generating an Arrow cache."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=DATASET_FILENAME,
        repo_type="dataset",
        local_dir=RAW_DATA_DIR,
    )

    frame = pd.read_csv(downloaded_path)
    return clean_dataset_frame(frame)


def create_splits(
    frame: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=random_state,
        stratify=frame["label"],
    )

    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=remainder["label"],
    )

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_frame in splits.items():
        split_frame.to_parquet(
            output_dir / f"{split_name}.parquet",
            index=False,
        )

    summary = pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": len(split_frame),
                "classes": split_frame["label"].nunique(),
            }
            for split_name, split_frame in splits.items()
        ]
    )

    summary.to_csv(
        output_dir / "split_summary.csv",
        index=False,
    )