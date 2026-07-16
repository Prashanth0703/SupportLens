from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from supportlens_ml.modeling import (
    build_majority_baseline,
    build_tfidf_logistic_regression,
    calculate_core_metrics,
)

import os

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
EXPERIMENT_NAME = "supportlens-intent-classification"


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python prepare_dataset.py` first."
        )
    return pd.read_parquet(path)


def save_classification_report(
    y_true: pd.Series,
    y_pred: object,
    output_path: Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: object,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 18))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        normalize="true",
        xticks_rotation="vertical",
        cmap="Blues",
        ax=axis,
        colorbar=False,
    )
    axis.set_title("Normalised confusion matrix — TF-IDF + Logistic Regression")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    majority = build_majority_baseline()
    majority.fit(train["text"], train["label"])
    majority_validation_predictions = majority.predict(validation["text"])
    majority_metrics = calculate_core_metrics(
        validation["label"],
        majority_validation_predictions,
    )

    model = build_tfidf_logistic_regression()
    model.fit(train["text"], train["label"])

    validation_predictions = model.predict(validation["text"])
    validation_metrics = calculate_core_metrics(
        validation["label"],
        validation_predictions,
    )

    test_predictions = model.predict(test["text"])
    test_metrics = calculate_core_metrics(
        test["label"],
        test_predictions,
    )

    model_path = ARTIFACT_DIR / "tfidf_logreg_v1.joblib"
    report_path = ARTIFACT_DIR / "test_classification_report.json"
    confusion_path = ARTIFACT_DIR / "test_confusion_matrix.png"
    metrics_path = ARTIFACT_DIR / "metrics.json"

    joblib.dump(model, model_path)
    save_classification_report(test["label"], test_predictions, report_path)
    save_confusion_matrix(test["label"], test_predictions, confusion_path)

    complete_metrics = {
        "majority_validation": majority_metrics,
        "tfidf_validation": validation_metrics,
        "tfidf_test": test_metrics,
        "dataset": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "classes": int(train["label"].nunique()),
        },
    }
    metrics_path.write_text(
        json.dumps(complete_metrics, indent=2),
        encoding="utf-8",
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="tfidf-logreg-v1"):
        mlflow.log_params(
            {
                "model": "logistic_regression",
                "features": "tfidf_word_1_2_grams",
                "max_features": 50_000,
                "class_weight": "balanced",
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "number_of_classes": int(train["label"].nunique()),
            }
        )
        mlflow.log_metrics(
            {
                "majority_validation_macro_f1": majority_metrics["macro_f1"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_weighted_f1": validation_metrics["weighted_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
            }
        )
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")

    print(json.dumps(complete_metrics, indent=2))
    print(f"Saved model and evaluation artifacts to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
