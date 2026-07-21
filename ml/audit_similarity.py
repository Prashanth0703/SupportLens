from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
    )

    train_vectors = vectorizer.fit_transform(train["text"])
    test_vectors = vectorizer.transform(test["text"])

    neighbours = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    neighbours.fit(train_vectors)

    distances, indices = neighbours.kneighbors(test_vectors)
    similarities = 1 - distances[:, 0]

    print("Test examples:", len(test))
    print("Mean nearest-train similarity:", similarities.mean())
    print("Median similarity:", pd.Series(similarities).median())

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        count = int((similarities >= threshold).sum())
        percentage = 100 * count / len(similarities)

        print(
            f"Similarity >= {threshold:.2f}: "
            f"{count} examples ({percentage:.2f}%)"
        )

    closest = similarities.argsort()[-20:][::-1]

    rows = []

    for test_index in closest:
        train_index = indices[test_index, 0]

        rows.append(
            {
                "similarity": similarities[test_index],
                "test_label": test.iloc[test_index]["label"],
                "train_label": train.iloc[train_index]["label"],
                "test_text": test.iloc[test_index]["text"],
                "nearest_train_text": train.iloc[train_index]["text"],
            }
        )

    output = pd.DataFrame(rows)

    output_path = Path(__file__).resolve().parent / "artifacts"
    output_path.mkdir(exist_ok=True)

    output.to_csv(
        output_path / "nearest_train_examples.csv",
        index=False,
    )

    print(
        "\nSaved closest train/test examples to "
        "artifacts/nearest_train_examples.csv"
    )


if __name__ == "__main__":
    main()