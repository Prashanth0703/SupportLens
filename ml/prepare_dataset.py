from pathlib import Path

from supportlens_ml.data import create_splits, download_dataset, save_splits

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    frame = download_dataset()
    splits = create_splits(frame)
    save_splits(splits, OUTPUT_DIR)

    print(f"Saved processed data to: {OUTPUT_DIR}")
    for name, split in splits.items():
        print(
            f"{name:10s} rows={len(split):5d} "
            f"classes={split['label'].nunique():2d}"
        )


if __name__ == "__main__":
    main()
