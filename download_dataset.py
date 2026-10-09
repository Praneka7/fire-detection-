"""Download the selected Kaggle fire dataset into Kaggle's local cache."""

from pathlib import Path

import kagglehub


def main() -> None:
    dataset_path = Path(kagglehub.dataset_download("phylake1337/fire-dataset"))
    print(f"Dataset downloaded to: {dataset_path}")
    print("Pass this path to train_model.py with --data-dir.")


if __name__ == "__main__":
    main()
