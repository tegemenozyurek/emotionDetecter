"""Step 2: Download the FER-2013 dataset from Kaggle into data/fer2013.

Requires a Kaggle API token in ~/.kaggle/access_token (or the KAGGLE_API_TOKEN
environment variable). The token is never stored inside this repository.
"""
import shutil
from pathlib import Path

import kagglehub

DATASET = "msambare/fer2013"
TARGET = Path(__file__).resolve().parent.parent / "data" / "fer2013"


def main():
    if TARGET.exists():
        print(f"Dataset already present at {TARGET}")
    else:
        cache_path = Path(kagglehub.dataset_download(DATASET))
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cache_path, TARGET)
        print(f"Downloaded to {TARGET}")

    for split in ("train", "test"):
        split_dir = TARGET / split
        print(f"\n[{split}]")
        total = 0
        for emotion_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            count = len(list(emotion_dir.glob("*.jpg")))
            total += count
            print(f"  {emotion_dir.name:<10} {count:>6}")
        print(f"  {'TOTAL':<10} {total:>6}")


if __name__ == "__main__":
    main()
