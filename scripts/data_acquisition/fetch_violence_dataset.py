"""
Dataset: Violence vs. Non-Violence: 11K Images Dataset
Source:  Kaggle `abdulmananraja/real-life-violence-situations`
Used by: models/image/violence_classifier

CHANGED FROM ORIGINAL PLAN: originally used
`mohamedmustafa/real-life-violence-situations-dataset`, which ships as
2000 VIDEOS (1000 violence / 1000 non-violence) requiring frame
extraction via OpenCV. That approach worked but was slow, noisy
(H264 decode warnings on corrupted frames -- harmless but ugly), and
added an unnecessary video-processing dependency.

This dataset is the image-only alternative: 11,000+ pre-extracted
images at 416x416, already labeled violence/non-violence. No video
decoding step needed -- just download and use directly.
"""

import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import KaggleDownloader

logger = logging.getLogger("fetch_violence_dataset")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    downloader = KaggleDownloader(
        slug="violence_images",
        kaggle_ref="abdulmananraja/real-life-violence-situations",
    )
    dest = downloader.fetch()

    class_dirs = [
        d for d in dest.rglob("*")
        if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir() if f.is_file())
    ]
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories with images found under {dest}")

    counts = Counter()
    for d in class_dirs:
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        counts[d.name] = n

    logger.info("Class counts: %s", dict(counts))
    logger.info("Total images: %d", sum(counts.values()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()