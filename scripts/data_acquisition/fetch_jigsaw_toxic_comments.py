"""
Dataset: Jigsaw Toxic Comment Classification Challenge
Source:  Kaggle competition `jigsaw-toxic-comment-classification-challenge`
Used by: models/text/toxicity_classifier

Prereq: accept competition rules at
  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
and have ~/.kaggle/kaggle.json configured.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import KaggleDownloader

logger = logging.getLogger("fetch_jigsaw_toxic")


def main():
    downloader = KaggleDownloader(
        slug="jigsaw_toxic_comments",
        kaggle_ref="jigsaw-toxic-comment-classification-challenge",
        is_competition=True,
    )
    dest = downloader.fetch()

    train_csv = dest / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Expected train.csv in {dest}, found: {list(dest.glob('*'))}")

    df = pd.read_csv(train_csv)
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Expected label columns missing from train.csv: {missing}")

    logger.info("Rows: %d", len(df))
    logger.info("Label positive rates:\n%s", df[label_cols].mean().round(4).to_string())
    logger.info("Any-label-positive rate: %.4f", (df[label_cols].sum(axis=1) > 0).mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()