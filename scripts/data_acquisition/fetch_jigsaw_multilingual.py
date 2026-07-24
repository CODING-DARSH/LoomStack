"""
Dataset: Jigsaw Multilingual Toxic Comment Classification
Source:  Kaggle competition `jigsaw-multilingual-toxic-comment-classification`
Used by: models/text/multilingual_toxicity_classifier (XLM-RoBERTa fine-tune)

Prereq: accept competition rules at
  https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import KaggleDownloader

logger = logging.getLogger("fetch_jigsaw_multilingual")


def main():
    downloader = KaggleDownloader(
        slug="jigsaw_multilingual_toxic",
        kaggle_ref="jigsaw-multilingual-toxic-comment-classification",
        is_competition=True,
    )
    dest = downloader.fetch()

    # validation.csv has multilingual labeled data (test.csv is unlabeled leaderboard data)
    val_csv = dest / "validation.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Expected validation.csv in {dest}, found: {list(dest.glob('*'))}")

    df = pd.read_csv(val_csv)
    logger.info("Rows: %d", len(df))
    if "lang" in df.columns:
        logger.info("Language distribution:\n%s", df["lang"].value_counts().to_string())
    if "toxic" in df.columns:
        logger.info("Toxic rate: %.4f", df["toxic"].mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()