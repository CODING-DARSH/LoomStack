"""
Dataset: Suicide and Depression Detection (Reddit posts)
Source:  Kaggle `nikhileswarkomati/suicide-watch`
Used by: models/text/self_harm_detector

NOTE: this model is treated as escalation-only in the pipeline (never
auto-removes content, never included in the automated retraining loop).
See docs/policy_rationale.md before wiring this into the decision layer.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import KaggleDownloader

logger = logging.getLogger("fetch_suicide_detection")


def main():
    downloader = KaggleDownloader(slug="suicide_watch", kaggle_ref="nikhileswarkomati/suicide-watch")
    dest = downloader.fetch()

    csv_candidates = list(dest.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found in {dest}")

    df = pd.read_csv(csv_candidates[0])
    expected_cols = {"text", "class"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Expected columns {expected_cols}, got {list(df.columns)}")

    logger.info("Rows: %d", len(df))
    logger.info("Class distribution:\n%s", df["class"].value_counts().to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()