"""
STEP 1: INSPECT ONLY. No cleaning happens here.

Jigsaw Toxic Comment Classification -- multi-label (6 toxicity
categories), unlike SMS Spam's single binary label. Run this first to
understand the label structure and text quality before writing
cleaning logic.

Usage:
    python 01_inspect.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inspect_jigsaw_toxic")

RAW_DIR = Path("data/raw/jigsaw_toxic_comments")
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def main():
    train_path = RAW_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found -- run the acquisition script first")

    df = pd.read_csv(train_path)
    logger.info(f"Reading: {train_path}")

    logger.info("\n=== SHAPE ===")
    logger.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    logger.info("\n=== ALL COLUMNS ===")
    logger.info(list(df.columns))

    logger.info("\n=== DTYPES ===")
    logger.info(df.dtypes.to_string())

    logger.info("\n=== NULL COUNTS PER COLUMN ===")
    logger.info(df.isnull().sum().to_string())

    missing_label_cols = [c for c in LABEL_COLS if c not in df.columns]
    if missing_label_cols:
        raise ValueError(f"Expected label columns missing: {missing_label_cols}")

    logger.info("\n=== LABEL VALUE RANGES (should all be 0/1) ===")
    for col in LABEL_COLS:
        uniques = df[col].unique()
        logger.info(f"{col}: {sorted(uniques)}")

    logger.info("\n=== PER-LABEL POSITIVE RATE ===")
    logger.info(df[LABEL_COLS].mean().round(4).to_string())

    logger.info("\n=== MULTI-LABEL OVERLAP ===")
    label_sum = df[LABEL_COLS].sum(axis=1)
    logger.info(f"Rows with 0 labels active: {(label_sum == 0).sum()}")
    logger.info(f"Rows with exactly 1 label active: {(label_sum == 1).sum()}")
    logger.info(f"Rows with 2+ labels active: {(label_sum >= 2).sum()}")
    logger.info(f"Max labels active on a single row: {label_sum.max()}")

    logger.info("\n=== TEXT COLUMN STATS (comment_text) ===")
    lengths = df["comment_text"].astype(str).str.len()
    logger.info(f"Length -- min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}")
    empty_texts = (df["comment_text"].astype(str).str.strip() == "").sum()
    logger.info(f"Empty/whitespace-only text rows: {empty_texts}")

    logger.info("\n=== NEWLINE / WHITESPACE ARTIFACTS ===")
    has_newline = df["comment_text"].astype(str).str.contains("\n", regex=False).sum()
    logger.info(f"Rows containing literal newlines: {has_newline}")

    logger.info("\n=== WIKI MARKUP ARTIFACTS (this is Wikipedia talk-page data) ===")
    has_wiki_markup = df["comment_text"].astype(str).str.contains(
        r"\[\[|\{\{|==\s*\w+\s*==", regex=True
    ).sum()
    logger.info(f"Rows containing likely wiki markup ([[ ]], {{ }}, == headers ==): {has_wiki_markup}")

    logger.info("\n=== IP ADDRESSES / USER SIGNATURES (common in wiki talk pages) ===")
    has_ip = df["comment_text"].astype(str).str.contains(
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", regex=True
    ).sum()
    logger.info(f"Rows containing IP-address-like patterns (often anonymous user signatures): {has_ip}")

    logger.info("\n=== EXACT DUPLICATE TEXT ===")
    logger.info(f"Duplicate comment_text values: {df['comment_text'].duplicated().sum()}")

    logger.info("\n=== ID COLUMN CHECK ===")
    if "id" in df.columns:
        logger.info(f"Duplicate ids: {df['id'].duplicated().sum()}")

    logger.info("\n=== SUMMARY: WHAT LIKELY NEEDS CLEANING ===")
    logger.info("Decide before writing 02_clean_and_split.py:")
    logger.info("  - Strip wiki markup ([[..]], {{..}}, == headers ==)?")
    logger.info("  - Strip/mask IP addresses (privacy + noise)?")
    logger.info("  - Normalize newlines to spaces?")
    logger.info("  - Deduplicate on exact text?")
    logger.info("  - How to split: random stratified, or by dominant label?")
    logger.info("    (multi-label stratification is harder than binary -- decide approach)")


if __name__ == "__main__":
    main()