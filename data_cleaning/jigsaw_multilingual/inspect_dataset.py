"""
INSPECT ONLY. No cleaning happens here.

Jigsaw Multilingual Toxic Comment Classification -- the labeled data
lives in validation.csv (test.csv is unlabeled leaderboard data, see
fetch_jigsaw_multilingual.py). Binary `toxic` label like jigsaw_toxic,
but multi-LANGUAGE instead of multi-label -- the thing to understand
here is per-language distribution and text quality, not label overlap.

Usage:
    python inspect_dataset.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inspect_jigsaw_multilingual")

RAW_DIR = Path("data/raw/jigsaw_multilingual_toxic")


def main():
    val_path = RAW_DIR / "validation.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"{val_path} not found -- run the acquisition script first")

    df = pd.read_csv(val_path)
    logger.info(f"Reading: {val_path}")

    logger.info("\n=== SHAPE ===")
    logger.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    logger.info("\n=== ALL COLUMNS ===")
    logger.info(list(df.columns))

    logger.info("\n=== DTYPES ===")
    logger.info(df.dtypes.to_string())

    logger.info("\n=== NULL COUNTS PER COLUMN ===")
    logger.info(df.isnull().sum().to_string())

    if "toxic" not in df.columns:
        raise ValueError("Expected 'toxic' label column missing")

    logger.info("\n=== LABEL VALUE RANGE (should be 0/1) ===")
    logger.info(f"toxic: {sorted(df['toxic'].unique())}")

    logger.info("\n=== OVERALL TOXIC RATE ===")
    logger.info(f"{df['toxic'].mean():.4f}")

    logger.info("\n=== LANGUAGE COVERAGE ===")
    if "lang" in df.columns:
        logger.info(f"Distinct languages: {df['lang'].nunique()}")
        logger.info(df["lang"].value_counts().to_string())
    else:
        logger.info("No 'lang' column found -- check schema, this dataset is language-labeled")

    logger.info("\n=== PER-LANGUAGE TOXIC RATE ===")
    if "lang" in df.columns:
        logger.info(df.groupby("lang")["toxic"].mean().round(4).to_string())
        logger.info("\n=== PER-LANGUAGE ROW COUNT (check class balance per split) ===")
        logger.info(df.groupby("lang").size().to_string())

    text_col = "comment_text" if "comment_text" in df.columns else None
    if text_col is None:
        logger.info("\n=== WARNING: no 'comment_text' column found ===")
        logger.info(f"Available columns: {list(df.columns)}")
    else:
        logger.info(f"\n=== TEXT COLUMN STATS ({text_col}) ===")
        lengths = df[text_col].astype(str).str.len()
        logger.info(f"Length -- min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}")
        empty_texts = (df[text_col].astype(str).str.strip() == "").sum()
        logger.info(f"Empty/whitespace-only text rows: {empty_texts}")

        logger.info("\n=== NEWLINE / WHITESPACE ARTIFACTS ===")
        has_newline = df[text_col].astype(str).str.contains("\n", regex=False).sum()
        logger.info(f"Rows containing literal newlines: {has_newline}")

        logger.info("\n=== NON-LATIN SCRIPT SANITY CHECK (expected, given multilingual) ===")
        has_non_ascii = df[text_col].astype(str).apply(lambda t: any(ord(c) > 127 for c in t)).sum()
        logger.info(f"Rows containing non-ASCII characters: {has_non_ascii} / {len(df)}")

        logger.info("\n=== ENCODING ARTIFACTS (mojibake heuristic) ===")
        has_mojibake = df[text_col].astype(str).str.contains(
            r"Ã.|â€.", regex=True
        ).sum()
        logger.info(f"Rows matching common mojibake byte patterns: {has_mojibake}")

        logger.info("\n=== HTML / MARKUP ARTIFACTS ===")
        has_html = df[text_col].astype(str).str.contains(
            r"<[^>]+>|&\w+;", regex=True
        ).sum()
        logger.info(f"Rows containing HTML tags or entities: {has_html}")

        logger.info("\n=== URL ARTIFACTS ===")
        has_url = df[text_col].astype(str).str.contains(
            r"https?://|www\.", regex=True
        ).sum()
        logger.info(f"Rows containing URLs: {has_url}")

        logger.info("\n=== EXACT DUPLICATE TEXT ===")
        logger.info(f"Duplicate {text_col} values: {df[text_col].duplicated().sum()}")

        logger.info("\n=== DUPLICATE TEXT ACROSS DIFFERENT LANGUAGES (mislabeled lang tag?) ===")
        if "lang" in df.columns:
            dup_text = df[df[text_col].duplicated(keep=False)]
            cross_lang = dup_text.groupby(text_col)["lang"].nunique()
            cross_lang = cross_lang[cross_lang > 1]
            logger.info(f"Texts duplicated across 2+ distinct 'lang' values: {len(cross_lang)}")

    if "id" in df.columns:
        logger.info("\n=== ID COLUMN CHECK ===")
        logger.info(f"Duplicate ids: {df['id'].duplicated().sum()}")

    logger.info("\n=== SUMMARY: WHAT LIKELY NEEDS CLEANING ===")
    logger.info("Decide before writing the cleaning script:")
    logger.info("  - Strip HTML tags/entities and URLs?")
    logger.info("  - Fix mojibake if present (re-decode with correct encoding)?")
    logger.info("  - Normalize newlines to spaces?")
    logger.info("  - Deduplicate on exact text (and check cross-language dupes)?")
    logger.info("  - Split strategy: stratify by 'toxic' only, or by 'toxic' + 'lang' jointly")
    logger.info("    to keep per-language balance across train/val/test?")
    logger.info("  - Is validation.csv large enough on its own, or does it need to merge with")
    logger.info("    another labeled source given test.csv is unlabeled?")


if __name__ == "__main__":
    main()