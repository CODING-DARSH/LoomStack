"""
STEP 1: INSPECT ONLY. No cleaning happens here.

Run this first to see exactly what's in the raw SMS Spam Collection
data before deciding what needs cleaning. Prints everything you'd
want to know before writing a single line of cleaning logic.

Usage:
    python 01_inspect.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inspect_sms_spam")

RAW_DIR = Path("data/raw/sms_spam_collection")


def main():
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR} -- run the acquisition script first")

    path = csv_files[0]
    logger.info(f"Reading: {path}")

    # SMS Spam Collection is known to use latin-1 encoding, not utf-8
    df = pd.read_csv(path, encoding="latin-1")

    logger.info("\n=== SHAPE ===")
    logger.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    logger.info("\n=== ALL COLUMNS (raw, before we decide what to keep) ===")
    logger.info(list(df.columns))

    logger.info("\n=== DTYPES ===")
    logger.info(df.dtypes.to_string())

    logger.info("\n=== FIRST 3 ROWS ===")
    logger.info(df.head(3).to_string())

    logger.info("\n=== NULL COUNTS PER COLUMN ===")
    logger.info(df.isnull().sum().to_string())

    logger.info("\n=== LABEL COLUMN VALUE COUNTS (v1) ===")
    if "v1" in df.columns:
        logger.info(df["v1"].value_counts().to_string())
        logger.info(f"Unique label values: {df['v1'].unique().tolist()}")

    logger.info("\n=== TEXT COLUMN STATS (v2) ===")
    if "v2" in df.columns:
        lengths = df["v2"].astype(str).str.len()
        logger.info(f"Text length -- min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}")
        empty_texts = (df["v2"].astype(str).str.strip() == "").sum()
        logger.info(f"Empty/whitespace-only text rows: {empty_texts}")

    logger.info("\n=== EXACT DUPLICATE ROWS ===")
    logger.info(f"Fully duplicated rows: {df.duplicated().sum()}")
    if "v2" in df.columns:
        logger.info(f"Duplicate text values (regardless of label): {df['v2'].duplicated().sum()}")

    logger.info("\n=== CHECKING FOR UNNAMED/JUNK COLUMNS ===")
    junk_cols = [c for c in df.columns if "Unnamed" in str(c)]
    if junk_cols:
        logger.info(f"Found junk columns to drop: {junk_cols}")
        for c in junk_cols:
            non_null = df[c].notna().sum()
            logger.info(f"  {c}: {non_null} non-null values (sample: {df[c].dropna().head(3).tolist()})")
    else:
        logger.info("None found.")

    logger.info("\n=== SUSPICIOUS CHARACTERS CHECK (encoding artifacts) ===")
    sample_weird = df["v2"].astype(str).str.contains(r"[Ã¢â€]", regex=True, na=False).sum()
    logger.info(f"Rows with likely mojibake/encoding artifacts: {sample_weird}")

    logger.info("\n=== SUMMARY: WHAT LIKELY NEEDS CLEANING ===")
    logger.info("Based on the above, decide before writing 02_clean.py:")
    logger.info("  - Drop junk/unnamed columns? (check output above)")
    logger.info("  - Rename v1 -> label, v2 -> text?")
    logger.info("  - Deduplicate on text?")
    logger.info("  - Fix any encoding artifacts?")
    logger.info("  - Standardize label values (spam/ham -> 1/0)?")


if __name__ == "__main__":
    main()