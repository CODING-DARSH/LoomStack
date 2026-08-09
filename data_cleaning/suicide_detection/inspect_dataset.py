"""
INSPECT ONLY. No cleaning happens here.

Suicide and Depression Detection (Reddit posts, Kaggle
nikhileswarkomati/suicide-watch). Feeds models/text/self_harm_detector,
which is explicitly escalation-only in this pipeline (never
auto-removes content, never in the automated retraining loop).

IMPORTANT: this script deliberately NEVER prints raw post text, not
even short previews/snippets. Everything here is aggregate statistics
only (counts, lengths, hashes for dedup) -- the content is real,
unfiltered crisis-related material, and there's no need to surface
individual rows to inspect data quality. If you want to eyeball
specific rows, open the raw CSV directly on your machine.

Usage:
    python inspect_dataset.py
"""

import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inspect_suicide_detection")

RAW_DIR = Path("data/raw/suicide_watch")


def main():
    csv_candidates = list(RAW_DIR.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR} -- run the acquisition script first")

    df = pd.read_csv(csv_candidates[0])
    logger.info(f"Reading: {csv_candidates[0]}")

    logger.info("\n=== SHAPE ===")
    logger.info(f"Rows: {len(df)}, Columns: {list(df.columns)}")

    logger.info("\n=== DTYPES ===")
    logger.info(df.dtypes.to_string())

    logger.info("\n=== NULL COUNTS ===")
    logger.info(df.isnull().sum().to_string())

    if not {"text", "class"}.issubset(df.columns):
        raise ValueError(f"Expected 'text' and 'class' columns, got {list(df.columns)}")

    logger.info("\n=== CLASS VALUES / DISTRIBUTION ===")
    logger.info(f"Distinct class values: {sorted(df['class'].unique())}")
    logger.info(df["class"].value_counts().to_string())
    logger.info(f"Class balance ratio: {df['class'].value_counts(normalize=True).to_string()}")

    logger.info("\n=== TEXT LENGTH STATS (aggregate only) ===")
    lengths = df["text"].astype(str).str.len()
    logger.info(f"Length -- min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}, "
                f"median: {lengths.median():.1f}")
    logger.info(f"Length by class:\n{df.groupby('class')['text'].apply(lambda s: s.astype(str).str.len().mean()).round(1).to_string()}")

    logger.info("\n=== EMPTY / WHITESPACE-ONLY TEXT ===")
    empty_count = (df["text"].astype(str).str.strip() == "").sum()
    logger.info(f"Empty/whitespace-only rows: {empty_count}")

    logger.info("\n=== VERY SHORT TEXT (potential low-signal rows) ===")
    very_short = (lengths < 10).sum()
    logger.info(f"Rows under 10 characters: {very_short}")

    logger.info("\n=== EXACT DUPLICATE TEXT ===")
    dupes = df["text"].duplicated().sum()
    logger.info(f"Duplicate text rows: {dupes}")
    logger.info(f"Duplicate rate by class:\n{df[df['text'].duplicated(keep=False)]['class'].value_counts().to_string()}")

    logger.info("\n=== NEAR-DUPLICATE VIA CONTENT HASH (normalized) ===")
    def normalize_for_hash(t):
        t = str(t).lower()
        t = re.sub(r"[^\w\s]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    normalized = df["text"].apply(normalize_for_hash)
    near_dup_count = normalized.duplicated().sum()
    logger.info(f"Near-duplicate rows (normalized, case/punctuation-insensitive): {near_dup_count}")

    logger.info("\n=== TITLE/BODY CONCATENATION ARTIFACT CHECK ===")
    # Reddit scrapes often concatenate title+body with no separator,
    # producing patterns like "wordWord" (lowercase directly followed
    # by uppercase) right where the title ends and body begins.
    concat_pattern = re.compile(r"[a-z][A-Z]")
    has_concat_artifact = df["text"].astype(str).str.contains(concat_pattern, regex=True).sum()
    logger.info(f"Rows matching a likely title/body concatenation pattern (lowercase directly "
                f"followed by uppercase, no space): {has_concat_artifact} / {len(df)}")

    logger.info("\n=== MARKUP / ENCODING ARTIFACTS ===")
    has_html_entity = df["text"].astype(str).str.contains(r"&\w+;", regex=True).sum()
    has_url = df["text"].astype(str).str.contains(r"https?://|www\.", regex=True).sum()
    has_markdown_link = df["text"].astype(str).str.contains(r"\[.+?\]\(.+?\)", regex=True).sum()
    has_removed_placeholder = df["text"].astype(str).str.lower().isin(["[removed]", "[deleted]"]).sum()
    has_mojibake = df["text"].astype(str).str.contains(r"Ã.|â€.", regex=True).sum()
    logger.info(f"Rows with HTML entities (e.g. &amp;): {has_html_entity}")
    logger.info(f"Rows with URLs: {has_url}")
    logger.info(f"Rows with markdown links: {has_markdown_link}")
    logger.info(f"Rows that are just [removed]/[deleted] placeholders: {has_removed_placeholder}")
    logger.info(f"Rows matching common mojibake byte patterns: {has_mojibake}")

    logger.info("\n=== NEWLINE / CONTROL CHAR ARTIFACTS ===")
    has_newline = df["text"].astype(str).str.contains("\n", regex=False).sum()
    logger.info(f"Rows with literal newlines: {has_newline}")

    logger.info("\n=== POTENTIAL PII PATTERNS (aggregate counts only, no content shown) ===")
    # phone-number-like and explicit-username-mention patterns -- just
    # counting matches, never printing what matched
    phone_pattern = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
    reddit_mention_pattern = re.compile(r"/u/\w+|u/\w+")
    has_phone_like = df["text"].astype(str).str.contains(phone_pattern, regex=True).sum()
    has_username_mention = df["text"].astype(str).str.contains(reddit_mention_pattern, regex=True).sum()
    logger.info(f"Rows matching a phone-number-like pattern: {has_phone_like}")
    logger.info(f"Rows containing a Reddit username mention (u/...): {has_username_mention}")

    logger.info("\n=== SUMMARY: WHAT LIKELY NEEDS DECIDING ===")
    logger.info("  - Confirm class label values map as expected (suicide / non-suicide)")
    logger.info("  - Decide handling for exact/near-duplicate rows (dupe rate by class matters --")
    logger.info("    if concentrated in one class, naive dedup could skew balance)")
    logger.info("  - Decide whether to strip title/body concatenation artifacts or leave as-is")
    logger.info("  - Decide phone-number/username pattern handling -- mask, or leave (these are")
    logger.info("    counts only here; a real redaction pass would need its own dedicated review,")
    logger.info("    not a slot in the standard regex-cleaning step)")
    logger.info("  - [removed]/[deleted] placeholder rows should likely be dropped as non-content")


if __name__ == "__main__":
    main()