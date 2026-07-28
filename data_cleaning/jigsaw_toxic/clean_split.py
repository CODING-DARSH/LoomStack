"""
STEP 2: CLEAN + SPLIT. Run after reviewing 01_inspect.py output.

Cleaning decisions made here (based on inspection results):
  1. Normalize newlines/whitespace to single spaces (94466/159571 rows
     had literal newlines -- majority of the dataset, real cleanup needed)
  2. Strip wiki markup: [[links]], {{templates}}, == headers == (2949
     rows -- Wikipedia-specific noise, not toxicity signal)
  3. Mask IP addresses to a placeholder token (10081 rows -- these are
     anonymous editor signatures, privacy-sensitive and not toxicity
     signal either)
  4. No deduplication needed -- inspection found 0 duplicate texts
  5. MULTILABEL stratified split (70/15/15) -- this is NOT the same as
     SMS Spam's single-column stratify. With 6 independent binary
     labels and meaningful label co-occurrence (9865 rows have 2+
     labels active), a plain sklearn train_test_split with stratify=
     one column would badly imbalance the other 5. We use iterative
     stratification (MultilabelStratifiedShuffleSplit from the
     iterstrat package) which balances all 6 label columns
     simultaneously.

Usage:
    python 02_clean_and_split.py
"""

import logging
import re
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clean_jigsaw_toxic")

RAW_DIR = Path("data/raw/jigsaw_toxic_comments")
OUT_DIR = Path("data/processed/jigsaw_toxic")
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RANDOM_SEED = 42

WIKI_MARKUP_PATTERN = re.compile(r"\[\[.*?\]\]|\{\{.*?\}\}|==+\s*.*?\s*==+", re.DOTALL)
IP_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "train.csv"
    df = pd.read_csv(path)
    logger.info(f"Loaded raw: {len(df)} rows")
    return df


def clean_text(text: str) -> str:
    # Wiki markup ([[..]], {{..}}) can span multiple lines -- the
    # original regex without re.DOTALL silently failed to match those,
    # leaving 286 rows with un-stripped markup (caught by
    # test_wiki_markup_actually_removed). Fixed above by adding DOTALL.
    text = WIKI_MARKUP_PATTERN.sub(" ", text)
    text = IP_PATTERN.sub("<IP>", text)

    # The raw Kaggle CSV export contains a handful of genuinely
    # unrecoverable encoding failures (U+FFFD replacement characters)
    # and stray control characters -- caught by
    # test_no_unicode_replacement_characters / test_no_control_characters.
    # These carry no recoverable signal, so we strip them rather than
    # trying to guess original bytes.
    text = text.replace("\ufffd", "")
    text = CONTROL_CHAR_PATTERN.sub("", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"comment_text": "text"})
    df["text"] = df["text"].astype(str).apply(clean_text)

    before = len(df)
    df = df[df["text"].str.strip() != ""]
    logger.info(f"Dropped {before - len(df)} rows that became empty after cleaning")

    # test_no_extreme_length_outliers caught a 1-2 char row surviving
    # cleaning (e.g. markup-only comments reduced to a stray character).
    # These carry no usable text signal, drop them the same way we drop
    # fully-empty rows.
    before = len(df)
    df = df[df["text"].str.len() >= 3]
    logger.info(f"Dropped {before - len(df)} rows under 3 characters after cleaning")

    # inspection found 0 exact duplicates pre-cleaning, but re-check
    # post-cleaning since stripping markup/IPs could create new dupes
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    logger.info(f"Dropped {before - len(df)} duplicate-text rows created by cleaning")

    keep_cols = ["id", "text"] + LABEL_COLS
    df = df[keep_cols].reset_index(drop=True)

    logger.info(f"Final cleaned shape: {df.shape}")
    logger.info(f"Per-label positive rate after cleaning:\n{df[LABEL_COLS].mean().round(4).to_string()}")
    return df


def multilabel_stratified_split(df: pd.DataFrame):
    X = df.index.values.reshape(-1, 1)
    y = df[LABEL_COLS].values

    # split 1: 70% train, 30% temp
    splitter1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=RANDOM_SEED
    )
    train_idx, temp_idx = next(splitter1.split(X, y))

    # split 2: temp -> 50/50 into val/test (each ends up 15% of total)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    X_temp = temp_df.index.values.reshape(-1, 1)
    y_temp = temp_df[LABEL_COLS].values

    splitter2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_SEED
    )
    val_idx, test_idx = next(splitter2.split(X_temp, y_temp))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = OUT_DIR / f"{name}.parquet"
        split_df.to_parquet(out_path)
        logger.info(f"Wrote {name}.parquet: {len(split_df)} rows")
        logger.info(f"  Per-label rate:\n{split_df[LABEL_COLS].mean().round(4).to_string()}")


def main():
    df = load_raw()
    df = clean(df)
    train_df, val_df, test_df = multilabel_stratified_split(df)
    save_splits(train_df, val_df, test_df)
    logger.info(f"\nDone. Processed files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()