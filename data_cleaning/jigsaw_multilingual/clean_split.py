"""
CLEAN + SPLIT. Run after reviewing inspect_dataset.py output.

Cleaning decisions made here (based on actual inspection results on
validation.csv, 8000 rows, 3 languages: es/it/tr):
  1. Strip HTML tags/entities (5 rows affected -- small but real, and
     residual markup is not toxicity signal)
  2. Mask URLs to a placeholder token (99 rows -- URLs carry no
     toxicity signal themselves and add noise/sparsity to the vocab)
  3. Strip unrecoverable encoding artifacts: U+FFFD replacement char
     and control characters, same policy as jigsaw_toxic (inspection's
     mojibake heuristic hit only 1 row -- too rare and ambiguous to
     attempt a re-decode guess, so we only strip what's provably
     broken rather than risk corrupting good text)
  4. Collapse newlines/repeated whitespace to single spaces
     (defensive -- inspection found 0 literal newlines pre-cleaning,
     but HTML/URL stripping can leave doubled whitespace behind)
  5. Deduplicate on exact text (inspection found 0 pre-cleaning, and
     0 cross-language dupes, but re-check post-cleaning since
     stripping HTML/URLs could create new dupes)
  6. JOINT stratified split on (lang x toxic) -- this is neither a
     single binary stratify (sms_spam) nor multilabel (jigsaw_toxic).
     Per-language toxic rates differ meaningfully (es 0.169, it 0.195,
     tr 0.107) and lang is imbalanced (tr 3000 vs es/it 2500 each), so
     stratifying on toxic alone could let per-language rates drift
     across splits. We stratify on a combined "lang|toxic" column
     (6 strata, all large enough for a 70/15/15 split) using plain
     sklearn train_test_split -- no need for iterative stratification
     since there's only one real label here.

Usage:
    python clean_split.py
"""

import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clean_jigsaw_multilingual")

RAW_DIR = Path("data/raw/jigsaw_multilingual_toxic")
OUT_DIR = Path("data/processed/jigsaw_multilingual")
RANDOM_SEED = 42

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ENTITY_PATTERN = re.compile(r"&\w+;")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the acquisition script first")
    df = pd.read_csv(path)
    logger.info(f"Loaded raw: {len(df)} rows, columns: {list(df.columns)}")
    return df


def clean_text(text: str) -> str:
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub("<URL>", text)

    # Unrecoverable encoding artifacts -- strip rather than guess at
    # re-decoding, same policy as jigsaw_toxic's clean_split.py
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

    before = len(df)
    df = df[df["text"].str.len() >= 3]
    logger.info(f"Dropped {before - len(df)} rows under 3 characters after cleaning")

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    logger.info(f"Dropped {before - len(df)} duplicate-text rows (post-cleaning check)")

    df = df[["id", "text", "lang", "toxic"]].reset_index(drop=True)

    logger.info(f"Final cleaned shape: {df.shape}")
    logger.info(f"Overall toxic rate: {df['toxic'].mean():.4f}")
    logger.info(f"Per-language row counts:\n{df['lang'].value_counts().to_string()}")
    logger.info(f"Per-language toxic rate:\n{df.groupby('lang')['toxic'].mean().round(4).to_string()}")
    return df


def joint_stratified_split(df: pd.DataFrame):
    strata = df["lang"] + "|" + df["toxic"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=strata, random_state=RANDOM_SEED
    )
    temp_strata = temp_df["lang"] + "|" + temp_df["toxic"].astype(str)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_strata, random_state=RANDOM_SEED
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df, val_df, test_df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = OUT_DIR / f"{name}.parquet"
        split_df.to_parquet(out_path)
        logger.info(f"Wrote {name}.parquet: {len(split_df)} rows")
        logger.info(f"  Per-language counts:\n{split_df['lang'].value_counts().to_string()}")
        logger.info(f"  Per-language toxic rate:\n{split_df.groupby('lang')['toxic'].mean().round(4).to_string()}")


def main():
    df = load_raw()
    df = clean(df)
    train_df, val_df, test_df = joint_stratified_split(df)
    save_splits(train_df, val_df, test_df)
    logger.info(f"\nDone. Processed files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()