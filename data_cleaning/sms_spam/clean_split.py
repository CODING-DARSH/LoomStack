"""
STEP 2: CLEAN + SPLIT. Run this after reviewing 01_inspect.py output.

Cleaning decisions made here (based on known structure of this dataset,
confirmed by inspection):
  1. Drop junk "Unnamed: 2/3/4" columns (known artifact of this CSV export)
  2. Rename v1 -> label, v2 -> text
  3. Standardize label: "spam" -> 1, "ham" -> 0
  4. Strip whitespace, drop empty text rows
  5. Deduplicate on exact text match (keep first occurrence)
  6. Stratified train/val/test split (70/15/15), fixed random seed
     for reproducibility -- this split is written to disk ONCE and
     reused, never regenerated randomly on each run.

Usage:
    python 02_clean_and_split.py
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clean_sms_spam")

RAW_DIR = Path("data/raw/sms_spam_collection")
OUT_DIR = Path("data/processed/sms_spam")
RANDOM_SEED = 42


def load_raw() -> pd.DataFrame:
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR}")
    df = pd.read_csv(csv_files[0], encoding="latin-1")
    logger.info(f"Loaded raw: {len(df)} rows, columns: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. The "Unnamed: 2/3/4" columns are NOT pure junk -- inspection
    #    showed they contain continuation fragments of the original
    #    message text. Some raw SMS messages contain unescaped commas,
    #    which broke the CSV parsing and spilled the tail of the
    #    message into these extra columns (~68 rows affected). Simply
    #    dropping them would silently truncate those messages, so we
    #    reconstruct the full text by concatenating non-null fragments
    #    back onto v2 before dropping the now-empty columns.
    junk_cols = [c for c in df.columns if "Unnamed" in str(c)]
    if junk_cols:
        for c in junk_cols:
            fragment = df[c].fillna("")
            df["v2"] = df["v2"].astype(str) + fragment.apply(lambda x: f",{x}" if x else "")
        df = df.drop(columns=junk_cols)
        logger.info(f"Reconstructed split text from columns: {junk_cols}, then dropped them")

    # 2. rename to standard schema used across all text datasets in this project
    df = df.rename(columns={"v1": "label", "v2": "text"})

    # 3. standardize label to int (0=ham, 1=spam) -- consistent with
    #    other binary text classifiers in this project
    label_map = {"ham": 0, "spam": 1}
    unmapped = set(df["label"].unique()) - set(label_map.keys())
    if unmapped:
        raise ValueError(f"Unexpected label values found, update label_map: {unmapped}")
    df["label"] = df["label"].map(label_map)

    # 4. strip whitespace, drop empty text
    df["text"] = df["text"].astype(str).str.strip()
    before = len(df)
    df = df[df["text"] != ""]
    logger.info(f"Dropped {before - len(df)} empty-text rows")

    # 5. deduplicate on exact text match
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    logger.info(f"Dropped {before - len(df)} duplicate-text rows")

    df = df[["text", "label"]].reset_index(drop=True)
    logger.info(f"Final cleaned shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def split_and_save(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # stratified split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = OUT_DIR / f"{name}.parquet"
        split_df.reset_index(drop=True).to_parquet(out_path)
        logger.info(
            f"Wrote {name}.parquet: {len(split_df)} rows, "
            f"spam rate={split_df['label'].mean():.3f}"
        )


def main():
    df = load_raw()
    df = clean(df)
    split_and_save(df)
    logger.info(f"\nDone. Processed files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()