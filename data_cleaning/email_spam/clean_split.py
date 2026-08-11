"""
CLEAN + SPLIT. Enron-Spam only. Nazario phishing corpus is deliberately
kept SEPARATE (own smaller clean_split.py, eval-only, never merged into
this train/val/test) -- see project decision: 414 phishing examples vs
31716 Enron rows is too size-imbalanced to support a real phishing
class, and Nazario needs its own HTML-heavy mbox parsing pipeline
anyway.

Cleaning decisions, confirmed via diagnose_enron_columns.py before
writing this:

  1. TEXT SOURCE: 'text' column is exactly subject + " " + message
     (confirmed 500/500 sampled rows). No hidden raw source being
     missed -- 'message' is just body-without-subject, nothing lost
     by using 'text' directly. Subject line carries real spam signal,
     so 'text' (fuller field) is the right column to clean, not
     'message'.

  2. Neither text nor message had any forwarded-headers / embedded
     Subject:/From:/To: lines / URLs / HTML tags -- confirmed genuinely
     clean at the source (HF's SetFit/enron_spam mirror), not a false
     negative from our checks. Standard artifact cleanup is still
     applied defensively (control chars, whitespace, mojibake) even
     though none were found, same policy as prior datasets.

  3. 51 rows are empty in BOTH text and message (confirmed length=0
     in message too, not a data-loss artifact) -- dropped as
     non-content, same as [removed]/[deleted] placeholders elsewhere.

  4. Duplicates concentrate in spam (4544) over ham (1081), ~4:1 --
     expected, spam campaigns send identical text; ham dupes are more
     likely accidental forwards/re-sends. Dedup is applied normally,
     with post-dedup class balance logged to confirm it doesn't skew
     too far given the uneven starting dupe rate.

  5. No native split exists -- fresh stratified 70/15/15 split on
     label_text, seeded.

  6. KNOWN LIMITATION (not a cleaning fix, a documented caveat):
     Enron-Spam is early-2000s corporate email. Spam patterns have
     changed substantially since then (no modern phishing kits, no
     crypto scams, no modern SEO spam). A model trained on this alone
     will generalize best to that era's spam style -- worth noting in
     any downstream model docs, not something cleaning can address.

Usage:
    python clean_split.py
"""

import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clean_enron_spam")

RAW_DIR = Path("data/raw/enron_spam")
OUT_DIR = Path("data/processed/email_spam")
RANDOM_SEED = 42

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the acquisition script first")
    df = pd.read_parquet(path)
    df = df[["text", "label_text"]].rename(columns={"label_text": "label"})
    logger.info(f"Loaded raw: {len(df)} rows")
    return df


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\ufffd", "")
    text = CONTROL_CHAR_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["text"].astype(str).str.strip() != ""].copy()
    logger.info(f"Dropped {before - len(df)} rows empty at the source (empty in both text and message)")

    df["text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["text"].str.strip() != ""]
    logger.info(f"Dropped {before - len(df)} rows that became empty after cleaning")

    before_by_label = df["label"].value_counts()
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    after_by_label = df["label"].value_counts()
    logger.info(f"Dropped {before - len(df)} duplicate-text rows")
    logger.info(f"Class counts before dedup:\n{before_by_label.to_string()}")
    logger.info(f"Class counts after dedup:\n{after_by_label.to_string()}")

    df = df.reset_index(drop=True)

    logger.info(f"Final cleaned shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"Class balance: {df['label'].value_counts(normalize=True).round(4).to_string()}")
    return df


def stratified_split(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED
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
        logger.info(f"  Class distribution:\n{split_df['label'].value_counts().to_string()}")


def main():
    df = load_raw()
    df = clean(df)
    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df)
    logger.info(f"\nDone. Processed files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()