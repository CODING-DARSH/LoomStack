"""
CLEAN + SPLIT. Run after reviewing inspect_dataset.py output.

Cleaning decisions made here, confirmed before writing (see project
history -- not re-litigated in code comments beyond a summary):

  1. PII MASKING: phone-number-like patterns -> <PHONE>, Reddit
     username mentions (u/... or /u/...) -> <REDDIT_USER>. 162 and
     1558 rows affected respectively. This is real people's
     unfiltered crisis-related content, so the bar for masking
     incidental PII is higher here than in prior datasets.

  2. LENGTH CONFOUND (suicide class mean 1050 chars vs non-suicide
     mean 329 chars) is DELIBERATELY NOT addressed by truncation here.
     Truncating risks cutting real crisis content out of long posts,
     which is a worse failure mode than leaving the confound in place.
     Instead, the test suite carries length-stratified checks so any
     future model relying on length as a shortcut is visible at eval
     time rather than silently passing on raw accuracy. See
     tests/test_cleaned_data.py.

  3. VERY SHORT ROWS (<10 chars, 29 rows) are left as-is, not dropped.
     Unlike prior datasets, a short post here can be maximally
     meaningful ("I can't do this anymore.") rather than noise --
     applying the usual "drop very short text" rule would be wrong
     for this dataset specifically.

  4. TITLE/BODY CONCATENATION FIX: 66271 rows (28.6%) have a scrape
     artifact where the post title runs directly into the body with
     no separator (e.g. "SuicideRecently I left..."). A single space
     is inserted at the lowercase-then-uppercase boundary. This is a
     formatting repair, not a content change.

  5. Standard artifact cleanup: HTML entities, URLs masked to <URL>,
     markdown links stripped to their visible text, newlines
     collapsed, control chars / mojibake-adjacent replacement chars
     stripped -- same policy as prior datasets.

  6. No native split exists (single raw CSV) -- fresh stratified
     70/15/15 split on class, seeded.

  7. [removed]/[deleted] placeholder drop rule is INCLUDED defensively
     even though inspection found 0 -- cheap insurance against a
     future re-fetch picking up rows that do contain them.

Usage:
    python clean_split.py
"""

import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clean_suicide_detection")

RAW_DIR = Path("data/raw/suicide_watch")
OUT_DIR = Path("data/processed/suicide_detection")
RANDOM_SEED = 42

HTML_ENTITY_PATTERN = re.compile(r"&\w+;")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_LINK_PATTERN = re.compile(r"\\?\[([^\]]+)\]\\?\(([^)]+)\)")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TITLE_BODY_BOUNDARY_PATTERN = re.compile(r"([a-z])([A-Z])")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
REDDIT_USERNAME_PATTERN = re.compile(r"/?u/\w+")
REMOVED_PLACEHOLDERS = {"[removed]", "[deleted]"}


def load_raw() -> pd.DataFrame:
    csv_candidates = list(RAW_DIR.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR} -- run the acquisition script first")
    df = pd.read_csv(csv_candidates[0])
    df = df[["text", "class"]].copy()
    logger.info(f"Loaded raw: {len(df)} rows")
    return df


def clean_text(text: str) -> str:
    text = str(text)

    # title/body concatenation repair -- insert a space at the
    # lowercase-then-uppercase boundary.
    text = TITLE_BODY_BOUNDARY_PATTERN.sub(r"\1 \2", text)

    # URL masking runs BEFORE markdown-link stripping. Raw URLs can
    # contain their own embedded ) or ] characters (tracking/CDN query
    # params), which breaks the markdown-link regex's [^)]+ / [^\]]+
    # character classes and lets the wrapper syntax survive untouched.
    # Masking first collapses any URL down to a short, simple <URL>
    # token with none of those problem characters, so the markdown
    # wrapper strip that follows always has clean, matchable content
    # to work with -- confirmed via diagnose_markdown_link.py against
    # a row where the original order left "[<URL> ...](<URL> ...)"
    # residue.
    text = URL_PATTERN.sub("<URL>", text)

    # markdown links -> keep visible text, drop the URL target.
    # Run as a FIXPOINT LOOP (repeat until no more matches) rather
    # than a single pass -- after repeated debugging, this proved
    # more reliable than trying to predict every ordering/nesting edge
    # case a single regex pass can hit on messy Reddit-scraped
    # markdown. Bounded iteration count as a safety net against any
    # pathological input causing an infinite loop.
    for _ in range(5):
        new_text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
        if new_text == text:
            break
        text = new_text

    text = HTML_ENTITY_PATTERN.sub(" ", text)

    # PII masking
    text = PHONE_PATTERN.sub("<PHONE>", text)
    text = REDDIT_USERNAME_PATTERN.sub("<REDDIT_USER>", text)

    text = text.replace("\ufffd", "")
    text = CONTROL_CHAR_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[~df["text"].astype(str).str.strip().str.lower().isin(REMOVED_PLACEHOLDERS)]
    logger.info(f"Dropped {before - len(df)} [removed]/[deleted] placeholder rows")

    df["text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["text"].str.strip() != ""]
    logger.info(f"Dropped {before - len(df)} rows that became empty after cleaning")

    # NOTE: deliberately no length-based drop here -- see docstring
    # point 3. Short rows are kept.

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    logger.info(f"Dropped {before - len(df)} duplicate-text rows (post-cleaning check)")

    df = df.reset_index(drop=True)

    logger.info(f"Final cleaned shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['class'].value_counts().to_string()}")
    logger.info(f"Mean length by class:\n{df.groupby('class')['text'].apply(lambda s: s.str.len().mean()).round(1).to_string()}")
    return df


def stratified_split(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["class"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["class"], random_state=RANDOM_SEED
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
        logger.info(f"  Class distribution:\n{split_df['class'].value_counts().to_string()}")


def main():
    df = load_raw()
    df = clean(df)
    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df)
    logger.info(f"\nDone. Processed files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()