"""
CLEAN + MERGE. Run after reviewing inspect_dataset.py output.

Merge decisions made here (confirmed with project owner before writing):
  1. Unified 3-class label schema: hatespeech / normal / offensive.
     Dynahate is natively binary -- mapped hate->hatespeech,
     nothate->normal. It never contributes "offensive" rows; that
     class comes from HateXplain only.
  2. HateXplain label = majority vote across its 3 annotators. In the
     rare case all 3 disagree (no majority possible), the row is
     DROPPED rather than guessed at.
  3. Dynahate ships train/validation/test.parquet as three IDENTICAL
     copies of the full 82399-row dataset (confirmed: t.equals(v) ==
     True) -- the real split lives in the row-level 'split' column
     (train/test/dev), not the filename. We load ONLY train.parquet
     and use its 'split' column as ground truth. Loading all three
     files as if they were distinct would have silently triplicated
     every row across train/val/test with total leakage.
  4. NATIVE SPLITS ARE KEPT, not re-shuffled. Both sources ship
     purpose-built splits (HateXplain: human-annotated split;
     Dynahate: split respects adversarial round balance across
     rounds 1-4) -- re-splitting from scratch would throw that away
     for no benefit. HateXplain's "validation" and Dynahate's "dev"
     are both renamed to "val" for a consistent 3-way naming scheme.
  5. HateXplain's post_tokens (list of words) is joined with " " to
     reconstruct text. This is NOT a perfect detokenization -- original
     punctuation spacing/contractions are lost (e.g. "can not" instead
     of "can't" -- that's how HateXplain itself tokenized it, not
     something we can recover). The <user> token is HateXplain's own
     anonymization marker for @mentions and is left as-is, same
     treatment as the <URL> placeholder in jigsaw_multilingual.
  6. A 'source' column (hatexplain / dynahate) is kept on every row so
     downstream tests/eval can check per-source drift and leakage
     separately -- these two corpora have very different collection
     methodologies (human-authored talk pages/tweets vs
     adversarially-generated-to-fool-a-model), so pooling them without
     a way to tell them apart would hide that.
  7. Deduplication is GLOBAL across both sources and all splits (not
     per-split, not per-source) -- this is the only thing that can
     introduce cross-source/cross-split leakage after merging, so it
     has to run on the full merged frame before anything is saved.
     Kept-first order is hatexplain-then-dynahate, so if the same text
     exists in both, we keep HateXplain's richer annotation.

Usage:
    python clean_split.py
"""

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clean_hatexplain_dynahate")

HATEXPLAIN_DIR = Path("data/raw/hatexplain")
DYNAHATE_DIR = Path("data/raw/dynahate")
OUT_DIR = Path("data/processed/hatexplain_dynahate")

LABEL_INT_TO_STR = {0: "hatespeech", 1: "normal", 2: "offensive"}
HATEXPLAIN_SPLIT_RENAME = {"train": "train", "validation": "val", "test": "test"}
DYNAHATE_SPLIT_RENAME = {"train": "train", "dev": "val", "test": "test"}

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------
# HateXplain
# ---------------------------------------------------------------------

def load_hatexplain() -> pd.DataFrame:
    rows = []
    dropped_no_majority = 0

    for split_file in sorted(HATEXPLAIN_DIR.glob("*.parquet")):
        raw_split_name = split_file.stem
        split_name = HATEXPLAIN_SPLIT_RENAME[raw_split_name]
        df = pd.read_parquet(split_file)

        for _, row in df.iterrows():
            labels = [int(v) for v in row["annotators"]["label"]]
            unique_labels = set(labels)

            if len(labels) == 3 and len(unique_labels) == 3:
                # true 3-way tie, no majority possible -- drop per
                # agreed policy rather than guessing
                dropped_no_majority += 1
                continue

            majority = max(unique_labels, key=labels.count)
            text = " ".join(row["post_tokens"])

            rows.append({
                "id": row["id"],
                "text": text,
                "label": LABEL_INT_TO_STR[majority],
                "source": "hatexplain",
                "split": split_name,
            })

    logger.info(f"HateXplain: loaded {len(rows)} rows, dropped {dropped_no_majority} with no annotator majority")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Dynahate
# ---------------------------------------------------------------------

def load_dynahate() -> pd.DataFrame:
    # train/validation/test.parquet are confirmed identical copies of
    # the full dataset -- load only one, and use the row-level 'split'
    # column (not the filename) as ground truth for split assignment.
    path = DYNAHATE_DIR / "train.parquet"
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    logger.info(f"Dynahate: loaded {len(df)} rows from {path.name} (single file, per-row 'split' column used)")

    unmapped_splits = set(df["split"].unique()) - set(DYNAHATE_SPLIT_RENAME.keys())
    if unmapped_splits:
        raise ValueError(f"Unexpected Dynahate split values, update DYNAHATE_SPLIT_RENAME: {unmapped_splits}")

    label_map = {1: "hatespeech", 0: "normal"}
    unmapped_labels = set(df["label"].unique()) - set(label_map.keys())
    if unmapped_labels:
        raise ValueError(f"Unexpected Dynahate label values, update label_map: {unmapped_labels}")

    out = pd.DataFrame({
        "id": df["acl.id"],
        "text": df["text"],
        "label": df["label"].map(label_map),
        "source": "dynahate",
        "split": df["split"].map(DYNAHATE_SPLIT_RENAME),
    })
    return out


# ---------------------------------------------------------------------
# Shared text cleaning
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\ufffd", "")
    text = CONTROL_CHAR_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_and_merge(hx_df: pd.DataFrame, dh_df: pd.DataFrame) -> pd.DataFrame:
    # hatexplain-then-dynahate order matters for dedup keep="first" below
    merged = pd.concat([hx_df, dh_df], ignore_index=True)
    logger.info(f"Merged (pre-clean): {len(merged)} rows")

    merged["text"] = merged["text"].apply(clean_text)

    before = len(merged)
    merged = merged[merged["text"].str.strip() != ""]
    logger.info(f"Dropped {before - len(merged)} rows that became empty after cleaning")

    before = len(merged)
    merged = merged[merged["text"].str.len() >= 3]
    logger.info(f"Dropped {before - len(merged)} rows under 3 characters after cleaning")

    # GLOBAL dedup across source AND split -- this is the only step
    # that can remove cross-source/cross-split leakage, must run on
    # the fully merged frame before saving
    before = len(merged)
    dupes_by_split = merged[merged.duplicated(subset=["text"], keep=False)]["split"].value_counts()
    merged = merged.drop_duplicates(subset=["text"], keep="first")
    logger.info(f"Dropped {before - len(merged)} duplicate-text rows (global, cross-source)")
    if len(dupes_by_split) > 0:
        logger.info(f"Duplicate text was distributed across splits as:\n{dupes_by_split.to_string()}")

    merged = merged[["id", "text", "label", "source", "split"]].reset_index(drop=True)

    logger.info(f"Final merged shape: {merged.shape}")
    logger.info(f"Label distribution:\n{merged['label'].value_counts().to_string()}")
    logger.info(f"Source distribution:\n{merged['source'].value_counts().to_string()}")
    logger.info(f"Split distribution:\n{merged['split'].value_counts().to_string()}")
    return merged


def save_splits(df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].drop(columns=["split"]).reset_index(drop=True)
        out_path = OUT_DIR / f"{split_name}.parquet"
        split_df.to_parquet(out_path)
        logger.info(f"Wrote {split_name}.parquet: {len(split_df)} rows")
        logger.info(f"  Label distribution:\n{split_df['label'].value_counts().to_string()}")
        logger.info(f"  Source distribution:\n{split_df['source'].value_counts().to_string()}")


def main():
    hx_df = load_hatexplain()
    dh_df = load_dynahate()
    merged = clean_and_merge(hx_df, dh_df)
    save_splits(merged)
    logger.info(f"\nDone. Processed files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()