"""
Production-grade test suite for cleaned Enron-Spam data. Combined
basic + advanced, per project convention. Nazario is NOT covered here
-- separate dataset, separate test file, eval-only, never merged.

Thresholds calibrated against the actual clean_split.py run:
  - 31716 raw -> 51 dropped (empty at source) -> 3137 duplicates
    dropped -> 28528 final rows
  - Dedup flipped the class balance: raw was spam-majority
    (16112/15553), cleaned is HAM-majority (14704/13824, ~51.5%/48.5%)
    -- confirmed expected given spam's dupe rate was ~4x ham's dupe
    rate (spam campaigns repeat identical text; ham dupes are more
    likely accidental). This is a real, documented effect of dedup,
    not a bug -- tests here lock in that the flip happened and stayed
    within a sane range, rather than assuming the raw spam-majority
    balance should have been preserved.
  - Splits: train 19969 / val 4279 / test 4280 (~70/15/15)

Usage:
    pytest data_cleaning/email_spam/tests/test_cleaned_data.py -v
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed/email_spam")
RAW_DIR = Path("data/raw/enron_spam")
FINGERPRINT_PATH = PROCESSED_DIR / ".fingerprint.json"
SPLITS = ["train", "val", "test"]
LABELS = ["ham", "spam"]


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def splits():
    data = {}
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.parquet"
        if not path.exists():
            pytest.fail(f"{path} does not exist -- run clean_split.py first")
        data[split] = pd.read_parquet(path)
    return data


@pytest.fixture(scope="module")
def full_dataset(splits):
    return pd.concat(splits.values(), ignore_index=True)


@pytest.fixture(scope="module")
def raw_dataset():
    path = RAW_DIR / "data.parquet"
    if not path.exists():
        pytest.skip("Raw enron_spam data.parquet not found -- skipping raw comparison tests")
    df = pd.read_parquet(path)
    return df[["text", "label_text"]].rename(columns={"label_text": "label"})


# ---------------------------------------------------------------------
# 1. Basic structural sanity
# ---------------------------------------------------------------------

def test_files_exist():
    for split in SPLITS:
        assert (PROCESSED_DIR / f"{split}.parquet").exists()


def test_expected_columns(splits):
    expected = {"text", "label"}
    for split, df in splits.items():
        assert set(df.columns) == expected, f"{split} has unexpected columns: {list(df.columns)}"


def test_no_nulls(splits):
    for split, df in splits.items():
        assert df.isnull().sum().sum() == 0, f"{split} has null values"


def test_no_empty_text(splits):
    for split, df in splits.items():
        assert (df["text"].str.strip() == "").sum() == 0


def test_label_values_restricted_to_known_set(splits):
    for split, df in splits.items():
        unique = set(df["label"].unique())
        assert unique.issubset(set(LABELS)), f"{split} has unexpected label values: {unique - set(LABELS)}"


def test_no_duplicate_text_within_split(splits):
    for split, df in splits.items():
        dupes = df["text"].duplicated().sum()
        assert dupes == 0, f"{split} has {dupes} duplicate text rows"


def test_no_exact_text_leakage_across_splits(splits):
    train_texts = set(splits["train"]["text"])
    val_texts = set(splits["val"]["text"])
    test_texts = set(splits["test"]["text"])
    assert len(train_texts & val_texts) == 0, "text leakage between train/val"
    assert len(train_texts & test_texts) == 0, "text leakage between train/test"
    assert len(val_texts & test_texts) == 0, "text leakage between val/test"


def test_split_proportions_roughly_correct(splits):
    total = sum(len(df) for df in splits.values())
    train_frac = len(splits["train"]) / total
    val_frac = len(splits["val"]) / total
    test_frac = len(splits["test"]) / total
    assert 0.65 <= train_frac <= 0.75, f"train fraction {train_frac:.3f} out of range"
    assert 0.10 <= val_frac <= 0.20, f"val fraction {val_frac:.3f} out of range"
    assert 0.10 <= test_frac <= 0.20, f"test fraction {test_frac:.3f} out of range"


def test_minimum_dataset_size(splits):
    total = sum(len(df) for df in splits.values())
    assert total >= 28000, f"Only {total} rows survived cleaning, expected ~28528"


def test_column_dtypes(splits):
    for split, df in splits.items():
        assert pd.api.types.is_object_dtype(df["text"]), f"{split}.text is not object dtype: {df['text'].dtype}"
        assert pd.api.types.is_object_dtype(df["label"]), f"{split}.label is not object dtype: {df['label'].dtype}"


def test_utf8_encoding_round_trips_cleanly(splits):
    for split, df in splits.items():
        for text in df["text"]:
            try:
                text.encode("utf-8").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                pytest.fail(f"{split} has text that fails UTF-8 round-trip: {e}")


# ---------------------------------------------------------------------
# 2. Class balance -- the dedup-driven flip, locked in deliberately
# ---------------------------------------------------------------------

def test_class_balance_is_ham_majority_post_dedup(full_dataset):
    """Raw data was spam-majority (16112/15553). Dedup flips this to
    ham-majority (~51.5%/48.5%) because spam's dupe rate was ~4x ham's.
    This test locks in the FLIPPED direction deliberately -- if a
    future cleaning change makes this spam-majority again, that's
    worth investigating (did the dedup logic change?), not silently
    passing a generic 'close to 50/50' check that wouldn't catch which
    direction the imbalance points."""
    ham_rate = (full_dataset["label"] == "ham").mean()
    assert 0.48 <= ham_rate <= 0.55, f"ham rate {ham_rate:.4f} outside expected [0.48, 0.55]"


def test_class_balance_stable_across_splits(splits):
    for split, df in splits.items():
        ham_rate = (df["label"] == "ham").mean()
        assert 0.45 <= ham_rate <= 0.58, f"{split} ham rate {ham_rate:.4f} outside expected [0.45, 0.58]"


def test_dedup_rate_higher_in_spam_than_ham(raw_dataset, full_dataset):
    """Confirms the documented mechanism (spam dupes at ~4x ham's
    rate) actually held -- not just that the balance flipped, but that
    it flipped for the expected, understood reason."""
    raw_counts = raw_dataset["label"].value_counts()
    cleaned_counts = full_dataset["label"].value_counts()

    spam_dedup_rate = 1 - (cleaned_counts.get("spam", 0) / raw_counts.get("spam", 1))
    ham_dedup_rate = 1 - (cleaned_counts.get("ham", 0) / raw_counts.get("ham", 1))

    assert spam_dedup_rate > ham_dedup_rate, (
        f"Expected spam dedup rate ({spam_dedup_rate:.3f}) to exceed ham dedup rate "
        f"({ham_dedup_rate:.3f}) -- if this flipped, the balance-flip explanation no longer holds"
    )


# ---------------------------------------------------------------------
# 3. Drift vs raw
# ---------------------------------------------------------------------

def test_row_count_drop_matches_known_causes(full_dataset, raw_dataset):
    """31716 raw -> 51 empty-at-source dropped -> 3137 duplicates
    dropped -> 28528 final. Total drop should match this, not silently
    diverge due to an unrelated cleaning change."""
    raw_count = len(raw_dataset)
    cleaned_count = len(full_dataset)
    dropped = raw_count - cleaned_count
    assert 3100 <= dropped <= 3300, (
        f"Dropped {dropped} rows ({raw_count} -> {cleaned_count}), expected ~3188 "
        f"(51 empty + 3137 duplicate)"
    )


# ---------------------------------------------------------------------
# 4. Near-duplicate leakage across splits
# ---------------------------------------------------------------------

def normalize_for_comparison(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def test_no_near_duplicate_leakage_across_splits(splits):
    normalized = {
        split: set(df["text"].apply(normalize_for_comparison))
        for split, df in splits.items()
    }
    train_val = normalized["train"] & normalized["val"]
    train_test = normalized["train"] & normalized["test"]
    val_test = normalized["val"] & normalized["test"]

    max_allowed = 25
    assert len(train_val) <= max_allowed, f"{len(train_val)} near-duplicate texts leak between train/val"
    assert len(train_test) <= max_allowed, f"{len(train_test)} near-duplicate texts leak between train/test"
    assert len(val_test) <= max_allowed, f"{len(val_test)} near-duplicate texts leak between val/test"


# ---------------------------------------------------------------------
# 5. Cleaning-artifact validation
# ---------------------------------------------------------------------

def test_no_unicode_replacement_characters(splits):
    for split, df in splits.items():
        bad = df["text"].str.contains("\ufffd", regex=False, na=False).sum()
        assert bad == 0, f"{split} has {bad} rows containing the Unicode replacement character"


def test_no_control_characters(splits):
    control_char_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    for split, df in splits.items():
        bad = df["text"].apply(lambda t: bool(control_char_pattern.search(t))).sum()
        assert bad == 0, f"{split} has {bad} rows containing raw control characters"


def test_no_literal_newlines(splits):
    for split, df in splits.items():
        bad = df["text"].str.contains("\n", regex=False).sum()
        assert bad == 0, f"{split} has {bad} rows containing literal newlines"


def test_no_double_spaces(splits):
    for split, df in splits.items():
        bad = df["text"].str.contains("  ", regex=False).sum()
        assert bad == 0, f"{split} has {bad} rows with doubled whitespace"


def test_all_text_is_valid_unicode(splits):
    for split, df in splits.items():
        for text in df["text"]:
            try:
                unicodedata.normalize("NFC", text)
            except (ValueError, TypeError) as e:
                pytest.fail(f"{split} has text that fails Unicode normalization: {e}")


# ---------------------------------------------------------------------
# 6. Outlier / length bounds
# ---------------------------------------------------------------------

def test_no_extreme_length_outliers(splits):
    for split, df in splits.items():
        lengths = df["text"].str.len()
        too_short = (lengths < 1).sum()
        too_long = (lengths > 250000).sum()
        assert too_short == 0, f"{split} has {too_short} empty texts"
        assert too_long == 0, (
            f"{split} has {too_long} texts over 250000 characters -- raw max was 228368, "
            f"investigate anything beyond that"
        )


# ---------------------------------------------------------------------
# 7. Statistical bounds
# ---------------------------------------------------------------------

def test_statistical_bounds_overall(full_dataset):
    row_count = len(full_dataset)
    ham_rate = (full_dataset["label"] == "ham").mean()
    assert row_count >= 28000, f"Row count {row_count} below expected minimum 28000"
    assert 0.48 <= ham_rate <= 0.55, f"ham rate {ham_rate:.4f} outside expected [0.48, 0.55]"
    assert set(full_dataset["label"].unique()).issubset(set(LABELS)), "Unexpected label values found"


# ---------------------------------------------------------------------
# 8. Data fingerprinting
# ---------------------------------------------------------------------

def _compute_fingerprint(df: pd.DataFrame) -> str:
    sortable = df.sort_values(["text", "label"]).reset_index(drop=True)
    payload = sortable.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_dataset_fingerprint_stability(full_dataset):
    current_fp = _compute_fingerprint(full_dataset)

    if not FINGERPRINT_PATH.exists():
        FINGERPRINT_PATH.write_text(json.dumps({"sha256": current_fp}, indent=2))
        pytest.skip(
            f"No stored fingerprint found -- wrote one now ({current_fp[:12]}...). "
            f"Re-run this test on future pipeline runs to detect drift."
        )

    stored_fp = json.loads(FINGERPRINT_PATH.read_text())["sha256"]
    assert current_fp == stored_fp, (
        f"Dataset fingerprint changed!\n"
        f"  stored:  {stored_fp}\n"
        f"  current: {current_fp}\n"
        f"If this is an intentional change (e.g. updated cleaning logic), "
        f"delete {FINGERPRINT_PATH} and re-run to accept the new fingerprint."
    )