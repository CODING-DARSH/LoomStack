"""
ADVANCED data quality tests -- these go beyond basic sanity checks
(file exists, no nulls, etc. -- see test_cleaned_data.py) and instead
validate the actual QUALITY and CONSISTENCY of the cleaned data.

Run after test_cleaned_data.py passes.

Usage:
    pytest data_cleaning/sms_spam/tests/test_advanced_quality.py -v
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed/sms_spam")
RAW_DIR = Path("data/raw/sms_spam_collection")
FINGERPRINT_PATH = PROCESSED_DIR / ".fingerprint.json"
SPLITS = ["train", "val", "test"]


@pytest.fixture(scope="module")
def splits():
    data = {}
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.parquet"
        if not path.exists():
            pytest.fail(f"{path} does not exist -- run 02_clean_and_split.py first")
        data[split] = pd.read_parquet(path)
    return data


@pytest.fixture(scope="module")
def full_dataset(splits):
    """All splits concatenated -- some checks (label consistency,
    dataset-wide fingerprinting) need to see everything at once."""
    return pd.concat(splits.values(), ignore_index=True)


@pytest.fixture(scope="module")
def raw_dataset():
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        pytest.skip("Raw CSV not found -- skipping drift comparison tests")
    return pd.read_csv(csv_files[0], encoding="latin-1")


def normalize_for_comparison(text: str) -> str:
    """Lowercase, strip punctuation/whitespace -- used to detect
    near-duplicates that exact-match dedup would miss (e.g. 'Win
    free iPhone!!' vs 'win free iphone')."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------
# 1. Label consistency -- same text should never have conflicting labels
# ---------------------------------------------------------------------

def test_no_label_conflicts_for_identical_text(full_dataset):
    """If the exact same message appears more than once (shouldn't,
    given dedup, but checking dataset-wide in case dedup was only
    applied per-split) it must always have the same label."""
    conflicts = (
        full_dataset.groupby("text")["label"]
        .nunique()
        .loc[lambda s: s > 1]
    )
    assert len(conflicts) == 0, (
        f"{len(conflicts)} texts have conflicting labels across the dataset. "
        f"Examples: {conflicts.index[:5].tolist()}"
    )


def test_no_label_conflicts_for_normalized_text(full_dataset):
    """Same check but after normalizing (lowercase, strip punctuation)
    -- catches near-identical messages with different labels, which
    likely indicates an annotation error upstream."""
    df = full_dataset.copy()
    df["normalized"] = df["text"].apply(normalize_for_comparison)
    conflicts = (
        df.groupby("normalized")["label"]
        .nunique()
        .loc[lambda s: s > 1]
    )
    # Report but don't hard-fail -- near-duplicate label conflicts are
    # worth knowing about but may be legitimate (e.g. same phrase used
    # in both a spam and a real conversational context)
    if len(conflicts) > 0:
        print(
            f"\nWARNING: {len(conflicts)} normalized-text groups have "
            f"conflicting labels (informational, not a hard failure): "
            f"{conflicts.index[:5].tolist()}"
        )


# ---------------------------------------------------------------------
# 2. Distribution drift -- cleaned data shouldn't look wildly different
#    from the raw data it came from
# ---------------------------------------------------------------------

def test_spam_rate_not_drifted_from_raw(full_dataset, raw_dataset):
    raw_spam_rate = (raw_dataset["v1"] == "spam").mean()
    cleaned_spam_rate = full_dataset["label"].mean()
    diff = abs(raw_spam_rate - cleaned_spam_rate)
    assert diff < 0.03, (
        f"Spam rate drifted from raw ({raw_spam_rate:.3f}) to cleaned "
        f"({cleaned_spam_rate:.3f}), diff={diff:.3f} -- cleaning may have "
        f"disproportionately removed one class"
    )


def test_mean_text_length_not_drifted_from_raw(full_dataset, raw_dataset):
    raw_mean_len = raw_dataset["v2"].astype(str).str.len().mean()
    cleaned_mean_len = full_dataset["text"].str.len().mean()
    # cleaned mean will be somewhat higher since we reconstructed
    # split-column text back onto v2 -- allow a wider tolerance for that
    pct_diff = abs(raw_mean_len - cleaned_mean_len) / raw_mean_len
    assert pct_diff < 0.15, (
        f"Mean text length drifted >15% from raw ({raw_mean_len:.1f} chars) "
        f"to cleaned ({cleaned_mean_len:.1f} chars)"
    )


# ---------------------------------------------------------------------
# 3. Near-duplicate leakage across splits (beyond exact-match dedup)
# ---------------------------------------------------------------------

def test_no_near_duplicate_leakage_across_splits(splits):
    """Exact-match dedup can still miss near-duplicates like 'Win free
    iPhone!!' vs 'win free iphone' -- these still leak information
    between splits even though they aren't byte-identical."""
    normalized = {
        split: set(df["text"].apply(normalize_for_comparison))
        for split, df in splits.items()
    }

    train_val = normalized["train"] & normalized["val"]
    train_test = normalized["train"] & normalized["test"]
    val_test = normalized["val"] & normalized["test"]

    # allow a small number -- very short common phrases ("ok", "yes")
    # will legitimately collide after normalization
    max_allowed = 15
    assert len(train_val) <= max_allowed, (
        f"{len(train_val)} near-duplicate texts leak between train/val "
        f"(threshold={max_allowed})"
    )
    assert len(train_test) <= max_allowed, (
        f"{len(train_test)} near-duplicate texts leak between train/test "
        f"(threshold={max_allowed})"
    )
    assert len(val_test) <= max_allowed, (
        f"{len(val_test)} near-duplicate texts leak between val/test "
        f"(threshold={max_allowed})"
    )


# ---------------------------------------------------------------------
# 4. Unicode / control character validation (beyond the basic mojibake
#    regex check in test_cleaned_data.py)
# ---------------------------------------------------------------------

def test_no_unicode_replacement_characters(splits):
    """The presence of U+FFFD (the replacement character) means some
    byte sequence failed to decode properly during cleaning."""
    for split, df in splits.items():
        bad = df["text"].str.contains("\ufffd", regex=False, na=False).sum()
        assert bad == 0, f"{split} has {bad} rows containing the Unicode replacement character"


def test_no_control_characters(splits):
    """Control characters (other than standard whitespace) shouldn't
    survive cleaning -- their presence usually indicates a parsing
    or encoding bug."""
    control_char_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    for split, df in splits.items():
        bad = df["text"].apply(lambda t: bool(control_char_pattern.search(t))).sum()
        assert bad == 0, f"{split} has {bad} rows containing raw control characters"


def test_all_text_is_valid_unicode(splits):
    for split, df in splits.items():
        for text in df["text"]:
            try:
                unicodedata.normalize("NFC", text)
            except (ValueError, TypeError) as e:
                pytest.fail(f"{split} has text that fails Unicode normalization: {e}")


# ---------------------------------------------------------------------
# 5. Outlier detection -- extreme lengths often indicate parsing issues
# ---------------------------------------------------------------------

def test_no_extreme_length_outliers(splits):
    for split, df in splits.items():
        lengths = df["text"].str.len()
        too_short = (lengths < 2).sum()
        too_long = (lengths > 1000).sum()
        assert too_short == 0, f"{split} has {too_short} texts under 2 characters"
        assert too_long == 0, (
            f"{split} has {too_long} texts over 1000 characters -- "
            f"check for a parsing/concatenation bug"
        )


# ---------------------------------------------------------------------
# 6. Statistical bounds -- lightweight equivalent of Great
#    Expectations / Pandera rules, without adding a new dependency
# ---------------------------------------------------------------------

def test_statistical_bounds(full_dataset):
    spam_rate = full_dataset["label"].mean()
    mean_len = full_dataset["text"].str.len().mean()
    row_count = len(full_dataset)

    assert 0.08 <= spam_rate <= 0.20, f"Spam rate {spam_rate:.3f} outside expected [0.08, 0.20]"
    assert 40 <= mean_len <= 120, f"Mean text length {mean_len:.1f} outside expected [40, 120]"
    assert row_count >= 4500, f"Row count {row_count} below expected minimum 4500"
    assert set(full_dataset["label"].unique()).issubset({0, 1}), "Labels outside {0, 1} found"


# ---------------------------------------------------------------------
# 7. Data fingerprinting -- detect unexpected changes between pipeline
#    runs (e.g. someone re-ran acquisition and the raw data changed,
#    or cleaning logic changed without updating the fingerprint)
# ---------------------------------------------------------------------

def _compute_fingerprint(df: pd.DataFrame) -> str:
    # sort for determinism regardless of row order, hash text+label pairs
    sortable = df.sort_values(["text", "label"]).reset_index(drop=True)
    payload = sortable.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_dataset_fingerprint_stability(full_dataset):
    """First run: writes the fingerprint to disk (informational).
    Subsequent runs: fails loudly if the cleaned dataset changed
    without anyone updating/deleting the stored fingerprint -- this
    catches silent changes to raw data or cleaning logic."""
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
        f"This means the cleaned data differs from the last verified run. "
        f"If this is an intentional change (e.g. updated cleaning logic), "
        f"delete {FINGERPRINT_PATH} and re-run to accept the new fingerprint."
    )


# ---------------------------------------------------------------------
# 8. Model-facing validation -- lightweight proxy without requiring a
#    real tokenizer/model dependency in the test suite
# ---------------------------------------------------------------------

def test_no_text_exceeds_reasonable_token_proxy_length(splits):
    """Real tokenizer-based validation belongs in the model training
    pipeline (it needs the actual tokenizer you'll train with), but we
    can catch obviously-too-long inputs here using a word-count proxy
    -- most transformer tokenizers produce roughly 1.3 tokens per word,
    so anything over ~300 words is already far beyond typical SMS
    length and worth flagging before it reaches a real tokenizer."""
    for split, df in splits.items():
        word_counts = df["text"].str.split().apply(len)
        too_long = (word_counts > 300).sum()
        assert too_long == 0, (
            f"{split} has {too_long} texts over 300 words -- verify these "
            f"aren't a cleaning/concatenation artifact before tokenizing"
        )