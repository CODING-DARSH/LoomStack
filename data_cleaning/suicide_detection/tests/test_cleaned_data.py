"""
Production-grade test suite for cleaned Suicide Detection data.
Combined basic + advanced, per project convention.

IMPORTANT: same policy as inspect_dataset.py -- this file never prints
raw text content on failure, only counts/lengths/aggregate stats.
Assertion messages are written to avoid leaking row content.

Thresholds calibrated against the actual clean_split.py run:
  - 232074 raw -> 231951 cleaned (123 duplicates emerged AFTER PII
    masking collapsed near-identical posts -- expected, not a bug)
  - Class balance: suicide 116031 / non-suicide 115920 (near-perfect,
    not forced/stratified from a raw imbalance -- the raw data was
    already exactly 50/50)
  - Mean length: non-suicide ~322 chars, suicide ~1045 chars -- a
    CONFIRMED, DELIBERATELY UNCORRECTED confound (see clean_split.py
    docstring point 2). This file's length-stratified tests exist
    specifically to keep that confound visible to anyone evaluating a
    model trained on this data, not to fix it.
  - Splits: train 162365 / val 34793 / test 34793 (~70/15/15)

Usage:
    pytest data_cleaning/suicide_detection/tests/test_cleaned_data.py -v
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed/suicide_detection")
FINGERPRINT_PATH = PROCESSED_DIR / ".fingerprint.json"
SPLITS = ["train", "val", "test"]
CLASSES = ["suicide", "non-suicide"]


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


# ---------------------------------------------------------------------
# 1. Basic structural sanity
# ---------------------------------------------------------------------

def test_files_exist():
    for split in SPLITS:
        assert (PROCESSED_DIR / f"{split}.parquet").exists()


def test_expected_columns(splits):
    expected = {"text", "class"}
    for split, df in splits.items():
        assert set(df.columns) == expected, f"{split} has unexpected columns: {list(df.columns)}"


def test_no_nulls(splits):
    for split, df in splits.items():
        assert df.isnull().sum().sum() == 0, f"{split} has null values"


def test_no_empty_text(splits):
    for split, df in splits.items():
        assert (df["text"].str.strip() == "").sum() == 0


def test_class_values_restricted_to_known_set(splits):
    for split, df in splits.items():
        unique = set(df["class"].unique())
        assert unique.issubset(set(CLASSES)), f"{split} has unexpected class values: {unique - set(CLASSES)}"


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
    assert total >= 230000, f"Only {total} rows survived cleaning, expected ~231951"


def test_column_dtypes(splits):
    for split, df in splits.items():
        assert pd.api.types.is_object_dtype(df["text"]), f"{split}.text is not object dtype: {df['text'].dtype}"
        assert pd.api.types.is_object_dtype(df["class"]), f"{split}.class is not object dtype: {df['class'].dtype}"


def test_utf8_encoding_round_trips_cleanly(splits):
    for split, df in splits.items():
        for text in df["text"]:
            try:
                text.encode("utf-8").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                pytest.fail(f"{split} has text that fails UTF-8 round-trip: {e}")


# ---------------------------------------------------------------------
# 2. Class balance
# ---------------------------------------------------------------------

def test_class_balance_close_to_fifty_fifty(splits):
    """Raw data was exactly 50/50 -- cleaning (dedup, PII masking)
    should not have meaningfully skewed this."""
    for split, df in splits.items():
        rate = (df["class"] == "suicide").mean()
        assert 0.48 <= rate <= 0.52, f"{split} suicide-class rate {rate:.4f} outside expected [0.48, 0.52]"


def test_dedup_did_not_disproportionately_affect_one_class(full_dataset):
    """123 duplicates emerged post-cleaning (from PII masking
    collapsing near-identical posts). This drop should be small and
    roughly even across classes -- if it were concentrated in one
    class, that would suggest PII patterns aren't evenly distributed
    in a way that's silently changing the effective class balance."""
    total = len(full_dataset)
    class_counts = full_dataset["class"].value_counts()
    diff = abs(class_counts["suicide"] - class_counts["non-suicide"])
    assert diff < 500, f"Class counts differ by {diff}, expected a small gap given raw data was exactly 50/50"


# ---------------------------------------------------------------------
# 3. Length confound -- DELIBERATELY VISIBLE, not fixed
# ---------------------------------------------------------------------

def test_length_confound_is_present_and_documented(full_dataset):
    """This test does NOT assert the confound is gone -- it asserts
    the confound is still there and roughly matches what we measured,
    so that anyone reading test output understands a model trained on
    this data risks learning length as a shortcut. If this ratio
    changes dramatically in a future cleaning pass, that's worth
    knowing (it would mean the confound characteristics shifted, for
    better or worse) -- so this is a locked-in observation, not a
    pass/fail quality gate in the usual sense."""
    mean_by_class = full_dataset.groupby("class")["text"].apply(lambda s: s.str.len().mean())
    ratio = mean_by_class["suicide"] / mean_by_class["non-suicide"]
    assert 2.0 <= ratio <= 5.0, (
        f"suicide/non-suicide mean length ratio is {ratio:.2f}, expected roughly 2-5x "
        f"(measured ~3.2x) -- if this changed significantly, understand why before training"
    )


def test_length_stratified_class_balance(full_dataset, splits):
    """The actual mitigation for the length confound: verify that
    WITHIN each length bucket, the class balance doesn't collapse to
    one class dominating. If short posts are almost all non-suicide
    and long posts are almost all suicide, a model evaluated only on
    overall accuracy could hide near-total reliance on length. This
    test forces that check to happen every time, for every split."""
    length_bins = [0, 200, 500, 1000, 2000, float("inf")]
    bin_labels = ["0-200", "200-500", "500-1000", "1000-2000", "2000+"]

    for split, df in splits.items():
        lengths = df["text"].str.len()
        bucket = pd.cut(lengths, bins=length_bins, labels=bin_labels)
        for label in bin_labels:
            bucket_df = df[bucket == label]
            if len(bucket_df) < 50:
                continue  # too few rows in this bucket to make a balance claim
            suicide_rate = (bucket_df["class"] == "suicide").mean()
            # NOTE: intentionally wide bounds -- we EXPECT some skew per
            # bucket (that's the confound), this just catches the
            # extreme case of a bucket being ~100% one class, which
            # would make within-bucket evaluation meaningless
            assert 0.05 <= suicide_rate <= 0.95, (
                f"{split} length bucket '{label}' ({len(bucket_df)} rows) is {suicide_rate:.2%} suicide-class -- "
                f"too skewed for any meaningful length-controlled evaluation in that bucket"
            )


# ---------------------------------------------------------------------
# 4. PII masking verification
# ---------------------------------------------------------------------

def test_no_raw_phone_numbers_remaining(splits):
    phone_pattern = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
    for split, df in splits.items():
        without_placeholder = df["text"].str.replace("<PHONE>", "", regex=False)
        bad = without_placeholder.str.contains(phone_pattern, regex=True).sum()
        assert bad == 0, f"{split} has {bad} rows with an un-masked phone-number-like pattern"


def test_no_raw_reddit_username_mentions_remaining(splits):
    username_pattern = re.compile(r"/?u/\w+")
    for split, df in splits.items():
        without_placeholder = df["text"].str.replace("<REDDIT_USER>", "", regex=False)
        bad = without_placeholder.str.contains(username_pattern, regex=True).sum()
        assert bad == 0, f"{split} has {bad} rows with an un-masked Reddit username mention"


def test_pii_placeholders_actually_present(full_dataset):
    """Confirms masking ran (not just that raw patterns are absent,
    which would trivially pass if the regex silently stopped
    matching)."""
    phone_placeholder_count = full_dataset["text"].str.contains("<PHONE>", regex=False).sum()
    user_placeholder_count = full_dataset["text"].str.contains("<REDDIT_USER>", regex=False).sum()
    assert phone_placeholder_count > 0, "No <PHONE> placeholders found -- phone masking may have silently broken"
    assert user_placeholder_count > 0, "No <REDDIT_USER> placeholders found -- username masking may have silently broken"


# ---------------------------------------------------------------------
# 5. Cleaning-artifact validation
# ---------------------------------------------------------------------

def test_no_html_entities_remaining(splits):
    entity_pattern = re.compile(r"&\w+;")
    for split, df in splits.items():
        bad = df["text"].str.contains(entity_pattern, regex=True).sum()
        assert bad == 0, f"{split} has {bad} rows with un-stripped HTML entities"


def test_no_raw_urls_remaining(splits):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    for split, df in splits.items():
        without_placeholder = df["text"].str.replace("<URL>", "", regex=False)
        bad = without_placeholder.str.contains(url_pattern, regex=True).sum()
        assert bad == 0, f"{split} has {bad} rows with an un-masked raw URL"


def test_no_markdown_links_remaining(splits):
    """Pattern tolerates Reddit's escaped-bracket markdown (\\[text\\](url))
    as well as the plain form -- see clean_split.py fix. Non-capturing
    groups used so str.contains doesn't warn about unused match groups."""
    md_pattern = re.compile(r"\\?\[(?:[^\]]+)\]\\?\((?:[^)]+)\)")
    for split, df in splits.items():
        bad = df["text"].str.contains(md_pattern, regex=True).sum()
        assert bad == 0, f"{split} has {bad} rows with un-stripped markdown link syntax"


def test_no_removed_deleted_placeholder_rows(splits):
    for split, df in splits.items():
        bad = df["text"].str.strip().str.lower().isin({"[removed]", "[deleted]"}).sum()
        assert bad == 0, f"{split} has {bad} rows that are just [removed]/[deleted] placeholders"


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


def test_title_body_concatenation_reduced(full_dataset):
    """66271/232074 (28.6%) raw rows had a title/body concatenation
    artifact. After inserting a space at the boundary, that specific
    pattern should be substantially reduced -- not necessarily zero
    (the pattern can occur in legitimate text, e.g. genuine camelCase
    or an unrelated lowercase-then-capital letter sequence), but the
    RATE should drop sharply, confirming the fix actually ran."""
    concat_pattern = re.compile(r"[a-z][A-Z]")
    rate = full_dataset["text"].str.contains(concat_pattern, regex=True).mean()
    assert rate < 0.10, (
        f"{rate:.2%} of rows still match the title/body concatenation pattern, "
        f"expected well under the raw 28.6% rate if the boundary fix worked"
    )


def test_all_text_is_valid_unicode(splits):
    for split, df in splits.items():
        for text in df["text"]:
            try:
                unicodedata.normalize("NFC", text)
            except (ValueError, TypeError) as e:
                pytest.fail(f"{split} has text that fails Unicode normalization: {e}")


# ---------------------------------------------------------------------
# 6. Very-short rows -- deliberately NOT dropped, confirm they survive
# ---------------------------------------------------------------------

def test_very_short_rows_are_preserved_not_dropped(full_dataset):
    """Deliberately different from every prior dataset in this repo:
    short posts here can be maximally meaningful, so they must NOT be
    filtered by a length floor. This test confirms that policy holds
    (some short rows exist), rather than checking they were removed."""
    very_short = (full_dataset["text"].str.len() < 10).sum()
    assert very_short > 0, (
        "No very-short rows (<10 chars) found -- if this used to be nonzero (raw had 29), "
        "check whether a length-based drop was accidentally reintroduced"
    )


# ---------------------------------------------------------------------
# 7. Outlier bounds
# ---------------------------------------------------------------------

def test_no_extreme_length_outliers(splits):
    for split, df in splits.items():
        lengths = df["text"].str.len()
        too_long = (lengths > 45000).sum()
        assert too_long == 0, (
            f"{split} has {too_long} texts over 45000 characters -- raw max was 40297, "
            f"investigate anything beyond that"
        )


# ---------------------------------------------------------------------
# 8. Statistical bounds
# ---------------------------------------------------------------------

def test_statistical_bounds_overall(full_dataset):
    row_count = len(full_dataset)
    class_rate = (full_dataset["class"] == "suicide").mean()
    assert row_count >= 230000, f"Row count {row_count} below expected minimum 230000"
    assert 0.48 <= class_rate <= 0.52, f"Overall suicide-class rate {class_rate:.4f} outside expected [0.48, 0.52]"
    assert set(full_dataset["class"].unique()).issubset(set(CLASSES)), "Unexpected class values found"


# ---------------------------------------------------------------------
# 9. Data fingerprinting
# ---------------------------------------------------------------------

def _compute_fingerprint(df: pd.DataFrame) -> str:
    sortable = df.sort_values(["text", "class"]).reset_index(drop=True)
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