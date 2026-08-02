"""
Production-grade test suite for cleaned Jigsaw Multilingual Toxic
Comment data. Combines basic sanity checks and advanced quality/drift
checks in one file (per-project convention going forward -- no more
separate test_cleaned_data.py / test_advanced.py split).

Thresholds below are calibrated against the actual clean_split.py run
on this dataset (8000 raw -> 7999 cleaned, 1 dup dropped, es/it/tr
languages, overall toxic rate 0.1538). Update them if you intentionally
change cleaning logic upstream.

Usage:
    pytest data_cleaning/jigsaw_multilingual/tests/test_cleaned_data.py -v
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed/jigsaw_multilingual")
RAW_DIR = Path("data/raw/jigsaw_multilingual_toxic")
FINGERPRINT_PATH = PROCESSED_DIR / ".fingerprint.json"
SPLITS = ["train", "val", "test"]
LANGS = ["es", "it", "tr"]


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
    path = RAW_DIR / "validation.csv"
    if not path.exists():
        pytest.skip("Raw validation.csv not found -- skipping drift comparison tests")
    return pd.read_csv(path)


def normalize_for_comparison(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------
# 1. Basic structural sanity
# ---------------------------------------------------------------------

def test_files_exist():
    for split in SPLITS:
        assert (PROCESSED_DIR / f"{split}.parquet").exists()


def test_expected_columns(splits):
    expected = {"id", "text", "lang", "toxic"}
    for split, df in splits.items():
        assert set(df.columns) == expected, f"{split} has unexpected columns: {list(df.columns)}"


def test_no_nulls(splits):
    for split, df in splits.items():
        assert df.isnull().sum().sum() == 0, f"{split} has null values"


def test_no_empty_text(splits):
    for split, df in splits.items():
        assert (df["text"].str.strip() == "").sum() == 0


def test_toxic_label_binary(splits):
    for split, df in splits.items():
        unique = set(df["toxic"].unique())
        assert unique.issubset({0, 1}), f"{split}.toxic has non-binary values: {unique}"


def test_lang_values_restricted_to_known_set(splits):
    for split, df in splits.items():
        unique = set(df["lang"].unique())
        assert unique.issubset(set(LANGS)), f"{split} has unexpected lang values: {unique - set(LANGS)}"


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


def test_no_id_leakage_across_splits(splits):
    train_ids = set(splits["train"]["id"])
    val_ids = set(splits["val"]["id"])
    test_ids = set(splits["test"]["id"])
    assert len(train_ids & val_ids) == 0, "id leakage between train/val"
    assert len(train_ids & test_ids) == 0, "id leakage between train/test"
    assert len(val_ids & test_ids) == 0, "id leakage between val/test"


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
    assert total >= 7900, f"Only {total} rows survived cleaning, expected ~7999"


def test_column_dtypes(splits):
    """A silent dtype flip (e.g. toxic becoming float, id becoming
    object) wouldn't fail any other test here but would break
    downstream training code that assumes int labels/ids."""
    for split, df in splits.items():
        assert pd.api.types.is_integer_dtype(df["id"]), f"{split}.id is not integer dtype: {df['id'].dtype}"
        assert pd.api.types.is_integer_dtype(df["toxic"]), f"{split}.toxic is not integer dtype: {df['toxic'].dtype}"
        assert pd.api.types.is_object_dtype(df["text"]), f"{split}.text is not object/string dtype: {df['text'].dtype}"
        assert pd.api.types.is_object_dtype(df["lang"]), f"{split}.lang is not object/string dtype: {df['lang'].dtype}"


def test_utf8_encoding_round_trips_cleanly(splits):
    """Catches lone surrogates or other broken-but-not-U+FFFD encoding
    damage that wouldn't show up in the replacement-character check --
    if a row can't round-trip through UTF-8 encode/decode, something
    upstream silently corrupted it."""
    for split, df in splits.items():
        for text in df["text"]:
            try:
                text.encode("utf-8").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                pytest.fail(f"{split} has text that fails UTF-8 round-trip: {e}")


# ---------------------------------------------------------------------
# 2. Joint (lang x toxic) stratification -- the real correctness bar
#    for this dataset's split strategy
# ---------------------------------------------------------------------

def test_per_language_row_counts_present_in_every_split(splits):
    """Every language must appear in every split with a non-trivial
    count -- catches a stratification bug that silently drops a
    minority stratum into only one split."""
    for split, df in splits.items():
        counts = df["lang"].value_counts()
        for lang in LANGS:
            assert lang in counts.index, f"{split} is missing language '{lang}' entirely"
            assert counts[lang] >= 20, f"{split} has only {counts.get(lang, 0)} rows for '{lang}'"


def test_val_test_per_language_size_sufficient_for_eval(splits):
    """Presence isn't enough for val/test -- per-language eval metrics
    (the whole point of a multilingual model) need a reasonable sample
    size per language, not just 'at least 20 rows'."""
    for split in ("val", "test"):
        counts = splits[split]["lang"].value_counts()
        for lang in LANGS:
            assert counts.get(lang, 0) >= 100, (
                f"{split} has only {counts.get(lang, 0)} rows for '{lang}', "
                f"want >=100 for a stable per-language eval metric"
            )


def test_per_language_toxic_rate_stable_across_splits(splits):
    """The whole point of stratifying on lang|toxic jointly instead of
    toxic alone: each language's toxic rate must stay stable across
    train/val/test, not just the overall rate."""
    for lang in LANGS:
        rates = {}
        for split, df in splits.items():
            lang_df = df[df["lang"] == lang]
            rates[split] = lang_df["toxic"].mean()
        max_diff = max(rates.values()) - min(rates.values())
        assert max_diff < 0.03, (
            f"Language '{lang}' toxic rate differs too much across splits "
            f"(max diff={max_diff:.4f}): {rates}"
        )


def test_per_language_proportions_stable_across_splits(splits):
    """Each language's SHARE of each split should also track its share
    of the full dataset (tr ~37.5%, es/it ~31.25% each) -- catches a
    split that's balanced on toxic rate but skewed on language mix."""
    total_lang_share = pd.concat(splits.values())["lang"].value_counts(normalize=True)
    for split, df in splits.items():
        split_lang_share = df["lang"].value_counts(normalize=True)
        for lang in LANGS:
            diff = abs(split_lang_share.get(lang, 0) - total_lang_share[lang])
            assert diff < 0.03, (
                f"{split} language share for '{lang}' ({split_lang_share.get(lang, 0):.3f}) "
                f"drifts from overall share ({total_lang_share[lang]:.3f})"
            )


def test_no_lang_conflicts_for_identical_text(full_dataset):
    """If the exact same string appears more than once (shouldn't,
    given dedup) it must not be tagged with two different languages --
    would indicate a labeling error upstream."""
    conflicts = (
        full_dataset.groupby("text")["lang"]
        .nunique()
        .loc[lambda s: s > 1]
    )
    assert len(conflicts) == 0, f"{len(conflicts)} texts have conflicting lang labels"


def test_no_label_conflicts_for_identical_text(full_dataset):
    conflicts = (
        full_dataset.groupby("text")["toxic"]
        .nunique()
        .loc[lambda s: s > 1]
    )
    assert len(conflicts) == 0, f"{len(conflicts)} texts have conflicting toxic labels"


# ---------------------------------------------------------------------
# 3. Distribution drift vs raw
# ---------------------------------------------------------------------

def test_toxic_rate_not_drifted_from_raw(full_dataset, raw_dataset):
    raw_rate = raw_dataset["toxic"].mean()
    cleaned_rate = full_dataset["toxic"].mean()
    diff = abs(raw_rate - cleaned_rate)
    assert diff < 0.01, (
        f"Toxic rate drifted from raw ({raw_rate:.4f}) to cleaned "
        f"({cleaned_rate:.4f}), diff={diff:.4f}"
    )


def test_row_count_not_drastically_reduced_from_raw(full_dataset, raw_dataset):
    """Cleaning should drop a handful of rows (dedup/empty-after-clean),
    not a meaningful fraction -- a large drop usually means a cleaning
    bug is nuking valid rows."""
    raw_count = len(raw_dataset)
    cleaned_count = len(full_dataset)
    dropped_frac = (raw_count - cleaned_count) / raw_count
    assert dropped_frac < 0.02, (
        f"Cleaning dropped {dropped_frac:.2%} of rows ({raw_count} -> {cleaned_count}), "
        f"expected under 2%"
    )


def test_per_language_row_counts_not_drifted_from_raw(full_dataset, raw_dataset):
    raw_counts = raw_dataset["lang"].value_counts()
    cleaned_counts = full_dataset["lang"].value_counts()
    for lang in LANGS:
        diff = abs(raw_counts.get(lang, 0) - cleaned_counts.get(lang, 0))
        assert diff <= 5, (
            f"Language '{lang}' row count drifted from raw ({raw_counts.get(lang, 0)}) "
            f"to cleaned ({cleaned_counts.get(lang, 0)}) by {diff} rows"
        )


# ---------------------------------------------------------------------
# 4. Near-duplicate leakage across splits (per-language, since
#    cross-language string collisions are meaningless here)
# ---------------------------------------------------------------------

def test_no_near_duplicate_leakage_across_splits(splits):
    for lang in LANGS:
        normalized = {
            split: set(
                df[df["lang"] == lang]["text"].apply(normalize_for_comparison)
            )
            for split, df in splits.items()
        }
        train_val = normalized["train"] & normalized["val"]
        train_test = normalized["train"] & normalized["test"]
        val_test = normalized["val"] & normalized["test"]

        max_allowed = 10
        assert len(train_val) <= max_allowed, (
            f"[{lang}] {len(train_val)} near-duplicate texts leak between train/val"
        )
        assert len(train_test) <= max_allowed, (
            f"[{lang}] {len(train_test)} near-duplicate texts leak between train/test"
        )
        assert len(val_test) <= max_allowed, (
            f"[{lang}] {len(val_test)} near-duplicate texts leak between val/test"
        )


# ---------------------------------------------------------------------
# 5. Cleaning-artifact validation -- confirm the specific things
#    clean_split.py claims to strip are actually gone
# ---------------------------------------------------------------------

def test_no_html_tags_remaining(splits):
    tag_pattern = re.compile(r"<[^>]+>")
    for split, df in splits.items():
        # <URL> is our own placeholder token, not leaked markup -- exclude it
        without_placeholder = df["text"].str.replace("<URL>", "", regex=False)
        bad = without_placeholder.str.contains(tag_pattern).sum()
        assert bad == 0, f"{split} has {bad} rows with un-stripped HTML tags"


def test_no_html_entities_remaining(splits):
    entity_pattern = re.compile(r"&\w+;")
    for split, df in splits.items():
        bad = df["text"].str.contains(entity_pattern).sum()
        assert bad == 0, f"{split} has {bad} rows with un-stripped HTML entities"


def test_no_raw_urls_remaining(splits):
    """Raw URLs should have been replaced with the <URL> placeholder --
    none should survive in http(s):// or www. form."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    for split, df in splits.items():
        bad = df["text"].str.contains(url_pattern).sum()
        assert bad == 0, f"{split} has {bad} rows with un-masked raw URLs"


def test_url_placeholder_count_matches_raw_url_count(full_dataset, raw_dataset):
    """Proves the URL masking actually ran, rather than just proving
    raw URLs are absent (which would also trivially pass if the regex
    silently stopped matching anything). Inspection found 99 raw rows
    containing URLs -- the <URL> placeholder count should land in that
    neighborhood, not zero and not wildly higher."""
    raw_url_rows = raw_dataset["comment_text"].astype(str).str.contains(
        r"https?://\S+|www\.\S+", regex=True
    ).sum()
    placeholder_count = full_dataset["text"].str.contains("<URL>", regex=False).sum()
    assert placeholder_count > 0, "No <URL> placeholders found -- URL masking may have silently broken"
    diff = abs(placeholder_count - raw_url_rows)
    assert diff <= 5, (
        f"<URL> placeholder count ({placeholder_count}) differs from raw URL-containing "
        f"rows ({raw_url_rows}) by more than expected (diff={diff})"
    )


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
    """Whitespace collapsing should leave no doubled spaces -- a common
    leftover artifact of regex substitution (e.g. stripping a tag
    leaves two adjacent spaces where the tag used to be)."""
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
        too_short = (lengths < 3).sum()
        too_long = (lengths > 2000).sum()
        assert too_short == 0, f"{split} has {too_short} texts under 3 characters"
        assert too_long == 0, (
            f"{split} has {too_long} texts over 2000 characters -- "
            f"raw max was 1785, investigate anything beyond that"
        )


def test_no_text_exceeds_reasonable_token_proxy_length(splits):
    """Word-count proxy for tokenizer length -- real tokenizer-based
    validation belongs in the training pipeline where the actual
    XLM-RoBERTa tokenizer is available."""
    for split, df in splits.items():
        word_counts = df["text"].str.split().apply(len)
        too_long = (word_counts > 400).sum()
        assert too_long == 0, (
            f"{split} has {too_long} texts over 400 words -- verify these "
            f"aren't a cleaning/concatenation artifact before tokenizing"
        )


# ---------------------------------------------------------------------
# 7. Statistical bounds -- lightweight equivalent of Great
#    Expectations / Pandera rules, calibrated per language
# ---------------------------------------------------------------------

def test_statistical_bounds_overall(full_dataset):
    toxic_rate = full_dataset["toxic"].mean()
    mean_len = full_dataset["text"].str.len().mean()
    row_count = len(full_dataset)

    assert 0.10 <= toxic_rate <= 0.20, f"Overall toxic rate {toxic_rate:.3f} outside expected [0.10, 0.20]"
    assert 250 <= mean_len <= 500, f"Mean text length {mean_len:.1f} outside expected [250, 500]"
    assert row_count >= 7900, f"Row count {row_count} below expected minimum 7900"
    assert set(full_dataset["toxic"].unique()).issubset({0, 1}), "toxic labels outside {0, 1} found"


def test_statistical_bounds_per_language(full_dataset):
    """Per-language toxic rate bounds, tight enough to catch a real
    regression but wide enough to tolerate normal sampling noise --
    calibrated from the observed rates (es 0.169, it 0.195, tr 0.107)."""
    expected_ranges = {
        "es": (0.12, 0.22),
        "it": (0.15, 0.25),
        "tr": (0.06, 0.16),
    }
    for lang, (low, high) in expected_ranges.items():
        rate = full_dataset[full_dataset["lang"] == lang]["toxic"].mean()
        assert low <= rate <= high, (
            f"Toxic rate for '{lang}' ({rate:.3f}) outside expected [{low}, {high}]"
        )


# ---------------------------------------------------------------------
# 8. Data fingerprinting -- detect unexpected changes between pipeline
#    runs
# ---------------------------------------------------------------------

def _compute_fingerprint(df: pd.DataFrame) -> str:
    sortable = df.sort_values(["text", "lang", "toxic"]).reset_index(drop=True)
    payload = sortable.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_dataset_fingerprint_stability(full_dataset):
    """First run: writes the fingerprint to disk (informational).
    Subsequent runs: fails loudly if the cleaned dataset changed
    without anyone updating/deleting the stored fingerprint."""
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