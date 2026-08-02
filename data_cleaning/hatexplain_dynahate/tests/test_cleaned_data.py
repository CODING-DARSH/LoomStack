"""
Production-grade test suite for the merged HateXplain + Dynahate
cleaned dataset. Single combined file (basic + advanced), per project
convention.

Thresholds below are calibrated against the actual clean_split.py run:
  - 101628 rows pre-clean -> 60322 post-dedup (41306 duplicates
    dropped, mostly intra-Dynahate)
  - label distribution: hatespeech 28094, normal 26757, offensive 5471
    (offensive ONLY comes from hatexplain -- structurally smaller
    class, not a bug)
  - source distribution: dynahate 41130 (~68%), hatexplain 19192 (~32%)
  - split sizes: train 48248 / val 6037 / test 6037 (~80/10/10,
    inherited from each source's native split)

Usage:
    pytest data_cleaning/hatexplain_dynahate/tests/test_cleaned_data.py -v
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed/hatexplain_dynahate")
HATEXPLAIN_RAW_DIR = Path("data/raw/hatexplain")
DYNAHATE_RAW_DIR = Path("data/raw/dynahate")
FINGERPRINT_PATH = PROCESSED_DIR / ".fingerprint.json"
SPLITS = ["train", "val", "test"]
SOURCES = ["hatexplain", "dynahate"]
LABELS = ["hatespeech", "normal", "offensive"]


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
def raw_hatexplain_row_count():
    if not HATEXPLAIN_RAW_DIR.exists():
        pytest.skip("Raw hatexplain dir not found -- skipping raw comparison tests")
    total = 0
    for f in HATEXPLAIN_RAW_DIR.glob("*.parquet"):
        total += len(pd.read_parquet(f))
    return total


@pytest.fixture(scope="module")
def raw_dynahate_row_count():
    path = DYNAHATE_RAW_DIR / "train.parquet"
    if not path.exists():
        pytest.skip("Raw dynahate/train.parquet not found -- skipping raw comparison tests")
    return len(pd.read_parquet(path))


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
    expected = {"id", "text", "label", "source"}
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


def test_source_values_restricted_to_known_set(splits):
    for split, df in splits.items():
        unique = set(df["source"].unique())
        assert unique.issubset(set(SOURCES)), f"{split} has unexpected source values: {unique - set(SOURCES)}"


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


def test_no_id_duplication_within_dataset(full_dataset):
    """ids come from two different source formats (HateXplain's
    twitter/gab-style string ids vs Dynahate's 'acl123' ids) -- they
    should never collide, but confirm rather than assume the two
    id schemes can't accidentally overlap."""
    dupes = full_dataset["id"].duplicated().sum()
    assert dupes == 0, f"{dupes} duplicate ids found across the merged dataset"


def test_split_proportions_roughly_correct(splits):
    total = sum(len(df) for df in splits.values())
    train_frac = len(splits["train"]) / total
    val_frac = len(splits["val"]) / total
    test_frac = len(splits["test"]) / total
    assert 0.75 <= train_frac <= 0.85, f"train fraction {train_frac:.3f} out of range"
    assert 0.05 <= val_frac <= 0.15, f"val fraction {val_frac:.3f} out of range"
    assert 0.05 <= test_frac <= 0.15, f"test fraction {test_frac:.3f} out of range"


def test_minimum_dataset_size(splits):
    total = sum(len(df) for df in splits.values())
    assert total >= 59000, f"Only {total} rows survived cleaning, expected ~60322"


def test_column_dtypes(splits):
    for split, df in splits.items():
        assert pd.api.types.is_object_dtype(df["id"]), f"{split}.id is not object/string dtype: {df['id'].dtype}"
        assert pd.api.types.is_object_dtype(df["text"]), f"{split}.text is not object/string dtype: {df['text'].dtype}"
        assert pd.api.types.is_object_dtype(df["label"]), f"{split}.label is not object/string dtype: {df['label'].dtype}"
        assert pd.api.types.is_object_dtype(df["source"]), f"{split}.source is not object/string dtype: {df['source'].dtype}"


def test_utf8_encoding_round_trips_cleanly(splits):
    for split, df in splits.items():
        for text in df["text"]:
            try:
                text.encode("utf-8").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                pytest.fail(f"{split} has text that fails UTF-8 round-trip: {e}")


# ---------------------------------------------------------------------
# 2. Merge-specific correctness -- the actual hard part of this dataset
# ---------------------------------------------------------------------

def test_offensive_label_only_comes_from_hatexplain(full_dataset):
    """Dynahate is binary (hate/nothate) and can never produce an
    'offensive' row -- if it does, the label mapping in clean_split.py
    broke."""
    offensive_sources = full_dataset[full_dataset["label"] == "offensive"]["source"].unique()
    assert set(offensive_sources) == {"hatexplain"}, (
        f"'offensive' label found in unexpected sources: {set(offensive_sources) - {'hatexplain'}}"
    )


def test_both_sources_present_in_every_split(splits):
    for split, df in splits.items():
        sources_present = set(df["source"].unique())
        assert sources_present == set(SOURCES), (
            f"{split} is missing a source entirely: expected {SOURCES}, got {sources_present}"
        )


def test_source_proportions_roughly_correct(full_dataset):
    """Dynahate should dominate (~68%) given its raw size advantage --
    a wildly different ratio would indicate the dedup step
    disproportionately wiped out one source."""
    source_frac = full_dataset["source"].value_counts(normalize=True)
    assert 0.60 <= source_frac["dynahate"] <= 0.75, (
        f"dynahate share {source_frac['dynahate']:.3f} outside expected [0.60, 0.75]"
    )
    assert 0.25 <= source_frac["hatexplain"] <= 0.40, (
        f"hatexplain share {source_frac['hatexplain']:.3f} outside expected [0.25, 0.40]"
    )


def test_source_proportions_stable_across_splits(splits):
    """Each split's source mix should track the overall mix -- since
    splits were inherited natively rather than freshly stratified,
    this isn't guaranteed by construction and is worth checking
    directly rather than assuming."""
    overall = pd.concat(splits.values())["source"].value_counts(normalize=True)
    for split, df in splits.items():
        split_frac = df["source"].value_counts(normalize=True)
        for source in SOURCES:
            diff = abs(split_frac.get(source, 0) - overall[source])
            assert diff < 0.05, (
                f"{split} source share for '{source}' ({split_frac.get(source, 0):.3f}) "
                f"drifts from overall share ({overall[source]:.3f})"
            )


def test_no_three_way_tie_ids_survive(full_dataset):
    """Cross-check against raw HateXplain: every hatexplain-source id
    in the cleaned data must have had a real annotator majority --
    confirms the tie-break-by-dropping policy in clean_split.py
    actually ran rather than silently no-op'ing."""
    if not HATEXPLAIN_RAW_DIR.exists():
        pytest.skip("Raw hatexplain dir not found -- skipping tie-drop verification")

    raw_frames = [pd.read_parquet(f) for f in HATEXPLAIN_RAW_DIR.glob("*.parquet")]
    raw = pd.concat(raw_frames, ignore_index=True)

    def is_three_way_tie(annotators):
        labels = [int(v) for v in annotators["label"]]
        return len(labels) == 3 and len(set(labels)) == 3

    tie_ids = set(raw[raw["annotators"].apply(is_three_way_tie)]["id"])
    cleaned_hx_ids = set(full_dataset[full_dataset["source"] == "hatexplain"]["id"])
    leaked_ties = tie_ids & cleaned_hx_ids
    assert len(leaked_ties) == 0, f"{len(leaked_ties)} 3-way-tie ids survived into cleaned data: {list(leaked_ties)[:5]}"


# ---------------------------------------------------------------------
# 3. Per-source, per-label stratification sanity
# ---------------------------------------------------------------------

def test_label_distribution_present_in_every_split(splits):
    for split, df in splits.items():
        labels_present = set(df["label"].unique())
        assert labels_present == set(LABELS), (
            f"{split} is missing a label entirely: expected {LABELS}, got {labels_present}"
        )


def test_label_rate_reasonably_stable_across_splits(splits):
    """Wider tolerance than a freshly-stratified dataset would need,
    since these splits were inherited natively rather than jointly
    re-stratified on label -- but a large drift would still indicate
    something is off."""
    for label in LABELS:
        rates = {split: (df["label"] == label).mean() for split, df in splits.items()}
        max_diff = max(rates.values()) - min(rates.values())
        assert max_diff < 0.05, (
            f"Label '{label}' rate differs too much across splits (max diff={max_diff:.4f}): {rates}"
        )


# ---------------------------------------------------------------------
# 4. Duplicate-drop plausibility vs raw
# ---------------------------------------------------------------------

def test_dedup_drop_count_plausible(full_dataset, raw_hatexplain_row_count, raw_dynahate_row_count):
    """Confirms the 41306-duplicate drop we saw is in the right
    ballpark, not evidence of an over-aggressive or broken dedup step
    silently eating a much larger fraction of the data."""
    raw_total = raw_hatexplain_row_count + raw_dynahate_row_count
    cleaned_total = len(full_dataset)
    dropped_frac = (raw_total - cleaned_total) / raw_total
    assert 0.30 <= dropped_frac <= 0.50, (
        f"Dedup dropped {dropped_frac:.2%} of merged rows ({raw_total} -> {cleaned_total}), "
        f"expected 30-50% given known heavy intra-Dynahate duplication"
    )


def test_cross_source_duplicates_resolved_in_favor_of_hatexplain(full_dataset):
    """clean_split.py's dedup order is hatexplain-then-dynahate, so any
    text appearing in BOTH raw sources should survive in cleaned data
    tagged source=hatexplain, never source=dynahate. This directly
    tests that the keep='first' ordering actually took effect, rather
    than just checking the total drop count is plausible (which would
    pass even if 100% of drops were intra-Dynahate and the
    keep-hatexplain-on-collision logic never fired)."""
    if not HATEXPLAIN_RAW_DIR.exists() or not (DYNAHATE_RAW_DIR / "train.parquet").exists():
        pytest.skip("Raw source dirs not found -- skipping cross-source dedup verification")

    def clean_text_for_compare(text: str) -> str:
        text = str(text)
        text = text.replace("\ufffd", "")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    hx_raw_frames = [pd.read_parquet(f) for f in HATEXPLAIN_RAW_DIR.glob("*.parquet")]
    hx_raw = pd.concat(hx_raw_frames, ignore_index=True)
    hx_texts = set(hx_raw["post_tokens"].apply(lambda t: clean_text_for_compare(" ".join(t))))

    dh_raw = pd.read_parquet(DYNAHATE_RAW_DIR / "train.parquet")
    dh_texts = set(dh_raw["text"].apply(clean_text_for_compare))

    cross_overlap = hx_texts & dh_texts
    if len(cross_overlap) == 0:
        pytest.skip("No cross-source text overlap in raw data -- nothing for this test to verify")

    overlap_rows = full_dataset[full_dataset["text"].isin(cross_overlap)]
    wrongly_kept_as_dynahate = overlap_rows[overlap_rows["source"] == "dynahate"]
    assert len(wrongly_kept_as_dynahate) == 0, (
        f"{len(wrongly_kept_as_dynahate)} cross-source duplicate texts survived tagged as "
        f"'dynahate' instead of 'hatexplain' -- dedup keep-order may be broken"
    )


def test_id_format_matches_claimed_source(full_dataset):
    """Dynahate ids follow the acl.id scheme (e.g. 'acl123'), HateXplain
    ids are twitter/gab post ids (e.g. '1178610029273976833_twitter').
    A row whose id pattern doesn't match its claimed source would
    indicate a silent column-swap bug in the merge."""
    dynahate_id_pattern = re.compile(r"^acl\d+")
    dynahate_rows = full_dataset[full_dataset["source"] == "dynahate"]
    mismatched_dynahate = dynahate_rows[~dynahate_rows["id"].astype(str).str.match(dynahate_id_pattern)]
    assert len(mismatched_dynahate) == 0, (
        f"{len(mismatched_dynahate)} rows tagged source=dynahate have an id not matching the "
        f"'acl<N>' pattern: {mismatched_dynahate['id'].head(5).tolist()}"
    )

    hatexplain_rows = full_dataset[full_dataset["source"] == "hatexplain"]
    mismatched_hatexplain = hatexplain_rows[hatexplain_rows["id"].astype(str).str.match(dynahate_id_pattern)]
    assert len(mismatched_hatexplain) == 0, (
        f"{len(mismatched_hatexplain)} rows tagged source=hatexplain have an id matching the "
        f"dynahate 'acl<N>' pattern instead: {mismatched_hatexplain['id'].head(5).tolist()}"
    )


# ---------------------------------------------------------------------
# 5. Near-duplicate leakage across splits (per source, since
#    cross-source string collisions were already handled by dedup)
# ---------------------------------------------------------------------

def test_no_near_duplicate_leakage_across_splits(splits):
    for source in SOURCES:
        normalized = {
            split: set(
                df[df["source"] == source]["text"].apply(normalize_for_comparison)
            )
            for split, df in splits.items()
        }
        train_val = normalized["train"] & normalized["val"]
        train_test = normalized["train"] & normalized["test"]
        val_test = normalized["val"] & normalized["test"]

        max_allowed = 25
        assert len(train_val) <= max_allowed, (
            f"[{source}] {len(train_val)} near-duplicate texts leak between train/val"
        )
        assert len(train_test) <= max_allowed, (
            f"[{source}] {len(train_test)} near-duplicate texts leak between train/test"
        )
        assert len(val_test) <= max_allowed, (
            f"[{source}] {len(val_test)} near-duplicate texts leak between val/test"
        )


# ---------------------------------------------------------------------
# 6. Cleaning-artifact validation
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


def test_all_text_is_valid_unicode(splits):
    for split, df in splits.items():
        for text in df["text"]:
            try:
                unicodedata.normalize("NFC", text)
            except (ValueError, TypeError) as e:
                pytest.fail(f"{split} has text that fails Unicode normalization: {e}")


# ---------------------------------------------------------------------
# 7. Outlier / length bounds
# ---------------------------------------------------------------------

def test_no_extreme_length_outliers(splits):
    for split, df in splits.items():
        lengths = df["text"].str.len()
        too_short = (lengths < 3).sum()
        too_long = (lengths > 2500).sum()
        assert too_short == 0, f"{split} has {too_short} texts under 3 characters"
        assert too_long == 0, (
            f"{split} has {too_long} texts over 2500 characters -- "
            f"raw dynahate max was 2374, investigate anything beyond that"
        )


def test_no_text_exceeds_reasonable_token_proxy_length(splits):
    for split, df in splits.items():
        word_counts = df["text"].str.split().apply(len)
        too_long = (word_counts > 500).sum()
        assert too_long == 0, (
            f"{split} has {too_long} texts over 500 words -- verify these "
            f"aren't a cleaning/concatenation artifact before tokenizing"
        )


def test_no_list_repr_artifacts_in_hatexplain_text(splits):
    """HateXplain text was reconstructed via ' '.join(post_tokens) --
    if that join ever silently fell back to str(list) instead (e.g.
    post_tokens wasn't the plain array expected), the text would
    contain Python list/array repr artifacts like "['", "']", "array(".
    Also guards against the 'target' field's literal 'None' strings
    leaking into text, which would happen if a future edit accidentally
    joined the wrong column."""
    artifact_pattern = re.compile(r"\['|'\]|array\(|dtype=")
    for split, df in splits.items():
        hx_rows = df[df["source"] == "hatexplain"]
        bad = hx_rows["text"].str.contains(artifact_pattern, regex=True).sum()
        assert bad == 0, f"{split} has {bad} hatexplain rows with list/array repr artifacts in text"


def test_user_placeholder_well_formed(splits):
    """HateXplain's <user> anonymization token should survive token
    joining as a clean, single-spaced token -- not '< user>', '<user >',
    or glued directly onto adjacent text without a space."""
    malformed_pattern = re.compile(r"<\s+user\s*>|<\s*user\s+>|\S<user>|<user>\S")
    for split, df in splits.items():
        hx_rows = df[df["source"] == "hatexplain"]
        bad = hx_rows["text"].str.contains(malformed_pattern, regex=True).sum()
        assert bad == 0, f"{split} has {bad} hatexplain rows with a malformed <user> placeholder"


# ---------------------------------------------------------------------
# 8. Statistical bounds
# ---------------------------------------------------------------------

def test_statistical_bounds_overall(full_dataset):
    row_count = len(full_dataset)
    assert row_count >= 59000, f"Row count {row_count} below expected minimum 59000"
    assert set(full_dataset["label"].unique()) == set(LABELS), "Unexpected label set found"
    assert set(full_dataset["source"].unique()) == set(SOURCES), "Unexpected source set found"


def test_statistical_bounds_per_label(full_dataset):
    """Calibrated from observed distribution: hatespeech 28094 (~47%),
    normal 26757 (~44%), offensive 5471 (~9%)."""
    label_rates = full_dataset["label"].value_counts(normalize=True)
    expected_ranges = {
        "hatespeech": (0.40, 0.55),
        "normal": (0.35, 0.50),
        "offensive": (0.05, 0.15),
    }
    for label, (low, high) in expected_ranges.items():
        rate = label_rates.get(label, 0)
        assert low <= rate <= high, f"Label '{label}' rate {rate:.3f} outside expected [{low}, {high}]"


# ---------------------------------------------------------------------
# 9. Data fingerprinting
# ---------------------------------------------------------------------

def _compute_fingerprint(df: pd.DataFrame) -> str:
    sortable = df.sort_values(["text", "label", "source"]).reset_index(drop=True)
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
        f"If this is an intentional change (e.g. updated cleaning/merge logic), "
        f"delete {FINGERPRINT_PATH} and re-run to accept the new fingerprint."
    )