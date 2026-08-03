"""
Production-grade test suite for cleaned Conversations Gone Awry
(Wikipedia corpus only) data. Combined basic + advanced, per project
convention.

Thresholds calibrated against the actual clean_split.py run:
  - 4188 conversations survived cleaning (0 dropped)
  - Perfect 50/50 label balance at every level: overall, train
    (1254/1254), val (420/420), test (420/420) -- this is a real,
    load-bearing property of the matched-pairs design, not
    approximate sampling noise, so bounds here are tight.
  - Splits: train 2508 / val 840 / test 840 (~60/20/20)
  - Pair integrity: 0 orphaned pairs, 0 cross-split pairs, 0 same-label
    pairs, verified both pre- and post-cleaning.

Usage:
    pytest data_cleaning/conversations_gone_awry/tests/test_cleaned_data.py -v
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed/conversations_gone_awry")
RAW_DIR = Path("data/raw/conversations_gone_awry")
FINGERPRINT_PATH = PROCESSED_DIR / ".fingerprint.json"
SPLITS = ["train", "val", "test"]
TURN_SEP = " <TURN> "


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
    frames = []
    for split_name, df in splits.items():
        tagged = df.copy()
        tagged["_split"] = split_name
        frames.append(tagged)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# 1. Basic structural sanity
# ---------------------------------------------------------------------

def test_files_exist():
    for split in SPLITS:
        assert (PROCESSED_DIR / f"{split}.parquet").exists()


def test_expected_columns(splits):
    expected = {"conversation_id", "pair_id", "text", "label", "num_utterances"}
    for split, df in splits.items():
        assert set(df.columns) == expected, f"{split} has unexpected columns: {list(df.columns)}"


def test_no_nulls(splits):
    for split, df in splits.items():
        assert df.isnull().sum().sum() == 0, f"{split} has null values"


def test_no_empty_text(splits):
    for split, df in splits.items():
        assert (df["text"].str.strip() == "").sum() == 0


def test_label_is_boolean(splits):
    for split, df in splits.items():
        assert df["label"].dtype == bool, f"{split}.label is not bool dtype: {df['label'].dtype}"


def test_no_duplicate_conversation_id_within_split(splits):
    for split, df in splits.items():
        dupes = df["conversation_id"].duplicated().sum()
        assert dupes == 0, f"{split} has {dupes} duplicate conversation_id rows"


def test_no_conversation_id_leakage_across_splits(splits):
    train_ids = set(splits["train"]["conversation_id"])
    val_ids = set(splits["val"]["conversation_id"])
    test_ids = set(splits["test"]["conversation_id"])
    assert len(train_ids & val_ids) == 0, "conversation_id leakage between train/val"
    assert len(train_ids & test_ids) == 0, "conversation_id leakage between train/test"
    assert len(val_ids & test_ids) == 0, "conversation_id leakage between val/test"


def test_split_proportions_roughly_correct(splits):
    total = sum(len(df) for df in splits.values())
    train_frac = len(splits["train"]) / total
    val_frac = len(splits["val"]) / total
    test_frac = len(splits["test"]) / total
    assert 0.55 <= train_frac <= 0.65, f"train fraction {train_frac:.3f} out of range"
    assert 0.15 <= val_frac <= 0.25, f"val fraction {val_frac:.3f} out of range"
    assert 0.15 <= test_frac <= 0.25, f"test fraction {test_frac:.3f} out of range"


def test_total_row_count_matches_raw_conversation_count(full_dataset):
    """0 conversations were dropped during cleaning -- the cleaned
    total should exactly match the raw conversation count, not just
    be 'close'."""
    assert len(full_dataset) == 4188, f"Expected exactly 4188 rows, got {len(full_dataset)}"


def test_column_dtypes(splits):
    for split, df in splits.items():
        assert pd.api.types.is_object_dtype(df["conversation_id"]), f"{split}.conversation_id wrong dtype"
        assert pd.api.types.is_object_dtype(df["pair_id"]), f"{split}.pair_id wrong dtype"
        assert pd.api.types.is_object_dtype(df["text"]), f"{split}.text wrong dtype"
        assert df["label"].dtype == bool, f"{split}.label wrong dtype"
        assert pd.api.types.is_integer_dtype(df["num_utterances"]), f"{split}.num_utterances wrong dtype"


def test_utf8_encoding_round_trips_cleanly(splits):
    for split, df in splits.items():
        for text in df["text"]:
            try:
                text.encode("utf-8").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                pytest.fail(f"{split} has text that fails UTF-8 round-trip: {e}")


# ---------------------------------------------------------------------
# 2. Label balance -- a load-bearing property, not approximate
# ---------------------------------------------------------------------

def test_exact_fifty_fifty_label_balance_per_split(splits):
    """Unlike every other dataset in this repo, this one has a
    GUARANTEED exact 50/50 balance by construction (matched pairs,
    one derailed + one non-derailed each). This should be exact, not
    within-a-tolerance -- any deviation means pair integrity broke
    during cleaning."""
    for split, df in splits.items():
        true_count = (df["label"] == True).sum()
        false_count = (df["label"] == False).sum()
        assert true_count == false_count, (
            f"{split} label balance is not exact: {true_count} True vs {false_count} False"
        )


def test_overall_label_count_matches_known_value(full_dataset):
    assert (full_dataset["label"] == True).sum() == 2094
    assert (full_dataset["label"] == False).sum() == 2094


# ---------------------------------------------------------------------
# 3. Pair integrity -- permanent regression tests, not one-time checks
# ---------------------------------------------------------------------

def test_every_pair_id_resolves_to_an_existing_conversation(full_dataset):
    all_ids = set(full_dataset["conversation_id"])
    missing = full_dataset[~full_dataset["pair_id"].isin(all_ids)]
    assert len(missing) == 0, (
        f"{len(missing)} rows have a pair_id that doesn't match any conversation_id in the dataset: "
        f"{missing['conversation_id'].head(5).tolist()}"
    )


def test_pair_id_pointers_are_reciprocal(full_dataset):
    """If A's pair_id points to B, B's pair_id must point back to A."""
    id_to_pair = dict(zip(full_dataset["conversation_id"], full_dataset["pair_id"]))
    non_reciprocal = [
        (cid, pid) for cid, pid in id_to_pair.items()
        if id_to_pair.get(pid) != cid
    ]
    assert len(non_reciprocal) == 0, f"{len(non_reciprocal)} non-reciprocal pair_id pointers: {non_reciprocal[:5]}"


def test_every_pair_has_complementary_labels(full_dataset):
    id_to_label = dict(zip(full_dataset["conversation_id"], full_dataset["label"]))
    id_to_pair = dict(zip(full_dataset["conversation_id"], full_dataset["pair_id"]))
    same_label_pairs = [
        cid for cid, pid in id_to_pair.items()
        if id_to_label.get(pid) == id_to_label[cid]
    ]
    assert len(same_label_pairs) == 0, (
        f"{len(same_label_pairs)} conversations share the same label as their pair partner "
        f"(should always be complementary): {same_label_pairs[:5]}"
    )


def test_every_pair_stays_within_the_same_split(full_dataset):
    """The single most important leakage guard for this dataset --
    if a pair is ever split across train/val/test, the model
    implicitly sees a near-duplicate-topic conversation during
    training that shares structure with an eval example."""
    id_to_split = dict(zip(full_dataset["conversation_id"], full_dataset["_split"]))
    id_to_pair = dict(zip(full_dataset["conversation_id"], full_dataset["pair_id"]))
    cross_split_pairs = [
        (cid, pid) for cid, pid in id_to_pair.items()
        if id_to_split.get(pid) != id_to_split[cid]
    ]
    assert len(cross_split_pairs) == 0, (
        f"{len(cross_split_pairs)} pairs are split across different splits -- LEAKAGE: {cross_split_pairs[:5]}"
    )


def test_pair_count_matches_known_value(full_dataset):
    """4188 conversations / 2 = 2094 pairs, verified via
    check_pair_structure.py before cleaning -- confirm cleaning
    preserved that exactly."""
    pairs_seen = set()
    for cid, pid in zip(full_dataset["conversation_id"], full_dataset["pair_id"]):
        pairs_seen.add(tuple(sorted([cid, pid])))
    assert len(pairs_seen) == 2094, f"Expected exactly 2094 resolved pairs, found {len(pairs_seen)}"


# ---------------------------------------------------------------------
# 4. Cleaning-artifact validation
# ---------------------------------------------------------------------

def test_no_raw_wiki_link_markup_remaining(splits):
    """[WIKI_LINK: ...] should have been fully normalized to
    <WIKILINK> -- none of the raw form should survive."""
    raw_pattern = re.compile(r"\[WIKI_LINK:")
    for split, df in splits.items():
        bad = df["text"].str.contains(raw_pattern, regex=True).sum()
        assert bad == 0, f"{split} has {bad} rows with un-normalized [WIKI_LINK: ...] markup"


def test_wikilink_placeholder_never_appears_by_design(full_dataset):
    """Confirmed via diagnose_test_failures.py: all 108 raw [WIKI_LINK:
    occurrences in the corpus are INSIDE section-header utterances (0
    in real reply bodies). Since section headers are always dropped
    from context, <WIKILINK> is expected to appear ZERO times in
    cleaned output -- this is correct behavior, not a broken
    normalization step. This test locks in that confirmed invariant;
    if it ever starts failing (count > 0), that would mean a future
    corpus update or upstream ConvoKit change put [WIKI_LINK: content
    into real utterance bodies, which would be worth knowing about."""
    convos_with_wikilink = full_dataset["text"].str.contains("<WIKILINK>", regex=False).sum()
    assert convos_with_wikilink == 0, (
        f"{convos_with_wikilink} conversations contain <WIKILINK>, expected exactly 0 -- "
        f"investigate whether [WIKI_LINK: content is now appearing outside section headers"
    )


def test_no_section_header_remnants(splits):
    """is_section_header utterances AND bare inline headers ConvoKit's
    flag misses (e.g. "==Source misrepresentation==" as a full,
    standalone reply -- see diagnose_test_failures.py) should both be
    dropped by clean_split.py. Checks per-TURN, not substring-anywhere
    across the whole joined text -- a longer reply legitimately
    quoting or referencing "==something==" inline as part of a larger
    message is real content, not a header remnant, and shouldn't be
    flagged."""
    header_pattern = re.compile(r"^==[^=]+==$")
    for split, df in splits.items():
        bad_rows = []
        for _, row in df.iterrows():
            turns = row["text"].split(TURN_SEP)
            if any(header_pattern.match(t.strip()) for t in turns):
                bad_rows.append(row["conversation_id"])
        assert len(bad_rows) == 0, (
            f"{split} has {len(bad_rows)} conversations with a standalone header-only turn: {bad_rows[:5]}"
        )


def test_turn_separator_present_for_multi_utterance_conversations(splits):
    """Any conversation with num_utterances > 1 must actually contain
    the <TURN> separator -- catches a join bug where utterances got
    silently concatenated without it."""
    for split, df in splits.items():
        multi_utt = df[df["num_utterances"] > 1]
        missing_sep = multi_utt[~multi_utt["text"].str.contains(TURN_SEP, regex=False)]
        assert len(missing_sep) == 0, (
            f"{split} has {len(missing_sep)} multi-utterance conversations missing the <TURN> separator"
        )


def test_num_utterances_consistent_with_turn_separator_count(splits):
    """num_utterances should equal the number of <TURN>-separated
    segments in text -- a direct structural consistency check between
    the metadata column and the actual text content."""
    for split, df in splits.items():
        computed = df["text"].apply(lambda t: t.count(TURN_SEP) + 1)
        mismatch = (computed != df["num_utterances"]).sum()
        assert mismatch == 0, f"{split} has {mismatch} rows where num_utterances doesn't match TURN_SEP count"


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
# 5. Outlier / length bounds
# ---------------------------------------------------------------------

def test_num_utterances_within_expected_bounds(splits):
    """Raw conversation length was min 3 / max 20 utterances (whole
    conversation). Post-truncation context should never exceed that
    upper bound, and should always be at least 1 (0-length contexts
    were dropped)."""
    for split, df in splits.items():
        assert (df["num_utterances"] < 1).sum() == 0, f"{split} has conversations with 0 utterances"
        assert (df["num_utterances"] > 20).sum() == 0, (
            f"{split} has conversations with more than 20 utterances, exceeding raw corpus max"
        )


def test_no_extreme_text_length_outliers(splits):
    """Raw utterance length went up to 140256 chars (a clear outlier
    in the raw data) -- at the conversation-context level post-join,
    bound generously but catch anything absurd."""
    for split, df in splits.items():
        lengths = df["text"].str.len()
        too_long = (lengths > 200000).sum()
        assert too_long == 0, f"{split} has {too_long} conversation contexts over 200000 characters"


# ---------------------------------------------------------------------
# 6. Statistical bounds
# ---------------------------------------------------------------------

def test_statistical_bounds_overall(full_dataset):
    row_count = len(full_dataset)
    label_rate = full_dataset["label"].mean()
    assert row_count == 4188, f"Row count {row_count} != expected 4188"
    assert abs(label_rate - 0.5) < 0.001, f"Label rate {label_rate:.4f} should be exactly 0.5"


def test_statistical_bounds_num_utterances(full_dataset):
    mean_utts = full_dataset["num_utterances"].mean()
    assert 2.0 <= mean_utts <= 8.0, f"Mean num_utterances {mean_utts:.2f} outside expected [2.0, 8.0]"


# ---------------------------------------------------------------------
# 7. Data fingerprinting
# ---------------------------------------------------------------------

def _compute_fingerprint(df: pd.DataFrame) -> str:
    sortable = df.sort_values(["conversation_id"]).reset_index(drop=True)
    payload = sortable.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_dataset_fingerprint_stability(full_dataset):
    compare_cols = ["conversation_id", "pair_id", "text", "label", "num_utterances"]
    current_fp = _compute_fingerprint(full_dataset[compare_cols])

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
        f"If this is an intentional change (e.g. updated cleaning/truncation logic), "
        f"delete {FINGERPRINT_PATH} and re-run to accept the new fingerprint."
    )