"""
Tests for the CLEANED SMS Spam data. Run these after 02_clean_and_split.py
to verify the output is actually correct before moving on to model training.

Usage:
    pytest data_cleaning/sms_spam/tests/test_cleaned_data.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed/sms_spam")
SPLITS = ["train", "val", "test"]


@pytest.fixture(scope="module")
def splits():
    """Load all three splits once for all tests in this file."""
    data = {}
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.parquet"
        if not path.exists():
            pytest.fail(f"{path} does not exist -- run 02_clean_and_split.py first")
        data[split] = pd.read_parquet(path)
    return data


def test_files_exist():
    for split in SPLITS:
        assert (PROCESSED_DIR / f"{split}.parquet").exists(), f"{split}.parquet is missing"


def test_expected_columns(splits):
    for split, df in splits.items():
        assert set(df.columns) == {"text", "label"}, (
            f"{split} has unexpected columns: {list(df.columns)}"
        )


def test_no_nulls(splits):
    for split, df in splits.items():
        assert df["text"].isnull().sum() == 0, f"{split} has null text values"
        assert df["label"].isnull().sum() == 0, f"{split} has null label values"


def test_no_empty_text(splits):
    for split, df in splits.items():
        empty = (df["text"].str.strip() == "").sum()
        assert empty == 0, f"{split} has {empty} empty-string text rows"


def test_labels_are_binary(splits):
    for split, df in splits.items():
        unique_labels = set(df["label"].unique())
        assert unique_labels.issubset({0, 1}), (
            f"{split} has unexpected label values: {unique_labels}"
        )


def test_no_duplicate_text_within_split(splits):
    for split, df in splits.items():
        dupes = df["text"].duplicated().sum()
        assert dupes == 0, f"{split} has {dupes} duplicate text rows"


def test_no_data_leakage_across_splits(splits):
    """The same text should never appear in more than one split -- this
    would let the model see test/val examples during training."""
    train_texts = set(splits["train"]["text"])
    val_texts = set(splits["val"]["text"])
    test_texts = set(splits["test"]["text"])

    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts

    assert len(train_val_overlap) == 0, f"{len(train_val_overlap)} texts leak between train/val"
    assert len(train_test_overlap) == 0, f"{len(train_test_overlap)} texts leak between train/test"
    assert len(val_test_overlap) == 0, f"{len(val_test_overlap)} texts leak between val/test"


def test_split_proportions_roughly_correct(splits):
    """Verify the 70/15/15 split actually landed close to that ratio."""
    total = sum(len(df) for df in splits.values())
    train_frac = len(splits["train"]) / total
    val_frac = len(splits["val"]) / total
    test_frac = len(splits["test"]) / total

    assert 0.65 <= train_frac <= 0.75, f"train fraction is {train_frac:.3f}, expected ~0.70"
    assert 0.10 <= val_frac <= 0.20, f"val fraction is {val_frac:.3f}, expected ~0.15"
    assert 0.10 <= test_frac <= 0.20, f"test fraction is {test_frac:.3f}, expected ~0.15"


def test_label_distribution_is_stratified(splits):
    """Each split should have a similar spam rate -- if stratification
    worked, these shouldn't differ by much."""
    spam_rates = {split: df["label"].mean() for split, df in splits.items()}
    rates = list(spam_rates.values())
    max_diff = max(rates) - min(rates)
    assert max_diff < 0.03, (
        f"Spam rates differ too much across splits (max diff={max_diff:.3f}): {spam_rates}"
    )


def test_minimum_dataset_size(splits):
    """Sanity check we didn't accidentally drop almost everything during
    cleaning (e.g. from an overly aggressive dedup or filter)."""
    total = sum(len(df) for df in splits.values())
    assert total > 4500, (
        f"Only {total} rows survived cleaning -- original dataset has ~5570, "
        f"something likely dropped too much data"
    )


def test_no_obvious_encoding_artifacts(splits):
    """Catch mojibake (e.g. 'Ã¢â€') that would indicate a botched
    encoding fix during cleaning."""
    for split, df in splits.items():
        bad = df["text"].str.contains(r"Ã¢â€", regex=True, na=False).sum()
        assert bad == 0, f"{split} still has {bad} rows with encoding artifacts"