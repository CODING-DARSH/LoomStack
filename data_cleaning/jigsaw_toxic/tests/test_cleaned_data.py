"""
Basic sanity tests for cleaned Jigsaw Toxic Comment data.
Multilabel-adapted version of the SMS Spam basic test suite.

Usage:
    pytest data_cleaning/jigsaw_toxic/tests/test_cleaned_data.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed/jigsaw_toxic")
SPLITS = ["train", "val", "test"]
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@pytest.fixture(scope="module")
def splits():
    data = {}
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.parquet"
        if not path.exists():
            pytest.fail(f"{path} does not exist -- run 02_clean_and_split.py first")
        data[split] = pd.read_parquet(path)
    return data


def test_files_exist():
    for split in SPLITS:
        assert (PROCESSED_DIR / f"{split}.parquet").exists()


def test_expected_columns(splits):
    expected = {"id", "text"} | set(LABEL_COLS)
    for split, df in splits.items():
        assert set(df.columns) == expected, f"{split} has unexpected columns: {list(df.columns)}"


def test_no_nulls(splits):
    for split, df in splits.items():
        assert df.isnull().sum().sum() == 0, f"{split} has null values"


def test_no_empty_text(splits):
    for split, df in splits.items():
        assert (df["text"].str.strip() == "").sum() == 0


def test_all_labels_binary(splits):
    for split, df in splits.items():
        for col in LABEL_COLS:
            unique = set(df[col].unique())
            assert unique.issubset({0, 1}), f"{split}.{col} has non-binary values: {unique}"


def test_no_duplicate_text_within_split(splits):
    for split, df in splits.items():
        dupes = df["text"].duplicated().sum()
        assert dupes == 0, f"{split} has {dupes} duplicate text rows"


def test_no_exact_leakage_across_splits(splits):
    train_texts = set(splits["train"]["text"])
    val_texts = set(splits["val"]["text"])
    test_texts = set(splits["test"]["text"])
    assert len(train_texts & val_texts) == 0
    assert len(train_texts & test_texts) == 0
    assert len(val_texts & test_texts) == 0


def test_no_id_leakage_across_splits(splits):
    """ids should also never repeat across splits -- a stronger
    leakage check than text alone, in case whitespace/casing differs
    but it's actually the same underlying comment."""
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
    assert 0.65 <= train_frac <= 0.75
    assert 0.10 <= val_frac <= 0.20
    assert 0.10 <= test_frac <= 0.20


def test_per_label_stratification_all_six_labels(splits):
    """Unlike SMS Spam's single label, this MUST check all 6 labels
    independently -- a plain single-column stratified split would
    pass a check on 'toxic' while badly imbalancing 'threat' or
    'identity_hate'."""
    for col in LABEL_COLS:
        rates = {split: df[col].mean() for split, df in splits.items()}
        max_diff = max(rates.values()) - min(rates.values())
        assert max_diff < 0.01, (
            f"Label '{col}' rate differs too much across splits "
            f"(max diff={max_diff:.4f}): {rates}"
        )


def test_minimum_dataset_size(splits):
    total = sum(len(df) for df in splits.values())
    assert total > 150000, f"Only {total} rows survived cleaning, expected ~159000"