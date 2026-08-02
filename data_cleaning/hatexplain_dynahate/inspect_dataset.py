"""
INSPECT ONLY. No cleaning happens here.

HateXplain + Dynahate -- these are two STRUCTURALLY DIFFERENT sources
that clean_split.py will eventually need to merge into one schema:

  - HateXplain: nested structure -- post_tokens (list of words, not a
    plain string), multiple annotators per post (each with their own
    label + target + rationale span), no single ground-truth label
    column -- majority vote has to be derived.
  - Dynahate: flat structure -- plain text + a single label column,
    much closer to the datasets already cleaned (sms_spam,
    jigsaw_multilingual).

This script profiles both sources SEPARATELY (different sections
below) since a shared profiling loop would hide exactly the
structural mismatch that matters here. It does not attempt to
reconcile or merge anything -- that's clean_split.py's job, once we
know what we're actually merging.

Usage:
    python inspect_dataset.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inspect_hatexplain_dynahate")

HATEXPLAIN_DIR = Path("data/raw/hatexplain")
DYNAHATE_DIR = Path("data/raw/dynahate")


def load_all_parquet(dir_path: Path) -> dict[str, pd.DataFrame]:
    if not dir_path.exists():
        raise FileNotFoundError(f"{dir_path} not found -- run the acquisition script first")
    files = sorted(dir_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {dir_path} -- run the acquisition script first")
    return {f.stem: pd.read_parquet(f) for f in files}


# ---------------------------------------------------------------------
# HateXplain
# ---------------------------------------------------------------------

def inspect_hatexplain():
    logger.info("\n" + "=" * 70)
    logger.info("HATEXPLAIN")
    logger.info("=" * 70)

    splits = load_all_parquet(HATEXPLAIN_DIR)
    logger.info(f"Splits found: {list(splits.keys())}")

    for name, df in splits.items():
        logger.info(f"\n--- split: {name} ---")
        logger.info(f"Rows: {len(df)}, Columns: {list(df.columns)}")
        logger.info(f"Dtypes:\n{df.dtypes.to_string()}")
        logger.info(f"Null counts:\n{df.isnull().sum().to_string()}")

    sample_df = next(iter(splits.values()))

    logger.info("\n=== SAMPLE ROW (raw structure) ===")
    logger.info(sample_df.iloc[0].to_dict())

    if "post_tokens" in sample_df.columns:
        logger.info("\n=== post_tokens FIELD ===")
        first_tokens = sample_df["post_tokens"].iloc[0]
        logger.info(f"Type: {type(first_tokens)}")
        logger.info(f"Example: {first_tokens}")
        token_counts = sample_df["post_tokens"].apply(len)
        logger.info(f"Token count -- min: {token_counts.min()}, max: {token_counts.max()}, mean: {token_counts.mean():.1f}")

    if "annotators" in sample_df.columns:
        logger.info("\n=== annotators FIELD (nested per-post) ===")
        first_annotators = sample_df["annotators"].iloc[0]
        logger.info(f"Type: {type(first_annotators)}")
        logger.info(f"Example: {first_annotators}")
        # Actual structure is a dict of parallel arrays (Arrow-style),
        # not a list of per-annotator dicts -- e.g.
        # {'label': array([1,1,1]), 'annotator_id': array([...]), 'target': array([...])}
        annotator_counts = sample_df["annotators"].apply(lambda a: len(a["label"]))
        logger.info(f"Annotators per post -- min: {annotator_counts.min()}, max: {annotator_counts.max()}, mean: {annotator_counts.mean():.2f}")

        logger.info("\n=== LABEL INTEGER -> MEANING CHECK ===")
        all_label_ints = sorted(set(
            int(v) for arr in sample_df["annotators"] for v in arr["label"]
        ))
        logger.info(f"Distinct raw label integers seen: {all_label_ints}")
        logger.info("(HateXplain's published schema is 0=hatespeech, 1=normal, 2=offensive -- "
                     "confirm this against dataset card, do not assume)")

        logger.info("\n=== DERIVED MAJORITY LABEL (info only -- not written anywhere yet) ===")
        def majority_label(annotators):
            labels = [int(v) for v in annotators["label"]]
            return max(set(labels), key=labels.count)

        derived = sample_df["annotators"].apply(majority_label)
        logger.info(f"Majority-vote label distribution:\n{derived.value_counts().to_string()}")

        logger.info("\n=== ANNOTATOR DISAGREEMENT RATE ===")
        def disagreement(annotators):
            labels = [int(v) for v in annotators["label"]]
            return len(set(labels)) > 1
        disagree_rate = sample_df["annotators"].apply(disagreement).mean()
        logger.info(f"Fraction of posts where annotators disagree on label: {disagree_rate:.4f}")

        logger.info("\n=== 3-WAY TIE CHECK (no majority possible) ===")
        def is_three_way_tie(annotators):
            labels = [int(v) for v in annotators["label"]]
            return len(labels) == 3 and len(set(labels)) == 3
        tie_rate = sample_df["annotators"].apply(is_three_way_tie).sum()
        logger.info(f"Posts with 3 annotators all disagreeing (no majority, needs a tie-break rule): {tie_rate} / {len(sample_df)}")

        logger.info("\n=== TARGET FIELD (identity group per annotator) ===")
        first_target = sample_df["annotators"].iloc[0]["target"]
        logger.info(f"Example target array: {first_target}")
        all_targets = []
        for arr in sample_df["annotators"]:
            for t in arr["target"]:
                all_targets.extend(list(t))
        target_counts = pd.Series(all_targets).value_counts()
        logger.info(f"Top target values across all annotators in this split:\n{target_counts.head(15).to_string()}")

    if "rationales" in sample_df.columns:
        logger.info("\n=== rationales FIELD ===")
        first_rationale = sample_df["rationales"].iloc[0]
        logger.info(f"Type: {type(first_rationale)}")
        logger.info(f"Example: {first_rationale}")
        empty_rationale = sample_df["rationales"].apply(lambda r: len(r) == 0).sum()
        logger.info(f"Posts with empty rationales (e.g. 'normal' posts, no highlighted span): {empty_rationale} / {len(sample_df)}")

    if "id" in sample_df.columns:
        logger.info("\n=== ID / DUPLICATE CHECK ACROSS SPLITS ===")
        all_ids = pd.concat([df["id"] for df in splits.values()])
        logger.info(f"Total ids: {len(all_ids)}, unique: {all_ids.nunique()}, duplicated: {all_ids.duplicated().sum()}")


# ---------------------------------------------------------------------
# Dynahate
# ---------------------------------------------------------------------

def inspect_dynahate():
    logger.info("\n" + "=" * 70)
    logger.info("DYNAHATE")
    logger.info("=" * 70)

    splits = load_all_parquet(DYNAHATE_DIR)
    logger.info(f"Splits found: {list(splits.keys())}")

    for name, df in splits.items():
        logger.info(f"\n--- split: {name} ---")
        logger.info(f"Rows: {len(df)}, Columns: {list(df.columns)}")
        logger.info(f"Dtypes:\n{df.dtypes.to_string()}")
        logger.info(f"Null counts:\n{df.isnull().sum().to_string()}")

    sample_df = next(iter(splits.values()))

    logger.info("\n=== SAMPLE ROW ===")
    logger.info(sample_df.iloc[0].to_dict())

    if "label" in sample_df.columns:
        logger.info("\n=== LABEL VALUES / DISTRIBUTION (per split) ===")
        for name, df in splits.items():
            logger.info(f"{name}: {df['label'].value_counts().to_string()}")

    text_col = "text" if "text" in sample_df.columns else None
    if text_col:
        logger.info(f"\n=== TEXT COLUMN STATS ({text_col}, all splits combined) ===")
        full = pd.concat(splits.values(), ignore_index=True)
        lengths = full[text_col].astype(str).str.len()
        logger.info(f"Length -- min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}")
        empty_texts = (full[text_col].astype(str).str.strip() == "").sum()
        logger.info(f"Empty/whitespace-only text rows: {empty_texts}")
        logger.info(f"Duplicate text values (across all splits combined): {full[text_col].duplicated().sum()}")

        logger.info("\n=== URL / HTML ARTIFACTS ===")
        has_url = full[text_col].astype(str).str.contains(r"https?://|www\.", regex=True).sum()
        has_html = full[text_col].astype(str).str.contains(r"<[^>]+>|&\w+;", regex=True).sum()
        logger.info(f"Rows containing URLs: {has_url}")
        logger.info(f"Rows containing HTML tags/entities: {has_html}")

    if "type" in sample_df.columns:
        logger.info("\n=== 'type' FIELD (adversarial perturbation category?) ===")
        full = pd.concat(splits.values(), ignore_index=True)
        logger.info(full["type"].value_counts().to_string())

    if "target" in sample_df.columns:
        logger.info("\n=== 'target' FIELD (identity group targeted?) ===")
        full = pd.concat(splits.values(), ignore_index=True)
        logger.info(full["target"].value_counts(dropna=False).head(20).to_string())

    if "round" in sample_df.columns or "round.base" in sample_df.columns:
        logger.info("\n=== ROUND FIELD (Dynabench adversarial data-collection round) ===")
        round_col = "round" if "round" in sample_df.columns else "round.base"
        full = pd.concat(splits.values(), ignore_index=True)
        logger.info(full[round_col].value_counts().sort_index().to_string())


# ---------------------------------------------------------------------
# Cross-source comparison -- the actual merge-planning questions
# ---------------------------------------------------------------------

def compare_sources():
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-SOURCE MERGE PLANNING")
    logger.info("=" * 70)
    logger.info("Before writing clean_split.py, decide:")
    logger.info("  - HateXplain has no single ground-truth label -- majority vote")
    logger.info("    across annotators needed. What's the tie-breaking rule for a")
    logger.info("    3-way split with no majority (e.g. hatespeech/offensive/normal,")
    logger.info("    each with 1 vote)?")
    logger.info("  - HateXplain's post_tokens is a list, Dynahate's text is a plain")
    logger.info("    string -- HateXplain tokens need joining (with what separator?")
    logger.info("    original whitespace isn't preserved in token lists).")
    logger.info("  - Label schemas differ: HateXplain is 3-class")
    logger.info("    (hatespeech/offensive/normal), Dynahate is binary")
    logger.info("    (hate/nothate). Decide the unified schema -- collapse")
    logger.info("    HateXplain's offensive into hate or normal? Keep 3-class and")
    logger.info("    make Dynahate's 'nothate' map to 'normal'?")
    logger.info("  - Should the merged dataset keep a 'source' column")
    logger.info("    (hatexplain vs dynahate) so drift/leakage tests and later")
    logger.info("    per-source evaluation are possible?")
    logger.info("  - HateXplain ships id per post; check no id/text overlap between")
    logger.info("    the two sources before assuming they're fully independent.")
    logger.info("  - Dynahate's 'round' field reflects adversarial data collection")
    logger.info("    rounds -- does the split need to preserve round balance, or is")
    logger.info("    a straight stratified split on label sufficient?")


def main():
    inspect_hatexplain()
    inspect_dynahate()
    compare_sources()


if __name__ == "__main__":
    main()