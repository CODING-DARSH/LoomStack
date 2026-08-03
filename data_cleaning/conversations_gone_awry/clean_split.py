"""
CLEAN + SPLIT. Wikipedia corpus ONLY (conversations-gone-awry-corpus).
CMV corpus is deliberately excluded -- its label (has_removed_comment)
isn't equivalent to conversation_has_personal_attack, and was
explicitly parked rather than merged (see conversation history / project
notes, not re-litigated here).

Cleaning + task-construction decisions made here, confirmed against
two verification scripts before writing this:

  1. TASK SHAPE: this is not flat text classification. Each row is a
     (conversation context, label) pair, where context = the
     utterances of a conversation and label = whether it derails into
     a personal attack. This is fundamentally different from every
     other dataset in this repo.

  2. TRUNCATION: verified via check_attack_position.py that the
     comment_has_personal_attack utterance is NOT always the last one
     in a derailed conversation (88.7% last, 11.3% not). So each
     derailed conversation is truncated to utterances STRICTLY BEFORE
     the actual attack utterance's position -- never just "drop the
     last utterance". Non-derailed conversations are kept at full
     length, unmodified.

  3. PAIR INTEGRITY: verified via check_pair_structure.py that:
       - pair_id is a per-conversation POINTER to its partner's
         conversation id (not a shared group key) -- resolved 2094
         reciprocal pairs from 4188 conversations, all consistent.
       - every pair is exactly one derailed + one non-derailed
       - every pair's two conversations are in the SAME native split
         (0 mismatches) -- the raw corpus's split assignment already
         respects pair integrity, so we use it as-is rather than
         re-deriving a split ourselves.
     No length-matching is imposed between pair partners (433/2094
     partners are naturally shorter than the derailed conversation's
     pre-attack context) -- forcing a match would deviate from how
     this benchmark is actually used, so we don't.
     pair_id is KEPT as an output column specifically so tests can
     verify no pair was broken across splits during cleaning, and so
     paired evaluation is possible downstream.

  4. TEXT CLEANING:
       - [WIKI_LINK: <target>] placeholders (ConvoKit's own
         pre-processing of raw [[wiki links]], not something we
         introduced) are normalized to a generic <WIKILINK> token --
         same treatment as <URL> in jigsaw_multilingual. Only ~0.3%
         of utterances affected.
       - is_section_header utterances are DROPPED from the context --
         they're structural markers ("== Section title ==""), not
         conversational turns, and would be noise for a model
         predicting conversational dynamics.
       - Utterances empty after cleaning are dropped.
       - Standard control-char/whitespace cleanup, same as prior
         datasets.
       - Utterances within a conversation are joined with a <TURN>
         separator token, in chronological order, so turn boundaries
         are preserved for the model instead of collapsing into one
         run-on paragraph.

Usage:
    python clean_split.py
"""

import logging
import re
from pathlib import Path

import pandas as pd
from convokit import Corpus

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clean_conversations_gone_awry")

RAW_DIR = Path("data/raw/conversations_gone_awry")
CORPUS_NAME = "conversations-gone-awry-corpus"
OUT_DIR = Path("data/processed/conversations_gone_awry")

WIKI_LINK_PATTERN = re.compile(r"\[WIKI_LINK:[^\]]*\]")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
BARE_HEADER_PATTERN = re.compile(r"^==[^=]+==$")
TURN_SEP = " <TURN> "


def clean_utterance_text(text: str) -> str:
    text = WIKI_LINK_PATTERN.sub("<WIKILINK>", text)
    text = text.replace("\ufffd", "")
    text = CONTROL_CHAR_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pre_attack_utterances(convo):
    """Returns the chronologically-ordered utterances to keep as
    context: for derailed conversations, everything strictly before
    the attack utterance; for non-derailed conversations, everything."""
    utts_sorted = sorted(convo.iter_utterances(), key=lambda u: u.timestamp)

    if convo.meta.get("conversation_has_personal_attack"):
        attack_positions = [
            i for i, u in enumerate(utts_sorted)
            if u.meta.get("comment_has_personal_attack")
        ]
        cutoff = attack_positions[0] if attack_positions else len(utts_sorted)
        return utts_sorted[:cutoff]

    return utts_sorted


def build_context_text(utts) -> tuple[str, int]:
    """Cleans each utterance, drops section headers and empties, joins
    with TURN_SEP. Returns (context_text, num_utterances_kept).

    Also drops "bare" inline headers that ConvoKit's is_section_header
    flag misses -- editors sometimes insert a subsection header (e.g.
    "==Source misrepresentation==") as the ENTIRE content of their own
    reply, to organize a discussion, rather than as a page-structure
    header. Functionally these carry zero conversational content, same
    as a flagged section header, so they're dropped the same way even
    though ConvoKit didn't tag them. Confirmed via
    diagnose_test_failures.py: found in 1/4188 conversations."""
    kept_texts = []
    for u in utts:
        if u.meta.get("is_section_header"):
            continue
        cleaned = clean_utterance_text(u.text)
        if not cleaned:
            continue
        if BARE_HEADER_PATTERN.match(cleaned):
            continue
        kept_texts.append(cleaned)
    return TURN_SEP.join(kept_texts), len(kept_texts)


def load_and_clean() -> pd.DataFrame:
    corpus = Corpus(filename=str(RAW_DIR / CORPUS_NAME))
    conversations = list(corpus.iter_conversations())
    logger.info(f"Loaded {len(conversations)} conversations")

    rows = []
    dropped_empty_context = 0

    for convo in conversations:
        utts = pre_attack_utterances(convo)
        context_text, num_kept = build_context_text(utts)

        if not context_text.strip():
            dropped_empty_context += 1
            continue

        rows.append({
            "conversation_id": convo.id,
            "pair_id": convo.meta.get("pair_id"),
            "text": context_text,
            "label": bool(convo.meta.get("conversation_has_personal_attack")),
            "split": convo.meta.get("split"),
            "num_utterances": num_kept,
        })

    logger.info(f"Dropped {dropped_empty_context} conversations with empty context after cleaning "
                f"(e.g. all utterances were section headers, or attack was the very first utterance)")

    df = pd.DataFrame(rows)
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"Split distribution:\n{df['split'].value_counts().to_string()}")
    return df


def verify_pair_integrity(df: pd.DataFrame):
    """Confirms cleaning didn't break the pair structure we verified
    pre-cleaning: every surviving conversation's partner should also
    survive, in the same split, with a complementary label. If a
    conversation was dropped for empty-context, its now-orphaned
    partner is logged (not necessarily an error -- see note below --
    but must be visible, not silent)."""
    id_to_row = {row["conversation_id"]: row for _, row in df.iterrows()}
    orphaned = []
    broken_split = []
    broken_label = []

    for _, row in df.iterrows():
        partner_id = row["pair_id"]
        partner = id_to_row.get(partner_id)
        if partner is None:
            orphaned.append(row["conversation_id"])
            continue
        if partner["split"] != row["split"]:
            broken_split.append((row["conversation_id"], partner_id))
        if partner["label"] == row["label"]:
            broken_label.append((row["conversation_id"], partner_id))

    logger.info(f"\n=== PAIR INTEGRITY POST-CLEANING ===")
    logger.info(f"Conversations whose partner did NOT survive cleaning (orphaned, partner likely "
                f"had empty context e.g. attack was the first utterance): {len(orphaned)}")
    logger.info(f"Surviving pairs split across DIFFERENT splits (should be 0): {len(broken_split)}")
    logger.info(f"Surviving pairs with the SAME label instead of complementary (should be 0): {len(broken_label)}")

    if broken_split:
        raise RuntimeError(f"Pair integrity broken across splits: {broken_split[:5]}")
    if broken_label:
        raise RuntimeError(f"Pair integrity broken on label composition: {broken_label[:5]}")


def save_splits(df: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_rename = {"train": "train", "val": "val", "test": "test"}
    for raw_split, out_split in split_rename.items():
        split_df = df[df["split"] == raw_split].drop(columns=["split"]).reset_index(drop=True)
        out_path = OUT_DIR / f"{out_split}.parquet"
        split_df.to_parquet(out_path)
        logger.info(f"Wrote {out_split}.parquet: {len(split_df)} rows")
        logger.info(f"  Label distribution:\n{split_df['label'].value_counts().to_string()}")


def main():
    df = load_and_clean()
    verify_pair_integrity(df)
    save_splits(df)
    logger.info(f"\nDone. Processed files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()