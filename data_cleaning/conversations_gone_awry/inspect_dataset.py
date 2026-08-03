"""
INSPECT ONLY. No cleaning happens here.

Conversations Gone Awry -- two ConvoKit Corpus objects, not flat
CSV/parquet like everything cleaned so far:
  - conversations-gone-awry-corpus (Wikipedia talk pages)
  - conversations-gone-awry-cmv-corpus (Reddit r/ChangeMyView)

ConvoKit corpora are THREADED: utterances reply to other utterances
within a conversation, and the prediction task here is fundamentally
different from anything cleaned so far -- it's not "classify this
text", it's "given the first N utterances of a conversation, will it
derail into a personal attack?" That means conversation-level
structure (utterance order, reply-to chains, how many utterances
precede the labeled outcome) is itself part of what needs inspecting,
not just text quality.

This script profiles both corpora SEPARATELY, at both the
conversation level and utterance level, since collapsing them into
one flat DataFrame view (like flat CSV datasets) would hide the
threading structure that's central to this dataset's actual task.

Usage:
    python inspect_dataset.py
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inspect_conversations_gone_awry")

RAW_DIR = Path("data/raw/conversations_gone_awry")
CORPORA = [
    "conversations-gone-awry-corpus",
    "conversations-gone-awry-cmv-corpus",
]


def inspect_corpus(name: str):
    from convokit import Corpus

    logger.info("\n" + "=" * 70)
    logger.info(name.upper())
    logger.info("=" * 70)

    corpus_dir = RAW_DIR / name
    if not corpus_dir.exists():
        raise FileNotFoundError(f"{corpus_dir} not found -- run the acquisition script first")

    corpus = Corpus(filename=str(corpus_dir))

    conversations = list(corpus.iter_conversations())
    utterances = list(corpus.iter_utterances())

    logger.info(f"\n=== SHAPE ===")
    logger.info(f"Conversations: {len(conversations)}")
    logger.info(f"Utterances: {len(utterances)}")
    logger.info(f"Speakers: {len(list(corpus.iter_speakers()))}")

    logger.info(f"\n=== CONVERSATION META KEYS (from first conversation) ===")
    sample_convo = conversations[0]
    logger.info(list(sample_convo.meta.keys()))

    logger.info(f"\n=== UTTERANCE META KEYS (from first utterance) ===")
    sample_utt = utterances[0]
    logger.info(list(sample_utt.meta.keys()))
    logger.info(f"Utterance fields: id={sample_utt.id!r}, speaker={sample_utt.speaker.id!r}, "
                f"reply_to={sample_utt.reply_to!r}, conversation_id={sample_utt.conversation_id!r}")
    logger.info(f"Sample text: {sample_utt.text[:200]!r}")

    logger.info(f"\n=== DERAILMENT LABEL (conversation_has_personal_attack) ===")
    label_key = "conversation_has_personal_attack"
    has_label = [c.meta.get(label_key) for c in conversations]
    label_present = sum(1 for v in has_label if v is not None)
    logger.info(f"Conversations with '{label_key}' set: {label_present} / {len(conversations)}")
    if label_present:
        from collections import Counter
        logger.info(f"Value distribution: {dict(Counter(v for v in has_label if v is not None))}")

    logger.info(f"\n=== PRE-EXISTING SPLIT (check for a 'split' meta field) ===")
    split_key_candidates = ["split", "annotation_year"]
    for key in split_key_candidates:
        values = [c.meta.get(key) for c in conversations]
        present = sum(1 for v in values if v is not None)
        if present:
            from collections import Counter
            logger.info(f"'{key}' present on {present}/{len(conversations)} conversations: "
                        f"{dict(Counter(v for v in values if v is not None))}")

    logger.info(f"\n=== CONVERSATION LENGTH (utterances per conversation) ===")
    convo_lengths = [len(list(c.iter_utterances())) for c in conversations]
    logger.info(f"Min: {min(convo_lengths)}, Max: {max(convo_lengths)}, "
                f"Mean: {sum(convo_lengths)/len(convo_lengths):.2f}")

    logger.info(f"\n=== UTTERANCE-LEVEL PERSONAL ATTACK LABEL ===")
    utt_label_candidates = ["comment_has_personal_attack", "is_section_header"]
    for key in utt_label_candidates:
        values = [u.meta.get(key) for u in utterances]
        present = sum(1 for v in values if v is not None)
        if present:
            from collections import Counter
            logger.info(f"'{key}' present on {present}/{len(utterances)} utterances: "
                        f"{dict(Counter(v for v in values if v is not None))}")

    logger.info(f"\n=== TEXT LENGTH / EMPTY CHECK ===")
    texts = [u.text for u in utterances]
    lengths = [len(t) for t in texts]
    empty = sum(1 for t in texts if not t.strip())
    logger.info(f"Length -- min: {min(lengths)}, max: {max(lengths)}, mean: {sum(lengths)/len(lengths):.1f}")
    logger.info(f"Empty/whitespace-only utterances: {empty} / {len(utterances)}")

    logger.info(f"\n=== SPEAKER ANONYMIZATION CHECK ===")
    speaker_ids = [u.speaker.id for u in utterances]
    logger.info(f"Sample speaker ids: {speaker_ids[:5]}")
    logger.info(f"Distinct speakers: {len(set(speaker_ids))}")

    logger.info(f"\n=== REPLY-TO / THREADING INTEGRITY ===")
    utt_ids = {u.id for u in utterances}
    dangling_replies = sum(
        1 for u in utterances
        if u.reply_to is not None and u.reply_to not in utt_ids
    )
    root_utterances = sum(1 for u in utterances if u.reply_to is None)
    logger.info(f"Root utterances (reply_to is None): {root_utterances}")
    logger.info(f"Dangling reply_to references (points to a non-existent utterance id): {dangling_replies}")

    logger.info(f"\n=== MARKUP / ARTIFACT CHECK ===")
    has_wiki_markup = sum(1 for t in texts if "[[" in t or "{{" in t)
    has_url = sum(1 for t in texts if "http://" in t or "https://" in t)
    has_newline = sum(1 for t in texts if "\n" in t)
    logger.info(f"Utterances with wiki markup ([[ ]], {{ }}): {has_wiki_markup}")
    logger.info(f"Utterances with URLs: {has_url}")
    logger.info(f"Utterances with literal newlines: {has_newline}")

    logger.info(f"\n=== DUPLICATE TEXT ===")
    logger.info(f"Duplicate utterance texts: {len(texts) - len(set(texts))}")


def compare_corpora():
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-CORPUS MERGE PLANNING")
    logger.info("=" * 70)
    logger.info("Before writing clean_split.py, decide:")
    logger.info("  - The actual prediction task: given the first K utterances of a")
    logger.info("    conversation, predict whether the LATER utterance derails into a")
    logger.info("    personal attack. This means clean_split.py needs to decide K")
    logger.info("    (how much context) and produce (context, label) pairs, not just")
    logger.info("    clean raw utterance text -- fundamentally different shape from")
    logger.info("    every dataset cleaned so far.")
    logger.info("  - Do Wikipedia talk pages and Reddit CMV threads get merged into")
    logger.info("    one dataset (with a 'source' column, same pattern as")
    logger.info("    hatexplain_dynahate), or kept as two separate outputs given how")
    logger.info("    different wiki markup vs Reddit formatting/quoting conventions are?")
    logger.info("  - Does either corpus ship a pre-existing train/val/test split in")
    logger.info("    conversation meta (check output above), or does this need a")
    logger.info("    fresh stratified split on the derailment label?")
    logger.info("  - Splitting must happen at the CONVERSATION level, never the")
    logger.info("    utterance level -- utterances from the same conversation must")
    logger.info("    never be split across train/val/test (that would leak the")
    logger.info("    conversation's outcome across splits).")
    logger.info("  - Wiki markup cleaning: reuse the same WIKI_MARKUP_PATTERN /")
    logger.info("    IP_PATTERN logic from jigsaw_toxic's clean_split.py for the")
    logger.info("    Wikipedia corpus specifically -- Reddit CMV won't have wiki markup.")


def main():
    for name in CORPORA:
        inspect_corpus(name)
    compare_corpora()


if __name__ == "__main__":
    main()