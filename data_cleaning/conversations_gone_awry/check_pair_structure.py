"""
ONE-OFF VERIFICATION -- not part of the pipeline.

Checks three things about pair_id in the Wikipedia corpus before
clean_split.py touches truncation or splitting logic:

  1. Does every pair_id link EXACTLY 2 conversations (one derailed,
     one not)? If pairing is looser than that, "truncate the partner
     to match" isn't well-defined.
  2. Are both conversations in a pair always in the SAME split
     (train/val/test)? If a pair is split across train and test, the
     model would implicitly see a near-duplicate-topic conversation
     during training that shares structure with a test example --
     a leakage path distinct from text-level leakage, and the kind
     that's easy to introduce silently by re-splitting without
     respecting pair_id.
  3. For each pair, what's the pre-attack utterance count of the
     derailed conversation vs the total utterance count of its
     non-derailed partner? This tells us whether "truncate the
     partner to the same length" is even usually possible (partner
     might be shorter than the pre-attack context needs).

Does not write or modify anything.

Usage:
    python check_pair_structure.py
"""

from collections import defaultdict
from pathlib import Path

from convokit import Corpus

RAW_DIR = Path("data/raw/conversations_gone_awry")
CORPUS_NAME = "conversations-gone-awry-corpus"


def pre_attack_length(convo) -> int:
    """Number of utterances strictly before the attack utterance.
    For non-derailed conversations, this is just the total length."""
    utts_sorted = sorted(convo.iter_utterances(), key=lambda u: u.timestamp)
    attack_positions = [
        i for i, u in enumerate(utts_sorted)
        if u.meta.get("comment_has_personal_attack")
    ]
    if attack_positions:
        return attack_positions[0]  # utterances before this index
    return len(utts_sorted)


def main():
    corpus = Corpus(filename=str(RAW_DIR / CORPUS_NAME))
    conversations = list(corpus.iter_conversations())
    convo_by_id = {c.id: c for c in conversations}

    print(f"Total conversations: {len(conversations)}")

    # pair_id is NOT a shared group key -- it's per-conversation and
    # points at the id of its PARTNER conversation. Confirmed by the
    # first run: grouping by pair_id directly gave 4188 singleton
    # groups (every conversation has a distinct pair_id), which only
    # makes sense if pair_id references another conversation's id.
    sample_pid = conversations[0].meta.get("pair_id")
    print(f"Sample pair_id: {sample_pid!r}")
    print(f"Is sample pair_id itself a conversation id in this corpus? {sample_pid in convo_by_id}")

    # Build pairs by following the pointer both ways
    pairs = {}
    unmatched = []
    reciprocal_mismatch = []
    for convo in conversations:
        partner_id = convo.meta.get("pair_id")
        partner = convo_by_id.get(partner_id)
        if partner is None:
            unmatched.append(convo.id)
            continue
        # confirm reciprocity: partner's own pair_id should point back
        if partner.meta.get("pair_id") != convo.id:
            reciprocal_mismatch.append((convo.id, partner_id, partner.meta.get("pair_id")))
            continue
        # canonical key so we don't double-count each pair from both sides
        key = tuple(sorted([convo.id, partner.id]))
        pairs[key] = (convo, partner)

    print(f"\n=== PAIR RESOLUTION VIA pair_id POINTER ===")
    print(f"Conversations whose pair_id doesn't match any conversation id in this corpus: {len(unmatched)}")
    print(f"Conversations whose partner's pair_id doesn't point back (non-reciprocal): {len(reciprocal_mismatch)}")
    print(f"Resolved pairs (deduplicated, each counted once): {len(pairs)}")
    print(f"Expected: {len(conversations)} conversations / 2 = {len(conversations)//2} pairs")

    if reciprocal_mismatch:
        print(f"  Example non-reciprocal: {reciprocal_mismatch[0]}")

    # 2. Derailed/non-derailed composition per pair
    wrong_composition = 0
    for key, (c1, c2) in pairs.items():
        labels = sorted([c1.meta.get("conversation_has_personal_attack"), c2.meta.get("conversation_has_personal_attack")])
        if labels != [False, True]:
            wrong_composition += 1
    print(f"\n=== PAIR LABEL COMPOSITION ===")
    print(f"Pairs that are NOT exactly one True + one False: {wrong_composition}")

    # 3. Split consistency within pairs
    split_mismatch = 0
    for key, (c1, c2) in pairs.items():
        if c1.meta.get("split") != c2.meta.get("split"):
            split_mismatch += 1
    print(f"\n=== SPLIT CONSISTENCY WITHIN PAIRS ===")
    print(f"Pairs whose two conversations land in DIFFERENT splits: {split_mismatch}")
    if split_mismatch > 0:
        print("  ^ THIS IS A LEAKAGE RISK if nonzero -- means the raw corpus itself")
        print("    already has cross-split pairs, or something about how 'split' was")
        print("    read is wrong. Investigate before trusting the native split.")

    # 4. Pre-attack length vs partner length
    print(f"\n=== PRE-ATTACK CONTEXT LENGTH vs PARTNER LENGTH ===")
    partner_shorter_count = 0
    checked = 0
    examples = []
    for key, (c1, c2) in pairs.items():
        labels = {c1.meta.get("conversation_has_personal_attack"): c1, c2.meta.get("conversation_has_personal_attack"): c2}
        if True not in labels or False not in labels:
            continue
        derailed = labels[True]
        clean = labels[False]

        derailed_ctx_len = pre_attack_length(derailed)
        clean_total_len = len(list(clean.iter_utterances()))

        checked += 1
        if clean_total_len < derailed_ctx_len:
            partner_shorter_count += 1
            if len(examples) < 5:
                examples.append({
                    "pair": key,
                    "derailed_pre_attack_len": derailed_ctx_len,
                    "partner_total_len": clean_total_len,
                })

    print(f"Pairs checked: {checked}")
    print(f"Pairs where the non-derailed partner is SHORTER than the derailed conversation's "
          f"pre-attack context (can't truncate partner to match): {partner_shorter_count}")
    if examples:
        print("Examples:")
        for ex in examples:
            print(f"  {ex}")


if __name__ == "__main__":
    main()