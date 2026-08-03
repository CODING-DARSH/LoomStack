"""
ONE-OFF VERIFICATION -- not part of the pipeline.

Checks whether the utterance flagged comment_has_personal_attack=True
is always the LAST utterance (by chronological order/timestamp) in its
conversation. This is the load-bearing assumption for the "predict
derailment before it happens" task: if the attack utterance isn't
reliably last, truncating "everything before the attack" isn't
well-defined, and clean_split.py needs a different strategy (e.g.
truncate at the attack's actual position, not just drop the last
utterance).

Usage:
    python check_attack_position.py
"""

from pathlib import Path
from convokit import Corpus

RAW_DIR = Path("data/raw/conversations_gone_awry")
CORPUS_NAME = "conversations-gone-awry-corpus"


def main():
    corpus = Corpus(filename=str(RAW_DIR / CORPUS_NAME))

    always_last = 0
    not_last = 0
    not_last_examples = []
    no_attack_found = 0

    for convo in corpus.iter_conversations():
        if not convo.meta.get("conversation_has_personal_attack"):
            continue

        utts = list(convo.iter_utterances())
        utts_sorted = sorted(utts, key=lambda u: u.timestamp)

        attack_positions = [
            i for i, u in enumerate(utts_sorted)
            if u.meta.get("comment_has_personal_attack")
        ]

        if not attack_positions:
            no_attack_found += 1
            continue

        last_index = len(utts_sorted) - 1
        if attack_positions == [last_index]:
            always_last += 1
        else:
            not_last += 1
            if len(not_last_examples) < 5:
                not_last_examples.append({
                    "conversation_id": convo.id,
                    "attack_positions": attack_positions,
                    "conversation_length": len(utts_sorted),
                })

    print(f"Derailed conversations where attack utterance IS the last one: {always_last}")
    print(f"Derailed conversations where attack utterance is NOT the last one: {not_last}")
    print(f"Derailed conversations with the flag set but no utterance-level match found: {no_attack_found}")

    if not_last_examples:
        print("\nExamples where attack is not last:")
        for ex in not_last_examples:
            print(f"  {ex}")


if __name__ == "__main__":
    main()