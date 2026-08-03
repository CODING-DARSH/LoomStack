"""
ONE-OFF DIAGNOSTIC -- investigates two test failures, changes nothing.

Usage:
    python diagnose_test_failures.py
"""

import re
from pathlib import Path

import pandas as pd
from convokit import Corpus

RAW_DIR = Path("data/raw/conversations_gone_awry")
CORPUS_NAME = "conversations-gone-awry-corpus"
PROCESSED_DIR = Path("data/processed/conversations_gone_awry")


def diagnose_wikilink():
    print("=" * 70)
    print("DIAGNOSIS 1: [WIKI_LINK: prevalence in headers vs non-headers")
    print("=" * 70)

    corpus = Corpus(filename=str(RAW_DIR / CORPUS_NAME))
    utterances = list(corpus.iter_utterances())

    pattern = re.compile(r"\[WIKI_LINK:")

    header_with_link = 0
    header_without_link = 0
    nonheader_with_link = 0
    nonheader_without_link = 0

    for u in utterances:
        is_header = bool(u.meta.get("is_section_header"))
        has_link = bool(pattern.search(u.text))
        if is_header and has_link:
            header_with_link += 1
        elif is_header:
            header_without_link += 1
        elif has_link:
            nonheader_with_link += 1
        else:
            nonheader_without_link += 1

    print(f"Section headers WITH [WIKI_LINK:  {header_with_link}")
    print(f"Section headers WITHOUT:          {header_without_link}")
    print(f"Non-headers WITH [WIKI_LINK:      {nonheader_with_link}")
    print(f"Non-headers WITHOUT:              {nonheader_without_link}")
    print(f"\nTotal utterances with [WIKI_LINK: anywhere: {header_with_link + nonheader_with_link}")


def diagnose_section_header_remnant():
    print("\n" + "=" * 70)
    print("DIAGNOSIS 2: the offending 'train' row with header-like text")
    print("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    header_pattern = re.compile(r"(?:^|<TURN> )==[^=]+==(?:\s|$)")
    matches = df[df["text"].str.contains(header_pattern, regex=True)]

    for _, row in matches.iterrows():
        print(f"\nconversation_id: {row['conversation_id']}")
        print(f"num_utterances: {row['num_utterances']}")
        print(f"FULL TEXT:\n{row['text']}")


if __name__ == "__main__":
    diagnose_wikilink()
    diagnose_section_header_remnant()