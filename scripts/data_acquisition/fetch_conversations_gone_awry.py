"""
Dataset: Conversations Gone Awry (Cornell ConvoKit)
Used by: models/text/conversational_context_classifier

Two corpora available via ConvoKit:
  - "conversations-gone-awry-corpus" (Wikipedia talk pages)
  - "conversations-gone-awry-cmv-corpus" (Reddit r/ChangeMyView)
We pull both; the model can train on either or a combined set.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import RAW_DIR, DatasetManifest

logger = logging.getLogger("fetch_conversations_gone_awry")

CORPORA = [
    "conversations-gone-awry-corpus",
    "conversations-gone-awry-cmv-corpus",
]


def fetch_corpus(name: str):
    from convokit import Corpus, download

    dest = RAW_DIR / "conversations_gone_awry" / name
    if DatasetManifest.load_if_exists(f"conversations_gone_awry_{name}"):
        logger.info("%s already fetched — skipping", name)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = download(name, data_dir=str(dest.parent))
    corpus = Corpus(filename=corpus_path)

    n_conversations = len(list(corpus.iter_conversations()))
    n_utterances = len(list(corpus.iter_utterances()))
    derailed = sum(
        1 for c in corpus.iter_conversations()
        if c.meta.get("conversation_has_personal_attack") is True
    )
    logger.info(
        "%s: %d conversations, %d utterances, %d flagged as derailed",
        name, n_conversations, n_utterances, derailed,
    )

    manifest = DatasetManifest(
        slug=f"conversations_gone_awry_{name}",
        source=f"convokit:{name}",
        files=[corpus_path],
        row_count=n_utterances,
    )
    manifest.save()


def main():
    for corpus_name in CORPORA:
        fetch_corpus(corpus_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()