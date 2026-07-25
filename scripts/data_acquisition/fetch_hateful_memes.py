"""
Dataset: Hateful Memes Challenge (Meta/Facebook AI)
Used by: models/multimodal/trimodal_shared_embedding (text+image leg)

NOTE: the original challenge site (hatefulmemeschallenge.com) and its
DrivenData download page have gone dead -- this is a known, widely
reported issue (see facebookresearch/mmf#1221), not specific to this
environment. HuggingFace hosts a faithful mirror with the same
structure (img/ + train.jsonl/dev_seen.jsonl/test_seen.jsonl/etc),
so we pull from there instead.

Still license-gated (research-use terms), so you may need to run
`huggingface-cli login` and accept terms on the dataset page first:
https://huggingface.co/datasets/neuralcatcher/hateful_memes

Fallback if the HF mirror is also unavailable: Kaggle mirror at
kaggle.com/datasets/williamberrios/hateful-memes (also gated, requires
manual acceptance on kaggle.com before API download works).
"""

import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import HFDownloader

logger = logging.getLogger("fetch_hateful_memes")

HF_PRIMARY = "neuralcatcher/hateful_memes"
HF_FALLBACK = "emily49/hateful-memes"


def try_fetch(hf_path: str):
    downloader = HFDownloader(slug="hateful_memes", hf_path=hf_path, split=None)
    return downloader.fetch()


def main():
    try:
        dest = try_fetch(HF_PRIMARY)
    except Exception as e:
        logger.warning("Primary mirror %s failed (%s), trying fallback %s", HF_PRIMARY, e, HF_FALLBACK)
        dest = try_fetch(HF_FALLBACK)

    parquet_files = list(dest.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {dest} after HF download. "
            f"If this dataset is gated, run `huggingface-cli login` and accept "
            f"terms at https://huggingface.co/datasets/{HF_PRIMARY} then retry."
        )

    import pandas as pd

    total_rows = 0
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        total_rows += len(df)
        logger.info("%s: %d rows, columns=%s", pf.name, len(df), list(df.columns))
        if "label" in df.columns:
            logger.info("  label distribution:\n%s", df["label"].value_counts().to_string())

    logger.info("Hateful Memes: %d total rows across %d split file(s)", total_rows, len(parquet_files))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()