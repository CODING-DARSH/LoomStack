"""
Datasets: HateXplain + Dynabench/Dynahate
Used by: models/text/coded_language_lora (flagship LoRA fine-tune)

HateXplain (Hate-speech-CNERG/hatexplain) still ships a legacy Python
loading script on its main branch. Newer versions of the `datasets`
library REMOVED support for scripts entirely -- `trust_remote_code`
no longer works around this (that flag still applies to models, not
datasets anymore).

Real fix: HuggingFace auto-converts every public dataset to Parquet
on a separate `refs/convert/parquet` branch, regardless of whether
the main branch has a script. Loading from that branch/revision
bypasses the script entirely -- no code execution, no flag needed.

Dynahate (aps/dynahate) already ships scriptless, no special handling
needed.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import RAW_DIR, DatasetManifest, HFDownloader

logger = logging.getLogger("fetch_hatexplain_dynahate")


def fetch_hatexplain():
    slug = "hatexplain"
    dest = RAW_DIR / slug
    if DatasetManifest.load_if_exists(slug):
        logger.info("%s already fetched -- skipping", slug)
        return

    from datasets import load_dataset

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Loading Hate-speech-CNERG/hatexplain via refs/convert/parquet branch")
    ds = load_dataset(
        "Hate-speech-CNERG/hatexplain",
        revision="refs/convert/parquet",
    )

    for split_name, split_ds in ds.items():
        out_file = dest / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_file))
        logger.info("HateXplain[%s]: %d rows -> %s", split_name, len(split_ds), out_file.name)

    files = list(dest.glob("*.parquet"))
    total_rows = sum(len(v) for v in ds.values())
    manifest = DatasetManifest(
        slug=slug,
        source="hf:Hate-speech-CNERG/hatexplain@refs/convert/parquet",
        files=[f.name for f in files],
        row_count=total_rows,
    )
    manifest.save()


def fetch_dynahate():
    slug = "dynahate"
    dest = RAW_DIR / slug
    if DatasetManifest.load_if_exists(slug):
        logger.info("%s already fetched -- skipping", slug)
        return

    from datasets import load_dataset

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Loading aps/dynahate via refs/convert/parquet branch")
    ds = load_dataset(
        "aps/dynahate",
        revision="refs/convert/parquet",
    )

    import pandas as pd

    all_files = []
    total_rows = 0
    for split_name, split_ds in ds.items():
        out_file = dest / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_file))
        total_rows += len(split_ds)
        all_files.append(out_file)
        df = split_ds.to_pandas()
        logger.info("Dynahate[%s]: %d rows, columns=%s", split_name, len(df), list(df.columns))
        if "label" in df.columns:
            logger.info("  label distribution:\n%s", df["label"].value_counts().to_string())

    manifest = DatasetManifest(
        slug=slug,
        source="hf:aps/dynahate@refs/convert/parquet",
        files=[f.name for f in all_files],
        row_count=total_rows,
    )
    manifest.save()


def main():
    fetch_hatexplain()
    fetch_dynahate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()