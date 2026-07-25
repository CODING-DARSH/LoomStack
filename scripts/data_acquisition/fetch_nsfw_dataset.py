"""
Dataset: NSFW image classification set
Source:  HuggingFace `DarkyMan/nsfw-image-classification`
Used by: models/image/nsfw_classifier

History of this dataset choice (kept here so it's not a mystery later):
  1. Original plan: Kaggle `drmariodominguez/nsfw-images-dataset` -> removed
     from Kaggle (NSFW-tagged datasets get periodically taken down there).
  2. Tried: HF `deepghs/nsfw_detect` -> OWNER-GATED, confirmed via
     HfApi().dataset_info(...).gated == True. Rejected.
  3. Tried: HF `DamarJati/NSFW-filter-DecentScan` -> ALSO gated despite the
     web page not showing an obvious gate banner. Rejected.
  4. Tried as fallback: `edwixx/NsFW-Dataset` -> wrong modality, it's a
     VIDEO dataset (10M-100M rows), not image classification. Rejected.
  5. Current: `DarkyMan/nsfw-image-classification` -> CONFIRMED not
     gated via:
         python -c "from huggingface_hub import HfApi; \
             print(HfApi().dataset_info('DarkyMan/nsfw-image-classification').gated)"
     which printed False. This is the verification step to run BEFORE
     trying to swap in any other HF dataset in the future -- don't
     trust the webpage banner alone, check the API field directly.

NOTE: this script only downloads and validates directory structure --
image content is never logged or displayed. Handle this dataset with
restricted repo/storage access; do not commit raw images.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import HFDownloader

logger = logging.getLogger("fetch_nsfw_dataset")

HF_PATH = "DarkyMan/nsfw-image-classification"


def main():
    downloader = HFDownloader(slug="nsfw_images", hf_path=HF_PATH, split=None)
    dest = downloader.fetch()

    import pandas as pd

    parquet_files = list(dest.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet file found in {dest}. Files present: "
            f"{[p.name for p in dest.rglob('*') if p.is_file()]}"
        )

    df = pd.read_parquet(parquet_files[0])
    logger.info("Rows: %d", len(df))
    logger.info("Columns: %s", list(df.columns))

    label_col = next((c for c in ("label", "labels", "category") if c in df.columns), None)
    if label_col:
        logger.info("Class distribution:\n%s", df[label_col].value_counts().to_string())
    else:
        logger.warning("No obvious label column found -- inspect columns manually: %s", list(df.columns))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()