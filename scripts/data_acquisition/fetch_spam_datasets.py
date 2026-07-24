"""
Dataset: SMS Spam Collection (UCI) + Enron-Spam / Nazario phishing corpus
Used by: models/text/spam_classifier

SMS Spam Collection -> Kaggle mirror `uciml/sms-spam-collection-dataset`
Enron-Spam           -> HF mirror `SetFit/enron_spam` (has ham/spam labels)
Nazario phishing     -> direct HTTP tarball from monkey.org (Jose Nazario's corpus)
"""

import logging
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from utils.downloader import HFDownloader, HTTPDownloader, KaggleDownloader

logger = logging.getLogger("fetch_spam_datasets")

NAZARIO_URL = "https://monkey.org/~jose/phishing/phishing0.mbox"


def fetch_sms_spam():
    downloader = KaggleDownloader(slug="sms_spam_collection", kaggle_ref="uciml/sms-spam-collection-dataset")
    dest = downloader.fetch()
    csv_candidates = list(dest.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found in {dest}")
    df = pd.read_csv(csv_candidates[0], encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    logger.info("SMS Spam: %d rows, spam rate=%.3f", len(df), (df.label == "spam").mean())
    return df


def fetch_enron_spam():
    downloader = HFDownloader(slug="enron_spam", hf_path="SetFit/enron_spam", split="train")
    dest = downloader.fetch()
    df = pd.read_parquet(dest / "data.parquet")
    logger.info("Enron-Spam: %d rows, columns=%s", len(df), list(df.columns))
    return df


def fetch_nazario_phishing():
    """
    Nazario's corpus ships as mbox files, not CSV. Downloaded raw here;
    parsing into (subject, body, label=phishing) rows happens in the
    preprocessing step, not acquisition, to keep this script idempotent
    and format-conversion-free.
    """
    downloader = HTTPDownloader(
        slug="nazario_phishing",
        url=NAZARIO_URL,
        filename="phishing0.mbox",
    )
    dest = downloader.fetch()
    mbox_path = dest / "phishing0.mbox"
    if mbox_path.exists():
        logger.info("Nazario mbox downloaded: %.2f MB", mbox_path.stat().st_size / 1e6)
    else:
        logger.warning("Nazario mbox not found after download — source URL may have moved")


def main():
    fetch_sms_spam()
    fetch_enron_spam()
    fetch_nazario_phishing()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()