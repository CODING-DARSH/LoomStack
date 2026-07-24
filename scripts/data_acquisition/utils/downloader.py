"""
Shared download utilities for dataset acquisition scripts.

Provides three backends:
  - KaggleDownloader     -> uses the Kaggle API (requires ~/.kaggle/kaggle.json)
  - HFDownloader         -> uses HuggingFace `datasets` library
  - HTTPDownloader       -> plain URL / git-lfs style downloads with retry + resume

All downloaders write into data/raw/<dataset_slug>/ and write a
manifest.json describing what was pulled, so re-runs are idempotent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

DATA_ROOT = Path(os.environ.get("MODPIPE_DATA_ROOT", "data")).resolve()
RAW_DIR = DATA_ROOT / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetManifest:
    slug: str
    source: str
    files: list[str] = field(default_factory=list)
    row_count: Optional[int] = None
    sha256: dict[str, str] = field(default_factory=dict)
    fetched_at: float = field(default_factory=time.time)

    def path(self) -> Path:
        return RAW_DIR / self.slug / "manifest.json"

    def save(self) -> None:
        self.path().parent.mkdir(parents=True, exist_ok=True)
        with open(self.path(), "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load_if_exists(cls, slug: str) -> Optional["DatasetManifest"]:
        p = RAW_DIR / slug / "manifest.json"
        if p.exists():
            with open(p) as f:
                return cls(**json.load(f))
        return None


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def recursive_unzip(root: Path, logger: logging.Logger, max_passes: int = 10) -> int:
    """
    Keep unzipping until no .zip files remain under root, or max_passes
    is hit. Needed because some Kaggle competitions (e.g. Jigsaw Toxic
    Comment) ship zips-inside-zips: the top-level competition zip
    contains train.csv.zip, test.csv.zip, etc rather than plain CSVs.
    """
    total_extracted = 0
    for _ in range(max_passes):
        zips = list(root.rglob("*.zip"))
        if not zips:
            break
        for zip_path in zips:
            logger.info("Extracting: %s", zip_path)
            subprocess.run(
                ["unzip", "-o", str(zip_path), "-d", str(zip_path.parent)],
                check=True,
                capture_output=True,
            )
            zip_path.unlink()
            total_extracted += 1
    remaining = list(root.rglob("*.zip"))
    if remaining:
        logger.warning("Stopped after %d passes, zips still remain: %s", max_passes, remaining)
    return total_extracted


class BaseDownloader:
    def __init__(self, slug: str, source: str):
        self.slug = slug
        self.source = source
        self.dest_dir = RAW_DIR / slug
        self.logger = logging.getLogger(slug)

    def already_fetched(self) -> bool:
        manifest = DatasetManifest.load_if_exists(self.slug)
        if manifest is None:
            return False
        for f in manifest.files:
            if not (self.dest_dir / f).exists():
                self.logger.warning("Manifest present but file %s missing, re-fetching", f)
                return False
        self.logger.info("Already fetched (%d files) — skipping download", len(manifest.files))
        return True

    def write_manifest(self, files: list[Path], row_count: Optional[int] = None) -> None:
        rel_files = [str(f.relative_to(self.dest_dir)) for f in files]
        checksums = {str(f.relative_to(self.dest_dir)): sha256_of(f) for f in files if f.stat().st_size < 500_000_000}
        manifest = DatasetManifest(
            slug=self.slug,
            source=self.source,
            files=rel_files,
            row_count=row_count,
            sha256=checksums,
        )
        manifest.save()
        self.logger.info("Wrote manifest with %d files", len(rel_files))


class KaggleDownloader(BaseDownloader):
    """
    Requires:
      pip install kaggle
      ~/.kaggle/kaggle.json with {"username": ..., "key": ...}
    """

    def __init__(self, slug: str, kaggle_ref: str, is_competition: bool = False):
        super().__init__(slug, source=f"kaggle:{kaggle_ref}")
        self.kaggle_ref = kaggle_ref
        self.is_competition = is_competition

    def fetch(self) -> Path:
        if self.already_fetched():
            return self.dest_dir
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["kaggle"]
        if self.is_competition:
            cmd += ["competitions", "download", "-c", self.kaggle_ref]
        else:
            cmd += ["datasets", "download", "-d", self.kaggle_ref]
        cmd += ["-p", str(self.dest_dir)]

        self.logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Kaggle download failed for {self.kaggle_ref}:\n{result.stderr}\n"
                f"Check that ~/.kaggle/kaggle.json exists and the ref is correct "
                f"(competitions require accepting rules on kaggle.com first)."
            )

        # unzip everything downloaded, including nested zips-inside-zips
        # (Kaggle competitions like jigsaw-toxic-comment ship train.csv.zip,
        # test.csv.zip, etc *inside* the top-level competition zip)
        recursive_unzip(self.dest_dir, logger=self.logger)

        files = [p for p in self.dest_dir.rglob("*") if p.is_file()]
        self.write_manifest(files)
        return self.dest_dir


class HFDownloader(BaseDownloader):
    """
    Requires: pip install datasets
    Some gated datasets need: huggingface-cli login
    """

    def __init__(
        self,
        slug: str,
        hf_path: str,
        hf_config: Optional[str] = None,
        split: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        super().__init__(slug, source=f"hf:{hf_path}" + (f"/{hf_config}" if hf_config else ""))
        self.hf_path = hf_path
        self.hf_config = hf_config
        self.split = split
        # Some older HF dataset repos still ship a Python loading script
        # instead of the newer scriptless parquet format. HF now requires
        # explicit opt-in to run that code (security change), or it fails
        # with "Dataset scripts are no longer supported". Only set this
        # True for datasets you've verified are legitimate/well-known.
        self.trust_remote_code = trust_remote_code

    def fetch(self) -> Path:
        if self.already_fetched():
            return self.dest_dir
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        from datasets import load_dataset

        self.logger.info(
            "Loading HF dataset %s (config=%s, split=%s, trust_remote_code=%s)",
            self.hf_path, self.hf_config, self.split, self.trust_remote_code,
        )
        ds = load_dataset(
            self.hf_path, self.hf_config, split=self.split,
            trust_remote_code=self.trust_remote_code,
        )

        out_file = self.dest_dir / "data.parquet"
        if hasattr(ds, "to_parquet"):
            ds.to_parquet(str(out_file))
        else:
            # DatasetDict without a fixed split: dump each split separately
            for split_name, split_ds in ds.items():
                split_ds.to_parquet(str(self.dest_dir / f"{split_name}.parquet"))

        files = [p for p in self.dest_dir.rglob("*.parquet")]
        row_count = len(ds) if hasattr(ds, "__len__") else None
        self.write_manifest(files, row_count=row_count)
        return self.dest_dir


class HTTPDownloader(BaseDownloader):
    def __init__(self, slug: str, url: str, filename: Optional[str] = None):
        super().__init__(slug, source=f"http:{url}")
        self.url = url
        self.filename = filename or url.split("/")[-1]

    def fetch(self, retries: int = 3, timeout: int = 60) -> Path:
        if self.already_fetched():
            return self.dest_dir
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.dest_dir / self.filename

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                self.logger.info("Downloading %s (attempt %d/%d)", self.url, attempt, retries)
                with requests.get(self.url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            f.write(chunk)
                break
            except Exception as e:
                last_err = e
                self.logger.warning("Attempt %d failed: %s", attempt, e)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Failed to download {self.url} after {retries} attempts: {last_err}")

        # auto-extract if archive
        if out_path.suffix in (".zip",):
            subprocess.run(["unzip", "-o", str(out_path), "-d", str(self.dest_dir)], check=True)
        elif out_path.suffixes[-2:] == [".tar", ".gz"] or out_path.suffix == ".tgz":
            subprocess.run(["tar", "-xzf", str(out_path), "-C", str(self.dest_dir)], check=True)

        files = [p for p in self.dest_dir.rglob("*") if p.is_file()]
        self.write_manifest(files)
        return self.dest_dir