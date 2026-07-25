"""
Run this AFTER data acquisition scripts have already downloaded data.

What it does, per dataset folder under data/raw/<slug>/:
  1. Recursively extracts any .zip found (including zips-inside-zips,
     e.g. Jigsaw ships train.csv.zip / test.csv.zip inside the main zip)
  2. Deletes the .zip files themselves once extracted (keeps only the
     actual data files)
  3. Removes common junk: __MACOSX/ folders, .DS_Store, Thumbs.db,
     empty directories left behind after extraction

Usage:
    python cleanup_extract.py                 # cleans everything under data/raw
    python cleanup_extract.py --only jigsaw   # only folders matching a filter
    python cleanup_extract.py --dry-run        # show what would happen, no changes
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("cleanup_extract")

JUNK_NAMES = {"__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini"}
RAW_DIR = Path("data/raw")


def recursive_unzip(root: Path, dry_run: bool = False, max_passes: int = 10) -> int:
    """Keep unzipping until no .zip files remain or max_passes is hit."""
    total_extracted = 0
    for _ in range(max_passes):
        zips = list(root.rglob("*.zip"))
        if not zips:
            break
        for zip_path in zips:
            logger.info("Extracting: %s", zip_path)
            if not dry_run:
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


def remove_junk(root: Path, dry_run: bool = False) -> int:
    removed = 0
    for path in list(root.rglob("*")):
        if path.name in JUNK_NAMES:
            logger.info("Removing junk: %s", path)
            if not dry_run:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            removed += 1
    return removed


def remove_empty_dirs(root: Path, dry_run: bool = False) -> int:
    removed = 0
    # walk bottom-up so nested empty dirs get removed too
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            logger.info("Removing empty dir: %s", path)
            if not dry_run:
                path.rmdir()
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Only process dataset folders whose name contains this substring")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without changing anything")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} does not exist — run this from the project root where data/raw/ lives")

    dataset_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if args.only:
        dataset_dirs = [d for d in dataset_dirs if args.only in d.name]

    if not dataset_dirs:
        logger.warning("No matching dataset folders found under %s", RAW_DIR)
        return

    grand_total_zips = 0
    grand_total_junk = 0
    grand_total_empty = 0

    for d in dataset_dirs:
        logger.info("=" * 70)
        logger.info("Processing: %s", d)
        n_zips = recursive_unzip(d, dry_run=args.dry_run)
        n_junk = remove_junk(d, dry_run=args.dry_run)
        n_empty = remove_empty_dirs(d, dry_run=args.dry_run)
        logger.info("%s -> extracted %d zip(s), removed %d junk file(s), %d empty dir(s)", d.name, n_zips, n_junk, n_empty)
        grand_total_zips += n_zips
        grand_total_junk += n_junk
        grand_total_empty += n_empty

    logger.info("=" * 70)
    logger.info(
        "DONE%s — total zips extracted: %d, junk removed: %d, empty dirs removed: %d",
        " (dry run, nothing changed)" if args.dry_run else "",
        grand_total_zips, grand_total_junk, grand_total_empty,
    )


if __name__ == "__main__":
    main()