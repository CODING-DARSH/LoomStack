"""
ONE-OFF DIAGNOSTIC -- prints ONLY the matched markdown-link substring,
never the surrounding row text. Changes nothing.

Usage:
    python diagnose_markdown_link.py
"""

import re
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed/suicide_detection")
MD_PATTERN = re.compile(r"\\?\[([^\]]+)\]\\?\(([^)]+)\)")


def main():
    df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    matches = df[df["text"].str.contains(MD_PATTERN, regex=True)]

    print(f"Rows matching: {len(matches)}")
    for _, row in matches.iterrows():
        found = MD_PATTERN.findall(row["text"])
        print(f"Matched groups (link text, url): {found}")

        for m in MD_PATTERN.finditer(row["text"]):
            start = max(0, m.start() - 15)
            end = min(len(row["text"]), m.end() + 15)
            span = row["text"][start:end]
            print(f"Span with minimal context: ...{span}...")
            print(f"Char codes of that span: {[hex(ord(c)) for c in span[:30]]}")


if __name__ == "__main__":
    main()