"""
One-off inspection script: dumps the FULL structure of the downloaded
NSFW parquet file, not just top-level column names -- checks dtypes,
a sample row, and specifically looks inside the 'image' field since
HF's imagefolder format sometimes encodes the label in the image's
file path rather than as a separate top-level column.
"""

from pathlib import Path
import pandas as pd

PARQUET_PATH = Path("data/raw/nsfw_images/data.parquet")

def main():
    if not PARQUET_PATH.exists():
        # fall back to searching for whatever parquet is actually there
        candidates = list(Path("data/raw/nsfw_images").rglob("*.parquet"))
        if not candidates:
            print(f"No parquet file found under data/raw/nsfw_images/")
            return
        path = candidates[0]
    else:
        path = PARQUET_PATH

    print(f"Reading: {path}")
    df = pd.read_parquet(path)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")

    print(f"\n--- First row, full detail ---")
    first_row = df.iloc[0]
    for col in df.columns:
        val = first_row[col]
        if isinstance(val, dict):
            print(f"{col} (dict): keys={list(val.keys())}")
            for k, v in val.items():
                v_repr = f"<bytes, len={len(v)}>" if isinstance(v, bytes) else v
                print(f"    {col}.{k} = {v_repr}")
        elif isinstance(val, bytes):
            print(f"{col} (bytes): len={len(val)}")
        else:
            print(f"{col}: {val}")

    print(f"\n--- Checking for any label-like info across all rows ---")
    for col in df.columns:
        sample_val = df[col].iloc[0]
        if isinstance(sample_val, dict) and "path" in sample_val:
            paths = df[col].apply(lambda x: x.get("path") if isinstance(x, dict) else None)
            print(f"Sample paths from '{col}.path':")
            print(paths.head(10).to_string())

if __name__ == "__main__":
    main()