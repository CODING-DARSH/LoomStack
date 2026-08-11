"""
INSPECT ONLY. No cleaning happens here.

Email spam sources -- separate from sms_spam (already cleaned).
Two structurally different sources, profiled separately:

  - Enron-Spam (HF: SetFit/enron_spam) -- clean parquet, already has
    ham/spam labels. Real corporate email text -- expect signature
    blocks, forwarded-message headers, legal disclaimers as recurring
    structural noise that SMS text never had.
  - Nazario phishing corpus -- raw .mbox file, NO labels and NO rows
    yet. Every message in this corpus is phishing by construction
    (it's a phishing-only collection), but the mbox needs to be
    PARSED (subject/body extraction from email headers) before it's
    even tabular data. This script parses it read-only for inspection
    -- actual parsing logic for the pipeline belongs in clean_split.py.

Usage:
    python inspect_dataset.py
"""

import logging
import mailbox
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inspect_email_spam")

ENRON_DIR = Path("data/raw/enron_spam")
NAZARIO_DIR = Path("data/raw/nazario_phishing")


def inspect_enron():
    logger.info("\n" + "=" * 70)
    logger.info("ENRON-SPAM")
    logger.info("=" * 70)

    path = ENRON_DIR / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the acquisition script first")

    df = pd.read_parquet(path)
    logger.info(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    logger.info(f"Dtypes:\n{df.dtypes.to_string()}")
    logger.info(f"Null counts:\n{df.isnull().sum().to_string()}")

    label_col_candidates = [c for c in df.columns if "label" in c.lower()]
    logger.info(f"\nLikely label column(s): {label_col_candidates}")
    for col in label_col_candidates:
        logger.info(f"{col} distinct values: {sorted(df[col].astype(str).unique())[:10]}")
        logger.info(f"{col} distribution:\n{df[col].value_counts().to_string()}")

    text_col_candidates = [c for c in df.columns if c.lower() in ("text", "message", "body", "email")]
    logger.info(f"\nLikely text column(s): {text_col_candidates}")

    if text_col_candidates:
        text_col = text_col_candidates[0]
        lengths = df[text_col].astype(str).str.len()
        logger.info(f"\n=== TEXT LENGTH ({text_col}) ===")
        logger.info(f"min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}")

        empty = (df[text_col].astype(str).str.strip() == "").sum()
        logger.info(f"Empty/whitespace-only rows: {empty}")

        logger.info(f"\n=== EXACT DUPLICATE TEXT ===")
        logger.info(f"Duplicate rows: {df[text_col].duplicated().sum()}")

        logger.info(f"\n=== EMAIL-SPECIFIC STRUCTURAL ARTIFACTS ===")
        forwarded = df[text_col].astype(str).str.contains(
            r"-{2,}\s*Forwarded [Mm]essage|-{2,}\s*Original [Mm]essage", regex=True
        ).sum()
        subject_header = df[text_col].astype(str).str.contains(r"^Subject:", regex=True).sum()
        from_header = df[text_col].astype(str).str.contains(r"^From:|^To:|^Sent:", regex=True).sum()
        has_url = df[text_col].astype(str).str.contains(r"https?://|www\.", regex=True).sum()
        has_html_tag = df[text_col].astype(str).str.contains(r"<[^>]+>", regex=True).sum()
        has_newline = df[text_col].astype(str).str.contains("\n", regex=False).sum()
        logger.info(f"Rows with 'Forwarded/Original Message' headers: {forwarded}")
        logger.info(f"Rows with embedded 'Subject:' line: {subject_header}")
        logger.info(f"Rows with embedded From:/To:/Sent: lines: {from_header}")
        logger.info(f"Rows with URLs: {has_url}")
        logger.info(f"Rows with HTML tags: {has_html_tag}")
        logger.info(f"Rows with literal newlines: {has_newline}")

        logger.info(f"\n=== LEGAL DISCLAIMER / SIGNATURE BOILERPLATE CHECK ===")
        disclaimer = df[text_col].astype(str).str.contains(
            r"confidential|proprietary|intended recipient|disclaimer", regex=True, case=False
        ).sum()
        logger.info(f"Rows containing common corporate-email disclaimer language: {disclaimer}")


def inspect_nazario():
    logger.info("\n" + "=" * 70)
    logger.info("NAZARIO PHISHING (raw .mbox -- parsed here for inspection only)")
    logger.info("=" * 70)

    mbox_path = NAZARIO_DIR / "phishing0.mbox"
    if not mbox_path.exists():
        raise FileNotFoundError(f"{mbox_path} not found -- run the acquisition script first")

    logger.info(f"File size: {mbox_path.stat().st_size / 1e6:.2f} MB")

    mbox = mailbox.mbox(str(mbox_path))
    messages = list(mbox)
    logger.info(f"Total messages in mbox: {len(messages)}")

    if not messages:
        logger.info("No messages parsed -- mbox may be empty or in an unexpected format")
        return

    header_keys_seen = set()
    for msg in messages[:50]:
        header_keys_seen.update(msg.keys())
    logger.info(f"\nHeader keys seen (first 50 messages): {sorted(header_keys_seen)}")

    subjects_present = sum(1 for m in messages if m.get("Subject"))
    logger.info(f"Messages with a Subject header: {subjects_present} / {len(messages)}")

    from_present = sum(1 for m in messages if m.get("From"))
    logger.info(f"Messages with a From header: {from_present} / {len(messages)}")

    date_present = sum(1 for m in messages if m.get("Date"))
    logger.info(f"Messages with a Date header: {date_present} / {len(messages)}")

    def extract_body(msg):
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            parts.append(payload.decode(errors="replace"))
                    except Exception:
                        pass
            return "\n".join(parts)
        else:
            try:
                payload = msg.get_payload(decode=True)
                return payload.decode(errors="replace") if payload else ""
            except Exception:
                return ""

    bodies = [extract_body(m) for m in messages]
    empty_bodies = sum(1 for b in bodies if not b.strip())
    logger.info(f"\nMessages with an empty extracted body: {empty_bodies} / {len(messages)}")

    body_lengths = [len(b) for b in bodies if b.strip()]
    if body_lengths:
        logger.info(f"Body length -- min: {min(body_lengths)}, max: {max(body_lengths)}, "
                     f"mean: {sum(body_lengths)/len(body_lengths):.1f}")

    content_types = set()
    multipart_count = 0
    for m in messages:
        content_types.add(m.get_content_type())
        if m.is_multipart():
            multipart_count += 1
    logger.info(f"\nDistinct content types seen: {sorted(content_types)}")
    logger.info(f"Multipart messages: {multipart_count} / {len(messages)}")

    html_only_count = 0
    for m in messages:
        if m.is_multipart():
            types = {part.get_content_type() for part in m.walk()}
            if "text/html" in types and "text/plain" not in types:
                html_only_count += 1
        elif m.get_content_type() == "text/html":
            html_only_count += 1
    logger.info(f"Messages with ONLY text/html (no text/plain part -- needs HTML stripping "
                f"to get body text): {html_only_count} / {len(messages)}")

    logger.info(f"\n=== DUPLICATE MESSAGE CHECK (by extracted body) ===")
    non_empty_bodies = [b for b in bodies if b.strip()]
    logger.info(f"Duplicate bodies: {len(non_empty_bodies) - len(set(non_empty_bodies))}")


def compare_and_plan():
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-SOURCE MERGE PLANNING")
    logger.info("=" * 70)
    logger.info("Before writing clean_split.py, decide:")
    logger.info("  - Enron-Spam already has ham/spam labels; Nazario is phishing-only")
    logger.info("    (single class, no ham examples of its own). Unified schema needs")
    logger.info("    deciding: 3-class (ham/spam/phishing)? Or binary, with phishing")
    logger.info("    folded into 'spam'? Phishing is arguably a distinct attack pattern")
    logger.info("    from bulk spam, worth its own label if the model needs to")
    logger.info("    distinguish them.")
    logger.info("  - Nazario has NO ham examples -- if merged with Enron, ham rows all")
    logger.info("    come from one source only. Check this doesn't silently bias what")
    logger.info("    'ham' looks like (Enron corporate email style) vs what 'spam' or")
    logger.info("    'phishing' look like (bulk marketing + Nazario's phishing style).")
    logger.info("  - Nazario body extraction: multipart/HTML-only messages need HTML")
    logger.info("    stripped to get real body text -- same HTML-tag-stripping approach")
    logger.info("    used in jigsaw_toxic, but this is EMAIL html, likely much heavier")
    logger.info("    markup (tables, inline styles) than a wiki comment ever had.")
    logger.info("  - Enron's forwarded-message / disclaimer boilerplate: strip, or")
    logger.info("    leave as legitimate structural signal for a corporate-email ham")
    logger.info("    class?")
    logger.info("  - No native train/val/test split on either source -- fresh")
    logger.info("    stratified split needed once schema is unified.")


def main():
    inspect_enron()
    inspect_nazario()
    compare_and_plan()


if __name__ == "__main__":
    main()