#!/usr/bin/env python3
"""List tagged pages where a structural rule disagrees with the tag, to re-check before training.

Rules used: never-junk pages (patient forms, clinical letters, signature pages, pages with
clinical values) say KEEP; header-only pages and record-retrieval paperwork say JUNK. A tag
that disagrees is either a labeling slip or a page tagged under an older policy (section
dividers, tables of contents, and EMR index pages are junk now).

Reads data/raw/annotated_pages.jsonl (built by build_annotated_pages.py) and writes a review
sheet b_jnk_creator/junk_tagger.py can open: every listed page has Junk Page = Yes, so the
tagger shows all of them, and pressing K, B or a junk type updates the label in place.

    python b_jnk/scripts/label_rule_conflicts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.page_subclass import admin_page_reason  # noqa: E402
from src.preprocessing.protocol import always_keep_tag  # noqa: E402

TAGGED = ROOT / "data" / "raw" / "annotated_pages.jsonl"
OUT_XLSX = ROOT / "reports" / "label_rule_conflicts.xlsx"


def main() -> int:
    if not TAGGED.is_file():
        raise SystemExit(f"{TAGGED} missing: run build_annotated_pages.py first")
    rows = []
    for line in TAGGED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        text, label = page["text"], page["label"]
        keep_tag = always_keep_tag(text)
        admin = None if keep_tag else admin_page_reason(text)
        if keep_tag and label != "KEEP":
            rule, why = "KEEP", keep_tag
        elif admin and label == "KEEP":
            rule, why = admin[0], admin[1]
        else:
            continue
        folder, _, image = page["id"].partition("/")
        rows.append([folder, image, "No", "Yes", label, rule, why, " ".join(text.split())[:200]])

    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Folder Name", "ID", "Blank Page", "Junk Page", "your_tag", "rule_says", "rule", "text"])
    for row in rows:
        ws.append(row)
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    print(f"{len(rows)} tagged pages disagree with a rule -> {OUT_XLSX}")
    print("Open it with b_jnk_creator/junk_tagger.py (REVIEW_SHEET), re-tag, then rerun build_annotated_pages.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
