#!/usr/bin/env python3
"""Add pages the model got wrong to data/raw/correction_pages.jsonl.

Each page's text is taken from the chart's Tesseract dump:

    OCR_DIR/<folder>/ocr/<folder>_prelim.txt     (page N is "===== N.png =====")

KEEP pages are labelled KEEP; JUNK pages get their junk subtype from the word rules
(assign_junk_subtype). Rows already in the file stay; a page listed again replaces its
old row. scripts/train_bert.py --continue-model trains on every row of the file.

Edit RUN CONFIG, then:

    python b_jnk/scripts/add_corrections.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_annotated_pages as bap  # noqa: E402
from src.preprocessing.page_subclass import assign_junk_subtype  # noqa: E402

# ============================== RUN CONFIG ==============================
# Main Tesseract output folder (or a list of them): OCR_DIR/<folder>/ocr/<folder>_prelim.txt
OCR_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_tesseract")
# End of the txt name after the folder name ("" for <folder>.txt).
TXT_SUFFIX = "_prelim"
OUT_JSONL = ROOT / "data" / "raw" / "correction_pages.jsonl"

# (chart folder, page numbers, KEEP or JUNK)
PAGES = [
    ("70090414_56516847", [16], "KEEP"),
    ("70098989_56517716", [80], "KEEP"),
    ("70088054_56514207", [3], "KEEP"),
    ("70085533_56513856", [8], "JUNK"),
    ("70085533_56513856", [18], "KEEP"),
    ("70088054_56514207", [39, 80, 82], "JUNK"),
    ("70084703_56513925", [1], "JUNK"),
    ("70084703_56513925", [10], "KEEP"),
    ("70075873_56510270", [23, 26], "KEEP"),
    ("70076113_56509933", [10], "KEEP"),
    ("67993286_56072069", [28], "JUNK"),
    ("68005641_56078298", [14], "JUNK"),
    ("68007166_56078189", [47], "JUNK"),
    ("68008419_56079696", [9, 38], "JUNK"),
    ("70070759_56510439", [53], "KEEP"),
    ("67992653_56072370", [87], "KEEP"),
    ("67992653_56072370", [89], "JUNK"),
    ("66945142_55331561", [34], "JUNK"),
    ("66945675_55331557", [1], "JUNK"),
    ("67989431_56075859", [43], "JUNK"),
    ("67989498_56074747", [2], "KEEP"),
    ("67989498_56074747", [34], "JUNK"),
    ("67989627_56076335", [33], "KEEP"),
    ("67989680_56076434", [63], "JUNK"),
    ("67992035_56071722", [36], "KEEP"),
    ("67992303_56072120", [6], "JUNK"),
    ("67992303_56072120", [15], "KEEP"),
    ("67992620_56072075", [11], "KEEP"),
]
# ========================================================================


def main() -> int:
    bap.TXT_SUFFIX = TXT_SUFFIX
    by_page: dict[tuple[str, str], str] = {}
    for d in OCR_DIR if isinstance(OCR_DIR, (list, tuple)) else [OCR_DIR]:
        if not Path(d).is_dir():
            raise SystemExit(f"OCR_DIR does not exist: {d}")
        pages, n = bap.index_ocr(Path(d))
        print(f"{d}: {n} charts")
        for key, text in pages.items():
            by_page.setdefault(key, text)

    new_rows: list[dict] = []
    missing: list[str] = []
    for folder, numbers, value in PAGES:
        for n in numbers:
            page_id = f"{folder}/{n}.png"
            text = bap.text_for(folder, f"{n}.png", by_page)
            if not text or not text.strip():
                missing.append(page_id)
                continue
            label = "KEEP" if value == "KEEP" else assign_junk_subtype(text).subclass
            new_rows.append({"id": page_id, "text": text, "label": label, "group": folder})
            preview = " ".join(text.split())[:110]
            print(f"  {page_id:28s} {label:20s} {preview}")

    new_ids = {r["id"].casefold() for r in new_rows}
    old_rows = []
    if OUT_JSONL.is_file():
        old_rows = [json.loads(line) for line in OUT_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in old_rows if str(r.get("id", "")).casefold() not in new_ids]
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for row in kept + new_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nadded {len(new_rows)} pages, kept {len(kept)} earlier rows -> {OUT_JSONL} ({len(kept) + len(new_rows)} rows)")
    if missing:
        print(f"NOT FOUND ({len(missing)}): no page text in OCR_DIR for {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
