#!/usr/bin/env python3
"""Join image labels to Tesseract page text for BERT and TF-IDF training.

The label CSV is the one b_jnk_creator writes (folder name, image name, value,
subtype). OCR_DIR is the main Tesseract output folder, with one text file per
chart folder:

    OCR_DIR/<folder name>/ocr/<folder name>.txt

The chart is found by the txt name, so a txt one level up or down still counts.

Pages inside it start with "===== 1.png =====". A label for 1.png takes that
page; if the extension differs (labels on .png, OCR on .jpg) the name without
extension is matched.

Writes data/raw/annotated_pages.jsonl. That file is added on top of the
existing training pages by scripts/train_bert.py and scripts/train_baseline.py.
Empty label rows are skipped.

Edit RUN CONFIG, then:

    python b_jnk/scripts/build_annotated_pages.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.tesseract_txt import split_pages  # noqa: E402

# ============================== RUN CONFIG ==============================
LABELS_CSV = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\blank_junk_labels.csv")
# Main Tesseract output folder: each chart is OCR_DIR/<folder>/ocr/<folder>.txt
OCR_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_tesseract")
OUT_JSONL = ROOT / "data" / "raw" / "annotated_pages.jsonl"
# ========================================================================

VALUES = {"KEEP", "BLANK", "JUNK"}
SUBTYPES = {
    "JUNK_INVOICE",
    "JUNK_COVER_PAGE",
    "JUNK_RECORD_REQUEST",
    "JUNK_INSTRUCTIONS",
    "JUNK_LETTER_FAX",
    "JUNK_OTHERS",
}


def _label(value: str, subtype: str) -> str | None:
    if value == "KEEP":
        return "KEEP"
    if value == "BLANK":
        return "BLANK"
    if value == "JUNK" and subtype in SUBTYPES:
        return subtype
    return None


def chart_files(ocr_dir: Path) -> dict[str, Path]:
    """Chart folder name -> its txt: OCR_DIR/<folder>/ocr/<folder>.txt.

    The txt name is the chart folder name. A txt without "===== page ====="
    markers is not a chart dump and is skipped.
    """
    found: dict[str, Path] = {}
    for path in sorted(ocr_dir.rglob("*.txt")):
        if not split_pages(path.read_text(encoding="utf-8", errors="replace")[:4000]):
            continue
        found.setdefault(path.stem.casefold(), path)
    return found


def index_ocr(ocr_dir: Path) -> tuple[dict[tuple[str, str], str], int]:
    """(folder, image name) and (folder, name without extension) -> page text, and chart count."""
    by_page: dict[tuple[str, str], str] = {}
    charts = chart_files(ocr_dir)
    for folder_key, path in charts.items():
        for name, text in split_pages(path.read_text(encoding="utf-8", errors="replace")):
            name = Path(name).name.casefold()
            by_page.setdefault((folder_key, name), text)
            by_page.setdefault((folder_key, Path(name).stem), text)
    return by_page, len(charts)


def text_for(folder: str, image_name: str, by_page: dict[tuple[str, str], str]) -> str | None:
    name = Path(image_name).name.casefold()
    folder_key = folder.casefold()
    found = by_page.get((folder_key, name))
    return found if found is not None else by_page.get((folder_key, Path(name).stem))


def build(labels_csv: Path, ocr_dir: Path, out_path: Path) -> dict[str, int]:
    if not labels_csv.is_file():
        raise FileNotFoundError(f"LABELS_CSV does not exist: {labels_csv}")
    if not ocr_dir.is_dir():
        raise FileNotFoundError(f"OCR_DIR does not exist: {ocr_dir}")
    by_page, chart_count = index_ocr(ocr_dir)
    ocr_folders = {folder for folder, _ in by_page}
    counts = {"written": 0, "unlabeled": 0, "bad_label": 0, "no_ocr": 0, "empty_text": 0, "charts": chart_count}
    missing_folders: dict[str, int] = {}
    missing_pages: list[str] = []
    rows: list[dict] = []
    with labels_csv.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            folder = (row.get("folder name") or "").strip()
            image = (row.get("image name") or "").strip()
            value = (row.get("value") or "").strip()
            subtype = (row.get("subtype") or "").strip()
            if not folder or not image:
                continue
            if not value:
                counts["unlabeled"] += 1
                continue
            if value not in VALUES:
                counts["bad_label"] += 1
                continue
            label = _label(value, subtype)
            if label is None:
                counts["bad_label"] += 1
                continue
            text = text_for(folder, image, by_page)
            if text is None:
                counts["no_ocr"] += 1
                if folder.casefold() in ocr_folders:
                    missing_pages.append(f"{folder}/{image}")
                else:
                    missing_folders[folder] = missing_folders.get(folder, 0) + 1
                continue
            if not text.strip():
                counts["empty_text"] += 1
                continue
            rows.append({
                "id": f"{folder}/{image}",
                "text": text,
                "label": label,
                "group": folder,
                "image": image,
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["written"] = len(rows)
    if missing_folders or missing_pages:
        print(f"txt charts found under OCR_DIR: {chart_count}")
        if missing_folders:
            print(f"{len(missing_folders)} label folders have no txt, e.g. {', '.join(list(missing_folders)[:5])}")
            sample = sorted(ocr_folders)[:5]
            print(f"  txt charts that do exist, e.g. {', '.join(sample) if sample else '(none)'}")
        if missing_pages:
            print(f"{len(missing_pages)} pages missing inside a chart txt, e.g. {', '.join(missing_pages[:5])}")
    return counts


def main() -> int:
    counts = build(LABELS_CSV, OCR_DIR, OUT_JSONL)
    print(f"wrote {counts['written']} pages -> {OUT_JSONL}")
    print(
        f"unlabeled {counts['unlabeled']}   bad label {counts['bad_label']}   "
        f"no ocr {counts['no_ocr']}   empty text {counts['empty_text']}"
    )
    return 0 if counts["written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
