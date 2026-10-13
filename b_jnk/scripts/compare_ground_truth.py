#!/usr/bin/env python3
"""Flag OCR pages like infer.py and compare BLANK / JUNK with a ground truth CSV.

Ground truth columns: Chart_Name, OriginalSequence, Blank Page, Junk Page (spaces,
underscores and case in the names do not matter).
Original Sequence is the page number, matched to the image name (3.png = page 3).
Blank Page / Junk Page count as set for Yes, Y, True, 1 or X.

Only charts that are in the ground truth are flagged. Inside those charts a page
the ground truth does not list counts as KEEP.

The CSV has one row per page that the model or the ground truth calls BLANK or
JUNK (KEEP pages on both sides are left out):

  folder name, image name, model, model subtype, ground truth, result

  result: correct     both say the same of BLANK / JUNK
          wrong type  one says BLANK, the other JUNK
          missed      ground truth BLANK / JUNK, model KEEP
          false flag  model BLANK / JUNK, ground truth KEEP
          no OCR page ground truth BLANK / JUNK, page not found in the OCR

Edit RUN CONFIG, then:
    python b_jnk/scripts/compare_ground_truth.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import infer  # noqa: E402
import infer_json  # noqa: E402
from src.inference.predict import PageClassifierService  # noqa: E402
from src.models.bert_classifier import bert_available  # noqa: E402
from src.models.decision import DecisionConfig  # noqa: E402

# ============================== RUN CONFIG ==============================
# Folder of chart folders: <folder>/<chart>/... with the chart's OCR inside.
OCR_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_tesseract")
# "txt": Tesseract txt with "===== 1.png =====" page markers.
# "json": Docling or Azure Document Intelligence JSON (page n = n.png).
OCR_FORMAT = "txt"
GROUND_TRUTH_CSV = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\ground_truth.csv")
# Empty = ground_truth_comparison.csv next to GROUND_TRUTH_CSV.
OUT_CSV: Path | None = None
# ========================================================================

YES = {"yes", "y", "true", "1", "x"}
COLUMNS = ["folder name", "image name", "model", "model subtype", "ground truth", "result"]


def _key(name: str) -> str:
    return name.strip().casefold()


def _page_number(image_name: str) -> int | None:
    digits = re.findall(r"\d+", Path(image_name).stem)
    return int(digits[-1]) if digits else None


def _column(header: list[str], want: str) -> str:
    # "Original Sequence", "OriginalSequence" and "original_sequence" are one column.
    squash = lambda s: re.sub(r"[^a-z0-9]", "", s.casefold())  # noqa: E731
    for name in header:
        if squash(name) == squash(want):
            return name
    raise SystemExit(f"{GROUND_TRUTH_CSV}: column '{want}' not found (columns: {', '.join(header)})")


def load_ground_truth(path: Path) -> tuple[dict[str, dict[int, str]], dict[str, str]]:
    """{chart: {page number: "BLANK" | "JUNK" | "BLANK+JUNK" | "KEEP"}}, and each chart's name as written."""
    if not path.is_file():
        raise SystemExit(f"Ground truth not found: {path}")
    truth: dict[str, dict[int, str]] = {}
    names: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        chart_col = _column(header, "chart name")
        seq_col = _column(header, "original sequence")
        blank_col = _column(header, "blank page")
        junk_col = _column(header, "junk page")
        for row in reader:
            chart = (row.get(chart_col) or "").strip()
            seq = (row.get(seq_col) or "").strip()
            if not chart or not re.fullmatch(r"\d+(\.0+)?", seq):
                continue
            blank = _key(row.get(blank_col) or "") in YES
            junk = _key(row.get(junk_col) or "") in YES
            label = "BLANK+JUNK" if blank and junk else "BLANK" if blank else "JUNK" if junk else "KEEP"
            truth.setdefault(_key(chart), {})[int(float(seq))] = label
            names.setdefault(_key(chart), chart)
    return truth, names


def chart_of(page_id: str) -> str:
    parts = [p for p in Path(page_id.replace("\\", "/")).parts[:-1] if p.casefold() != "ocr"]
    return parts[0] if parts else ""


def load_pages() -> list[dict]:
    if OCR_FORMAT == "json":
        return infer_json.load_pages(OCR_DIR)
    if OCR_FORMAT != "txt":
        raise SystemExit(f'OCR_FORMAT must be "txt" or "json", got: {OCR_FORMAT!r}')
    pages: list[dict] = []
    for fp in sorted(OCR_DIR.rglob("*.txt")):
        rel = fp.relative_to(OCR_DIR)
        prefix = f"{rel.parent.as_posix()}/" if rel.parent != Path(".") else ""
        pages.extend(infer._pages_from_rapid_txt(fp.read_text(encoding="utf-8", errors="replace"), prefix=prefix))
    return pages


def build_service() -> PageClassifierService:
    cfg = infer._load_cfg(Path(infer.CONFIG_PATH))
    if not bert_available(infer.BERT_DIR):
        raise SystemExit(f"BERT model not found: {infer.BERT_DIR}")
    from src.models.bert_classifier import BertPageClassifier

    print(f"Loading BERT model: {infer.BERT_DIR}", flush=True)
    return PageClassifierService(
        BertPageClassifier.load(infer.BERT_DIR),
        model_version=cfg["model_version"] + "+bert",
        decision=DecisionConfig(**cfg["decision"]),
        min_dictionary_words=cfg["routing"]["min_dictionary_words"],
        route_short_pages=not cfg.get("bert", {}).get("decide_short_pages", False),
    )


def result_of(model: str, truth: str) -> str:
    if model == "KEEP":
        return "missed"
    if truth == "KEEP":
        return "false flag"
    return "correct" if model in truth.split("+") else "wrong type"


def main() -> int:
    if not OCR_DIR.is_dir():
        raise SystemExit(f"OCR folder not found: {OCR_DIR}")
    truth, names = load_ground_truth(GROUND_TRUTH_CSV)
    out = OUT_CSV or GROUND_TRUTH_CSV.with_name("ground_truth_comparison.csv")
    print(f"Ground truth: {len(truth)} charts", flush=True)

    print(f"Reading OCR: {OCR_DIR}", flush=True)
    pages = [p for p in load_pages() if _key(chart_of(p["page_id"])) in truth]
    charts_found = {_key(chart_of(p["page_id"])) for p in pages}
    print(f"{len(pages)} pages in {len(charts_found)} ground-truth charts "
          f"({len(truth) - len(charts_found)} ground-truth charts have no OCR)", flush=True)

    service = build_service()
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for i, page in enumerate(pages, start=1):
        folder = chart_of(page["page_id"])
        image = Path(page["page_id"]).name
        number = _page_number(image)
        r = service.predict_one(page["page_id"], page.get("ocr_text", ""), content_meta=page.get("content_meta"))
        gt = truth[_key(folder)].get(number, "KEEP") if number is not None else "KEEP"
        if number is not None:
            seen.add((_key(folder), number))
        if r.flag != "KEEP" or gt != "KEEP":
            rows.append({
                "folder name": folder, "image name": image,
                "model": r.flag, "model subtype": r.subclass or "",
                "ground truth": gt, "result": result_of(r.flag, gt),
            })
        if i % 50 == 0 or i == len(pages):
            print(f"  [{i}/{len(pages)}] flagged", flush=True)

    for chart, chart_pages in truth.items():
        if chart not in charts_found:
            continue
        for number, gt in sorted(chart_pages.items()):
            if gt != "KEEP" and (chart, number) not in seen:
                rows.append({
                    "folder name": names[chart], "image name": f"page {number}",
                    "model": "", "model subtype": "", "ground truth": gt, "result": "no OCR page",
                })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["result"] for row in rows)
    truth_flagged = sum(1 for row in rows if row["ground truth"] != "KEEP")
    caught = counts["correct"] + counts["wrong type"]
    print(f"\nWrote {len(rows)} rows -> {out}")
    print(f"Ground truth BLANK/JUNK pages: {truth_flagged}  caught: {caught}  "
          f"missed: {counts['missed']}  no OCR page: {counts['no OCR page']}")
    print(f"Model BLANK/JUNK on ground-truth KEEP pages (false flags): {counts['false flag']}")
    for name in ("correct", "wrong type", "missed", "false flag", "no OCR page"):
        print(f"  {name}: {counts[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
