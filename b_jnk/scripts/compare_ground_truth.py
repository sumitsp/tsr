#!/usr/bin/env python3
"""Flag OCR pages like infer.py with two models and compare them with a ground truth CSV.

Ground truth columns: Chart_Name, OriginalSequence, Blank Page, Junk Page (spaces,
underscores and case in the names do not matter).
Original Sequence is the page number, matched to the image name (3.png = page 3).
Blank Page / Junk Page count as set for Yes, Y, True, 1 or X.

Only charts that are in the ground truth are flagged. Inside those charts a page
the ground truth does not list counts as KEEP. For txt OCR, only
<chart>/<chart>_prelim.txt is read (not other txt files in the folder).

Both models (NEW_MODEL_DIR: the latest, OLD_MODEL_DIR: the one saved before it) flag
every page, with the word rules, as infer.py runs them. The CSV has one row per page
where the two models give a different flag (KEEP / BLANK / JUNK):

  folder name, image name, new model, new subtype, old model, old subtype,
  ground truth, new result, old result, right

  result: correct     agrees with the ground truth (KEEP, or the same of BLANK / JUNK)
          wrong type  one says BLANK, the other JUNK
          missed      ground truth BLANK / JUNK, model KEEP
          false flag  model BLANK / JUNK, ground truth KEEP
  right:  new / old / neither

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
# The latest model (models/bert_page.candidate if it was not saved over bert_page).
NEW_MODEL_DIR = ROOT / "models" / "bert_page"
# The model saved before it.
OLD_MODEL_DIR = ROOT / "models" / "bert_page_1014_backup"
# Empty = model_comparison.csv next to GROUND_TRUTH_CSV.
OUT_CSV: Path | None = None
# How many ground-truth chart folders to run. None or 0 = all of them.
MAX_FOLDERS: int | None = 100
# ========================================================================

YES = {"yes", "y", "true", "1", "x"}
COLUMNS = ["folder name", "image name", "new model", "new subtype", "old model", "old subtype",
           "ground truth", "new result", "old result", "right"]


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


def _prelim_txt(chart_dir: Path) -> Path | None:
    name = f"{chart_dir.name}_prelim.txt"
    direct = chart_dir / name
    if direct.is_file():
        return direct
    hits = sorted(p for p in chart_dir.rglob(name) if p.is_file())
    return hits[0] if hits else None


def selected_charts(truth: dict[str, dict[int, str]]) -> list[str]:
    folders = sorted(
        p.name for p in OCR_DIR.iterdir() if p.is_dir() and _key(p.name) in truth
    )
    if MAX_FOLDERS:
        folders = folders[:MAX_FOLDERS]
    return folders


def load_pages(folders: list[str]) -> list[dict]:
    if OCR_FORMAT == "json":
        wanted = {_key(name) for name in folders}
        return [p for p in infer_json.load_pages(OCR_DIR) if _key(chart_of(p["page_id"])) in wanted]
    if OCR_FORMAT != "txt":
        raise SystemExit(f'OCR_FORMAT must be "txt" or "json", got: {OCR_FORMAT!r}')
    pages: list[dict] = []
    missing: list[str] = []
    for name in folders:
        fp = _prelim_txt(OCR_DIR / name)
        if fp is None:
            missing.append(name)
            continue
        rel = fp.relative_to(OCR_DIR)
        prefix = f"{rel.parent.as_posix()}/" if rel.parent != Path(".") else ""
        pages.extend(infer._pages_from_rapid_txt(fp.read_text(encoding="utf-8", errors="replace"), prefix=prefix))
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        print(f"{len(missing)} folders have no <folder>_prelim.txt: {shown}", flush=True)
    return pages


def build_service(model_dir: Path) -> PageClassifierService:
    cfg = infer._load_cfg(Path(infer.CONFIG_PATH))
    if not bert_available(model_dir):
        raise SystemExit(f"BERT model not found: {model_dir}")
    from src.models.bert_classifier import BertPageClassifier

    print(f"Loading BERT model: {model_dir}", flush=True)
    return PageClassifierService(
        BertPageClassifier.load(model_dir),
        model_version=cfg["model_version"] + "+bert",
        decision=DecisionConfig(**cfg["decision"]),
        min_dictionary_words=cfg["routing"]["min_dictionary_words"],
        route_short_pages=not cfg.get("bert", {}).get("decide_short_pages", False),
    )


def result_of(model: str, truth: str) -> str:
    if model == "KEEP":
        return "correct" if truth == "KEEP" else "missed"
    if truth == "KEEP":
        return "false flag"
    return "correct" if model in truth.split("+") else "wrong type"


def main() -> int:
    if not OCR_DIR.is_dir():
        raise SystemExit(f"OCR folder not found: {OCR_DIR}")
    truth, _ = load_ground_truth(GROUND_TRUTH_CSV)
    out = OUT_CSV or GROUND_TRUTH_CSV.with_name("model_comparison.csv")
    folders = selected_charts(truth)
    cap = f"first {MAX_FOLDERS}" if MAX_FOLDERS else "all"
    print(f"Ground truth: {len(truth)} charts  running: {cap} ({len(folders)} folders)", flush=True)

    print(f"Reading OCR: {OCR_DIR}  (*_prelim.txt only)", flush=True)
    pages = load_pages(folders)
    charts_found = {_key(chart_of(p["page_id"])) for p in pages}
    print(f"{len(pages)} pages in {len(charts_found)} folders "
          f"({len(folders) - len(charts_found)} selected folders have no OCR)", flush=True)

    new_service = build_service(NEW_MODEL_DIR)
    old_service = build_service(OLD_MODEL_DIR)
    rows: list[dict] = []
    tally = {"new": Counter(), "old": Counter()}
    for i, page in enumerate(pages, start=1):
        folder = chart_of(page["page_id"])
        image = Path(page["page_id"]).name
        number = _page_number(image)
        text, meta = page.get("ocr_text", ""), page.get("content_meta")
        new = new_service.predict_one(page["page_id"], text, content_meta=meta)
        old = old_service.predict_one(page["page_id"], text, content_meta=meta)
        gt = truth[_key(folder)].get(number, "KEEP") if number is not None else "KEEP"
        new_result, old_result = result_of(new.flag, gt), result_of(old.flag, gt)
        tally["new"][new_result] += 1
        tally["old"][old_result] += 1
        if new.flag != old.flag:
            right = "new" if new_result == "correct" else "old" if old_result == "correct" else "neither"
            rows.append({
                "folder name": folder, "image name": image,
                "new model": new.flag, "new subtype": new.subclass or "",
                "old model": old.flag, "old subtype": old.subclass or "",
                "ground truth": gt, "new result": new_result, "old result": old_result, "right": right,
            })
        if i % 50 == 0 or i == len(pages):
            print(f"  [{i}/{len(pages)}] flagged, {len(rows)} differ", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    right = Counter(row["right"] for row in rows)
    print(f"\nWrote {len(rows)} pages where the models differ -> {out}")
    print(f"  new model right: {right['new']}   old model right: {right['old']}   neither: {right['neither']}")
    print(f"\nAll {len(pages)} pages, against the ground truth:")
    print(f"  {'':12s} {'correct':>8s} {'missed':>8s} {'false flag':>11s} {'wrong type':>11s}")
    for name in ("new", "old"):
        c = tally[name]
        print(f"  {name + ' model':12s} {c['correct']:8d} {c['missed']:8d} {c['false flag']:11d} {c['wrong type']:11d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
