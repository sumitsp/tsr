#!/usr/bin/env python3
"""Pick the untagged pages most worth a human look, as a review sheet for b_jnk_creator.

Runs the current pipeline (BERT and the word rules, as in infer.py)
over Tesseract chart txt files and keeps a page when the code is unsure or calls
it junk / blank, plus a small random share of confident KEEP pages, which shows
the junk the models still miss. Pages already in LABELS_CSV are skipped.

The sheet has Folder Name, ID, Junk Page, Blank Page columns: every picked page
is Junk Page = Yes, so b_jnk_creator/junk_tagger.py opens all of them and you
press K, B, or the junk type. Tagging these teaches the model far more per page
than tagging pages in chart order.

Edit RUN CONFIG, then:
    python b_jnk/scripts/find_junk_candidates.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import infer  # noqa: E402
from build_annotated_pages import chart_files  # noqa: E402
from src.inference.predict import PageClassifierService  # noqa: E402
from src.models.bert_classifier import bert_available  # noqa: E402
from src.models.decision import DecisionConfig  # noqa: E402
from src.preprocessing.page_subclass import junk_trigger_groups  # noqa: E402
from src.preprocessing.tesseract_txt import split_pages  # noqa: E402

# ============================== RUN CONFIG ==============================
# Main Tesseract output folder: each chart is OCR_DIR/<folder>/ocr/<folder>_prelim.txt
OCR_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_tesseract")
LABELS_CSV = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\blank_junk_labels.csv")
OUT_XLSX = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\junk_candidates.xlsx")
# Model junk probability at or above this makes a KEEP page a candidate.
MIN_P_JUNK = 0.15
# Share of confident KEEP pages added at random (the misses nobody looks for).
RANDOM_KEEP_SHARE = 0.02
MAX_PAGES = 3000
# ========================================================================


def labeled_pages(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {
            ((r.get("folder name") or "").strip().casefold(), Path((r.get("image name") or "").strip()).stem.casefold())
            for r in csv.DictReader(fh)
        }


def service(cfg: dict) -> PageClassifierService:
    if not bert_available(infer.BERT_DIR):
        raise SystemExit(f"BERT model not found: {infer.BERT_DIR}")
    from src.models.bert_classifier import BertPageClassifier

    return PageClassifierService(
        BertPageClassifier.load(infer.BERT_DIR),
        model_version="candidates",
        decision=DecisionConfig(**cfg["decision"]),
        min_dictionary_words=cfg["routing"]["min_dictionary_words"],
        route_short_pages=not cfg.get("bert", {}).get("decide_short_pages", False),
    )


def main() -> int:
    cfg = infer._load_cfg(Path(infer.CONFIG_PATH))
    main_service = service(cfg)
    done = labeled_pages(LABELS_CSV)
    charts = chart_files(OCR_DIR)
    print(f"{len(set(charts.values()))} charts, {len(done)} pages already labeled", flush=True)

    rng = random.Random(5)
    picked: list[dict] = []
    seen_files: set[Path] = set()
    for folder_key, path in sorted(charts.items()):
        if path in seen_files:
            continue
        seen_files.add(path)
        folder = path.parent.parent.name if path.parent.name.lower() == "ocr" else path.parent.name
        for name, text in split_pages(path.read_text(encoding="utf-8", errors="replace")):
            if (folder.casefold(), Path(name).stem.casefold()) in done:
                continue
            r = main_service.predict_one(name, text)
            triggers = junk_trigger_groups(text)
            reasons = []
            if r.flag != "KEEP":
                reasons.append(f"called {r.flag}")
            if r.review_required:
                reasons.append("review")
            if r.flag == "KEEP" and (r.p_junk or 0) >= MIN_P_JUNK:
                reasons.append(f"p_junk {r.p_junk:.2f}")
            if triggers:
                reasons.append("words: " + ", ".join(triggers))
            if not reasons and rng.random() < RANDOM_KEEP_SHARE:
                reasons.append("random KEEP check")
            if reasons:
                picked.append({
                    "Folder Name": folder, "ID": name, "Blank Page": "No", "Junk Page": "Yes",
                    "model_flag": r.flag, "model_type": r.subclass, "model_score": r.p_junk,
                    "reason": "; ".join(reasons),
                })
        print(f"  {folder}: {len(picked)} picked so far", flush=True)
        if len(picked) >= MAX_PAGES:
            break

    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    header = ["Folder Name", "ID", "Blank Page", "Junk Page", "model_flag", "model_type", "model_score", "reason"]
    ws.append(header)
    for row in picked[:MAX_PAGES]:
        ws.append([row[k] if row[k] is not None else "" for k in header])
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    print(f"Wrote {min(len(picked), MAX_PAGES)} pages -> {OUT_XLSX}\n"
          f"Set REVIEW_SHEET to it in b_jnk_creator/junk_tagger.py and tag them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
