#!/usr/bin/env python3
"""Why did this page get its flag? Shows each layer for one page.

    python b_jnk/scripts/explain_page.py 66945142_55331561/3.png
    python b_jnk/scripts/explain_page.py --text-file page.txt

A page id is looked up in data/raw/annotated_pages.jsonl (the Tesseract text the
models trained on), so the training label is shown next to what the code does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import infer  # noqa: E402
from src.inference.predict import PageClassifierService  # noqa: E402
from src.models.bert_classifier import bert_available  # noqa: E402
from src.models.decision import DecisionConfig  # noqa: E402
from src.preprocessing.page_subclass import (  # noqa: E402
    NOTE_CONTENT_RE,
    _candidate_hits,
    end_record_reason,
    invoice_page_reason,
    shell_page_reason,
)
from src.preprocessing.protocol import analyze_protocol  # noqa: E402


def service(model, cfg: dict, route_short: bool = True) -> PageClassifierService:
    return PageClassifierService(
        model,
        model_version="explain",
        decision=DecisionConfig(**cfg["decision"]),
        min_dictionary_words=cfg["routing"]["min_dictionary_words"],
        route_short_pages=route_short,
    )


def show(name: str, r) -> None:
    probs = f"p_keep={r.p_keep} p_blank={r.p_blank} p_junk={r.p_junk}" if r.p_keep is not None else "no model call"
    print(f"  {name}: {r.flag} {r.subclass or ''}  {probs}\n    reason: {r.decision_reason}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("page_id", nargs="?", help="<folder>/<image>, as in annotated_pages.jsonl")
    ap.add_argument("--text-file")
    ap.add_argument("--txt", help="a chart's Tesseract txt (<folder>_prelim.txt); give the image name as page_id, e.g. 42.png")
    args = ap.parse_args()
    cfg = infer._load_cfg(Path(infer.CONFIG_PATH))

    label = None
    if args.txt:
        from src.preprocessing.tesseract_txt import split_pages

        want = Path((args.page_id or "").replace("\\", "/")).name.casefold()
        pages = split_pages(Path(args.txt).read_text(encoding="utf-8", errors="replace"))
        match = [t for name, t in pages if Path(name).name.casefold() == want or Path(name).stem.casefold() == Path(want).stem]
        if not match:
            print(f"{args.page_id} is not in {args.txt}. Pages there: {', '.join(n for n, _ in pages[:15])} ...")
            return 1
        text = match[0]
    elif args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8", errors="replace")
    else:
        path = ROOT / cfg["bert"].get("annotated_jsonl", "data/raw/annotated_pages.jsonl")
        want = (args.page_id or "").replace("\\", "/").casefold()
        row = next(
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and json.loads(line)["id"].casefold() == want),
            None,
        )
        if row is None:
            print(f"{args.page_id} is not in {path}. Use --text-file with the page text.")
            return 1
        text, label = row["text"], row["label"]

    print(f"training label: {label or '(not given)'}")
    print(f"text ({len(text.split())} words), exactly as OCR read it:")
    print("-" * 60)
    print(text)
    print("-" * 60)

    print("\nword rules:")
    if analyze_protocol(text).retain_clinical_image:
        print("  clinical image: rules never mark it junk")
    print(f"  phrases found: {_candidate_hits(text) or 'none'}")
    guard = NOTE_CONTENT_RE.search(text)
    print(f"  clinical word that can hold the cover/letter rule back: {guard.group(0) if guard else 'none'}")
    print(f"  invoice rule: {invoice_page_reason(text)}   cover/letter rule: {shell_page_reason(text)}"
          f"   end-of-record rule: {end_record_reason(text)}")

    print("\nBERT with the rules on top, as infer.py runs it:")
    bert_dir = Path(infer.BERT_DIR)
    if not bert_available(bert_dir):
        print(f"  BERT: not found at {bert_dir}")
        return 1
    from src.models.bert_classifier import BertPageClassifier

    route_short = not cfg.get("bert", {}).get("decide_short_pages", False)
    show(f"BERT ({bert_dir.name})", service(BertPageClassifier.load(bert_dir), cfg, route_short).predict_one("page", text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
