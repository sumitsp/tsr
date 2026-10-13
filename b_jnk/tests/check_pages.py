#!/usr/bin/env python3
"""Regression check: every page in tests/pages/junk must flag JUNK or BLANK, every page in
tests/pages/keep must stay KEEP. Uses the real pipeline (BERT + rules), as infer.py runs it.

    python b_jnk/tests/check_pages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
sys.argv = sys.argv[:1]

import infer  # noqa: E402
from src.inference.predict import PageClassifierService  # noqa: E402
from src.models.bert_classifier import BertPageClassifier  # noqa: E402
from src.models.decision import DecisionConfig  # noqa: E402

PAGES = Path(__file__).resolve().parent / "pages"


def check(svc: PageClassifierService, verbose: bool = True) -> list[str]:
    """Run every test page through svc. Returns the pages that came out wrong."""
    failed: list[str] = []
    for want in ("junk", "keep"):
        for path in sorted((PAGES / want).glob("*.txt")):
            r = svc.predict_one(path.name, path.read_text(encoding="utf-8"))
            ok = (r.flag == "KEEP") == (want == "keep")
            if not ok:
                failed.append(f"{want}/{path.name}")
            if verbose:
                print(f"{'ok  ' if ok else 'FAIL'} {want}/{path.name:28s} -> {r.flag:5s} {r.subclass:24s} {r.decision_reason}")
    return failed


def main() -> int:
    cfg = infer._load_cfg(Path(infer.CONFIG_PATH))
    model = BertPageClassifier.load(infer.BERT_DIR)
    route_short = not cfg["bert"].get("decide_short_pages", False)
    svc = PageClassifierService(model, model_version="check", decision=DecisionConfig(**cfg["decision"]),
                                min_dictionary_words=cfg["routing"]["min_dictionary_words"], route_short_pages=route_short)
    failed = check(svc)
    print(f"\n{len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
