#!/usr/bin/env python3
"""Summarise the model's mistakes on real tagged pages, with no page text in it.

Reads the held-out results that train_bert.py wrote (reports/bert_eval_pages.csv,
real_2fold runs: every page judged by a model that did not train on its chart)
and the tagged page text (data/raw/annotated_pages.jsonl). Writes counts only:
which labels get confused, which rule decided, page length, fax header or not,
which clinical words held a rule back, and which junk words were on the page.
No page text, names, ids, or numbers are written, so the summary can be read
out to plan new synthetic pages on another machine.

    python b_jnk/scripts/error_summary.py      # -> reports/error_summary.txt
"""

from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.page_subclass import NOTE_CONTENT_RE, junk_trigger_groups, word_count  # noqa: E402

EVAL_CSV = ROOT / "reports" / "bert_eval_pages.csv"
TAGGED = ROOT / "data" / "raw" / "annotated_pages.jsonl"
OUT = ROOT / "reports" / "error_summary.txt"
FAX_LINE_RE = re.compile(r"(?i)\b(?:fax|facsimile)\b.{0,40}\b(?:p\.?\s*\d|page\s*\d)|\bP\.\d{3}/\d{3}")


def length_bucket(n: int) -> str:
    return "0-19 words" if n < 20 else "20-99 words" if n < 100 else "100-299 words" if n < 300 else "300+ words"


def main() -> int:
    if not EVAL_CSV.is_file():
        raise SystemExit(f"{EVAL_CSV} missing: run train_bert.py (without --skip-eval) first")
    text = {}
    if TAGGED.is_file():
        for line in TAGGED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                text[row["id"]] = row["text"]
    with EVAL_CSV.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run"].startswith("real_2fold") and r["page_id"] in text]
    if not rows:
        raise SystemExit("No held-out results for tagged pages in the eval file.")

    confusions = collections.Counter()
    groups: dict[str, dict[str, collections.Counter]] = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for r in rows:
        if r["true_flag"] == r["flag"] and (r["flag"] != "JUNK" or r["subclass"] == r["true_label"]):
            continue
        kind = "KEEP lost (clinical page called junk/blank)" if r["true_flag"] == "KEEP" else (
            "junk/blank missed (called KEEP)" if r["flag"] == "KEEP" else "wrong junk type")
        key = f"{kind}: {r['true_label']} -> {r['subclass'] or r['flag']}"
        confusions[key] += 1
        t = text[r["page_id"]]
        g = groups[key]
        g["decided by"][re.sub(r":.*", "", r["decision_reason"].split("+")[-1])] += 1
        g["length"][length_bucket(word_count(t))] += 1
        g["fax header line"]["yes" if FAX_LINE_RE.search(t) else "no"] += 1
        for word in {m.group(0).lower() for m in NOTE_CONTENT_RE.finditer(t)} or {"(none)"}:
            g["clinical words on page"][word if word.isalpha() or word == "(none)" else "(code or dose)"] += 1
        for trig in junk_trigger_groups(t) or ["(none)"]:
            g["junk word groups on page"][trig] += 1

    lines = [f"Held-out real tagged pages checked: {len(rows)}", f"Mistakes: {sum(confusions.values())}", ""]
    for key, n in confusions.most_common():
        lines.append(f"{n:5d}  {key}")
        for name, counter in groups[key].items():
            top = ", ".join(f"{k} {v}" for k, v in counter.most_common(8))
            lines.append(f"         {name}: {top}")
        lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
