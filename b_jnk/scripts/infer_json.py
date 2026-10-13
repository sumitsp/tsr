#!/usr/bin/env python3
"""Docling or Azure Document Intelligence JSON -> page text -> KEEP / BLANK / JUNK Excel.

Each page of each JSON is turned into plain text in memory (no files are
written), then flagged by scripts/infer.py with BERT and the word rules.
The Excel is saved next to JSON_INPUT. Nothing is deleted.

  Docling (os_ocr.py):  {"1.jpg": {"markdown": ..., "document": {...}}, ...}
  Document Intelligence: an analyze result, with or without the service envelope;
                         page N is named DI_PAGE_NAME (1.png, 2.png, ...).

Edit RUN CONFIG, then run:
  python b_jnk/scripts/infer_json.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import infer  # noqa: E402
from src.preprocessing.docling_page import extract_docling_page  # noqa: E402
from src.preprocessing.document_intelligence import iter_analyze_results, pages_in_result  # noqa: E402

# ============================== RUN CONFIG ==============================
# One JSON, or a folder searched at any depth for chart JSON files.
JSON_INPUT = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_docling")
# Document Intelligence numbers pages; page n is written as this image name.
DI_PAGE_NAME = "{n}.png"
# ========================================================================


def plain_text(markdown: str) -> str:
    """Docling markdown -> plain text like Tesseract's: no #, table pipes, or placeholders."""
    text = re.sub(r"<!--.*?-->", " ", markdown or "", flags=re.S)
    if re.fullmatch(r"\s*\[(?:no text detected|ERROR extracting[^\]]*)\]\s*", text):
        return ""
    lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*", line):
            continue  # table divider row
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = line.replace("**", "").strip()
        if line.startswith("|") or line.endswith("|"):
            line = " ".join(cell.strip() for cell in line.strip("|").split("|") if cell.strip())
        lines.append(line)
    return "\n".join(lines).strip()


def _is_docling(payload: object) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    values = list(payload.values())[:12]
    return all(isinstance(v, dict) and ("markdown" in v or "document" in v) for v in values)


def json_pages(payload: object) -> tuple[str, list[tuple[str, str]]] | None:
    """(format, [(image name, text)]) or None when the JSON is neither format."""
    if _is_docling(payload):
        items = sorted(payload.items(), key=lambda kv: infer._natural_page_key(str(kv[0])))
        return "docling", [(str(name), plain_text(extract_docling_page(value).text)) for name, value in items]
    pages = [
        (DI_PAGE_NAME.format(n=number), text)
        for result in iter_analyze_results(payload)
        for number, text in pages_in_result(result)
    ]
    return ("document_intelligence", pages) if pages else None


def load_pages(json_input: Path) -> list[dict]:
    """Every page of every Docling / Document Intelligence JSON under json_input."""
    files = [json_input] if json_input.is_file() else sorted(json_input.rglob("*.json"))
    base = json_input.parent if json_input.is_file() else json_input
    pages: list[dict] = []
    for path in files:
        rel = path.relative_to(base)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skip {rel}: {exc}", flush=True)
            continue
        found = json_pages(payload)
        if found is None:
            print(f"  skip {rel}: not Docling or Document Intelligence", flush=True)
            continue
        kind, chart_pages = found
        prefix = f"{rel.parent.as_posix()}/" if rel.parent != Path(".") else ""
        for name, text in chart_pages:
            pages.append({"page_id": f"{prefix}{name}", "ocr_text": infer._clean_ocr_text(text)})
        print(f"  loaded {len(chart_pages)} pages from {rel} ({kind})", flush=True)
    return pages


def main() -> int:
    source = Path(JSON_INPUT).expanduser()
    if not source.exists():
        raise SystemExit(f"Input not found: {source}\nEdit JSON_INPUT at the top of scripts/infer_json.py")
    # infer.py runs as configured; only its page reader is swapped for the JSON one.
    infer.INPUT_SOURCE = "local"
    infer.LOCAL_INPUT = source
    infer._load_pages = load_pages
    return infer.main()


if __name__ == "__main__":
    raise SystemExit(main())
