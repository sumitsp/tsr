#!/usr/bin/env python3
"""Find ID cards and US health insurance cards in charts and write them to a CSV.

On its own: nothing in b_jnk calls it, and it changes no model or rule. Run it only
on pages already marked printed or visual.

Page text comes from the chart's Tesseract dump (OCR_DIR/<chart>/.../<chart>_prelim.txt,
page N is "===== N.png ====="). Cards are often scanned sideways and then read as
nonsense, so when a page's text is weak and IMAGE_DIR is set, the image
(IMAGE_DIR/<chart>/.../N.png) is read again turned 90, 270 and 180 degrees.

With OCR_DIR = None every image in IMAGE_DIR is read with Tesseract (upright first).
A folder of loose images (no chart folders) is one chart named after the folder.

CSV columns:
  folder name, image name, card (yes / no), card type, confidence, rotated,
  matched words, blocked by, words

Only cards and near misses (card words, but form / letter / clinical words too) are
written, unless WRITE_ALL_PAGES is True.

Edit RUN CONFIG, then:
    python b_jnk/id_card/run_id_cards.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_rules import detect, word_count  # noqa: E402

# ============================== RUN CONFIG ==============================
# Tesseract txt folder: OCR_DIR/<chart>/.../<chart>_prelim.txt. None = OCR the images.
OCR_DIR: Path | None = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_tesseract")
TXT_SUFFIX = "_prelim"
# Page images: IMAGE_DIR/<chart>/.../N.png. None = text only (sideways cards are missed).
IMAGE_DIR: Path | None = None
# Chart folders to run. Empty = every chart folder found.
CHARTS: list[str] = []
# How many chart folders to run. None or 0 = all of them.
MAX_FOLDERS: int | None = 100
# Read weak pages again turned sideways (needs IMAGE_DIR and pytesseract).
ROTATE_WEAK_PAGES = True
TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
# Empty = id_cards.csv in OCR_DIR (or IMAGE_DIR).
OUT_CSV: Path | None = None
WRITE_ALL_PAGES = False
# ========================================================================

PAGE_SPLIT = re.compile(r"=====+\s*([^\n=]+?)\s*=====+")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
COLUMNS = ["folder name", "image name", "card", "card type", "confidence", "rotated",
           "matched words", "blocked by", "words"]


def _natural(name: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", name))


def split_pages(text: str) -> list[tuple[str, str]]:
    parts = PAGE_SPLIT.split(text or "")
    return [(Path(parts[i].strip()).name, parts[i + 1] if i + 1 < len(parts) else "")
            for i in range(1, len(parts), 2)]


def prelim_txt(chart_dir: Path) -> Path | None:
    name = f"{chart_dir.name}{TXT_SUFFIX}.txt"
    direct = chart_dir / name
    if direct.is_file():
        return direct
    hits = sorted(p for p in chart_dir.rglob(name) if p.is_file())
    return hits[0] if hits else None


def chart_images(chart_dir: Path) -> dict[str, Path]:
    """Image name (and name without extension), lower case -> path."""
    found: dict[str, Path] = {}
    if not chart_dir.is_dir():
        return found
    for p in sorted(chart_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            found.setdefault(p.name.casefold(), p)
            found.setdefault(p.stem.casefold(), p)
    return found


def weak_text(text: str) -> bool:
    """Few real words: likely a sideways scan read as nonsense."""
    tokens = re.findall(r"\S+", text or "")
    real = [t for t in tokens if re.fullmatch(r"[A-Za-z]{3,}[.,:;]?", t) and re.search(r"[aeiouyAEIOUY]", t)]
    return len(real) < 40 or len(real) < 0.6 * max(len(tokens), 1)


_tess = None


def tesseract():
    global _tess
    if _tess is None:
        try:
            import pytesseract
        except ImportError as exc:
            raise SystemExit("pytesseract is not installed (pip install pytesseract pillow)") from exc
        if TESSERACT_CMD and Path(TESSERACT_CMD).is_file():
            pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_CMD)
        _tess = pytesseract
    return _tess


def ocr(image: Path, angle: int) -> str:
    from PIL import Image

    with Image.open(image) as im:
        im = im.convert("L")
        if angle:
            im = im.rotate(angle, expand=True)
        return tesseract().image_to_string(im)


def check_page(text: str | None, image: Path | None):
    """(result, angle). Tries the turned image only when the text is weak and no card was found."""
    if text is None:
        text = ocr(image, 0) if image else ""
    result, angle = detect(text), 0
    if result.is_card or result.blocked_by or not (ROTATE_WEAK_PAGES and image and weak_text(text)):
        return result, angle
    for turn in (90, 270, 180):
        turned = detect(ocr(image, turn))
        if turned.is_card:
            return turned, turn
    return result, angle


def charts_to_run() -> list[Path]:
    root = OCR_DIR or IMAGE_DIR
    if root is None:
        raise SystemExit("Set OCR_DIR or IMAGE_DIR")
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")
    folders = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: _natural(p.name))
    if OCR_DIR is None and not folders:
        folders = [root]  # loose images
    if CHARTS:
        wanted = {c.casefold() for c in CHARTS}
        folders = [p for p in folders if p.name.casefold() in wanted]
    if MAX_FOLDERS:
        folders = folders[:MAX_FOLDERS]
    return folders


def pages_of(chart: Path) -> list[tuple[str, str | None, Path | None]]:
    """(image name, text or None to OCR, image path or None)."""
    images = chart_images(IMAGE_DIR / chart.name) if IMAGE_DIR and OCR_DIR else {}
    if OCR_DIR is None:
        imgs = sorted({p for p in chart_images(chart).values()}, key=lambda p: _natural(p.name))
        return [(p.name, None, p) for p in imgs]
    fp = prelim_txt(chart)
    if fp is None:
        return []
    pages = split_pages(fp.read_text(encoding="utf-8", errors="replace"))
    out = []
    for name, text in sorted(pages, key=lambda p: _natural(p[0])):
        image = images.get(name.casefold()) or images.get(Path(name).stem.casefold())
        out.append((name, text, image))
    return out


def main() -> int:
    charts = charts_to_run()
    root = OCR_DIR or IMAGE_DIR
    out = OUT_CSV or root / "id_cards.csv"
    print(f"Charts: {len(charts)}  text: {OCR_DIR or 'OCR the images'}  images: {IMAGE_DIR or 'none'}", flush=True)

    rows: list[dict] = []
    tally: Counter = Counter()
    no_txt: list[str] = []
    n_pages = 0
    for c_i, chart in enumerate(charts, start=1):
        pages = pages_of(chart)
        if not pages:
            no_txt.append(chart.name)
            continue
        for name, text, image in pages:
            n_pages += 1
            result, angle = check_page(text, image)
            if result.is_card:
                tally[result.card_type] += 1
            elif result.blocked_by:
                tally["near miss"] += 1
            if result.is_card or result.blocked_by or WRITE_ALL_PAGES:
                rows.append({
                    "folder name": chart.name, "image name": name,
                    "card": "yes" if result.is_card else "no",
                    "card type": result.card_type, "confidence": result.confidence,
                    "rotated": angle or "",
                    "matched words": " | ".join(result.matched),
                    "blocked by": " | ".join(result.blocked_by),
                    "words": result.words or word_count(text or ""),
                })
        print(f"  [{c_i}/{len(charts)}] {chart.name}: {len(pages)} pages, "
              f"{sum(v for k, v in tally.items() if k != 'near miss')} cards so far", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if no_txt:
        shown = ", ".join(no_txt[:10]) + (" ..." if len(no_txt) > 10 else "")
        print(f"\n{len(no_txt)} folders have no pages: {shown}")
    cards = sum(v for k, v in tally.items() if k != "near miss")
    print(f"\n{n_pages} pages, {cards} cards, {tally['near miss']} near misses -> {out}")
    for kind, n in tally.most_common():
        print(f"  {kind:22s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
