#!/usr/bin/env python3
"""Print synthetic pages to images, scan-damage them, and read them back with Tesseract.

The models read Tesseract text in production, so synthetic pages go through the
same Tesseract call as ts_ocr.py (pytesseract.image_to_string, default settings)
after being drawn like a real page: letterhead, fonts, margins, a fax header line,
and one damage profile (clean scan, fax, photocopy, or heavy). BLANK pages are made
here too (no LLM): speckles, a lone letterhead or footer, "intentionally left blank".

  python scripts/render_synthetic_v2.py            # all of synthetic_v2/raw -> synthetic_v2/clean.jsonl
  python scripts/render_synthetic_v2.py --limit 20 --save-images  # a quick look, images in synthetic_v2/preview

Run on the Mac (needs tesseract). The output file is what training reads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "synthetic_v2"
RAW = V2 / "raw"
OUT = V2 / "clean.jsonl"
VERIFIED = V2 / "verified.jsonl"
PREVIEW = V2 / "preview"

BLANK_PAGES = 450
PAGE_W, PAGE_H = 1700, 2200  # US letter at 200 dpi
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONTS = [
    ("Arial.ttf", "Arial Bold.ttf"),
    ("Times New Roman.ttf", "Times New Roman Bold.ttf"),
    ("Courier New.ttf", "Courier New Bold.ttf"),
    ("Verdana.ttf", "Verdana Bold.ttf"),
    ("Georgia.ttf", "Georgia Bold.ttf"),
    ("Tahoma.ttf", "Tahoma Bold.ttf"),
    ("Trebuchet MS.ttf", "Trebuchet MS Bold.ttf"),
]
PROFILES = [("clean", 0.30), ("fax", 0.35), ("photocopy", 0.25), ("heavy", 0.10)]
FLAG = {"KEEP": "KEEP", "BLANK": "BLANK"}


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    return ImageFont.truetype(str(path if path.is_file() else FONT_DIR / "Arial.ttf"), size)


def _wrap(line: str, font, width: int) -> list[str]:
    if not line.strip() or font.getlength(line) <= width:
        return [line]
    out, cur = [], ""
    for word in line.split(" "):
        test = f"{cur} {word}" if cur else word
        if font.getlength(test) <= width:
            cur = test
        else:
            if cur:
                out.append(cur)
            cur = word
    out.append(cur)
    return out


def _fax_header(rng: random.Random) -> str:
    def phone() -> str:
        return f"({rng.randint(201, 989)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
    page, of = rng.randint(1, 12), rng.randint(12, 40)
    date = f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(2022, 2026)} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    return rng.choice([
        f"From: {phone()}    To: {phone()}    P.{page:03d}/{of:03d}    {date}",
        f"{date}  FAX  {phone()}  PAGE {page:03d}/{of:03d}",
        f"Received {date} via Fax  Page {page} of {of}",
    ])


def draw_page(text: str, rng: random.Random, fax_header: bool) -> Image.Image:
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(img)
    regular, bold = rng.choice(FONTS)
    size = rng.randint(24, 34)
    font, font_bold = _font(regular, size), _font(bold, size + rng.randint(0, 6))
    left = rng.randint(90, 190)
    width = PAGE_W - left - rng.randint(90, 190)
    y = rng.randint(70, 150)
    gap = int(size * rng.uniform(1.15, 1.5))
    if fax_header:
        small = _font("Courier New.ttf", rng.randint(20, 24))
        draw.text((left - 40, 25), _fax_header(rng), font=small, fill=0)
        y += 20
    center_head = rng.random() < 0.45
    for n, line in enumerate(text.splitlines()):
        head = n < 3 and center_head
        f = font_bold if n == 0 and rng.random() < 0.6 else font
        for part in _wrap(line, f, width):
            if y > PAGE_H - 90:
                return img
            x = left + (width - f.getlength(part)) / 2 if head else left
            draw.text((x, y), part, font=f, fill=rng.randint(0, 40))
            y += gap
    return img


def damage(img: Image.Image, profile: str, rng: random.Random) -> Image.Image:
    img = img.rotate(rng.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor=255)
    if profile == "clean":
        return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.6)))
    if profile in {"fax", "heavy"}:
        w, h = img.size
        img = img.resize((w, h // 2)).resize((w, h))  # 200 x 100 dpi fine-mode fax
        cut = rng.randint(120, 175)
        img = img.point(lambda p: 0 if p < cut else 255)
        img = _speckle(img, rng, rng.randint(300, 1500))
    if profile in {"photocopy", "heavy"}:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.3)))
        img = img.point(lambda p: int(40 + p * rng.uniform(0.7, 0.85)))
        img = _speckle(img, rng, rng.randint(200, 800))
    if profile == "heavy":
        w, h = img.size
        scale = rng.uniform(0.38, 0.55)
        img = img.resize((int(w * scale), int(h * scale))).resize((w, h))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(12, 30))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return img


def _speckle(img: Image.Image, rng: random.Random, n: int) -> Image.Image:
    px = img.load()
    w, h = img.size
    for _ in range(n):
        x, y = rng.randrange(w), rng.randrange(h)
        px[x, y] = 0 if rng.random() < 0.7 else 255
    return img


def pick_profile(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for name, weight in PROFILES:
        acc += weight
        if r < acc:
            return name
    return PROFILES[-1][0]


def render(row: dict) -> dict | None:
    import pytesseract

    rng = random.Random(int(hashlib.sha1(row["id"].encode()).hexdigest()[:8], 16))
    profile = row.get("profile") or pick_profile(rng)
    fax_header = row.get("fax_header", rng.random() < (0.45 if profile == "fax" else 0.12))
    img = damage(draw_page(row["text"], rng, fax_header), profile, rng)
    ocr = pytesseract.image_to_string(img).strip()
    if row.get("save_dir"):
        img.save(Path(row["save_dir"]) / f"{row['id']}.png")
    src_words = set(re.findall(r"[a-z]{3,}", row["text"].lower()))
    got_words = set(re.findall(r"[a-z]{3,}", ocr.lower()))
    recall = len(src_words & got_words) / len(src_words) if src_words else 1.0
    # A page the damage made unreadable no longer shows its label: leave it out.
    if row["label"] != "BLANK" and len(src_words) >= 5 and recall < 0.3:
        return None
    return {
        "id": f"v2-{row['id']}",
        "label": row["label"],
        "flag": FLAG.get(row["label"], "JUNK"),
        "page_kind": row["page_kind"],
        "hard_negative": bool(row.get("hard_negative")),
        "noise": profile,
        "fax_header": fax_header,
        "text": ocr,
        "source_text": row["text"],
    }


def blank_rows(pages: list[dict], n: int, rng: random.Random) -> list[dict]:
    """BLANK pages from code: nothing on them but scan marks, a letterhead, a footer, or a stock phrase."""
    heads = [p["text"].splitlines()[:rng.randint(1, 3)] for p in rng.sample(pages, min(len(pages), n))] or [["Clinic"]]
    rows = []
    for i in range(n):
        kind = rng.choice(["speckles", "letterhead", "footer", "letterhead_footer", "system", "fax_line"])
        page_no = f"Page {rng.randint(2, 30)} of {rng.randint(30, 60)}"
        text = {
            "speckles": "",
            "letterhead": "\n".join(heads[i % len(heads)]),
            "footer": "\n" * 55 + page_no,
            "letterhead_footer": "\n".join(heads[i % len(heads)]) + "\n" * 50 + page_no,
            "system": "\n" * 20 + rng.choice(["This page intentionally left blank", "Intentionally Left Blank", "No content on this page"]),
            "fax_line": "",
        }[kind]
        rows.append({
            "id": f"blank-{i:04d}",
            "label": "BLANK",
            "page_kind": f"blank {kind}",
            "text": text,
            "fax_header": kind == "fax_line",
            "profile": rng.choice(["clean", "fax", "photocopy", "heavy"]),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="render only this many pages (a quick look)")
    ap.add_argument("--save-images", action="store_true", help=f"also save the page images to {PREVIEW}")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    pages, seen = [], set()
    for path in sorted(RAW.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["text_sha1"] not in seen:
                seen.add(row["text_sha1"])
                pages.append(row)
    if not pages:
        raise SystemExit(f"No pages in {RAW}. Run generate_synthetic_v2.py first.")
    if VERIFIED.is_file():
        checks = [json.loads(line) for line in VERIFIED.read_text(encoding="utf-8").splitlines() if line.strip()]
        agreed = {c["id"] for c in checks if c["agree"]}
        before = len(pages)
        pages = [p for p in pages if p["id"] in agreed]
        print(f"{len(pages)} of {before} pages passed the blind check "
              f"({sum(not c['agree'] for c in checks)} disagreed, {before - len(checks)} not checked)", flush=True)
    elif not args.limit:
        raise SystemExit("No blind check yet: run generate_synthetic_v2.py verify, then verify-collect. "
                         "(--limit works without it, for a preview.)")
    rng = random.Random(11)
    rows = pages + blank_rows(pages, BLANK_PAGES, rng)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    if args.save_images:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        for row in rows:
            row["save_dir"] = str(PREVIEW)
    print(f"Rendering {len(rows)} pages with {args.workers} workers…", flush=True)

    kept, dropped = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, out in enumerate(pool.map(render, rows, chunksize=8), start=1):
            if out is None:
                dropped += 1
            else:
                kept.append(out)
            if n % 250 == 0:
                print(f"  {n}/{len(rows)}", flush=True)
    target = OUT if not args.limit else V2 / "clean_preview.jsonl"
    with target.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in kept:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Wrote {len(kept)} pages -> {target} ({dropped} unreadable after damage, left out)\n{counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
