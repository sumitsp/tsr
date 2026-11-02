#!/usr/bin/env python3
"""Which way is each page turned? Give an image or a folder (sub folders are read too).

    python page_rotation/run_rotation.py C:\\path\\to\\13.png
    python page_rotation/run_rotation.py C:\\path\\to\\folder

Prints each image's rotation: the degrees to turn it CLOCKWISE to make it upright.

Writes, so you can look at the result:
  image  -> <name>_upright.<ext> next to it
  folder -> <folder>_upright\\ (same sub folders and names) and <folder>_rotation.csv
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rotation_cv import detect_rotation, make_upright  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
COLUMNS = ["image", "rotation", "sideways score", "flip score", "letters", "lines", "note"]


def _natural(path: Path) -> tuple:
    return tuple(int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", str(path)))


def check(path: Path, out_image: Path) -> dict | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"  cannot read: {path}")
        return None
    r = detect_rotation(image)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), make_upright(image, r))
    return {"rotation": r.rotation, "sideways score": r.sideways_score, "flip score": r.flip_score,
            "letters": r.letters, "lines": r.lines, "note": r.note}


def main(args: list[str]) -> int:
    if len(args) != 1:
        print(__doc__)
        return 2
    target = Path(args[0].strip('"'))
    if target.is_file():
        row = check(target, target.with_name(f"{target.stem}_upright{target.suffix}"))
        if row:
            print(f"{target.name}: rotation {row['rotation']}  (sideways {row['sideways score']:+.2f}, "
                  f"flip {row['flip score']:+.2f}) {row['note']}")
            print(f"upright copy -> {target.with_name(f'{target.stem}_upright{target.suffix}')}")
        return 0
    if not target.is_dir():
        raise SystemExit(f"Not found: {target}")

    images = sorted((p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                    key=_natural)
    out_dir = target.with_name(f"{target.name}_upright")
    rows = []
    for path in images:
        rel = path.relative_to(target)
        row = check(path, out_dir / rel)
        if row:
            rows.append({"image": str(rel), **row})
            print(f"  {str(rel):40s} rotation {row['rotation']:3d}  {row['note']}", flush=True)

    out_csv = target.with_name(f"{target.name}_rotation.csv")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {deg: sum(r["rotation"] == deg for r in rows) for deg in (0, 90, 180, 270)}
    print(f"\n{len(rows)} images: " + "  ".join(f"{d}: {n}" for d, n in counts.items()))
    print(f"upright copies -> {out_dir}\nCSV -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
