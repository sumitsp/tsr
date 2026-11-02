#!/usr/bin/env python3
"""Check rotation_cv on pages you know are upright: each page is turned 0, 90, 180 and
270 degrees and the detector has to find the turn back.

    python page_rotation/check_rotation.py <folder of upright page images> [more folders]

Prints each miss and a summary: right / wrong.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rotation_cv import detect_rotation, make_upright  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# Max images per folder (0 = all).
MAX_PER_FOLDER = 0


def main(folders: list[str]) -> int:
    if not folders:
        print(__doc__)
        return 2
    tally: Counter = Counter()
    for folder in folders:
        images = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if MAX_PER_FOLDER:
            images = images[:MAX_PER_FOLDER]
        for path in images:
            page = cv2.imread(str(path))
            if page is None:
                continue
            for turned_by in (0, 90, 180, 270):
                # Turn the upright page counter-clockwise; the fix is the same turn clockwise.
                turned = make_upright(page, (360 - turned_by) % 360)
                want = turned_by
                r = detect_rotation(turned)
                result = "right" if r.rotation == want else "wrong"
                tally[result] += 1
                if result != "right":
                    print(f"wrong  {path.name:22s} want {want:3d} got {r.rotation:3d} "
                          f"sideways {r.sideways_score:+.2f} flip {r.flip_score:+.2f} {r.note} {r.details}")
    total = tally["right"] + tally["wrong"]
    print(f"\n{total} turned pages: right {tally['right']}  wrong {tally['wrong']}"
          f"  ({100 * tally['right'] / max(total, 1):.1f}% right)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
