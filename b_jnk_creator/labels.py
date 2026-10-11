"""Local CSV of keep / blank / junk labels. One row per folder + image."""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

COLUMNS = ["folder name", "image name", "value", "subtype"]
VALUES = {"KEEP", "BLANK", "JUNK"}
SUBTYPES = {
    "JUNK_INVOICE",
    "JUNK_COVER_PAGE",
    "JUNK_RECORD_REQUEST",
    "JUNK_INSTRUCTIONS",
    "JUNK_LETTER_FAX",
    "JUNK_OTHERS",
}


class LabelStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: list[dict[str, str]] = []
        self._index: dict[tuple[str, str], int] = {}
        self._load()

    def _blank(self, folder: str, image_name: str) -> dict[str, str]:
        return {"folder name": folder, "image name": image_name, "value": "", "subtype": ""}

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                folder = (row.get("folder name") or "").strip()
                name = (row.get("image name") or "").strip()
                value = (row.get("value") or "").strip()
                subtype = (row.get("subtype") or "").strip()
                if not folder or not name:
                    continue
                if value and value not in VALUES:
                    continue
                if subtype and subtype not in SUBTYPES:
                    continue
                if value in {"KEEP", "BLANK"}:
                    subtype = ""
                record = self._blank(folder, name)
                record["value"] = value
                record["subtype"] = subtype
                key = (folder, name)
                if key in self._index:
                    self._rows[self._index[key]] = record
                else:
                    self._index[key] = len(self._rows)
                    self._rows.append(record)

    def get(self, folder: str, image_name: str) -> dict[str, str] | None:
        with self._lock:
            pos = self._index.get((folder, image_name))
            return None if pos is None else dict(self._rows[pos])

    def has_row(self, folder: str, image_name: str) -> bool:
        return self.get(folder, image_name) is not None

    def replace(
        self,
        folder: str,
        image_name: str,
        *,
        value: str | None,
        subtype: str | None,
    ) -> dict[str, str]:
        """Write the whole mark. Value and subtype may both be empty."""
        if value is not None and value not in VALUES:
            raise ValueError(f"value must be KEEP, BLANK, or JUNK, got {value!r}")
        if subtype is not None and subtype not in SUBTYPES:
            raise ValueError(f"unknown junk subtype {subtype!r}")
        if value in {"KEEP", "BLANK"}:
            subtype = None
        if value == "JUNK" and not subtype:
            raise ValueError("a junk page needs a subtype")
        if subtype and value not in {None, "JUNK"}:
            raise ValueError("subtype is only stored on a junk page")
        with self._lock:
            key = (folder, image_name)
            pos = self._index.get(key)
            created = pos is None
            if created:
                self._index[key] = len(self._rows)
                self._rows.append(self._blank(folder, image_name))
                pos = self._index[key]
            previous = dict(self._rows[pos])
            row = self._rows[pos]
            row["value"] = value or ""
            row["subtype"] = subtype or ""
            try:
                self._write()
            except Exception:
                if created:
                    self._rows.pop()
                    del self._index[key]
                else:
                    self._rows[pos] = previous
                raise
            return dict(row)

    def add_missing(self, rows: list[tuple[str, str, str, str]]) -> int:
        """Append (folder, image, value, subtype) rows that have no row yet. One write.

        A page that already has a row keeps it, so a hand label is never overwritten.
        """
        added: list[tuple[str, str]] = []
        with self._lock:
            for folder, image_name, value, subtype in rows:
                if value not in VALUES or (subtype and subtype not in SUBTYPES):
                    raise ValueError(f"bad label {value!r} / {subtype!r} for {folder}/{image_name}")
                key = (folder, image_name)
                if key in self._index:
                    continue
                record = self._blank(folder, image_name)
                record["value"] = value
                record["subtype"] = subtype if value == "JUNK" else ""
                self._index[key] = len(self._rows)
                self._rows.append(record)
                added.append(key)
            if not added:
                return 0
            try:
                self._write()
            except Exception:
                del self._rows[len(self._rows) - len(added):]
                for key in added:
                    del self._index[key]
                raise
        return len(added)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)
        os.replace(tmp, self.path)
