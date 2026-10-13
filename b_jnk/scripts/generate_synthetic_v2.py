#!/usr/bin/env python3
"""Generate synthetic page texts with Claude (run on the Mac only, never on the
Windows training machine).

synthetic_v2/plan.csv says how many pages of each kind to write; the system
prompt is synthetic_v2/label_guide.md. Every page gets its own brief (US state,
specialty, EMR or vendor style, length) so the set does not repeat itself.

  python scripts/generate_synthetic_v2.py plan              # what would be sent, and the rough cost
  python scripts/generate_synthetic_v2.py sample --kinds 3  # a few requests right now, to read before paying for all
  python scripts/generate_synthetic_v2.py submit            # all requests as one Message Batch (half price)
  python scripts/generate_synthetic_v2.py collect           # wait for the batch, write synthetic_v2/raw/*.jsonl
  python scripts/generate_synthetic_v2.py verify            # a second, blind pass labels every page (one batch)
  python scripts/generate_synthetic_v2.py verify-collect    # writes synthetic_v2/verified.jsonl

The blind pass sees each page without its intended label. render_synthetic_v2.py
keeps only pages where both passes agree, so a "junk" page that reads like a
clinical note (or the reverse) never reaches training.

The API key is read from ANTHROPIC_API_KEY or from b_jnk/.env (git-ignored).
The pages are printed and OCR'd afterwards by scripts/render_synthetic_v2.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "synthetic_v2"
PLAN = V2 / "plan.csv"
GUIDE = V2 / "label_guide.md"
RAW = V2 / "raw"
STATE = V2 / "batches.json"
VERIFIED = V2 / "verified.jsonl"
VERIFY_PAGES_PER_REQUEST = 20
LABELS = ["KEEP", "JUNK_INVOICE", "JUNK_COVER_PAGE", "JUNK_RECORD_REQUEST", "JUNK_INSTRUCTIONS",
          "JUNK_LETTER_FAX", "JUNK_OTHERS"]
VERIFY_INSTRUCTIONS = """

# Your task now: label pages

You are now the reviewer, not the writer. For each page below, give the one label
a careful human reviewer would give it under the definitions above. Judge only the
page text. When a page could be clinical content a coder needs, answer KEEP."""
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"page": {"type": "integer"}, "label": {"type": "string", "enum": LABELS}},
                "required": ["page", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

MODEL = "claude-opus-5-5"
EFFORT = "medium"
PAGES_PER_REQUEST = 10
MAX_TOKENS = 32000
SEED = 7

STATES = [
    "Texas", "California", "Florida", "New York", "Pennsylvania", "Ohio", "Georgia", "North Carolina",
    "Michigan", "New Jersey", "Virginia", "Washington", "Arizona", "Tennessee", "Utah", "Colorado",
    "Indiana", "Missouri", "Maryland", "Wisconsin", "Minnesota", "South Carolina", "Alabama", "Louisiana",
    "Kentucky", "Oregon", "Oklahoma", "Connecticut", "Iowa", "Nevada", "Kansas", "Nebraska", "New Mexico",
    "Illinois", "Massachusetts", "Arkansas", "Mississippi", "Idaho", "Maine", "West Virginia",
]
SPECIALTIES = [
    "family medicine", "internal medicine", "cardiology", "gastroenterology", "endocrinology", "nephrology",
    "pulmonology", "oncology", "orthopedics", "neurology", "psychiatry", "dermatology", "ophthalmology",
    "urology", "OB/GYN", "pediatrics", "geriatrics", "rheumatology", "hospital medicine", "emergency medicine",
    "general surgery", "podiatry", "infectious disease", "pain management", "physical therapy",
]
SYSTEMS = [
    "Epic", "Oracle Cerner", "athenahealth", "eClinicalWorks", "NextGen", "Allscripts", "Greenway",
    "Meditech", "Practice Fusion", "Kareo", "AdvancedMD", "a small practice's own Word template",
    "a hospital's own print layout", "a typed letter on letterhead",
]
VENDORS = [
    "a national records retrieval company", "a health plan's HEDIS team", "a risk adjustment vendor",
    "a release-of-information copy service", "a Medicare Advantage plan", "a Medicaid managed care plan",
    "a coding review company", "a hospital HIM department",
]
LENGTHS = ["short", "medium", "medium", "long"]

SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"brief": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["brief", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pages"],
    "additionalProperties": False,
}


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_API_KEY" and value.strip():
                os.environ["ANTHROPIC_API_KEY"] = value.strip().strip('"').strip("'")
                return
    raise SystemExit("Put ANTHROPIC_API_KEY=... in b_jnk/.env (it is git-ignored), or export it.")


def read_plan() -> list[dict]:
    with PLAN.open(newline="", encoding="utf-8") as fh:
        return [dict(row, count=int(row["count"])) for row in csv.DictReader(fh)]


def page_brief(rng: random.Random, row: dict) -> str:
    length = "short" if "short page" in (row["twist"] or "") else rng.choice(LENGTHS)
    who = rng.choice(VENDORS) if row["label"] in {"JUNK_RECORD_REQUEST", "JUNK_INSTRUCTIONS"} else rng.choice(SYSTEMS)
    return f"{length}; {rng.choice(STATES)}; {rng.choice(SPECIALTIES)}; style of {who}"


def build_requests() -> list[dict]:
    """One request per PAGES_PER_REQUEST pages of one kind. custom_id names the label and kind."""
    rng = random.Random(SEED)
    requests = []
    for k, row in enumerate(read_plan()):
        left = row["count"]
        part = 0
        while left > 0:
            n = min(PAGES_PER_REQUEST, left)
            briefs = [page_brief(rng, row) for _ in range(n)]
            lines = [
                f"Write {n} pages, all labeled {row['label']}, page kind: {row['page_kind']}.",
            ]
            if row["twist"]:
                lines.append(f"Every page in this set: {row['twist']}.")
            lines.append("One page per brief, in order (length; US state; specialty; style):")
            lines += [f"{i}. {b}" for i, b in enumerate(briefs, start=1)]
            lines.append('Return JSON: {"pages": [{"brief": 1, "text": "..."}, ...]}.')
            requests.append({
                "custom_id": f"k{k:02d}-p{part:02d}",
                "label": row["label"],
                "page_kind": row["page_kind"],
                "hard_negative": row["label"] == "KEEP" and bool(row["twist"]),
                "prompt": "\n".join(lines),
                "pages": n,
            })
            left -= n
            part += 1
    return requests


def params(prompt: str) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        "system": [{"type": "text", "text": GUIDE.read_text(encoding="utf-8"), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": prompt}],
    }


def raw_pages() -> list[dict]:
    rows = []
    for path in sorted(RAW.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def verify_requests() -> list[dict]:
    """Pages not yet verified, VERIFY_PAGES_PER_REQUEST to a request, shuffled across labels."""
    done = set()
    if VERIFIED.is_file():
        done = {json.loads(line)["id"] for line in VERIFIED.read_text(encoding="utf-8").splitlines() if line.strip()}
    pages = [p for p in raw_pages() if p["id"] not in done]
    random.Random(3).shuffle(pages)
    out = []
    for start in range(0, len(pages), VERIFY_PAGES_PER_REQUEST):
        chunk = pages[start:start + VERIFY_PAGES_PER_REQUEST]
        body = "\n\n".join(f"=== PAGE {i} ===\n{p['text']}" for i, p in enumerate(chunk, start=1))
        out.append({
            "custom_id": f"verify-{start // VERIFY_PAGES_PER_REQUEST:04d}",
            "pages": chunk,
            "params": {
                "model": MODEL,
                "max_tokens": 16000,
                "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": VERIFY_SCHEMA}},
                "system": [{
                    "type": "text",
                    "text": GUIDE.read_text(encoding="utf-8") + VERIFY_INSTRUCTIONS,
                    "cache_control": {"type": "ephemeral"},
                }],
                "messages": [{"role": "user", "content": f"Label these {len(chunk)} pages.\n\n{body}"}],
            },
        })
    return out


def save_state(batch_id: str, kind: str, requests: int) -> None:
    state = json.loads(STATE.read_text()) if STATE.is_file() else []
    state.append({"batch_id": batch_id, "kind": kind, "requests": requests,
                  "created": time.strftime("%Y-%m-%d %H:%M")})
    STATE.write_text(json.dumps(state, indent=2))


def last_batch(kind: str) -> str:
    if not STATE.is_file():
        raise SystemExit("No batch submitted yet.")
    found = [b for b in json.loads(STATE.read_text()) if b.get("kind", "generate") == kind]
    if not found:
        raise SystemExit(f"No {kind} batch submitted yet.")
    return found[-1]["batch_id"]


def wait(client, batch_id: str):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        c = batch.request_counts
        print(f"{batch_id}: {c.processing} processing, {c.succeeded} done, {c.errored} errored", flush=True)
        time.sleep(60)


def write_pages(req: dict, message) -> int:
    """Save one response as synthetic_v2/raw/<custom_id>.jsonl. Returns pages written."""
    if message.stop_reason in {"refusal", "max_tokens"}:
        print(f"  {req['custom_id']}: stopped ({message.stop_reason}), skipped", flush=True)
        return 0
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        pages = json.loads(text)["pages"]
    except (json.JSONDecodeError, KeyError):
        print(f"  {req['custom_id']}: unreadable JSON, skipped", flush=True)
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, page in enumerate(pages):
        body = str(page.get("text") or "").strip()
        if not body:
            continue
        rows.append({
            "id": f"{req['custom_id']}-{i:02d}",
            "label": req["label"],
            "page_kind": req["page_kind"],
            "hard_negative": req["hard_negative"],
            "text": body,
            "text_sha1": hashlib.sha1(" ".join(body.lower().split()).encode()).hexdigest()[:12],
        })
    with (RAW / f"{req['custom_id']}.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def cmd_plan(_args) -> int:
    reqs = build_requests()
    by_label: dict[str, int] = {}
    for r in reqs:
        by_label[r["label"]] = by_label.get(r["label"], 0) + r["pages"]
    total = sum(by_label.values())
    print(f"{len(reqs)} requests, {total} pages: {by_label}")
    # ~450 output tokens a page (text + JSON) plus thinking; batch price is half of $20 / M output.
    low, high = total * 450 * 1.3 / 1e6 * 10, total * 450 * 2.5 / 1e6 * 10
    print(f"rough cost with the batch discount: ${low:.0f} to ${high:.0f}")
    print("\nfirst request:\n" + reqs[0]["prompt"])
    return 0


def cmd_sample(args) -> int:
    import anthropic

    load_key()
    client = anthropic.Anthropic()
    reqs = build_requests()
    rng = random.Random()
    # One request from a few different kinds, so the sample shows variety.
    by_kind: dict[str, dict] = {}
    for r in reqs:
        by_kind.setdefault(r["page_kind"], r)
    for req in rng.sample(list(by_kind.values()), args.kinds):
        print(f"{req['custom_id']} {req['label']} / {req['page_kind']} ...", flush=True)
        message = client.messages.create(**params(req["prompt"]))
        print(f"  {write_pages(req, message)} pages, {message.usage.output_tokens} output tokens", flush=True)
    print(f"Read them in {RAW}")
    return 0


def cmd_submit(_args) -> int:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    load_key()
    client = anthropic.Anthropic()
    done = {p.stem for p in RAW.glob("*.jsonl")} if RAW.is_dir() else set()
    reqs = [r for r in build_requests() if r["custom_id"] not in done]
    if not reqs:
        print("Every request already has a raw file.")
        return 0
    batch = client.messages.batches.create(requests=[
        Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**params(r["prompt"])))
        for r in reqs
    ])
    save_state(batch.id, "generate", len(reqs))
    print(f"Submitted {len(reqs)} requests as batch {batch.id}. Run `collect` to wait for it.")
    return 0


def cmd_collect(_args) -> int:
    import anthropic

    load_key()
    client = anthropic.Anthropic()
    batch_id = last_batch("generate")
    wait(client, batch_id)
    by_id = {r["custom_id"]: r for r in build_requests()}
    pages = failed = 0
    for result in client.messages.batches.results(batch_id):
        req = by_id.get(result.custom_id)
        if req is None:
            continue
        if result.result.type == "succeeded":
            pages += write_pages(req, result.result.message)
        else:
            failed += 1
    print(f"Wrote {pages} pages to {RAW}; {failed} requests failed (run `submit` again to redo only those).")
    return 0


def cmd_verify(_args) -> int:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    load_key()
    reqs = verify_requests()
    if not reqs:
        print("Every page is already verified.")
        return 0
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=[
        Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"])) for r in reqs
    ])
    save_state(batch.id, "verify", len(reqs))
    print(f"Submitted {sum(len(r['pages']) for r in reqs)} pages in {len(reqs)} requests as batch {batch.id}. "
          f"Run `verify-collect` to wait for it.")
    return 0


def cmd_verify_collect(_args) -> int:
    import anthropic

    load_key()
    client = anthropic.Anthropic()
    batch_id = last_batch("verify")
    wait(client, batch_id)
    by_id = {r["custom_id"]: r for r in verify_requests()}
    agree = disagree = 0
    with VERIFIED.open("a", encoding="utf-8") as fh:
        for result in client.messages.batches.results(batch_id):
            req = by_id.get(result.custom_id)
            if req is None or result.result.type != "succeeded":
                continue
            message = result.result.message
            if message.stop_reason in {"refusal", "max_tokens"}:
                continue
            text = next((b.text for b in message.content if b.type == "text"), "")
            try:
                labels = {int(x["page"]): x["label"] for x in json.loads(text)["labels"]}
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            for i, page in enumerate(req["pages"], start=1):
                if i not in labels:
                    continue
                ok = labels[i] == page["label"]
                agree += ok
                disagree += not ok
                fh.write(json.dumps({"id": page["id"], "intended": page["label"], "verified": labels[i],
                                     "page_kind": page["page_kind"], "agree": ok}) + "\n")
    print(f"{agree} pages agree, {disagree} disagree (left out of training) -> {VERIFIED}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sample = sub.add_parser("sample")
    sample.add_argument("--kinds", type=int, default=3)
    sub.add_parser("submit")
    sub.add_parser("collect")
    sub.add_parser("verify")
    sub.add_parser("verify-collect")
    args = ap.parse_args()
    return {"plan": cmd_plan, "sample": cmd_sample, "submit": cmd_submit, "collect": cmd_collect,
            "verify": cmd_verify, "verify-collect": cmd_verify_collect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
