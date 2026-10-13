# Retrain the page model

Two machines, and nothing private crosses between them:

- **Mac**: makes synthetic pages (Claude API + Tesseract). Never sees real pages.
- **Windows**: holds the real pages and labels, trains, and runs the model. No LLM.
- **git** carries code and synthetic pages Mac -> Windows. From Windows back to the Mac only a text-free mistake summary goes, read out in words.

## 1. Mac: make synthetic pages

```bash
cd b_jnk
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env                         # git-ignored
.venv/bin/python scripts/generate_synthetic_v2.py plan              # pages per label, rough cost
.venv/bin/python scripts/generate_synthetic_v2.py sample --kinds 4  # a few requests now; read synthetic_v2/raw/*.jsonl
.venv/bin/python scripts/generate_synthetic_v2.py submit            # all of synthetic_v2/plan.csv, one batch, half price
.venv/bin/python scripts/generate_synthetic_v2.py collect           # waits (usually under an hour) -> synthetic_v2/raw/
.venv/bin/python scripts/generate_synthetic_v2.py verify            # blind second pass labels every page
.venv/bin/python scripts/generate_synthetic_v2.py verify-collect    # -> synthetic_v2/verified.jsonl
.venv/bin/python scripts/render_synthetic_v2.py --limit 20 --save-images   # look at synthetic_v2/preview/*.png
.venv/bin/python scripts/render_synthetic_v2.py                     # agreed pages, printed + Tesseract -> synthetic_v2/clean.jsonl
```

Commit and push `synthetic_v2/`.

Why the pages can be trusted:
- `label_guide.md` defines labels by purpose, the way pages are tagged in b_jnk_creator, and tells the writer to use real, correct wording with no typos.
- The blind pass labels each page without knowing what it was meant to be. A page the two passes disagree on is left out, so a "junk" page that reads like a clinical note never reaches training.
- Pages are OCR'd by the same Tesseract call as `ts_ocr.py`, so the model sees synthetic text in the same form as real text.

## 2. Windows: more real labels

```bash
git pull
python b_jnk/scripts/find_junk_candidates.py        # OCR_DIR, LABELS_CSV, OUT_XLSX at the top
python b_jnk_creator/junk_tagger.py                 # REVIEW_SHEET = the junk_candidates.xlsx it wrote
```

Every page in the sheet opens in the tagger: press K, B, or the junk type. These are the pages the model is unsure about or calls junk, so each one teaches more than a page tagged in chart order.

## 3. Windows: build and train

Back up `b_jnk/models/bert_page` first: training replaces it at the end.

```bash
python b_jnk/scripts/build_annotated_pages.py       # labels CSV + Tesseract txt -> data/raw/annotated_pages.jsonl
python b_jnk/scripts/train_bert.py                  # checks on held-out real charts, then trains and saves models/bert_page
python b_jnk/scripts/error_summary.py               # mistakes on held-out real pages, counts only
```

`train_bert.py` prints the device: `xpu` is the Intel Arc GPU (needs PyTorch with Intel GPU support), `cpu` is an overnight run. To shorten a CPU run, lower `bert.max_length` (512 -> 256) or `bert.epochs` (4 -> 3) in `configs/default.json`.

## 4. Windows: keep it only if it is better

In the `real_2fold, tagged pages only` block, compared with the previous run:
- KEEP pages lost must not go up.
- junk/blank caught should go up.

If it is worse, put the backed-up `bert_page` back.

## 5. Close the loop

Read `reports/error_summary.txt` on Windows. It has no page text: only which labels get confused, which rule decided, page length, fax header, and which word groups were on the page. Describe the failing page types in words on the Mac (for example "fax covers from GI practices called KEEP", "insurance cards missed"), add or raise those rows in `synthetic_v2/plan.csv`, and run step 1 again: `submit` and `verify` only send what is new.
