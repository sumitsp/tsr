# b_jnk_creator

Label chart images as Keep, Blank, or a junk subtype. The page runs on this machine. Images are read from a local folder or from Azure Blob. The CSV is written locally.

Each row is one image: `value` (`KEEP`, `BLANK`, or `JUNK`) and `subtype` (empty for Keep and Blank; for Junk one of Invoice, Cover page, Record request, Instructions, Letter / fax, Others). Labeling it again updates that row. Empty fields are allowed.

## Setup

```bash
pip install -r b_jnk_creator/requirements.txt
az login
```

Edit the RUN CONFIG block at the top of `main.py`:

- `INPUT_SOURCE`: `"local"` or `"blob"`.
- Local: `LOCAL_INPUT` is the folder that contains the chart folders, so `LOCAL_INPUT\<folder name>\<image name>` is the file.
- Blob: `CONTAINER_NAME` and `PREFIX` (`PREFIX/<chart folder>/<image>`). The account is `azsadve2aipoc`. Sign in with `az login` first.
- `START_FROM`: leave empty to begin at the first folder. Set a folder name to begin at that folder and continue through the later folders.
- `CSV_PATH`: where `blank_junk_labels.csv` is written. Close it in Excel while labeling, or a save will fail until you close it.

## Run

```bash
python b_jnk_creator/main.py
```

Open the URL it prints (http://127.0.0.1:8767). Images play in order: folder 1 image 1, folder 1 image 2, then the next folder.

Marks stay on the current image until Enter. Enter writes the whole row, including empty fields, and opens the next image. Press the same key again to clear that mark. Going back loads the saved row so it can be changed or cleared.

| Key | Action |
| --- | --- |
| K | Keep |
| B | Blank |
| I | Junk / Invoice |
| C | Junk / Cover page |
| R | Junk / Record request |
| N | Junk / Instructions |
| F | Junk / Letter or fax |
| O | Junk / Others |
| Enter or Right | Save this page, empty fields included, and open the next image |
| Left | Previous image. Unsaved marks on this page are saved first. |
| Up | First image of the previous folder |

Back and Previous folder do the same thing with the mouse. Folder and Image, or an image number (1 is the first image of the first folder), jump to a specific page.

Refresh keeps your place. With no saved place, it opens on the first image that has no row yet.

## Tag junk pages from a review sheet

`junk_tagger.py` opens only the pages a review sheet marked as junk, so you only pick the junk type.

The review sheet is an `.xlsx` or `.csv` with a folder column, an id column (`1`, `1.png`), a `Junk Page` column and a `Blank Page` column (Yes / No). The columns are found by name. If a name differs, set `FOLDER_COLUMN`, `ID_COLUMN`, `JUNK_COLUMN` or `BLANK_COLUMN` in RUN CONFIG.

- Junk Page = Yes: the image opens and you tag its type (I, C, R, N, F, O). K and B still work if the sheet was wrong.
- Blank Page = Yes: written as `BLANK` when the tool starts, without opening the image.
- Rows go into the same `CSV_PATH` that `main.py` writes. A page that already has a row is never overwritten.

Edit RUN CONFIG at the top of `junk_tagger.py` (`REVIEW_SHEET`, `LOCAL_INPUT` or blob settings, `CSV_PATH`). `LOCAL_INPUT` can be one folder or a list of folders; each is searched at any depth for the chart folders the sheet names. Then:

```bash
python b_jnk_creator/junk_tagger.py
```

Open http://127.0.0.1:8768. Keys and buttons are the same as `main.py`. The terminal says how many blank rows were appended and which junk rows had no image.

## Retrain the page model on these labels

```bash
python b_jnk/scripts/build_annotated_pages.py   # labels CSV + Tesseract txt -> data/raw/annotated_pages.jsonl
python b_jnk/scripts/train_bert.py              # synthetic + real + annotated pages; checks on real pages, then saves models/bert_page
python b_jnk/scripts/train_baseline.py          # TF-IDF word model on real + annotated pages (used when BERT is missing)
```

Set `LABELS_CSV` and `OCR_DIR` (the Tesseract output folder, `<folder>/<folder>.txt`) in `build_annotated_pages.py` first. `infer.py` then uses the new model, and the word rules (invoice, cover sheet, letter / fax, end of record) still run on top of it.
