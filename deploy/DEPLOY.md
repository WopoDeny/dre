# Deploying the Document Reconstruction Engine

The module turns a PDF, scan, or photo of a document into an editable DOCX. If the module is not
confident in the result, it refuses with a code and a reason; a document with errors is never issued.

## 1. Machine requirements

| | target server | minimum |
|---|---|---|
| CPU | 1 core x86-64 (Xeon) | 1 core x86-64 (2015+) |
| Memory | 128 GB (memory is not a constraint) | 4 GB (OS + 1 worker; no warm spare worker) |
| Container memory limit | do not set (`--cpus=1` without `--memory`) | `--memory=2g` (process peak 1.3 GB) |
| Disk | 1.5 GB (image) + input and output files | same |
| GPU / network | not needed, everything runs offline | — |

OCR models, dictionaries, and word lists stay loaded in the worker permanently (a "warm pool"); on 1 core
one document is processed at a time, plus one warm spare worker (with ~6 GB of memory or more) that picks up
the work immediately if the first is stopped on deadline.

Time on a single core of the reference laptop (after the 26.09 optimization): scan 1 page — 2.7 s, 3 pages — 8.6 s,
7 pages — 18.9 s, 10 pages — 26.5 s; an ordinary PDF with a text layer — 0.1–0.5 s. A server Xeon may be slower
per core — the module measures this at start-up (section 11).

Time on a single core (reference laptop): an ordinary PDF with a text layer — 0.1–0.5 s per document,
scan/photo — 3–8 s per page. The limit is 45 s per document and no more than 10 pages (`DRE_MAX_PAGES`).

## 2. Installation

### Docker (main method)

```sh
docker build -t dre:1 .                       # from the repository root
docker run --rm --cpus=1 dre:1 selfcheck
```

With no internet on the server: build on a machine that has internet and transfer the image as a file.

```sh
docker save dre:1 | gzip > dre-1.tar.gz       # on the build machine
docker load < dre-1.tar.gz                    # on the server
```

Verify the image with a single command (build, self-check on 1 core, byte-for-byte DOCX comparison
against a non-Docker install): `deploy/docker-check.sh`. From WSL without Docker Desktop integration:
`DOCKER=".../docker.exe" deploy/docker-check.sh` (files are streamed into the container, without mounting).

### Without Docker (Ubuntu/Debian)

```sh
sudo deploy/install.sh            # installs into /opt/dre, commands into /usr/local/bin
```

The script installs the packages (Tesseract and its models, the Hunspell ru/kk/uz/en dictionaries, the
Liberation fonts, Python), creates the `/opt/dre/venv` environment, builds the word lists in
`/opt/dre/lexicon`, and runs the self-check. An identical result to Docker is guaranteed on the same
package versions (Ubuntu 26.04, Tesseract 5.5.0); on a different version the script warns.

## 3. Running

| what | Docker | without Docker |
|---|---|---|
| HTTP service on port 8080 | `docker run -d --restart=unless-stopped -p 8080:8080 -e DRE_API_KEY=… --cpus=1 dre:1` | `DRE_API_KEY=… DRE_BIND=0.0.0.0:8080 dre-serve` |
| a single file | `docker run --rm --user $(id -u):$(id -g) -v $PWD:/data dre:1 convert /data/a.pdf --output /data/a.docx` | `dre-convert a.pdf --output a.docx` |
| a folder | `docker run --rm --user $(id -u):$(id -g) -v $PWD:/data dre:1 batch /data/in /data/out` | `dre-batch in out` |
| folder summary | `docker run --rm -v $PWD:/data dre:1 summary /data/out` | `dre-summary out` |
| self-check | `docker run --rm dre:1 selfcheck` | `dre-selfcheck` |

`--user $(id -u):$(id -g)` — so the results in the mounted folder belong to you, not to the container user.
The service is ready 5–10 s after start (loading models and calibrating speed); until then `/ready` returns 503.

## 4. Self-check

`selfcheck` verifies Tesseract and its models, the dictionaries, the word lists, the fonts, the cores and
memory, measures machine speed, and runs 6 control files: an ordinary PDF, three scans (ru, kk, Uzbek Latin),
a photo of a sheet at an angle with its text compared against a reference, and a low-resolution photo that must
be refused. The output is in English; at the end it prints "OK" or a list of PROBLEMS; the return code is 0 or 1.
Run it after installation, after an update, and after moving to a new machine.

## 5. Batch mode (folder)

`batch IN OUT` processes all PDF, JPEG, PNG, WEBP, and HEIC files from the `IN` folder, including subfolders:

- `OUT/<name>.docx` — the issued documents; for files from subfolders the name is assembled as `a/b/x.pdf` → `a__b__x.docx`;
- `OUT/_refused/` — refusals: a copy of the source file and `<name>.json` with the code, reason, and time (`--refused DIR` sets a different folder);
- `OUT/journal.jsonl` — one line per document: file, outcome, reason, seconds, pages.

`summary OUT` — a summary of the journal: issued, refusals by reason, time, a list of the issued DOCX files to review.
On a repeat run, already-processed files are skipped (`--again` — process again).
`--workers N` — how many documents to process at once; there is no point setting more than the number of cores (on 1 core — 1).

## 6. API

Every response carries an `X-Request-ID` header.

| request | response |
|---|---|
| `GET /health` | 200 `{"status": "alive"}` — the process is alive |
| `GET /ready` | 200 — ready to accept documents, 503 — still loading or no `DRE_API_KEY`. In the body: OCR languages, `ocr_capabilities.speed` (calibration), `page_threads` |
| `POST /convert` | body — the whole file (not multipart), up to 20 MB |

Conversion request:

```sh
curl -H "Authorization: Bearer $DRE_API_KEY" -H "Content-Type: application/pdf" \
     --data-binary @letter.pdf -o letter.docx http://server:8080/convert
```

`Content-Type`: `application/pdf`, `image/jpeg`, `image/png`, `image/webp`, `image/heic`, `image/heif`.

Success: 200, the body is a DOCX (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`),
and the `X-Conversion-MS` header is the time spent on the server.

Refusal or error: JSON

```json
{"error": {"code": "UNREADABLE_DOCUMENT", "message": "…", "stage": "qa", "request_id": "…",
           "details": {"page_index": 0, "reason": "unreliable_text", "words": "…"}},
 "metrics": {"total_ms": 4120.5, "pages_total": 1}}
```

## 7. Refusal codes

| HTTP | `code` | what it means | what to do |
|---|---|---|---|
| 400 | `INVALID_PDF` | not a PDF and not an image, file is corrupt, empty body | send a valid file |
| 401 | `UNAUTHORIZED` | no key or wrong key | add the `Authorization: Bearer <DRE_API_KEY>` header |
| 413 | `UPLOAD_TOO_LARGE` | larger than 20 MB | reduce the file |
| 415 | `INVALID_PDF` | unsupported `Content-Type` | see the list of types above |
| 422 | `UNREADABLE_DOCUMENT` | the document cannot be issued without risk of errors, the reason is in `details.reason` | see the table below; usually — rescan |
| 422 | `LANGUAGE_UNAVAILABLE` | the script is outside those allowed (stage 1: Cyrillic and Latin) | — |
| 422 | `DOCUMENT_TOO_EXPENSIVE` | projected not to finish within 45 s (many pages, slow machine) or the memory limit was exceeded | split the document, give more cores |
| 503 | `SERVICE_BUSY` | all workers are busy | retry the request later |
| 503 | `NOT_READY` | the service is loading or no key is set | wait for `/ready` |
| 504 | `FAST_SLA_EXCEEDED` | did not finish within 45 s | split the document, give more cores |
| 500 | `CONVERSION_FAILED` | internal error | send the file and the `request_id` to the developers |

`details.reason` values for `UNREADABLE_DOCUMENT`:

| `reason` | what it means |
|---|---|
| `low_resolution` | letters smaller than 8 px (`letter_height_px`): a low-resolution scan or a small font on a photo |
| `unreliable_text` | there are words that could not be read reliably; `words` holds those words (text\|second reading) |
| `unreliable_text_near_handwriting` | printed text next to a handwritten date or number was read with low confidence |
| `unverifiable_text` | too many words that cannot be checked against the dictionary |
| `text_not_read` | most of the text on the page was not recognized (`letters_read_share`) |
| `unrecognized_text` | part of the page looks like text but was not recognized (a different alphabet, text under a stamp) |
| `unresolved_graphic_text_overlap` | a stamp or signature covers text that could not be read |
| `unresolved_table_cells`, `unresolved_ruled_raster_table`, `unresolved_borderless_raster_table` | the table could not be reconstructed reliably |
| `script_not_allowed_ocr`, `script_not_allowed_native` | text in a forbidden alphabet (Arabic, Han at stage 1) |
| `too_many_pages` | more pages than `DRE_MAX_PAGES` |

`details.page_index` — the page number (0-based) where the problem was found.

## 8. Logs

The service writes to stderr (`docker logs`, the gunicorn journal) one JSON line per conversion:

```json
{"time": "2026-09-25T18:57:48+00:00", "request_id": "7b68…", "status": 422, "ms": 714,
 "code": "UNREADABLE_DOCUMENT", "reason": "low_resolution", "stage": "geometry", "page": 0, "pages": 1, "docx_bytes": null}
```

Document contents never reach the journal. Batch mode writes `OUT/journal.jsonl` and a JSON file next to each refusal.

## 9. Settings (environment variables)

| variable | default | what it does |
|---|---|---|
| `DRE_API_KEY` | — (required for the service) | access key for `/convert` |
| `DRE_OCR_ALLOWED_SCRIPTS` | `Latin,Cyrillic` | alphabets for scans and photos |
| `DRE_NATIVE_ALLOWED_SCRIPTS` | `Latin,Cyrillic` | alphabets for PDFs with a text layer |
| `DRE_MAX_PAGES` | `10` | more pages — refusal before recognition |
| `DRE_MIN_LETTER_PX` | `8` | letter-height threshold (px) for the `low_resolution` refusal |
| `DRE_PAGE_THREADS` | `auto` | pages at once (by cores and memory, at most 4) |
| `DRE_WORKERS` | `auto` | worker processes: by cores + 1 warm spare if memory allows |
| `DRE_CONCURRENCY` | by cores | documents at once (on 1 core — 1; extra requests get `SERVICE_BUSY`) |
| `DRE_WORKER_RSS_MB` | 8000 with memory ≥ 16 GB, otherwise 2200 + 400 per thread | worker memory limit (leak protection) |
| `DRE_SPEED_FACTOR` | measured at start | set the speed multiplier manually |
| `DRE_LEXICON_DIR` | `/app/lexicon` (Docker) | where the word lists live |

## 10. Updating

Docker: build a new image (`docker build -t dre:2 .`) or load it (`docker load`),
run `docker run --rm dre:2 selfcheck`, then replace the container. Without Docker: fetch the new
repository version and re-run `sudo deploy/install.sh`; it reinstalls the environment and runs the self-check.
The same input gives the same DOCX on any machine with the same image version.

## 11. A slow core (server Xeon)

- At start-up the module recognizes a reference page and compares the time with the laptop core the
  measurements were taken on (the value is in `/ready`, `ocr_capabilities.speed.factor`; 1.0 — as fast as the
  reference, 1.5 — one and a half times slower). The self-check prints this multiplier and the projection for a single-page scan.
- Before recognition: projection = 3.3 s per scan page × multiplier × 1.1 (on the laptop the median is 2.8 s, 95% of pages —
  up to 4.8 s). More than 45 s — an immediate `DOCUMENT_TOO_EXPENSIVE` refusal, `reason: estimated_time_over_budget`.
  How many scan pages fit: multiplier 1.0 — up to 11, 1.3 — up to 9, 1.5 — up to 7, 2.0 — up to 5 (with `DRE_MAX_PAGES` 10).
- During processing: before each next page — the slowest page of this document × the remaining pages
  × 1.1 + 1.5 s must fit in the remaining time, otherwise the same refusal immediately, rather than a timeout at the end.
- If a required verification step does not have time to run (re-reading, a third reading of numbers), the document is not
  issued with unverified text: it is refused on projection. A slow machine issues fewer long documents,
  but the DOCX files it does issue are the same as on a fast one.
- Verified on a half-loaded core (multiplier 2.08): 1 and 3 pages issued in 5.5 and 17.5 s,
  7 and 10 pages — refused in 0.0 s; with a deliberately under-estimated projection — refusal after the first page, not a timeout.

## 12. Monday checklist (production server: 1 Xeon core, 128 GB, no GPU)

1. Clone (private repository; private samples are not in the repository — they are in `samples-private.zip`):
   ```
   git clone git@github.com:<ORG>/dre.git && cd dre && git checkout v0.2.1-stage1
   ```
2. Build the image (5–10 min; only Docker is needed, sudo is not required if the user is in the docker group):
   ```
   docker build -t dre:0.2.1 .
   ```
3. Self-check on a single core — the last line must be `OK`:
   ```
   docker run --rm --cpus=1 dre:0.2.1 selfcheck
   ```
   Record the `speed: {'factor': …}` line — the Xeon's speed multiplier relative to the laptop (1.0 — as fast as the laptop).
   If it is above 1.3, long scans will be rejected on time more often (section 11); the projection for a single-page
   scan must be below 45 s, otherwise the output will say `NO`.
4. Service key — only through the environment (template: `.env.example`), never in repository files:
   ```
   python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/.dre_key && chmod 600 ~/.dre_key
   docker run -d --restart=unless-stopped --cpus=1 -p 8080:8080 -e DRE_API_KEY="$(cat ~/.dre_key)" dre:0.2.1
   curl -s localhost:8080/ready
   ```
5. Pilot in folder mode (without the service): put the documents in `~/pilot/in`, then
   ```
   mkdir -p ~/pilot/out
   docker run --rm --cpus=1 --user $(id -u):$(id -g) -v ~/pilot:/data dre:0.2.1 batch /data/in /data/out
   docker run --rm -v ~/pilot:/data dre:0.2.1 summary /data/out
   ```
   The issued DOCX files are in `~/pilot/out`, the refusals with a reason in `~/pilot/out/_refused`, the journal in `journal.jsonl`.
   Review the first 50–100 issued documents by eye (the risks are in `handoff/FINAL_REPORT.md`).
6. Optional: comparison against a reference on the hires test set — unpack `hires-rasters.zip` into the repository
   root and run the `handoff/samples/hires/scan_mild` folder (and others) with the command from step 5: 74 of 75 are expected to be issued.
