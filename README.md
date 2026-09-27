# Document Reconstruction Engine

Turns PDFs, scans and photos of official documents into editable DOCX files that look typed in Word.
Stage 1: Cyrillic and Latin text (Russian, Kazakh, Kyrgyz, Uzbek, Tajik, Azerbaijani, Turkmen, English),
native PDF text layers, one CPU core, 45 s per document. A document that cannot be rebuilt reliably is refused
with a reason instead of being issued with errors.

## Run

```
docker build -t dre .
docker run --rm --cpus=1 dre selfcheck                      # must end with "OK"
docker run -d --restart=unless-stopped --cpus=1 -p 8080:8080 -e DRE_API_KEY=... dre
curl -H "Authorization: Bearer $DRE_API_KEY" --data-binary @doc.pdf -o doc.docx localhost:8080/convert
```

Folder mode, one file and all settings: see the documentation.

## Documentation

- `docs/DEPLOY.md` — server setup from scratch, API, settings, refusal codes, logs.
- `docs/PILOT.md` — pilot in folder mode.
- `docs/FINAL_REPORT.md` — measured results, accuracy, time, known risks.
- `docs/TASK.md` — the task and the product rules.

## Layout

- `src/` — the module; `tests/` — tests (`python -m pytest tests`).
- `deploy/` — container entrypoint and install scripts; `Dockerfile`, `.env.example`.
- `dev/` — development material only (samples, tools, runs, fixtures); it is not part of the Docker image.
