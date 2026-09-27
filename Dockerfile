# Document Reconstruction Engine: the same system packages as the reference machine
# (Ubuntu 26.04: Tesseract 5.5.0, tessdata 4.1.0, Hunspell dictionaries, Python 3.14),
# so the image gives the same DOCX as an installation without Docker.
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_THREAD_LIMIT=1 \
    OPENBLAS_NUM_THREADS=1 \
    DRE_WORKERS=1 \
    DRE_LEXICON_DIR=/app/lexicon \
    PATH=/app/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates \
        libtesseract5 tesseract-ocr tesseract-ocr-osd tesseract-ocr-eng \
        tesseract-ocr-rus tesseract-ocr-kaz tesseract-ocr-kir tesseract-ocr-uzb tesseract-ocr-uzb-cyrl \
        tesseract-ocr-tgk tesseract-ocr-aze tesseract-ocr-aze-cyrl \
        tesseract-ocr-script-latn tesseract-ocr-script-cyrl \
        hunspell-ru hunspell-kk hunspell-uz hunspell-en-us \
        fonts-liberation fonts-liberation2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python3 -m venv /app/venv \
    && pip install --no-cache-dir '.[server]' \
    && python -c "from pathlib import Path; from document_reconstruction.recognition.lexicon import build; print(build(Path('/usr/share/tesseract-ocr/5/tessdata'), Path('/app/lexicon')))" \
    && useradd --create-home --uid 10001 converter \
    && mkdir -p /data && chown converter /data
USER converter
EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=3s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=2)"
# Service by default; `docker run IMAGE selfcheck`, `docker run IMAGE batch /data/in /data/out`,
# `docker run IMAGE convert FILE --output OUT` run the other commands.
COPY deploy/entrypoint.sh /usr/local/bin/dre
ENTRYPOINT ["/usr/local/bin/dre"]
CMD ["serve"]
