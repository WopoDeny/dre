#!/bin/sh
# Install Document Reconstruction Engine without Docker (Ubuntu / Debian), one command:
#
#   sudo deploy/install.sh [TARGET]          (TARGET defaults to /opt/dre)
#
# Installs the system packages (Tesseract, models, Hunspell dictionaries, fonts, Python),
# creates TARGET/venv with the engine, builds the word lists in TARGET/lexicon, puts
# the commands dre-convert, dre-batch, dre-summary, dre-selfcheck, dre-serve into /usr/local/bin and
# runs the self-check. The same DOCX as the Docker image is guaranteed with the same
# package versions (Ubuntu 26.04: Tesseract 5.5.0); other versions are reported.
set -eu

TARGET="${1:-/opt/dre}"
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "Root rights are needed: sudo $0 $*" >&2
    exit 1
fi
if ! command -v apt-get >/dev/null 2>&1; then
    echo "The script is for Ubuntu/Debian (apt-get). Use Docker on other systems." >&2
    exit 1
fi

echo "== System packages"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates \
    libtesseract5 tesseract-ocr tesseract-ocr-osd tesseract-ocr-eng \
    tesseract-ocr-rus tesseract-ocr-kaz tesseract-ocr-kir tesseract-ocr-uzb tesseract-ocr-uzb-cyrl \
    tesseract-ocr-tgk tesseract-ocr-aze tesseract-ocr-aze-cyrl \
    tesseract-ocr-script-latn tesseract-ocr-script-cyrl \
    hunspell-ru hunspell-kk hunspell-uz hunspell-en-us \
    fonts-liberation fonts-liberation2

VERSION="$(tesseract --version 2>&1 | head -1)"
case "$VERSION" in
    "tesseract 5.5.0"*) ;;
    *) echo "WARNING: $VERSION (reference 5.5.0): recognition may differ slightly from the Docker image." ;;
esac

echo "== Engine in $TARGET"
mkdir -p "$TARGET"
python3 -m venv "$TARGET/venv"
"$TARGET/venv/bin/pip" install --no-cache-dir --upgrade pip >/dev/null
"$TARGET/venv/bin/pip" install --no-cache-dir "$SOURCE[server]"
DRE_LEXICON_DIR="$TARGET/lexicon" "$TARGET/venv/bin/python" -c \
    "from pathlib import Path; from document_reconstruction.recognition.lexicon import build; print('word lists:', build(Path('/usr/share/tesseract-ocr/5/tessdata'), Path('$TARGET/lexicon')))"

for command in convert batch summary selfcheck serve; do
    cat > "/usr/local/bin/dre-$command" <<EOF
#!/bin/sh
export DRE_LEXICON_DIR="$TARGET/lexicon" OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1
case "$command" in
    convert) exec "$TARGET/venv/bin/python" -m document_reconstruction "\$@" ;;
    serve) exec "$TARGET/venv/bin/gunicorn" --bind "\${DRE_BIND:-127.0.0.1:8080}" --workers 1 --threads 4 --timeout 60 \\
               --graceful-timeout 55 --keep-alive 2 document_reconstruction.service.wsgi:application ;;
    *) exec "$TARGET/venv/bin/python" -m document_reconstruction $command "\$@" ;;
esac
EOF
    chmod 755 "/usr/local/bin/dre-$command"
done

echo "== Self-check"
/usr/local/bin/dre-selfcheck
