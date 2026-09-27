#!/bin/sh
# Build the image and check it the way the target server runs it: one core.
#
#   deploy/docker-check.sh [IMAGE]            (IMAGE defaults to dre:latest)
#
# 1. docker build; 2. self-check in the container (--cpus=1);
# 3. the same folder converted in the container and by the local installation (.venv):
#    every DOCX must be byte-identical, every refusal must have the same reason.
# DOCKER=docker.exe (Docker Desktop's Windows client, from WSL without the WSL integration) is supported:
# the build context is then passed as a Windows path.
set -eu

DOCKER="${DOCKER:-docker}"
host_path() {
    case "$DOCKER" in
        *.exe) wslpath -w "$1" ;;
        *) echo "$1" ;;
    esac
}

IMAGE="${1:-dre:latest}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== Build $IMAGE"
"$DOCKER" build -t "$IMAGE" "$(host_path "$ROOT")"

echo "== Self-check in the container (1 core)"
"$DOCKER" run --rm --cpus=1 "$IMAGE" selfcheck

echo "== Comparison with the installation without Docker"
mkdir -p "$WORK/in" "$WORK/docker" "$WORK/local"
for name in scan_mild/ru_letter_ministry scan_medium/ky_letter_mayor scan_mild/kz_order_school photo_angle/ky_certificate \
            scan_mild/uz_latn_order scan_medium/tg_letter photo_angle/ru_act_acceptance scan_mild/ru_protocol_two_pages; do
    cp "$ROOT/handoff/samples/hires/$name.pdf" "$WORK/in/$(echo "$name" | tr / _).pdf"
done
cp "$ROOT/handoff/samples/native/01_uz_mvd_letter_from_etalon.pdf" "$WORK/in/native_01.pdf"
chmod -R a+rwX "$WORK"
# The documents go in and the results come out as a tar stream: no bind mount needed (works with docker.exe too).
tar -C "$WORK" -cf - in | "$DOCKER" run --rm -i --cpus=1 --entrypoint sh "$IMAGE" -c \
    'mkdir -p /tmp/w && tar -xf - -C /tmp/w && python -m document_reconstruction batch /tmp/w/in /tmp/w/docker >&2 && tar -cf - -C /tmp/w docker' \
    | tar -xf - -C "$WORK"
OMP_THREAD_LIMIT=1 taskset -c 0 "$ROOT/.venv/bin/python" -m document_reconstruction batch "$WORK/in" "$WORK/local"

"$ROOT/.venv/bin/python" - "$WORK/docker" "$WORK/local" <<'PY'
import hashlib, json, sys
from pathlib import Path
docker, local = Path(sys.argv[1]), Path(sys.argv[2])
def outcomes(folder):
    result = {}
    for line in (folder / "journal.jsonl").read_text().splitlines():
        record = json.loads(line)
        name = Path(record["file"]).stem
        if record["outcome"] == "issued":
            result[name] = "docx " + hashlib.sha256((folder / record["docx"]).read_bytes()).hexdigest()[:16]
        else:
            result[name] = "refused " + str(record.get("reason"))
    return result
a, b = outcomes(docker), outcomes(local)
different = sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))
for name in sorted(a):
    print(f"{name:40} docker: {a.get(name):28} local: {b.get(name)}")
print("SAME" if not different else "DIFFERENCES: " + ", ".join(different))
sys.exit(1 if different else 0)
PY
