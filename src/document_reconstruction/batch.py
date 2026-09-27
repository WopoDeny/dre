"""Folder mode: every document of a folder to DOCX.

    dre-batch IN_DIR OUT_DIR [--refused DIR] [--workers N] [--again]
    (or: python -m document_reconstruction batch IN_DIR OUT_DIR)

OUT_DIR/<name>.docx for issued documents; refused documents are copied to the
refused folder (default OUT_DIR/_refused) next to <name>.json with the reason.
OUT_DIR/journal.jsonl has one line per document (outcome, reason, seconds, pages).
Documents already converted or refused are skipped unless --again is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .core.errors import public_error
from .resources import worker_rss_mb
from .service.workers import WarmWorkerPool

ACCEPTED = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dre-batch", description="Convert every document of a folder to DOCX.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--refused", type=Path, default=None, help="folder for refused documents (default OUTPUT/_refused)")
    parser.add_argument("--workers", type=int, default=1, help="documents converted at once (1-4, each needs about 2 GB)")
    parser.add_argument("--again", action="store_true", help="convert documents that already have a result")
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f"not a folder: {args.input}")
    refused_dir = args.refused or args.output / "_refused"
    args.output.mkdir(parents=True, exist_ok=True)
    refused_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in ACCEPTED)

    def target(path: Path) -> str:
        # Subfolders keep documents with equal names apart: a/b/x.pdf -> a__b__x
        return "__".join(path.relative_to(args.input).with_suffix("").parts)

    todo = [path for path in files if args.again or not (
        (args.output / f"{target(path)}.docx").exists() or (refused_dir / f"{target(path)}.json").exists())]
    print(f"Documents: {len(files)}, to convert: {len(todo)}", flush=True)
    if not todo:
        return 0
    workers = max(1, min(4, args.workers))
    pool = WarmWorkerPool(workers=workers, max_worker_rss_mb=worker_rss_mb())
    journal = (args.output / "journal.jsonl").open("a", encoding="utf-8")
    counts = {"issued": 0, "refused": 0}

    def convert(path: Path) -> dict[str, object]:
        name = target(path)
        started = time.perf_counter()
        try:
            result = pool.convert(path.read_bytes(), request_id=name)
            error = result.error
        except Exception as cause:  # noqa: BLE001 - a broken file must not stop the folder
            result, error = None, public_error(cause).to_dict(request_id=name)
        seconds = round(time.perf_counter() - started, 2)
        record = {"file": str(path.relative_to(args.input)), "time": datetime.now().isoformat(timespec="seconds"), "seconds": seconds,
                  "pages": (result.metrics or {}).get("pages_total") if result else None}
        if result is not None and result.ok:
            (args.output / f"{name}.docx").write_bytes(result.docx)
            record.update(outcome="issued", docx=f"{name}.docx")
        else:
            details = (error or {}).get("details") or {}
            record.update(outcome="refused", code=(error or {}).get("code"), reason=details.get("reason") or (error or {}).get("code"), message=(error or {}).get("message"),
                          details=details)
            shutil.copy2(path, refused_dir / f"{name}{path.suffix.lower()}")
            (refused_dir / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return record

    try:
        with ThreadPoolExecutor(workers) as executor:
            for record in executor.map(convert, todo):
                counts[record["outcome"]] += 1
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                journal.flush()
                mark = "issued" if record["outcome"] == "issued" else f"refused ({record.get('reason')})"
                print(f"{record['file']}: {mark}, {record['seconds']} s", flush=True)
    finally:
        journal.close()
        pool.close()
    print(f"Issued: {counts['issued']}, refused: {counts['refused']} (folder {refused_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def summary(argv: list[str] | None = None) -> int:
    """dre-batch summary OUT_DIR: issued / refused by reason / time, from OUT_DIR/journal.jsonl."""
    parser = argparse.ArgumentParser(prog="dre-summary", description="Summary of a folder conversion journal.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    latest: dict[str, dict] = {}
    for line in (args.output / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["file"]] = record  # a document converted again counts once, with its last result
    records = list(latest.values())
    issued = [record for record in records if record["outcome"] == "issued"]
    refused = [record for record in records if record["outcome"] != "issued"]
    total = max(1, len(records))
    print(f"Documents: {len(records)}")
    print(f"Issued:     {len(issued)} ({100 * len(issued) / total:.0f}%)")
    print(f"Refused:    {len(refused)} ({100 * len(refused) / total:.0f}%)")
    reasons: dict[str, int] = {}
    for record in refused:
        reasons[str(record.get("reason"))] = reasons.get(str(record.get("reason")), 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  {count:4}  {reason}")
    seconds = sorted(float(record["seconds"]) for record in records)
    if seconds:
        print(f"Time, s: median {seconds[len(seconds) // 2]:.1f}, maximum {seconds[-1]:.1f}")
    print("\nTo review (issued DOCX):")
    for record in sorted(issued, key=lambda record: record["file"]):
        print(f"  {record['docx']}    <- {record['file']}")
    return 0
