"""Real PDF benchmarks, JSONL observations, summaries, and opt-in model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import unicodedata
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..core import ConversionContext, EngineError, PageRoute
from ..engine import ConversionEngine


def _canonical(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def character_error_rate(expected: str, actual: str) -> float | None:
    expected, actual = _canonical(expected), _canonical(actual)
    denominator = max(1, len(expected))
    while expected and actual and expected[0] == actual[0]:
        expected, actual = expected[1:], actual[1:]
    while expected and actual and expected[-1] == actual[-1]:
        expected, actual = expected[:-1], actual[:-1]
    if not expected or not actual:
        return (len(expected) + len(actual)) / denominator
    if len(expected) * len(actual) > 4_000_000:
        return None
    row = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        next_row = [i]
        for j, right in enumerate(actual, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = next_row
    return row[-1] / denominator



def unicode_error_rate(expected: str, actual: str) -> float | None:
    """Edit distance over NFC Unicode text while preserving punctuation/digits.

    Whitespace runs are normalized because reconstruction intentionally allows
    Word to reflow content. Unlike ``character_error_rate``, this metric keeps
    script-specific marks, punctuation, and number separators for language
    acceptance evidence.
    """

    expected = re.sub(r"\s+", " ", unicodedata.normalize("NFC", expected)).strip()
    actual = re.sub(r"\s+", " ", unicodedata.normalize("NFC", actual)).strip()
    denominator = max(1, len(expected))
    if len(expected) * len(actual) > 4_000_000:
        return None
    row = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        next_row = [i]
        for j, right in enumerate(actual, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = next_row
    return row[-1] / denominator

def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    measured = [record for record in records if record["phase"] == "measured"]
    cases = {}
    for case_id in sorted({record["case_id"] for record in measured}):
        selected = [record for record in measured if record["case_id"] == case_id]
        times = sorted(float(record["total_ms"]) for record in selected)
        cases[case_id] = {
            "runs": len(selected), "fastest_ms": min(times), "median_ms": statistics.median(times), "slowest_ms": max(times),
            "quality_failures": sum(not record["expectation_met"] for record in selected),
            "errors": sum(record["error"] is not None for record in selected),
            "retry_pages": sum(record.get("retry_pages", 0) for record in selected),
        }
    return {"schema_version": 1, "measured_runs": len(measured), "expectation_failures": sum(not record["expectation_met"] for record in measured), "cases": cases}


def infrastructure_record(case_id: str) -> dict[str, object]:
    context = ConversionContext()
    try:
        with context:
            with context.stage("preflight"):
                context.set_page_routes([PageRoute.NATIVE, PageRoute.OCR])
                if case_id == "early_rejection":
                    # The rejection probe must exceed the currently configured
                    # conversion deadline even after that deadline changes.
                    context.budget.require_fit(context.budget.timeout_ms + 1_000)
            with context.stage("qa"):
                context.workspace.file("probe.txt").write_text("Infrastructure probe", encoding="utf-8")
            if case_id == "targeted_retry":
                context.claim_retry([1], estimated_ms=1)
    except EngineError:
        pass
    expected = "rejected" if case_id == "early_rejection" else "ok"
    return {"schema_version": 1, "suite_kind": "infrastructure", "case_id": case_id, "request_id": context.request_id, "status": context.status, "error": context.error.to_dict(request_id=context.request_id) if context.error else None, "expectation_met": context.status == expected, **context.metrics.snapshot()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/runs.jsonl"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--debug-models", action="store_true", help="Write document text to local debug JSON files; use only for an authorized corpus.")
    parser.add_argument("--infrastructure", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 100 or not 0 <= args.warmup <= 10:
        parser.error("Repeat must be 1 through 100 and warmup 0 through 10.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    engine = ConversionEngine()
    if args.infrastructure:
        cases = [{"case_id": name} for name in ("lifecycle", "targeted_retry", "early_rejection")]
    else:
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1 or not isinstance(manifest.get("cases"), list):
                raise ValueError("Invalid manifest schema")
            cases = manifest["cases"]
            identifiers = [case["case_id"] for case in cases]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("Duplicate case identifiers")
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"The benchmark manifest is invalid: {type(error).__name__}.")
        if args.case:
            cases = [case for case in cases if case["case_id"] in args.case]
        if not cases:
            parser.error("No benchmark cases were selected.")
        startup = time.perf_counter()
        engine.warm()
        startup_ms = (time.perf_counter() - startup) * 1000
    environment = {"python": platform.python_version(), "platform": platform.platform(), "packages": {name: importlib.metadata.version(name) for name in ("PyMuPDF", "python-docx", "Pillow", "numpy", "scipy")}}
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            for run in range(args.warmup + args.repeat):
                phase = "warmup" if run < args.warmup else "measured"
                for case in cases:
                    if args.infrastructure:
                        record = infrastructure_record(case["case_id"])
                    else:
                        path = (args.manifest.parent / case["input"]).resolve()
                        payload = path.read_bytes()
                        result = engine.convert(payload)
                        text = result.model.text() if result.model else ""
                        cer = character_error_rate(case.get("expected_text", ""), text) if case.get("expected_text") else None
                        expected_routes = case.get("expected_routes", [])
                        routes = {
                            entry["page"]: "OCR" if entry.get("route") == "OCR_TEXT_LAYER" else entry.get("route", "OCR")
                            for entry in result.diagnostics
                            if not entry.get("retry")
                        }
                        classification = sum(routes.get(i) == route for i, route in enumerate(expected_routes)) / len(expected_routes) if expected_routes else None
                        table_match = result.quality is not None and result.quality["tables"] == case.get("expected_tables", 0)
                        record = {
                            "schema_version": 1, "suite_kind": "conversion", "case_id": case["case_id"], "request_id": result.request_id,
                            "input_sha256": hashlib.sha256(payload).hexdigest(), "status": "ok" if result.ok else "error", "error": result.error,
                            "quality": result.quality, "character_error_rate": cer, "classification_accuracy": classification,
                            "expectation_met": result.ok and (cer is None or cer <= 0.03) and (classification is None or classification == 1) and table_match,
                            "model_startup_ms": startup_ms, "diagnostics": result.diagnostics, **result.metrics,
                        }
                        if result.ok and phase == "measured":
                            artifact_dir = args.output.parent / "outputs"
                            artifact_dir.mkdir(exist_ok=True)
                            safe_name = str(case["case_id"])
                            if not safe_name.replace("_", "").replace("-", "").isalnum():
                                raise ValueError("Case identifiers must be safe artifact names.")
                            (artifact_dir / f"{safe_name}.docx").write_bytes(result.docx)
                            if args.debug_models:
                                (artifact_dir / f"{safe_name}.model.json").write_text(json.dumps(result.model.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
                    record.update({"phase": phase, "run_index": run, "timestamp": datetime.now(timezone.utc).isoformat(), "environment": environment})
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    print(json.dumps({"case": record["case_id"], "phase": phase, "total_ms": round(record["total_ms"], 2), "passed": record["expectation_met"]}), flush=True)
    finally:
        engine.close()
    summary = summarize(records)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 1 if summary["expectation_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
