"""Offline comparison of exactly two local OCR candidates on identical prepared pages.

No candidate is downloaded or auto-installed. PaddleOCR is measured only when its
runtime and explicit local model directories are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean

import numpy as np
import pymupdf
from PIL import Image

from ..core import DeadlineBudget
from ..core.memory import read_process_high_water_bytes
from ..recognition.geometry import recover
from ..recognition.tesseract import TesseractBackend
from .runner import character_error_rate


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tesseract_version() -> str:
    try:
        return subprocess.check_output(["tesseract", "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _hardware() -> dict[str, object]:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL, timeout=2).strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"cpu": cpu, "gpu": gpu, "device": "cpu"}



def _directory_identity(path: Path | None) -> dict[str, object] | None:
    """Hash an explicit local model directory for reproducible offline evidence."""
    if path is None or not path.is_dir():
        return None
    digest = hashlib.sha256()
    total = 0
    files = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with child.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
        files += 1
    return {"sha256": digest.hexdigest(), "bytes": total, "files": files}

def _valid_geometry_ratio(boxes: list[tuple[float, float, float, float]], width: int, height: int) -> float | None:
    if not boxes:
        return None
    valid = sum(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height for x0, y0, x1, y1 in boxes)
    return valid / len(boxes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/ocr-bakeoff.jsonl"))
    parser.add_argument("--paddle-det-model-dir", type=Path)
    parser.add_argument("--paddle-rec-model-dir", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [case for case in manifest["cases"] if case["expected_routes"] == ["OCR"]]
    hardware = _hardware()

    tess_started = time.perf_counter()
    tesseract = TesseractBackend()
    tess_init_ms = (time.perf_counter() - tess_started) * 1000
    paddle = None
    paddle_init_ms = None
    blocked = "paddleocr_runtime_not_installed"
    if importlib.util.find_spec("paddleocr") is not None and importlib.util.find_spec("paddle") is not None:
        blocked = "local_paddle_model_paths_required"
        if args.paddle_det_model_dir and args.paddle_rec_model_dir and args.paddle_det_model_dir.is_dir() and args.paddle_rec_model_dir.is_dir():
            from paddleocr import PaddleOCR
            paddle_started = time.perf_counter()
            paddle = PaddleOCR(
                text_detection_model_dir=str(args.paddle_det_model_dir),
                text_recognition_model_dir=str(args.paddle_rec_model_dir),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
            )
            paddle_init_ms = (time.perf_counter() - paddle_started) * 1000

    paddle_model_identity = {
        "detection": _directory_identity(args.paddle_det_model_dir),
        "recognition": _directory_identity(args.paddle_rec_model_dir),
    }

    runtime = {
        "python": platform.python_version(),
        "tesseract": _tesseract_version(),
        "tesseract_c_api": tesseract.runtime_version,
        "paddleocr": _version("paddleocr"),
        "paddlepaddle": _version("paddlepaddle"),
        **hardware,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            for case in cases:
                page_start = time.perf_counter()
                budget = DeadlineBudget()
                with pymupdf.open(args.manifest.parent / case["input"]) as pdf:
                    preflight_ms = (time.perf_counter() - page_start) * 1000
                    stage_start = time.perf_counter()
                    pixmap = pdf[0].get_pixmap(dpi=200, alpha=False)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    render_ms = (time.perf_counter() - stage_start) * 1000
                stage_start = time.perf_counter()
                recovery = recover(image, budget)
                image = recovery.image
                geometry_ms = (time.perf_counter() - stage_start) * 1000
                stage_start = time.perf_counter()
                orientation = tesseract.detect_script(image, budget)
                script_ms = (time.perf_counter() - stage_start) * 1000
                if orientation.clockwise_orientation and orientation.orientation_confidence >= 0.5:
                    stage_start = time.perf_counter()
                    image = image.rotate(orientation.clockwise_orientation, expand=True, fillcolor="white")
                    geometry_ms += (time.perf_counter() - stage_start) * 1000
                prepared_ms = (time.perf_counter() - page_start) * 1000
                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                for candidate in ("tesseract", "paddleocr"):
                    record = {
                        "schema_version": 2,
                        "case_id": case["case_id"],
                        "candidate": candidate,
                        "prepared_image_sha256": image_hash,
                        "shared_preparation": True,
                        "preparation_operations": recovery.operations,
                        "preparation_ms": prepared_ms,
                        "preflight_ms": preflight_ms,
                        "render_ms": render_ms,
                        "geometry_ms": geometry_ms,
                        "script_ms": script_ms,
                        "ocr_ms": 0.0,
                        "candidate_initialization_ms": tess_init_ms if candidate == "tesseract" else paddle_init_ms,
                        "profile_loading_ms": tesseract.profile_load_ms.get("eng") if candidate == "tesseract" else paddle_init_ms,
                        "warm_processing_ms": None,
                        "runtime": runtime,
                        "candidate_model_identity": tesseract.profile_identity("eng") if candidate == "tesseract" else paddle_model_identity,
                        "character_error_rate": None,
                        "mean_confidence": None,
                        "geometry_valid_ratio": None,
                        "geometry_reference_iou": None,
                        "ram_process_high_water_bytes": read_process_high_water_bytes(),
                        "vram_peak_bytes": None,
                    }
                    if candidate == "paddleocr" and paddle is None:
                        record.update(status="blocked", reason=blocked)
                    else:
                        start = time.perf_counter()
                        boxes: list[tuple[float, float, float, float]] = []
                        if candidate == "tesseract":
                            words = tesseract.recognize(image, "eng", DeadlineBudget(), page_width=image.width, page_height=image.height)
                            text = " ".join(word.text for word in words)
                            confidence = mean(word.confidence for word in words) if words else 0
                            line_count = len({word.line_id for word in words})
                            boxes = [(w.box.x0, w.box.y0, w.box.x1, w.box.y1) for w in words]
                        else:
                            result = list(paddle.predict(np.asarray(image)))[0]
                            text = " ".join(result["rec_texts"])
                            confidence = float(np.mean(result["rec_scores"])) if len(result["rec_scores"]) else 0
                            line_count = len(result["rec_texts"])
                            for polygon in result.get("dt_polys", []):
                                points = np.asarray(polygon)
                                boxes.append((float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())))
                        elapsed = (time.perf_counter() - start) * 1000
                        normalized = " ".join(text.split())
                        expected_lines = [" ".join(line.split()) for line in case.get("expected_text", "").splitlines() if line.strip()]
                        record.update(
                            status="measured",
                            ocr_ms=elapsed,
                            warm_processing_ms=elapsed,
                            total_ms=prepared_ms + elapsed,
                            character_error_rate=character_error_rate(case.get("expected_text", ""), text),
                            mean_confidence=confidence,
                            lines_detected=line_count,
                            missing_reference_lines=sum(line not in normalized for line in expected_lines),
                            geometry_valid_ratio=_valid_geometry_ratio(boxes, image.width, image.height),
                            ram_process_high_water_bytes=read_process_high_water_bytes(),
                        )
                    stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
                    stream.flush()
    finally:
        tesseract.close()
    print(json.dumps({
        "candidates": ["tesseract", "paddleocr"],
        "pages_per_candidate": len(cases),
        "comparison_complete": paddle is not None,
        "winner": None,
        "blocked_candidate": None if paddle is not None else "paddleocr",
        "blocked_reason": None if paddle is not None else blocked,
        "runtime": runtime,
    }))
    return 0 if paddle is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
