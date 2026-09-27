"""Persistent local Tesseract C API with per-recognition deadline monitors.

An instance belongs to exactly one worker process and is not thread-safe. C API
handles retain model weights; request images and text are cleared after use.
"""

from __future__ import annotations

import csv
import ctypes as c
import ctypes.util
import hashlib
import io
import os
import threading
import time
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass

from PIL import Image

from ..core import DeadlineBudget, EngineError, ErrorCode
from ..model import Box
from .registry import capability_report
from .types import Word


@dataclass(frozen=True)
class ScriptObservation:
    script: str = "Latin"
    script_confidence: float = 0
    clockwise_orientation: int = 0
    orientation_confidence: float = 0


class TesseractBackend:
    name = "tesseract"

    def __init__(self, *, tessdata_dir: str | None = None, max_profiles: int = 6) -> None:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        library = ctypes.util.find_library("tesseract")
        if library is None:
            raise EngineError(ErrorCode.NOT_READY, "The local Tesseract library is not installed.")
        self.lib = c.CDLL(library)
        self._bind()
        self._path = tessdata_dir.encode() if tessdata_dir else None
        # A Tesseract handle is not thread-safe: every thread (parallel pages) gets its own set.
        self._local = threading.local()
        self._all_handles: list[OrderedDict[str, int]] = []
        self._registry_lock = threading.Lock()
        self.profile_load_ms: dict[str, float] = {}
        self._model_identity_cache: dict[str, dict[str, object]] = {}
        self.runtime_version = "unknown"
        self.tessdata_dir: str | None = tessdata_dir
        init_started = time.perf_counter()
        self._max_profiles = max(2, max_profiles)
        try:
            handle = self._handle("eng")
            version = self.lib.TessVersion()
            self.runtime_version = version.decode("utf-8", errors="replace") if version else "unknown"
            datapath = self.lib.TessBaseAPIGetDatapath(handle)
            if datapath:
                self.tessdata_dir = datapath.decode("utf-8", errors="replace")
            languages = self.lib.TessBaseAPIGetAvailableLanguagesAsVector(handle)
            self.available: set[str] = set()
            if languages:
                index = 0
                while languages[index]:
                    self.available.add(languages[index].decode("utf-8"))
                    index += 1
                self.lib.TessDeleteTextArray(languages)
            if "osd" not in self.available:
                raise EngineError(ErrorCode.NOT_READY, "The orientation and script model is not installed.")
            self._handle("osd")
            self.initialization_ms = (time.perf_counter() - init_started) * 1000
        except BaseException:
            self.close()
            raise


    @property
    def capabilities(self) -> dict[str, object]:
        report = capability_report(self.available)
        report["runtime_version"] = self.runtime_version
        report["tessdata_dir"] = self.tessdata_dir
        report["initialized_profiles"] = list(self._handles)
        report["profile_load_ms"] = dict(self.profile_load_ms)
        report["initialization_ms"] = self.initialization_ms
        return report

    def model_identity(self, model: str) -> dict[str, object]:
        """Return a stable local traineddata identity for offline evidence."""
        if model in self._model_identity_cache:
            return dict(self._model_identity_cache[model])
        record: dict[str, object] = {"model": model, "sha256": None, "bytes": None}
        if self.tessdata_dir:
            path = Path(self.tessdata_dir) / f"{model}.traineddata"
            try:
                data = path.read_bytes()
            except OSError:
                pass
            else:
                record.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        self._model_identity_cache[model] = record
        return dict(record)

    def profile_identity(self, languages: str) -> list[dict[str, object]]:
        return [self.model_identity(model) for model in languages.split("+") if model]

    def prewarm_profiles(self, scripts: tuple[str, ...] = ("Latin", "Cyrillic", "Arabic", "Han")) -> None:
        """Load ordinary production script profiles outside conversion requests."""
        from .languages import profile

        for script in scripts:
            try:
                languages = profile(script, self.available)
            except EngineError:
                continue
            self._handle(languages)

    def _bind(self) -> None:
        pointer = c.c_void_p
        definitions = {
            "TessVersion": (c.c_char_p, []),
            "TessBaseAPICreate": (pointer, []),
            "TessBaseAPIDelete": (None, [pointer]),
            "TessBaseAPIInit3": (c.c_int, [pointer, c.c_char_p, c.c_char_p]),
            "TessBaseAPISetVariable": (c.c_int, [pointer, c.c_char_p, c.c_char_p]),
            "TessBaseAPISetPageSegMode": (None, [pointer, c.c_int]),
            "TessBaseAPISetSourceResolution": (None, [pointer, c.c_int]),
            "TessBaseAPISetImage": (None, [pointer, pointer, c.c_int, c.c_int, c.c_int, c.c_int]),
            "TessBaseAPIRecognize": (c.c_int, [pointer, pointer]),
            "TessBaseAPIGetTsvText": (pointer, [pointer, c.c_int]),
            "TessDeleteText": (None, [pointer]),
            "TessBaseAPIClear": (None, [pointer]),
            "TessBaseAPIClearAdaptiveClassifier": (None, [pointer]),
            "TessMonitorCreate": (pointer, []),
            "TessMonitorDelete": (None, [pointer]),
            "TessMonitorSetDeadlineMSecs": (None, [pointer, c.c_int]),
            "TessBaseAPIGetAvailableLanguagesAsVector": (c.POINTER(c.c_char_p), [pointer]),
            "TessBaseAPIGetDatapath": (c.c_char_p, [pointer]),
            "TessDeleteTextArray": (None, [c.POINTER(c.c_char_p)]),
            "TessBaseAPIDetectOrientationScript": (c.c_int, [pointer, c.POINTER(c.c_int), c.POINTER(c.c_float), c.POINTER(c.c_char_p), c.POINTER(c.c_float)]),
        }
        for name, (result, arguments) in definitions.items():
            function = getattr(self.lib, name)
            function.restype = result
            function.argtypes = arguments

    @property
    def _handles(self) -> OrderedDict[str, int]:
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = OrderedDict()
            with self._registry_lock:
                self._all_handles.append(handles)
        return handles

    def _handle(self, languages: str) -> int:
        if languages in self._handles:
            self._handles.move_to_end(languages)
            return self._handles[languages]
        load_started = time.perf_counter()
        handle = self.lib.TessBaseAPICreate()
        if not handle:
            raise EngineError(ErrorCode.NOT_READY, "The OCR engine could not allocate a model handle.")
        if self.lib.TessBaseAPIInit3(handle, self._path, languages.encode()) != 0:
            self.lib.TessBaseAPIDelete(handle)
            raise EngineError(ErrorCode.LANGUAGE_UNAVAILABLE, "A local OCR profile could not be initialized.", details={"profile": languages})
        self.lib.TessBaseAPISetVariable(handle, b"debug_file", os.devnull.encode())
        self.lib.TessBaseAPISetVariable(handle, b"classify_enable_learning", b"0")
        self._handles[languages] = handle
        self.profile_load_ms[languages] = (time.perf_counter() - load_started) * 1000
        while len(self._handles) > self._max_profiles:
            _, oldest = self._handles.popitem(last=False)
            self.lib.TessBaseAPIDelete(oldest)
        return handle

    def _set_image(self, handle: int, image: Image.Image) -> object:
        gray = image.convert("L")
        buffer = c.create_string_buffer(gray.tobytes())
        self.lib.TessBaseAPISetImage(handle, buffer, gray.width, gray.height, 1, gray.width)
        self.lib.TessBaseAPISetSourceResolution(handle, 200)
        return buffer

    def detect_script(self, image: Image.Image, budget: DeadlineBudget) -> ScriptObservation:
        budget.check(stage="script")
        preview = image.copy()
        preview.thumbnail((1400, 1400))
        handle = self._handle("osd")
        buffer = self._set_image(handle, preview)
        orientation, orientation_confidence = c.c_int(), c.c_float()
        script, script_confidence = c.c_char_p(), c.c_float()
        try:
            detected = self.lib.TessBaseAPIDetectOrientationScript(handle, c.byref(orientation), c.byref(orientation_confidence), c.byref(script), c.byref(script_confidence))
            budget.check(stage="script")
            if not detected or not script.value:
                return ScriptObservation()
            return ScriptObservation(script.value.decode(), script_confidence.value, orientation.value, orientation_confidence.value)
        finally:
            self.lib.TessBaseAPIClear(handle)
            del buffer

    def detect_script_regions(self, image: Image.Image, budget: DeadlineBudget) -> list[ScriptObservation]:
        """Collect bounded regional OSD evidence for pages that may mix scripts."""
        budget.check(stage="script")
        width, height = image.size
        if width < 400 or height < 400:
            return []
        from .regions import detect_text_columns

        columns = detect_text_columns(image, max_columns=4)
        observations: list[ScriptObservation] = []
        if columns:
            for left, right in columns:
                budget.check(stage="script")
                observations.append(self.detect_script(image.crop((left, 0, right, height)), budget))
            return observations

        # When no recurring vertical gutters exist, use the established bounded
        # horizontal strategy. This covers vertically stacked multilingual pages
        # without an unbounded region search.
        bands = 4 if height >= 800 else 2
        for index in range(bands):
            top = round(height * index / bands)
            bottom = round(height * (index + 1) / bands)
            budget.check(stage="script")
            observations.append(self.detect_script(image.crop((0, top, width, bottom)), budget))
        return observations

    def recognize(self, image: Image.Image, languages: str, budget: DeadlineBudget, *, page_width: float, page_height: float, psm: int = 3) -> list[Word]:
        budget.check(stage="ocr")
        handle = self._handle(languages)
        self.lib.TessBaseAPISetPageSegMode(handle, psm)
        buffer = self._set_image(handle, image)
        monitor = self.lib.TessMonitorCreate()
        pointer = None
        try:
            self.lib.TessMonitorSetDeadlineMSecs(monitor, max(1, int(budget.timeout_seconds(reserve_ms=500, stage="ocr") * 1000)))
            if self.lib.TessBaseAPIRecognize(handle, monitor) != 0:
                budget.check(stage="ocr")
                if budget.remaining_ms <= 1_000:
                    # Tesseract's own deadline (half a second before ours) stopped it: out of time, not unreadable.
                    raise EngineError(ErrorCode.DEADLINE_EXCEEDED, "The request deadline has been reached.", stage="ocr")
                raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "Local recognition did not complete successfully.", stage="ocr")
            pointer = self.lib.TessBaseAPIGetTsvText(handle, 0)
            if not pointer:
                raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "Local recognition produced no text observations.", stage="ocr")
            tsv = c.string_at(pointer).decode("utf-8", errors="replace")
            words = []
            sx, sy = page_width / image.width, page_height / image.height
            fields = ("level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text")
            # The C API returns data rows without the CLI's TSV header.
            for row in csv.DictReader(io.StringIO(tsv), fieldnames=fields, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row.get("level") != "5" or not row.get("text", "").strip():
                    continue
                x, y, w, h = (int(row[key]) for key in ("left", "top", "width", "height"))
                confidence = max(0, min(1, float(row["conf"]) / 100))
                words.append(Word(row["text"].strip(), Box(x * sx, y * sy, (x + w) * sx, (y + h) * sy), confidence, size=max(6, h * sy), line_id=tuple(int(row[key]) for key in ("block_num", "par_num", "line_num"))))
            budget.check(stage="ocr")
            return words
        finally:
            if pointer:
                self.lib.TessDeleteText(pointer)
            self.lib.TessMonitorDelete(monitor)
            self.lib.TessBaseAPIClear(handle)
            self.lib.TessBaseAPIClearAdaptiveClassifier(handle)
            del buffer

    def release_thread(self) -> None:
        """Free the handles of the calling thread (a finished page worker)."""
        handles = getattr(self._local, "handles", None)
        if handles is None:
            return
        for handle in handles.values():
            self.lib.TessBaseAPIDelete(handle)
        handles.clear()
        with self._registry_lock:
            self._all_handles = [other for other in self._all_handles if other is not handles]
        self._local.handles = None

    def close(self) -> None:
        with self._registry_lock:
            groups, self._all_handles = self._all_handles, []
        for handles in groups:
            for handle in handles.values():
                self.lib.TessBaseAPIDelete(handle)
            handles.clear()
