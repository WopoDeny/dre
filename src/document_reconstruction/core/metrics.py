"""Request-local wall-clock timings; repeated stages accumulate without overlap."""

from __future__ import annotations

import threading
import time
from enum import StrEnum
from typing import Any

from .deadline import Clock
from .memory import MemoryTracker


class Stage(StrEnum):
    PREFLIGHT = "preflight"
    RENDER = "render"
    GEOMETRY = "geometry"
    SCRIPT = "script"
    OCR = "ocr"
    UNDERSTANDING = "understanding"
    DOCX = "docx"
    QA = "qa"
    CLEANUP = "cleanup"


class Metrics:
    """One sequential stage owner per request; parallel requests use separate objects."""

    def __init__(self, started_ns: int, *, clock: Clock = time.monotonic_ns, memory: MemoryTracker | None = None) -> None:
        self.clock = clock
        self.started_ns = started_ns
        self.finished_ns: int | None = None
        self.memory = memory if memory is not None else MemoryTracker()
        self._durations = {stage: 0 for stage in Stage}
        self._calls = {stage: 0 for stage in Stage}
        # The stage running in each thread (pages are recognized in parallel threads).
        self._local = threading.local()
        self._lock = threading.Lock()
        self.pages_total: int | None = None
        self.pages_native: int | None = None
        self.pages_ocr: int | None = None
        self.pages_blank: int | None = None
        self.pages_text_layer_reused = 0
        self.pages_ocr_performed = 0
        self.retry_pages = 0
        self.workspace_cleanup_failed = False
        self.admission_estimated_ms: float | None = None
        self.admission_recorded_elapsed_ms: float | None = None
        self.admission_budget_remaining_ms: float | None = None
        self.estimated_ocr_pages: int | None = None
        self.estimated_searchable_layer_pages: int | None = None
        self.estimated_render_megapixels: float | None = None
        self.render_dpi: int | None = None
        self.render_pixel_limit: int | None = None
        self.memory.sample()

    def time(self, stage: Stage | str) -> StageTimer:
        return StageTimer(self, Stage(stage))

    def record_admission_estimate(
        self,
        estimated_ms: float,
        *,
        ocr_pages: int,
        searchable_layer_pages: int,
        render_megapixels: float,
        budget_remaining_ms: float,
        render_dpi: int,
        render_pixel_limit: int,
    ) -> None:
        """Record the post-preflight estimate and its observable complexity inputs."""

        self.admission_estimated_ms = float(estimated_ms)
        self.admission_recorded_elapsed_ms = max(0, self.clock() - self.started_ns) / 1_000_000
        self.admission_budget_remaining_ms = float(budget_remaining_ms)
        self.estimated_ocr_pages = int(ocr_pages)
        self.estimated_searchable_layer_pages = int(searchable_layer_pages)
        self.estimated_render_megapixels = float(render_megapixels)
        self.render_dpi = int(render_dpi)
        self.render_pixel_limit = int(render_pixel_limit)

    def finish(self) -> None:
        if _active(self) is not None:
            raise RuntimeError("An active stage must finish before the request.")
        if self.finished_ns is None:
            self.memory.sample()
            self.finished_ns = self.clock()

    def snapshot(self) -> dict[str, Any]:
        end = self.finished_ns if self.finished_ns is not None else self.clock()
        total_ms = max(0, end - self.started_ns) / 1_000_000
        actual_after_admission = None
        estimate_error = None
        estimate_ratio = None
        if self.admission_estimated_ms is not None and self.admission_recorded_elapsed_ms is not None:
            actual_after_admission = max(0.0, total_ms - self.admission_recorded_elapsed_ms)
            estimate_error = actual_after_admission - self.admission_estimated_ms
            if self.admission_estimated_ms > 0:
                estimate_ratio = actual_after_admission / self.admission_estimated_ms
        return {
            "total_ms": total_ms,
            **{f"{stage.value}_ms": elapsed / 1_000_000 for stage, elapsed in self._durations.items()},
            "stage_calls": {stage.value: count for stage, count in self._calls.items()},
            "pages_total": self.pages_total,
            "pages_native": self.pages_native,
            "pages_ocr": self.pages_ocr,
            "pages_blank": self.pages_blank,
            "pages_text_layer_reused": self.pages_text_layer_reused,
            "pages_ocr_performed": self.pages_ocr_performed,
            "retry_pages": self.retry_pages,
            "admission_estimated_ms": self.admission_estimated_ms,
            "admission_budget_remaining_ms": self.admission_budget_remaining_ms,
            "estimated_ocr_pages": self.estimated_ocr_pages,
            "estimated_searchable_layer_pages": self.estimated_searchable_layer_pages,
            "estimated_render_megapixels": self.estimated_render_megapixels,
            "render_dpi": self.render_dpi,
            "render_pixel_limit": self.render_pixel_limit,
            "admission_actual_ms": actual_after_admission,
            "estimate_error_ms": estimate_error,
            "estimate_actual_ratio": estimate_ratio,
            "workspace_cleanup_failed": self.workspace_cleanup_failed,
            **self.memory.snapshot(),
        }


def _active(metrics: Metrics) -> Stage | None:
    return getattr(metrics._local, "stage", None)


class StageTimer:
    def __init__(self, metrics: Metrics, stage: Stage) -> None:
        self.metrics = metrics
        self.stage = stage
        self._started: int | None = None
        self._finished = False

    def __enter__(self) -> StageTimer:
        if self._started is not None or self.metrics.finished_ns is not None:
            raise RuntimeError("Stage timers are single-use and require a live request.")
        if _active(self.metrics) is not None:
            raise RuntimeError("Nested or overlapping stage timers are not supported.")
        self.metrics._local.stage = self.stage
        self.metrics.memory.sample()
        self._started = self.metrics.clock()
        return self

    def __exit__(self, *_: object) -> bool:
        if self._started is None or self._finished:
            raise RuntimeError("The stage timer is not active.")
        with self.metrics._lock:
            self.metrics._durations[self.stage] += max(0, self.metrics.clock() - self._started)
            self.metrics._calls[self.stage] += 1
        self.metrics._local.stage = None
        self._finished = True
        self.metrics.memory.sample()
        return False
