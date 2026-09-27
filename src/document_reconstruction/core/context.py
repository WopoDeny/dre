"""Conversion lifecycle, deadline, measurements, and a single targeted retry."""

from __future__ import annotations

import time
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Sequence
from uuid import uuid4

from .deadline import Clock, DeadlineBudget, INTERNAL_STOP_MS
from .errors import EngineError, ErrorCode, public_error
from .memory import MemoryTracker
from .metrics import Metrics, Stage
from .workspace import TemporaryWorkspace


class PageRoute(StrEnum):
    NATIVE = "NATIVE"
    OCR = "OCR"
    BLANK = "BLANK"


class ConversionContext:
    def __init__(
        self,
        *,
        timeout_ms: float = INTERNAL_STOP_MS,
        request_id: str | None = None,
        workspace_root: Path | None = None,
        clock: Clock = time.monotonic_ns,
        memory: MemoryTracker | None = None,
    ) -> None:
        self.budget = DeadlineBudget(timeout_ms, clock=clock)
        self.request_id = request_id or uuid4().hex
        self.metrics = Metrics(self.budget.started_ns, clock=clock, memory=memory)
        self.workspace = TemporaryWorkspace(workspace_root)
        self.status = "created"
        self.error: EngineError | None = None
        self._routes: tuple[PageRoute, ...] | None = None
        self._retry_used = False

    def _finish(self, error: BaseException | None) -> None:
        self.metrics.workspace_cleanup_failed = self.workspace.cleanup_error is not None
        self.error = public_error(error) if error is not None else None
        self.status = "ok" if error is None else "error"
        if self.error is not None and self.error.code == ErrorCode.SLA_REJECTED:
            self.status = "rejected"
        self.metrics.finish()

    def __enter__(self) -> ConversionContext:
        if self.status != "created":
            raise RuntimeError("A conversion context is single-use.")
        try:
            self.budget.check(stage="preflight")
            self.workspace.__enter__()
        except BaseException as error:
            self._finish(error)
            raise
        self.status = "running"
        return self

    def __exit__(self, exc_type: object, error: BaseException | None, traceback: object) -> bool:
        try:
            with self.metrics.time("cleanup"):
                self.workspace.__exit__(exc_type, error, traceback)
            if error is None:
                self.budget.check(stage="cleanup")
        except BaseException as final_error:
            self._finish(final_error)
            raise
        self._finish(error)
        return False

    def _require_running(self) -> None:
        if self.status != "running":
            raise RuntimeError("This operation requires an active conversion context.")

    @contextmanager
    def stage(self, stage: Stage | str) -> Iterator[None]:
        self._require_running()
        selected = Stage(stage)
        self.budget.check(stage=selected.value)
        with self.metrics.time(selected):
            yield
            self.budget.check(stage=selected.value)

    def set_page_routes(self, routes: Sequence[PageRoute]) -> None:
        self._require_running()
        if self._routes is not None:
            raise EngineError(ErrorCode.INVARIANT_VIOLATION, "Page routes may only be assigned once.")
        self._routes = tuple(PageRoute(route) for route in routes)
        self.metrics.pages_total = len(self._routes)
        self.metrics.pages_native = self._routes.count(PageRoute.NATIVE)
        self.metrics.pages_ocr = self._routes.count(PageRoute.OCR)
        self.metrics.pages_blank = self._routes.count(PageRoute.BLANK)

    def assert_ocr_page(self, page_index: int) -> None:
        self._require_running()
        if (
            self._routes is None or isinstance(page_index, bool) or not isinstance(page_index, int)
            or not 0 <= page_index < len(self._routes) or self._routes[page_index] != PageRoute.OCR
        ):
            raise EngineError(ErrorCode.INVARIANT_VIOLATION, "OCR is permitted only for an OCR-routed page.", stage="ocr")

    def claim_retry(self, page_indices: Sequence[int], *, estimated_ms: float, reserve_ms: float = 0) -> None:
        self._require_running()
        if self._retry_used:
            raise EngineError(ErrorCode.RETRY_LIMIT_EXCEEDED, "Only one expensive targeted retry is allowed.", stage="qa")
        if not page_indices or len(set(page_indices)) != len(page_indices):
            raise EngineError(ErrorCode.INVARIANT_VIOLATION, "Retry pages must be nonempty and unique.", stage="qa")
        for index in page_indices:
            self.assert_ocr_page(index)
        self.budget.require_fit(estimated_ms, reserve_ms=reserve_ms, stage="qa")
        self._retry_used = True
        self.metrics.retry_pages = len(page_indices)
