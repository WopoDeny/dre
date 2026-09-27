"""Warm process isolation makes stuck native/OCR calls stoppable at the deadline."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path
from uuid import uuid4

from ..core import DeadlineBudget, EngineError, ErrorCode
from ..core.workspace import TemporaryWorkspace
from ..engine import ConversionEngine, ConversionResult


def _worker(connection: object) -> None:
    os.environ["OMP_THREAD_LIMIT"] = "1"
    engine = ConversionEngine()
    try:
        engine.warm()
        capabilities = dict(engine.ocr.capabilities, speed=engine.speed, page_threads=engine.settings.page_threads)
        connection.send({"ready": True, "languages": sorted(engine.ocr.available), "ocr_capabilities": capabilities})
        while True:
            message = connection.recv()
            if message is None:
                break
            root = Path(message["directory"])
            result = engine.convert((root / "input.pdf").read_bytes(), timeout_ms=message["timeout_ms"], request_id=message["request_id"], workspace_root=root)
            if result.ok:
                (root / "output.tmp").write_bytes(result.docx)
                (root / "output.tmp").replace(root / "output.docx")
            connection.send({"request_id": result.request_id, "metrics": result.metrics, "error": result.error, "quality": result.quality, "diagnostics": result.diagnostics})
            del result
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send({"ready": False})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        engine.close()
        connection.close()


class _Slot:
    def __init__(self) -> None:
        self.process: mp.Process | None = None
        self.connection = None
        self.languages: list[str] = []
        self.ready = False
        self.ocr_capabilities: dict[str, object] = {}
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(30):
            self.stop()
            raise RuntimeError("The conversion worker did not initialize in time.")
        message = self.connection.recv()
        self.ready = bool(message.get("ready"))
        self.languages = message.get("languages", [])
        self.ocr_capabilities = message.get("ocr_capabilities", {})
        if not self.ready:
            self.stop()
            raise RuntimeError("The conversion worker could not initialize its models.")

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        self.ready = False
        self.ocr_capabilities = {}
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(0.15)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(0.15)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class WarmWorkerPool:
    _running: threading.BoundedSemaphore | None = None
    def __init__(self, *, workers: int = 1, max_worker_rss_mb: int = 1200, workspace_root: Path | None = None,
                 concurrency: int | None = None) -> None:
        if not 1 <= workers <= 8 or max_worker_rss_mb < 128:
            raise ValueError("Worker count must be 1 through 8 and the RSS limit at least 128 MiB.")
        # Workers beyond `concurrency` are warm spares: a worker stopped at a deadline is replaced at once
        # instead of after its restart, while no more documents run at a time than the cores allow.
        self._running = threading.BoundedSemaphore(max(1, min(workers, concurrency or workers)))
        self._slots = [_Slot() for _ in range(workers)]
        self._available: queue.Queue[_Slot] = queue.Queue(maxsize=workers)
        self._max_rss = max_worker_rss_mb * 1024 * 1024
        self._workspace_root = workspace_root
        self._closed = False
        self._state_lock = threading.RLock()
        self._restart_threads: list[threading.Thread] = []
        try:
            for slot in self._slots:
                slot.start()
                self._available.put_nowait(slot)
        except BaseException:
            self.close()
            raise

    @property
    def ready(self) -> bool:
        return not self._closed and all(slot.ready and slot.process is not None and slot.process.is_alive() for slot in self._slots)

    @property
    def languages(self) -> list[str]:
        return sorted(set.intersection(*(set(slot.languages) for slot in self._slots)))

    @property
    def ocr_capabilities(self) -> dict[str, object]:
        if not self._slots or not self.ready:
            return {}
        # Workers are initialized from the same immutable deployment image.
        # Return one actual worker's initialized capability report.
        return dict(self._slots[0].ocr_capabilities)

    def _restart(self, slot: _Slot) -> None:
        try:
            with self._state_lock:
                if self._closed:
                    return
            slot.start()
            with self._state_lock:
                if self._closed:
                    stop = True
                else:
                    self._available.put_nowait(slot)
                    stop = False
            if stop:
                slot.stop()
        except Exception:
            slot.stop()

    def _schedule_restart(self, slot: _Slot) -> None:
        try:
            with self._state_lock:
                if self._closed:
                    return
                self._restart_threads = [previous for previous in self._restart_threads if previous.is_alive()]
                thread = threading.Thread(target=self._restart, args=(slot,), daemon=True)
                self._restart_threads.append(thread)
                thread.start()
        except BaseException:
            try:
                slot.stop()
            except BaseException:
                pass

    @staticmethod
    def _stop_abandoned(slot: _Slot, primary: BaseException) -> None:
        try:
            slot.stop()
        except BaseException as cleanup_error:
            primary.add_note(f"Stopping the abandoned conversion worker also failed: {type(cleanup_error).__name__}.")

    def convert(self, payload: bytes, *, budget: DeadlineBudget | None = None, request_id: str | None = None) -> ConversionResult:
        running = self._running
        if running is None:
            return self._convert(payload, budget=budget, request_id=request_id)
        if not running.acquire(blocking=False):
            raise EngineError(ErrorCode.SERVICE_BUSY, "All conversion slots are currently occupied.")
        try:
            return self._convert(payload, budget=budget, request_id=request_id)
        finally:
            running.release()

    def _convert(self, payload: bytes, *, budget: DeadlineBudget | None = None, request_id: str | None = None) -> ConversionResult:
        budget = budget or DeadlineBudget()
        request_id = request_id or uuid4().hex
        if self._closed:
            raise EngineError(ErrorCode.NOT_READY, "The conversion service is shutting down.")
        try:
            slot = self._available.get_nowait()
        except queue.Empty as error:
            raise EngineError(ErrorCode.SERVICE_BUSY, "All conversion slots are currently occupied.") from error
        reusable = True
        dispatched = False
        synchronized = False
        try:
            with TemporaryWorkspace(self._workspace_root) as workspace:
                try:
                    (workspace.path / "input.pdf").write_bytes(payload)
                    budget.check()
                    dispatched = True
                    reusable = False
                    slot.connection.send({"directory": str(workspace.path), "timeout_ms": budget.remaining_ms, "request_id": request_id})
                    while True:
                        if budget.remaining_ms <= 0:
                            raise EngineError(ErrorCode.FAST_SLA_EXCEEDED, "The conversion worker was stopped at the request deadline.")
                        if slot.connection.poll(min(0.05, budget.remaining_ms / 1000)):
                            break
                        if not slot.process.is_alive():
                            raise EngineError(ErrorCode.CONVERSION_FAILED, "The conversion worker exited unexpectedly.")
                        try:
                            rss = int(Path(f"/proc/{slot.process.pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                        except (OSError, ValueError, IndexError):
                            rss = 0
                        if rss > self._max_rss:
                            raise EngineError(ErrorCode.DOCUMENT_TOO_EXPENSIVE, "The conversion worker exceeded its memory limit.")
                    result = slot.connection.recv()
                    required = {"request_id", "metrics", "error", "quality", "diagnostics"}
                    if not isinstance(result, dict) or not required.issubset(result):
                        raise EngineError(ErrorCode.CONVERSION_FAILED, "The conversion worker did not return a complete result.")
                    if result["request_id"] != request_id:
                        raise EngineError(ErrorCode.CONVERSION_FAILED, "The conversion worker returned a response for a different request.")
                    synchronized = True
                    reusable = True
                    output = None
                    if result["error"] is None:
                        path = workspace.path / "output.docx"
                        if not path.is_file() or path.stat().st_size > 40 * 1024 * 1024:
                            raise EngineError(ErrorCode.CONVERSION_FAILED, "The conversion output is missing or oversized.")
                        output = path.read_bytes()
                    budget.check()
                    result["metrics"]["server_total_ms"] = budget.elapsed_ms
                    converted = ConversionResult(result["request_id"], output, result["metrics"], result["error"], quality=result["quality"], diagnostics=result["diagnostics"])
                except BaseException as primary:
                    if dispatched and not synchronized:
                        reusable = False
                        self._stop_abandoned(slot, primary)
                    raise
            converted.metrics["server_total_ms"] = budget.elapsed_ms
            # Deleting request files is part of conversion, not a free tail.
            # A verified DOCX must not turn into a successful response if
            # workspace cleanup consumed the last remaining time.
            budget.check(stage="cleanup")
            return converted
        except (EOFError, BrokenPipeError, OSError) as error:
            reusable = False
            raise EngineError(ErrorCode.CONVERSION_FAILED, "The conversion worker connection failed.") from error
        finally:
            if reusable:
                with self._state_lock:
                    if not self._closed:
                        self._available.put_nowait(slot)
            else:
                self._schedule_restart(slot)

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            threads = list(self._restart_threads)
        for slot in self._slots:
            slot.stop()
        for thread in threads:
            thread.join(timeout=31)
