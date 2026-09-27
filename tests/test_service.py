"""API contract and real process-watchdog behavior."""

import io
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from document_reconstruction.core import DeadlineBudget, EngineError, ErrorCode
from document_reconstruction.core.workspace import TemporaryWorkspace
from document_reconstruction.engine import ConversionEngine, ConversionResult
from document_reconstruction.service.app import Application
from document_reconstruction.service.workers import WarmWorkerPool


class LocalPool:
    ready = True
    languages = ["eng", "osd"]

    def convert(self, payload, *, budget, request_id):
        return ConversionEngine().convert(payload, request_id=request_id, timeout_ms=budget.remaining_ms)


def blocked_worker(connection):
    connection.send({"ready": True, "languages": ["eng", "osd"]})
    connection.recv()
    time.sleep(10)


class RequestCancelled(BaseException):
    pass


class FakeProcess:
    pid = 999999

    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


class ScriptedConnection:
    def __init__(self, mode, events):
        self.mode = mode
        self.events = events
        self.message = None

    def send(self, message):
        self.message = message
        self.events.append("dispatch")

    def poll(self, timeout):
        if self.mode == "cancel":
            raise RequestCancelled()
        return True

    def recv(self):
        request_id = self.message["request_id"]
        root = Path(self.message["directory"])
        if self.mode == "success":
            (root / "output.docx").write_bytes(b"PK fake docx")
            response_id = request_id
        elif self.mode == "stale":
            (root / "output.docx").write_bytes(b"PK stale docx")
            response_id = "stale-request"
        else:
            raise AssertionError(self.mode)
        return {"request_id": response_id, "metrics": {}, "error": None, "quality": {}, "diagnostics": {}}


class FakeSlot:
    def __init__(self, mode, events, replacement_event=None, stop_error=None):
        self.events = events
        self.connection = ScriptedConnection(mode, events)
        self.process = FakeProcess()
        self.ready = True
        self.languages = ["eng", "osd"]
        self.replacement_event = replacement_event
        self.stop_error = stop_error

    def stop(self):
        self.events.append("stop")
        self.ready = False
        self.process.alive = False
        if self.stop_error is not None:
            raise self.stop_error

    def start(self):
        self.events.append("start")
        self.connection = ScriptedConnection("success", self.events)
        self.process = FakeProcess()
        self.ready = True
        if self.replacement_event is not None:
            self.replacement_event.set()


def fake_pool(root, slot):
    pool = WarmWorkerPool.__new__(WarmWorkerPool)
    pool._slots = [slot]
    pool._available = queue.Queue(maxsize=1)
    pool._available.put_nowait(slot)
    pool._max_rss = 1200 * 1024 * 1024
    pool._workspace_root = Path(root)
    pool._closed = False
    pool._state_lock = threading.RLock()
    pool._restart_threads = []
    return pool


class APITests(unittest.TestCase):
    def call(self, path, method="GET", body=b"", **overrides):
        environ = {"PATH_INFO": path, "REQUEST_METHOD": method, "CONTENT_TYPE": "application/pdf", "CONTENT_LENGTH": str(len(body)), "wsgi.input": io.BytesIO(body), "HTTP_AUTHORIZATION": "Bearer test-key", **overrides}
        captured = {}
        def start(status, headers):
            captured.update(status=status, headers=dict(headers))
        output = b"".join(Application(LocalPool(), api_key="test-key")(environ, start))
        return captured, output

    def test_health_and_readiness(self):
        for path in ("/health", "/ready"):
            metadata, body = self.call(path)
            self.assertTrue(metadata["status"].startswith("200"))
            self.assertIsInstance(json.loads(body), dict)

    def test_raw_pdf_returns_word_document(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks/corpus/native_small.pdf"
        metadata, body = self.call("/convert", "POST", path.read_bytes())
        self.assertTrue(metadata["status"].startswith("200"))
        self.assertTrue(body.startswith(b"PK"))
        self.assertIn("wordprocessingml.document", metadata["headers"]["Content-Type"])

    def test_invalid_pdf_is_json_failure(self):
        metadata, body = self.call("/convert", "POST", b"invalid")
        self.assertTrue(metadata["status"].startswith("400"))
        self.assertEqual(json.loads(body)["error"]["code"], "INVALID_PDF")

    def test_auth_size_and_media_type(self):
        for overrides, expected in (({"HTTP_AUTHORIZATION": "Bearer wrong"}, "401"), ({"CONTENT_LENGTH": str(21 * 1024 * 1024)}, "413"), ({"CONTENT_TYPE": "text/plain"}, "415")):
            metadata, _ = self.call("/convert", "POST", b"test", **overrides)
            self.assertTrue(metadata["status"].startswith(expected))

    def test_incomplete_upload_and_wrong_method(self):
        metadata, _ = self.call("/convert", "POST", b"small", CONTENT_LENGTH="100")
        self.assertTrue(metadata["status"].startswith("400"))
        metadata, _ = self.call("/convert")
        self.assertTrue(metadata["status"].startswith("405"))

    def test_api_never_returns_success_after_conversion_budget_is_exhausted(self):
        class FakeClock:
            now = 0

            def __call__(self):
                return self.now

        clock = FakeClock()
        budget = DeadlineBudget(10, clock=clock)

        class DelayedPool:
            def convert(self, payload, *, budget, request_id):
                clock.now += 11_000_000
                return ConversionResult(request_id, b'PK fake Word output', {}, None)

        captured = {}
        environ = {"PATH_INFO": "/convert", "REQUEST_METHOD": "POST", "CONTENT_TYPE": "application/pdf",
                   "CONTENT_LENGTH": "4", "wsgi.input": io.BytesIO(b"test"),
                   "HTTP_AUTHORIZATION": "Bearer test-key"}
        with patch('document_reconstruction.service.app.DeadlineBudget', return_value=budget):
            response = Application(DelayedPool(), api_key='test-key')(
                environ, lambda status, headers: captured.update(status=status, headers=dict(headers)))
        self.assertEqual(captured['status'], '504 Gateway Timeout')
        self.assertEqual(json.loads(b''.join(response))['error']['code'], ErrorCode.FAST_SLA_EXCEEDED)


class WorkerTests(unittest.TestCase):
    def test_cleanup_overrun_must_not_return_successful_docx(self):
        class FakeClock:
            now = 0

            def __call__(self):
                return self.now

        clock = FakeClock()

        class DelayedWorkspace(TemporaryWorkspace):
            def __exit__(self, exc_type, error, traceback):
                outcome = super().__exit__(exc_type, error, traceback)
                clock.now += 1_100_000_000
                return outcome

        with tempfile.TemporaryDirectory() as root:
            slot = FakeSlot('success', [])
            pool = fake_pool(root, slot)
            try:
                with patch('document_reconstruction.service.workers.TemporaryWorkspace', DelayedWorkspace):
                    with self.assertRaises(EngineError) as caught:
                        pool.convert(b'test', budget=DeadlineBudget(1_000, clock=clock), request_id='slow-cleanup')
                self.assertEqual(caught.exception.code, ErrorCode.DEADLINE_EXCEEDED)
                self.assertEqual(list(Path(root).iterdir()), [])
                self.assertTrue(slot.ready, 'Completed idle workers need not be killed when cleanup is slow')
            finally:
                pool.close()

    def test_real_warm_worker_conversion_and_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            pool = WarmWorkerPool(workspace_root=Path(root))
            try:
                self.assertTrue(pool.ready)
                slot = pool._available.get_nowait()
                try:
                    with self.assertRaises(EngineError) as caught:
                        pool.convert(b"test")
                    self.assertEqual(caught.exception.code, ErrorCode.SERVICE_BUSY)
                finally:
                    pool._available.put_nowait(slot)
                source = Path(__file__).resolve().parents[1] / "benchmarks/corpus/native_small.pdf"
                result = pool.convert(source.read_bytes())
                self.assertTrue(result.ok, result.error)
                self.assertEqual(list(Path(root).iterdir()), [])
            finally:
                pool.close()

    def test_stuck_worker_is_stopped_and_files_are_removed(self):
        with tempfile.TemporaryDirectory() as root, patch("document_reconstruction.service.workers._worker", blocked_worker):
            pool = WarmWorkerPool(workspace_root=Path(root))
            start = time.monotonic()
            try:
                with self.assertRaises(EngineError) as caught:
                    pool.convert(b"test", budget=DeadlineBudget(100))
                self.assertEqual(caught.exception.code, ErrorCode.FAST_SLA_EXCEEDED)
                self.assertLess(time.monotonic() - start, 1.5)
                self.assertEqual(list(Path(root).iterdir()), [])
            finally:
                pool.close()

    def test_cancellation_after_dispatch_stops_before_workspace_cleanup(self):
        events = []
        with tempfile.TemporaryDirectory() as root:
            slot = FakeSlot("cancel", events)
            pool = fake_pool(root, slot)
            original_cleanup = TemporaryWorkspace.cleanup

            def tracked_cleanup(workspace):
                events.append("cleanup")
                return original_cleanup(workspace)

            try:
                with patch("document_reconstruction.service.workers.TemporaryWorkspace.cleanup", tracked_cleanup):
                    with self.assertRaises(RequestCancelled):
                        pool.convert(b"test", request_id="cancelled-request")
                self.assertIn("stop", events)
                self.assertLess(events.index("stop"), events.index("cleanup"))
                self.assertTrue(pool._available.empty())
                self.assertEqual(list(Path(root).iterdir()), [])
            finally:
                pool.close()

    def test_cancellation_preserves_primary_failure_if_worker_stop_fails(self):
        events = []
        with tempfile.TemporaryDirectory() as root:
            slot = FakeSlot("cancel", events, stop_error=RuntimeError("stop failed"))
            pool = fake_pool(root, slot)
            pool._schedule_restart = lambda abandoned: None
            with self.assertRaises(RequestCancelled) as caught:
                pool.convert(b"test", request_id="cancelled-request")
            self.assertIn("stop", events)
            self.assertTrue(any("Stopping the abandoned conversion worker also failed" in note for note in getattr(caught.exception, "__notes__", [])))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_stale_response_is_rejected_and_worker_replaced(self):
        events = []
        replacement_ready = threading.Event()
        with tempfile.TemporaryDirectory() as root:
            slot = FakeSlot("stale", events, replacement_ready)
            pool = fake_pool(root, slot)
            try:
                with self.assertRaises(EngineError) as caught:
                    pool.convert(b"test", request_id="current-request")
                self.assertEqual(caught.exception.code, ErrorCode.CONVERSION_FAILED)
                self.assertIn("stop", events)
                self.assertTrue(replacement_ready.wait(2), events)
                recovered = pool.convert(b"test", request_id="recovered-request")
                self.assertTrue(recovered.ok)
                self.assertEqual(recovered.request_id, "recovered-request")
                self.assertEqual(recovered.docx, b"PK fake docx")
            finally:
                pool.close()

    def test_memory_guard_stops_the_worker(self):
        original = Path.read_text
        def observe(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                return "1000000 1000000"
            return original(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as root, patch("document_reconstruction.service.workers._worker", blocked_worker):
            pool = WarmWorkerPool(workspace_root=Path(root), max_worker_rss_mb=128)
            try:
                with patch.object(Path, "read_text", observe), self.assertRaises(EngineError) as caught:
                    pool.convert(b"test", budget=DeadlineBudget(2000))
                self.assertEqual(caught.exception.code, ErrorCode.DOCUMENT_TOO_EXPENSIVE)
                self.assertEqual(list(Path(root).iterdir()), [])
            finally:
                pool.close()


if __name__ == "__main__":
    unittest.main()
