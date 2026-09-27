"""Deterministic lifecycle, timing, failure, and isolation tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_reconstruction.core import ConversionContext, DeadlineBudget, EngineError, ErrorCode, Metrics, PageRoute
from document_reconstruction.core.errors import public_error
from document_reconstruction.core.memory import MemoryTracker
from document_reconstruction.core.workspace import TemporaryWorkspace


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance(self, milliseconds):
        self.now += int(milliseconds * 1_000_000)


class DeadlineTests(unittest.TestCase):
    def test_global_cap_and_remaining(self):
        clock = Clock()
        budget = DeadlineBudget(55_000, clock=clock)
        self.assertEqual(budget.timeout_ms, 45_000)
        clock.advance(1200)
        self.assertEqual(budget.remaining_ms, 43_800)
        self.assertEqual(budget.timeout_seconds(reserve_ms=800), 43)

    def test_exact_deadline_and_nonnegative_remaining(self):
        clock = Clock()
        budget = DeadlineBudget(10, clock=clock)
        clock.advance(10)
        with self.assertRaises(EngineError) as caught:
            budget.check(stage="ocr")
        self.assertEqual(caught.exception.code, ErrorCode.DEADLINE_EXCEEDED)
        clock.advance(5)
        self.assertEqual(budget.remaining_ms, 0)

    def test_rejection_before_work(self):
        budget = DeadlineBudget(1000, clock=Clock())
        budget.require_fit(700, reserve_ms=200)
        with self.assertRaises(EngineError) as caught:
            budget.require_fit(900, reserve_ms=100)
        self.assertEqual(caught.exception.code, ErrorCode.SLA_REJECTED)

    def test_invalid_numbers(self):
        for value in (0, -1, True, float("nan"), float("inf"), "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DeadlineBudget(value)


class MetricsTests(unittest.TestCase):
    def test_repeated_stage_and_exception_accounting(self):
        clock = Clock()
        metrics = Metrics(0, clock=clock)
        with metrics.time("ocr"):
            clock.advance(5)
        with self.assertRaises(RuntimeError), metrics.time("ocr"):
            clock.advance(3)
            raise RuntimeError("Expected test failure")
        clock.advance(2)
        metrics.finish()
        clock.advance(9)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["total_ms"], 10)
        self.assertEqual(snapshot["ocr_ms"], 8)
        self.assertEqual(snapshot["stage_calls"]["ocr"], 2)
        self.assertEqual(snapshot["docx_ms"], 0)
        self.assertIsNone(snapshot["pages_total"])

    def test_overlap_is_rejected(self):
        metrics = Metrics(0, clock=Clock())
        with metrics.time("ocr"):
            with self.assertRaises(RuntimeError):
                with metrics.time("qa"):
                    pass

    def test_optional_memory_failures_are_not_fatal(self):
        def unavailable():
            raise OSError("Unavailable test provider")
        memory = MemoryTracker(rss_reader=unavailable)
        memory.sample()
        self.assertIsNone(memory.snapshot()["ram_peak_bytes"])
        self.assertIsNone(memory.snapshot()["vram_peak_bytes"])

    def test_memory_peak_is_sampled_and_zero_is_valid(self):
        values = iter((10, 50, 20))
        tracker = MemoryTracker(rss_reader=lambda: next(values), vram_reader=lambda: 0)
        for _ in range(3):
            tracker.sample()
        self.assertEqual(tracker.snapshot()["ram_peak_bytes"], 50)
        self.assertEqual(tracker.snapshot()["vram_peak_bytes"], 0)


class WorkspaceTests(unittest.TestCase):
    def test_removal_on_success_and_failure(self):
        for fail in (False, True):
            path = None
            try:
                with TemporaryWorkspace() as workspace:
                    path = workspace.path
                    workspace.file("sample.txt").write_text("Temporary test data")
                    if fail:
                        raise ValueError("Expected test failure")
            except ValueError:
                pass
            self.assertFalse(path.exists())

    def test_traversal_absolute_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as outside, TemporaryWorkspace() as workspace:
            for value in ("../escape", outside, "."):
                with self.assertRaises(EngineError):
                    workspace.file(value)
            workspace.file("link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(EngineError):
                workspace.file("link/escape")

    def test_cleanup_does_not_mask_primary_failure(self):
        workspace = TemporaryWorkspace()
        with self.assertRaisesRegex(ValueError, "Primary"):
            with workspace:
                with patch.object(workspace._directory, "cleanup", side_effect=PermissionError()):
                    workspace.__exit__(ValueError, ValueError("Primary"), None)
                    self.assertEqual(workspace.cleanup_error.code, ErrorCode.WORKSPACE_ERROR)
                    raise ValueError("Primary")

    def test_cleanup_failure_is_visible_on_success(self):
        workspace = TemporaryWorkspace().__enter__()
        try:
            with patch.object(workspace._directory, "cleanup", side_effect=PermissionError()):
                with self.assertRaises(EngineError):
                    workspace.__exit__(None, None, None)
        finally:
            workspace.cleanup()


class ContextTests(unittest.TestCase):
    def test_native_guard_and_single_targeted_retry(self):
        with ConversionContext() as context:
            context.set_page_routes([PageRoute.NATIVE, PageRoute.OCR])
            with self.assertRaises(EngineError):
                context.assert_ocr_page(0)
            context.claim_retry([1], estimated_ms=1)
            with self.assertRaises(EngineError) as caught:
                context.claim_retry([1], estimated_ms=1)
            self.assertEqual(caught.exception.code, ErrorCode.RETRY_LIMIT_EXCEEDED)
        self.assertEqual(context.metrics.snapshot()["retry_pages"], 1)

    def test_rejected_retry_does_not_consume_retry(self):
        with ConversionContext() as context:
            context.set_page_routes([PageRoute.OCR])
            with self.assertRaises(EngineError):
                context.claim_retry([0], estimated_ms=50_000)
            context.claim_retry([0], estimated_ms=1)

    def test_stage_overrun_cleans_workspace(self):
        clock = Clock()
        context = ConversionContext(timeout_ms=10, clock=clock)
        with self.assertRaises(EngineError):
            with context:
                path = context.workspace.path
                with context.stage("ocr"):
                    clock.advance(11)
        self.assertFalse(path.exists())
        self.assertEqual(context.status, "error")
        self.assertEqual(context.metrics.snapshot()["ocr_ms"], 11)

    def test_two_requests_are_isolated(self):
        with ConversionContext() as first, ConversionContext() as second:
            self.assertNotEqual(first.request_id, second.request_id)
            self.assertNotEqual(first.workspace.path, second.workspace.path)
            first.set_page_routes([PageRoute.OCR])
            self.assertIsNone(second.metrics.pages_total)

    def test_unknown_error_is_redacted_and_serializable(self):
        error = public_error(ValueError("Private document text"))
        encoded = json.dumps(error.to_dict(request_id="test"), allow_nan=False)
        self.assertNotIn("Private document", encoded)
        self.assertIn("INTERNAL_ERROR", encoded)


if __name__ == "__main__":
    unittest.main()
