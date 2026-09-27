"""A document that will not finish in time is refused before the deadline, not stopped by it."""

import dataclasses
import time
import unittest
from pathlib import Path
from unittest import mock

from document_reconstruction.config import Settings
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.preflight import CostPredictor
from document_reconstruction.recognition.types import RecognizedPage

SCAN = Path(__file__).resolve().parents[1] / "handoff/samples/speed/scan_03p.pdf"


class TimeForecastTests(unittest.TestCase):
    def test_slow_pages_are_refused_after_the_first_one(self):
        if not SCAN.exists():
            self.skipTest("speed samples are not available")

        def slow_scan(self, pdf, plan, context, source=None, **hints):
            time.sleep(2.0)  # the machine turns out slower than forecast
            return RecognizedPage(plan.index, plan.width, plan.height, []), {"page": plan.index}

        settings = dataclasses.replace(Settings.from_env(), page_threads=1)  # the target server: one core
        engine = ConversionEngine(ocr=mock.Mock(), predictor=CostPredictor(speed_factor=0.01), settings=settings)
        with mock.patch.object(ConversionEngine, "_scan", slow_scan):
            started = time.perf_counter()
            result = engine.convert(SCAN.read_bytes(), timeout_ms=6000)
            elapsed = time.perf_counter() - started
        self.assertFalse(result.ok)
        self.assertEqual(result.error["code"], "DOCUMENT_TOO_EXPENSIVE")
        self.assertEqual(result.error["details"]["reason"], "estimated_time_over_budget")
        self.assertLess(elapsed, 4.0)  # refused after one page, not at the 6 s deadline


if __name__ == "__main__":
    unittest.main()
