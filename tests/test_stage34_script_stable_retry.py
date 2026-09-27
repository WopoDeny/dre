"""Do not broaden a strongly observed single script solely because PSM 3 returned zero words."""
from __future__ import annotations

import unittest
from pathlib import Path

from document_reconstruction.engine import ConversionEngine
from document_reconstruction.model import Box
from document_reconstruction.recognition.tesseract import ScriptObservation
from document_reconstruction.recognition.types import Word

ROOT = Path(__file__).resolve().parents[1]
SCAN = (ROOT / "benchmarks" / "corpus" / "scan_1.pdf").read_bytes()


class StableScriptBackend:
    available = {"Latin", "Cyrillic", "Arabic", "HanS", "HanT", "osd", "eng"}

    def __init__(self, *, mixed_evidence: bool = False):
        self.mixed_evidence = mixed_evidence
        self.calls: list[tuple[str, int]] = []

    def detect_script(self, image, budget):
        return ScriptObservation("Latin", 3.1, 0, 9.0)

    def detect_script_regions(self, image, budget):
        if self.mixed_evidence:
            return [ScriptObservation("Latin", 2.5), ScriptObservation("Cyrillic", 2.5)]
        return [ScriptObservation("Latin", 2.5), ScriptObservation("Latin", 2.5)]

    def recognize(self, image, languages, budget, *, page_width, page_height, psm=3):
        self.calls.append((languages, psm))
        if psm == 3:
            return []
        text = "привет" if languages.startswith("Latin+Cyrillic") else "English"
        return [Word(text, Box(35, 80 + 34 * i, 140, 98 + 34 * i), .96, line_id=(i + 1, 1, 1)) for i in range(10)]

    def close(self):
        pass


class ScriptStableRetryTests(unittest.TestCase):
    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')
    def test_empty_first_pass_preserves_strong_single_script_on_only_retry(self):
        backend = StableScriptBackend()
        result = ConversionEngine(ocr=backend).convert(SCAN)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(backend.calls, [("Latin", 3), ("Latin", 6)])
        self.assertEqual(result.metrics["retry_pages"], 1)
        self.assertEqual([d["profile"] for d in result.diagnostics if d.get("ocr_performed")], ["Latin", "Latin"])
        self.assertIn("English", result.model.text())
        self.assertNotIn("привет", result.model.text())

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_corroborated_multiple_scripts_still_use_mixed_profile(self):
        backend = StableScriptBackend(mixed_evidence=True)
        result = ConversionEngine(ocr=backend).convert(SCAN)
        self.assertEqual(backend.calls[0][0], "Latin+Cyrillic+Arabic+HanS+HanT")
        self.assertEqual(backend.calls[1][0], "Latin+Cyrillic+Arabic+HanS+HanT")
        self.assertFalse(result.ok)
        self.assertEqual(result.error["code"], "UNREADABLE_DOCUMENT")
        self.assertIn("Latin", result.error["details"].get("missing_scripts", ""))


if __name__ == "__main__":
    unittest.main()
