"""Measured OCR regression coverage for the frozen Tajik acceptance source."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from document_reconstruction.benchmark.runner import unicode_error_rate
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.recognition.languages import profile, tajik_profile_evidence
from document_reconstruction.recognition.tesseract import TesseractBackend

ROOT = Path(__file__).resolve().parents[1]


class TajikTargetedOCRTests(unittest.TestCase):
    def test_tajik_evidence_requires_distinctive_script_marks(self):
        self.assertTrue(tajik_profile_evidence("Тоҷикӣ: Ғғ, Йй, Ққ, Уӯ, Ҳҳ, Ҷҷ; № 7."))
        self.assertFalse(tajik_profile_evidence("Қазақ тілі: Ә ә, Ғ ғ, Қ қ, Ң ң, Ө ө, Ұ ұ, Ү ү, Һ һ, І і."))
        self.assertFalse(tajik_profile_evidence("Русский текст с обычными кириллическими буквами."))
        self.assertFalse(tajik_profile_evidence("Ўзбекистон: Ў ў, Қ қ, Ғ ғ, Ҳ ҳ; 2026 йил."))
        self.assertFalse(tajik_profile_evidence("Ҷҷ"))  # One distinctive form is not sufficient.

    def test_exact_tajik_profile_requires_installed_local_model(self):
        self.assertEqual(profile("Tajik", {"tgk", "Cyrillic"}), "tgk")
        from document_reconstruction.core import EngineError, ErrorCode
        with self.assertRaises(EngineError) as failure:
            profile("Tajik", {"Cyrillic"})
        self.assertEqual(failure.exception.code, ErrorCode.LANGUAGE_UNAVAILABLE)

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_actual_tajik_ocr_recovers_distinctive_letters_via_one_retry(self):
        backend = TesseractBackend()
        if "tgk" not in backend.available or "Cyrillic" not in backend.available:
            backend.close()
            self.skipTest("The local exact Tajik and generic Cyrillic traineddata are required.")
        engine = ConversionEngine(ocr=backend)
        try:
            manifest = json.loads((ROOT / "benchmarks/stage7/synthetic/language/manifest.json").read_text())
            record = next(entry for entry in manifest["cases"] if entry["language"] == "Tajik")
            result = engine.convert((ROOT / record["input"]).read_bytes())
        finally:
            engine.close()
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["retry_pages"], 1)
        self.assertEqual([entry["profile"] for entry in result.diagnostics if entry.get("ocr_performed")], ["Cyrillic", "tgk"])
        self.assertLessEqual(unicode_error_rate(record["ground_truth"], result.model.text()), 0.10)
        self.assertIn("Ӣ", result.model.text())
        self.assertIn("Ӯ", result.model.text())


if __name__ == "__main__":
    unittest.main()
