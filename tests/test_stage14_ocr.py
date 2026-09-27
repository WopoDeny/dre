"""Source-visible RTL numeric acceptance and low-confidence OCR geometry tests."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from PIL import Image
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.model import Box, Table
from document_reconstruction.recognition.regions import _unresolved_borderless_table_issue
from document_reconstruction.recognition.types import Word

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks/stage14/synthetic/mixed_direction"


class VisibleSourceTests(unittest.TestCase):
    def test_critical_numeric_tokens_override_low_aggregate_error(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from stage14_mixed_direction_acceptance import critical_token_recall
        tokens = ["۱۲۳/45", "۲۰۲۶-۰۹-۱۹"]
        reference = "سند شماره ۱۲۳/45، تاریخ ۲۰۲۶-۰۹-۱۹؛ بررسی نهایی. " * 3
        corrupted = reference.replace("۱۲۳/45", "۱۲۳/۳")
        self.assertEqual(critical_token_recall(reference, tokens), 1.0)
        self.assertEqual(critical_token_recall(corrupted, tokens), 0.5)

    def test_checked_in_rtl_source_hashes_match_immutable_manifest(self):
        manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["cases"]), 2)
        for record in manifest["cases"]:
            with self.subTest(case=record["case_id"]):
                image = (CORPUS / record["prepared_source_image"]).read_bytes()
                source = (CORPUS / record["input"]).read_bytes()
                self.assertEqual(hashlib.sha256(image).hexdigest(), record["prepared_source_image_sha256"])
                self.assertEqual(hashlib.sha256(source).hexdigest(), record["input_sha256"])
                self.assertEqual(record["class"], "synthetic_scanned_pdf")
                self.assertEqual(len(record["required_visible_tokens"]), 2)

    def test_low_confidence_words_are_not_invisible_grid_gutters(self):
        words = []
        for row in range(3):
            y = 100 + row * 70
            for index, (text, left, right, confidence) in enumerate((
                ("نهاية.", 80, 120, 0.94),
                ("مراجعة", 126, 174, 0.9),
                ("2026-09-19", 180, 270, 0.0),
                ("موعد", 277, 316, 0.91),
                ("123/45", 323, 378, 0.0),
                ("رقم", 385, 421, 0.9),
                ("الوثيقة", 427, 470, 0.94),
            )):
                words.append(Word(text, Box(left, y, right, y + 17), confidence, line_id=(row, 1, 1)))
        self.assertIsNone(_unresolved_borderless_table_issue(words, [], width=595, height=842))

    def test_genuine_borderless_three_column_grid_remains_unresolved(self):
        words = []
        for row in range(4):
            y = 110 + row * 60
            for column, text in enumerate(("Item", "17", "Approved")):
                x = 50 + 190 * column
                words.append(Word(text, Box(x, y, x + 65, y + 17), 0.93, line_id=(row, column, 1)))
        issue = _unresolved_borderless_table_issue(words, [], width=595, height=842)
        self.assertIsNotNone(issue)
        self.assertEqual(issue.code, "unresolved_borderless_raster_table")

    def test_persian_visible_mixed_direction_raster_not_misclassified_as_table(self):
        from document_reconstruction.recognition.tesseract import TesseractBackend
        backend = TesseractBackend()
        if "Arabic" not in backend.available:
            backend.close()
            self.skipTest("The local Arabic script profile is not installed.")
        engine = ConversionEngine(ocr=backend)
        try:
            result = engine.convert((CORPUS / "persian.pdf").read_bytes())
        finally:
            engine.close()
        self.assertTrue(result.ok, result.error)
        self.assertIsNotNone(result.model)
        self.assertFalse(any(isinstance(block, Table) for page in result.model.pages for block in page.blocks))
        if "eng" in backend.available:
            # Stage-17 source-backed mixed-digit recovery intentionally spends
            # the sole request-wide retry on this previously weak identifier.
            self.assertEqual(result.metrics["pages_ocr_performed"], 2)
            self.assertEqual(result.metrics["retry_pages"], 1)
            self.assertEqual(result.model.text().count("۱۲۳/45"), 3)
        else:
            self.assertEqual(result.metrics["pages_ocr_performed"], 1)
            self.assertEqual(result.metrics["retry_pages"], 0)


if __name__ == "__main__":
    unittest.main()
