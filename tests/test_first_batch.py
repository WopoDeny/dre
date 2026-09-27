"""Acceptance regressions from the immutable, fictional first PDF batch."""

import hashlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from docx import Document as WordDocument

from document_reconstruction.engine import ConversionEngine
from document_reconstruction.model import Table
from document_reconstruction.recognition.native import _explicit_native_spaces


CORPUS = Path(__file__).resolve().parents[1] / "benchmarks" / "first_batch"
CASES = {case["id"]: case for case in json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))["cases"]}


class FirstBatchNativeSafetyTests(unittest.TestCase):
    def test_prose_in_kazakh_native_pdf_is_not_converted_into_borderless_table(self):
        case = CASES["03_native_kazakh"]
        payload = (CORPUS / "pdfs" / case["filename"]).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), case["sha256"])
        ocr = Mock()
        result = ConversionEngine(ocr=ocr).convert(payload)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["ocr_ms"], 0)
        ocr.detect_script.assert_not_called()
        ocr.recognize.assert_not_called()
        self.assertFalse(any(isinstance(block, Table) for block in result.model.pages[0].blocks))
        self.assertIn(case["title"], result.model.text())
        self.assertIn("Қоймаға 14 қорап жеткізілді.", result.model.text())
        self.assertIn("ҚҚ-2048/Ә.", result.model.text())

    def test_prose_in_tajik_native_pdf_is_not_converted_into_borderless_table(self):
        case = CASES["06_native_tajik"]
        payload = (CORPUS / "pdfs" / case["filename"]).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), case["sha256"])
        result = ConversionEngine(ocr=Mock()).convert(payload)
        self.assertTrue(result.ok, result.error)
        self.assertFalse(any(isinstance(block, Table) for block in result.model.pages[0].blocks))
        self.assertIn(case["title"], result.model.text())
        self.assertIn("ТҶ-2048/Ғ.", result.model.text())

class FirstBatchNativeWhitespaceTests(unittest.TestCase):
    def test_native_traditional_chinese_explicit_title_spaces_survive_to_word(self):
        """A native PDF's explicit spaces must override CJK OCR-tight joining."""
        case = CASES["13_native_chinese_traditional"]
        payload = (CORPUS / "pdfs" / case["filename"]).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), case["sha256"])
        ocr = Mock()
        result = ConversionEngine(ocr=ocr).convert(payload)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["ocr_ms"], 0)
        ocr.recognize.assert_not_called()
        self.assertIn(case["title"], result.model.text())
        self.assertIn(case["title"], WordDocument(io.BytesIO(result.docx)).paragraphs[0].text)

    def test_multispan_native_spaces_are_preserved_when_source_order_is_proven(self):
        """Native font changes must not erase real spaces around Han/number text."""
        text_dict = {"blocks": [{"type": 0, "lines": [{"dir": (1.0, 0.0), "spans": [
            {"text": "收貨確認單 ", "bbox": (10, 20, 100, 36)},
            {"text": "編號 27", "bbox": (102, 20, 160, 36)},
        ]}]}]}
        words = [
            (10, 20, 99, 36, "收貨確認單", 0, 0, 0),
            (102, 20, 131, 36, "編號", 0, 0, 1),
            (135, 20, 160, 36, "27", 0, 0, 2),
        ]
        self.assertEqual(_explicit_native_spaces(text_dict, words), {1, 2})

    def test_multispan_reordered_geometry_does_not_invent_source_spaces(self):
        """An ambiguous run order is not trustworthy spacing evidence."""
        text_dict = {"blocks": [{"type": 0, "lines": [{"dir": (1.0, 0.0), "spans": [
            {"text": "收貨確認單 ", "bbox": (110, 20, 190, 36)},
            {"text": "編號 27", "bbox": (10, 20, 90, 36)},
        ]}]}]}
        words = [
            (110, 20, 190, 36, "收貨確認單", 0, 0, 0),
            (10, 20, 50, 36, "編號", 0, 0, 1),
            (54, 20, 90, 36, "27", 0, 0, 2),
        ]
        self.assertEqual(_explicit_native_spaces(text_dict, words), set())


if __name__ == "__main__":
    unittest.main()
