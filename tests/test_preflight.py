"""Regression coverage for cheap preflight, routing, native extraction, and admission."""

from __future__ import annotations

import io
import unittest
from unittest.mock import Mock, patch

import pymupdf
from PIL import Image, ImageDraw

from document_reconstruction.core import ConversionContext, PageRoute
from document_reconstruction.config import Settings
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.recognition.coverage import Coverage
from document_reconstruction.preflight import CostPredictor, analyze
from document_reconstruction.recognition.native import extract_native
from document_reconstruction.recognition.tesseract import ScriptObservation
from document_reconstruction.recognition.types import Word
from document_reconstruction.model import Box


def native_pdf(*, pages: int = 1, rotation: int = 0, short: bool = False) -> bytes:
    with pymupdf.open() as pdf:
        for index in range(pages):
            page = pdf.new_page(width=612, height=792)
            text = "Short" if short else f"Native page {index + 1} has useful editable text."
            page.insert_text((72, 96), text, fontsize=12)
            if rotation:
                page.set_rotation(rotation)
        return pdf.tobytes()


def raster_pdf(*, blank: bool = False) -> bytes:
    image = Image.new("RGB", (612, 792), "white")
    if not blank:
        draw = ImageDraw.Draw(image)
        draw.text((70, 90), "Raster-only page content", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=buffer.getvalue())
        return pdf.tobytes()


def searchable_scan_pdf(*, layer_lines: int | None = None, duplicate_layer: bool = False) -> bytes:
    lines = [
        "Searchable scan heading",
        "First complete line of raster text",
        "Second complete line of raster text",
        "Third complete line of raster text",
        "Fourth complete line of raster text",
    ]
    points = [(72, 100 + 42 * index) for index in range(len(lines))]
    with pymupdf.open() as source:
        page = source.new_page(width=612, height=792)
        for point, text in zip(points, lines):
            page.insert_text(point, text, fontsize=16)
        pixmap = page.get_pixmap(dpi=72, colorspace=pymupdf.csRGB, alpha=False)
        raster = pixmap.tobytes("png")
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=raster)
        count = len(lines) if layer_lines is None else layer_lines
        for point, text in zip(points[:count], lines[:count]):
            page.insert_text(point, text, fontsize=16, render_mode=3)
            if duplicate_layer:
                page.insert_text(point, text, fontsize=16, render_mode=3)
        return pdf.tobytes()


class PreflightTests(unittest.TestCase):
    def plans(self, payload: bytes, *, timeout_ms: float = 45_000):
        with ConversionContext(timeout_ms=timeout_ms) as context:
            with pymupdf.open(stream=payload, filetype="pdf") as pdf:
                plans = analyze(pdf, context, CostPredictor())
            snapshot = context.metrics.snapshot()
        return plans, snapshot

    def test_rotated_native_page_uses_normalized_dimensions(self):
        plans, _ = self.plans(native_pdf(rotation=90))
        self.assertEqual(plans[0].rotation, 90)
        self.assertEqual((plans[0].width, plans[0].height), (612, 792))
        self.assertEqual(plans[0].route, PageRoute.NATIVE)

    def test_blank_and_image_only_pages_are_distinguished(self):
        with pymupdf.open() as pdf:
            pdf.new_page(width=612, height=792)
            blank = pdf.tobytes()
        blank_plans, _ = self.plans(blank)
        image_plans, _ = self.plans(raster_pdf())
        self.assertEqual(blank_plans[0].route, PageRoute.BLANK)
        self.assertEqual(image_plans[0].route, PageRoute.OCR)
        self.assertEqual(image_plans[0].native_characters, 0)
        self.assertGreater(image_plans[0].image_coverage, 0.95)

    def test_invisible_complete_text_layer_is_trusted_without_font_name_dependency(self):
        plans, _ = self.plans(searchable_scan_pdf())
        plan = plans[0]
        self.assertEqual(plan.route, PageRoute.OCR)
        self.assertTrue(plan.searchable_scan)
        self.assertEqual(plan.text_layer_status, "TRUSTED")
        self.assertGreaterEqual(plan.text_layer_confidence, 0.70)
        self.assertGreaterEqual(plan.text_layer_ink_coverage, 0.65)
        self.assertEqual(plan.render_megapixels, 0)

    def test_incomplete_or_duplicate_text_layer_uses_scan_recovery_not_reuse(self):
        partial, _ = self.plans(searchable_scan_pdf(layer_lines=1))
        duplicate, _ = self.plans(searchable_scan_pdf(duplicate_layer=True))
        for plan in (partial[0], duplicate[0]):
            self.assertEqual(plan.route, PageRoute.OCR)
            self.assertFalse(plan.searchable_scan)
            self.assertEqual(plan.text_layer_status, "UNTRUSTED")
            self.assertGreater(plan.render_megapixels, 0)
        self.assertLess(partial[0].text_layer_ink_coverage, 0.65)
        self.assertGreater(duplicate[0].text_layer_duplicate_ratio, 0.05)

    def test_native_extraction_keeps_reliable_size_and_emphasis_observations(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=612, height=792)
            page.insert_text((72, 96), "BoldTitle", fontsize=18, fontname="hebo")
            page.insert_text((72, 130), "Regular", fontsize=11, fontname="helv")
            recognized = extract_native(page, 0)
            self.assertTrue(recognized.words)
            self.assertTrue(all(word.source_page == 0 for word in recognized.words))
        words = {word.text: word for word in recognized.words}
        self.assertTrue(words["BoldTitle"].bold)
        self.assertAlmostEqual(words["BoldTitle"].size, 18, delta=0.5)
        self.assertFalse(words["Regular"].bold)
        self.assertAlmostEqual(words["Regular"].size, 11, delta=0.5)

    def test_multpage_and_short_native_pages_never_invoke_ocr(self):
        backend = Mock()
        result = ConversionEngine(ocr=backend).convert(native_pdf(pages=2, short=True))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["pages_native"], 2)
        self.assertEqual(result.metrics["pages_ocr"], 0)
        backend.detect_script.assert_not_called()
        backend.recognize.assert_not_called()
        self.assertEqual(result.metrics["ocr_ms"], 0)


    def test_native_preflight_never_renders_a_thumbnail(self):
        with patch("document_reconstruction.preflight._ink_coverage", side_effect=AssertionError("native preflight rendered")) as coverage:
            plans, _ = self.plans(native_pdf())
        self.assertEqual(plans[0].route, PageRoute.NATIVE)
        coverage.assert_not_called()

    def test_untrusted_existing_layer_runs_local_ocr_and_is_counted_separately(self):
        backend = Mock()
        backend.available = {"eng", "osd"}
        backend.detect_script.return_value = ScriptObservation(script="Latin", script_confidence=3.0)
        backend.recognize.return_value = [
            Word("Recovered", Box(72, 90, 150, 108), confidence=0.92, line_id=(0,)),
            Word("raster", Box(160, 90, 215, 108), confidence=0.92, line_id=(0,)),
            Word("content", Box(225, 90, 290, 108), confidence=0.92, line_id=(0,)),
        ]
        # The mocked words do not match the raster ink; this test is about routing only.
        with patch("document_reconstruction.engine.uncovered_text", return_value=Coverage([], 0)):
            # A tiny synthetic raster: the resolution gate is not what this test is about.
            settings = Settings(("Latin", "Cyrillic"), ("Latin", "Cyrillic"), 10, min_letter_px=0)
            result = ConversionEngine(ocr=backend, settings=settings).convert(searchable_scan_pdf(layer_lines=1))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["pages_ocr"], 1)
        self.assertEqual(result.metrics["pages_ocr_performed"], 1)
        self.assertEqual(result.metrics["pages_text_layer_reused"], 0)
        self.assertEqual(result.diagnostics[0]["recognition_source"], "local_ocr")
        self.assertTrue(result.diagnostics[0]["ocr_performed"])
        self.assertEqual(backend.recognize.call_count, 2)  # two independent readings

    def test_trusted_existing_layer_reuse_is_counted_without_ocr(self):
        backend = Mock()
        result = ConversionEngine(ocr=backend).convert(searchable_scan_pdf())
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["pages_ocr"], 1)
        self.assertEqual(result.metrics["pages_ocr_performed"], 0)
        self.assertEqual(result.metrics["pages_text_layer_reused"], 1)
        self.assertEqual(result.diagnostics[0]["recognition_source"], "existing_text_layer")
        self.assertFalse(result.diagnostics[0]["ocr_performed"])
        self.assertLess(result.diagnostics[0]["confidence"], 1.0)
        backend.detect_script.assert_not_called()
        backend.recognize.assert_not_called()

    def test_metrics_record_admission_estimate_and_zero_unused_stages(self):
        result = ConversionEngine(ocr=Mock()).convert(native_pdf())
        self.assertTrue(result.ok, result.error)
        self.assertGreater(result.metrics["admission_estimated_ms"], 0)
        self.assertIsNotNone(result.metrics["estimate_error_ms"])
        self.assertEqual(result.metrics["render_ms"], 0)
        self.assertEqual(result.metrics["geometry_ms"], 0)
        self.assertEqual(result.metrics["script_ms"], 0)
        self.assertEqual(result.metrics["ocr_ms"], 0)
        self.assertEqual(result.metrics["render_dpi"], 300)
        self.assertEqual(result.metrics["render_pixel_limit"], 36_000_000)


if __name__ == "__main__":
    unittest.main()
