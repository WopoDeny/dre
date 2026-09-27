"""Small real-conversion regressions against generated public fixtures."""

import unittest
import io
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from document_reconstruction.engine import ConversionEngine
from document_reconstruction.preflight import CostPredictor
from document_reconstruction.benchmark.runner import character_error_rate, infrastructure_record, summarize, unicode_error_rate
from document_reconstruction.core.metrics import Stage
from document_reconstruction.recognition.tesseract import ScriptObservation
from document_reconstruction.recognition.types import Word
from document_reconstruction.model import Box, Document, InlineRun, Page, Paragraph, Table
import pymupdf
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas

CORPUS = Path(__file__).resolve().parents[1] / "benchmarks" / "corpus"


class _ReserveExhaustionBackend:
    available = {"eng", "osd"}

    def detect_script(self, image, budget):
        return ScriptObservation("Latin", 3.0, 0, 3.0)

    def recognize(self, image, languages, budget, **kwargs):
        budget.timeout_seconds(reserve_ms=10_000, stage="ocr")
        return [Word("never", Box(10, 10, 40, 25), confidence=0.99)]

    def close(self):
        pass


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ConversionEngine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_invalid_input_is_structured_and_has_no_docx(self):
        result = self.engine.convert(b"invalid")
        self.assertEqual(result.error["code"], "INVALID_PDF")
        self.assertIsNone(result.docx)
        self.assertIn("total_ms", result.metrics)

    def test_partial_native_overlay_never_silently_drops_raster_text(self):
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100), "gray").save(buffer, format="PNG")
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((48, 48), "Native header")
            page.insert_image(pymupdf.Rect(30, 100, 560, 800), stream=buffer.getvalue())
            payload = pdf.tobytes()
        result = self.engine.convert(payload)
        self.assertEqual(result.error["code"], "CONVERSION_FAILED")
        self.assertIsNone(result.docx)
        self.assertEqual(result.metrics["ocr_ms"], 0)

    def test_searchable_scan_text_layer_is_recovered_without_full_page_image(self):
        # Tesseract searchable PDFs use GlyphLessFont for their hidden OCR
        # layer. This fixture exercises that compatibility route without
        # invoking OCR a second time.
        source = CORPUS / "searchable_scan.pdf"
        # A raster at this page size would exceed the old OCR cost prior under
        # a short budget; direct text-layer recovery should still fit easily.
        result = self.engine.convert(source.read_bytes(), timeout_ms=2_000)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["pages_ocr"], 1)
        self.assertEqual(result.metrics["ocr_ms"], 0)
        self.assertEqual(result.diagnostics[0]["route"], "OCR_TEXT_LAYER")
        self.assertIn("searchable scan", result.model.text().lower())
        self.assertEqual(result.quality["text_coverage"], 1.0)
        self.assertTrue(any(run.source_region == "searchable_text_layer" for block in result.model.pages[0].blocks if isinstance(block, Paragraph) for run in block.runs))
        with zipfile.ZipFile(io.BytesIO(result.docx)) as package:
            self.assertEqual([name for name in package.namelist() if name.startswith("word/media/")], [])

    def test_native_pages_never_call_ocr(self):
        backend = Mock()
        result = ConversionEngine(ocr=backend).convert((CORPUS / "native_small.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)
        backend.recognize.assert_not_called()
        backend.detect_script.assert_not_called()
        self.assertEqual(result.metrics["ocr_ms"], 0)
        self.assertLess(len(result.model.pages[0].blocks), 11)


    def test_native_columns_keep_parallel_relationship_and_sentence_boundaries(self):
        result = self.engine.convert((CORPUS / "native_columns.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)
        page = result.model.pages[0]
        heading, layout = page.blocks
        self.assertEqual(heading.kind, "heading")
        self.assertEqual(heading.text, "Business Review")
        left, right = [cell.text.split("\n") for cell in layout.cells]
        self.assertEqual(left, [f"Left column statement {number}." for number in range(1, 8)])
        self.assertEqual(right, [f"Right column statement {number}." for number in range(1, 8)])


    def test_native_borderless_table_with_bold_header_is_reconstructed_once(self):
        buffer = io.BytesIO()
        writer = canvas.Canvas(buffer, pagesize=(595, 842), invariant=1)
        writer.setFont("Helvetica-Bold", 14)
        writer.drawString(48, 790, "Borderless Schedule")
        writer.setFont("Helvetica-Bold", 11)
        for x, value in ((48, "Item"), (260, "Qty"), (360, "Status")):
            writer.drawString(x, 745, value)
        writer.setFont("Helvetica", 11)
        for row_index, values in enumerate((("Core service", "12", "Approved"), ("Integration guide", "3", "Pending"), ("Review notes", "8", "Complete"))):
            y = 715 - row_index * 28
            for x, value in zip((48, 260, 360), values):
                writer.drawString(x, y, value)
        writer.drawString(48, 600, "Notes below the table remain separate.")
        writer.save()
        result = self.engine.convert(buffer.getvalue())
        self.assertTrue(result.ok, result.error)
        tables = [block for block in result.model.pages[0].blocks if isinstance(block, Table)]
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table.source_region, "native_borderless_table")
        self.assertEqual((table.rows, table.columns), (4, 3))
        expected = {"Item", "Qty", "Status", "Core service", "12", "Approved", "Integration guide", "3", "Pending", "Review notes", "8", "Complete"}
        self.assertEqual({cell.text for cell in table.cells}, expected)
        model_text = result.model.text()
        for value in expected:
            self.assertEqual(model_text.count(value), 1)
        self.assertIn("Notes below the table remain separate.", model_text)

    def test_parallel_native_columns_are_not_misclassified_as_borderless_table(self):
        result = self.engine.convert((CORPUS / "native_columns.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)
        tables = [block for block in result.model.pages[0].blocks if isinstance(block, Table)]
        # Side-by-side text columns are a borderless layout table, never a data table.
        self.assertTrue(all(table.borderless for table in tables))
        self.assertEqual([table.columns for table in tables], [2])

    def test_native_table_is_editable(self):
        result = self.engine.convert((CORPUS / "native_table.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.quality["tables"], 1)

    def test_native_rotation_metadata_preserves_logical_reading_order(self):
        source = (CORPUS / "native_small.pdf").read_bytes()
        reference = self.engine.convert(source).model.text()
        for angle in (90, 180, 270):
            with self.subTest(angle=angle), pymupdf.open(stream=source, filetype="pdf") as pdf:
                pdf[0].set_rotation(angle)
                result = self.engine.convert(pdf.tobytes())
                self.assertTrue(result.ok, result.error)
                self.assertEqual(result.model.text(), reference)
                self.assertEqual(result.model.pages[0].source_rotation, angle)
                self.assertEqual(result.metrics["ocr_ms"], 0)

    def test_real_scan_and_rotation_preserve_text(self):
        for name in ("scan_1", "scan_rotated_90", "scan_rotated_270"):
            with self.subTest(name=name):
                result = self.engine.convert((CORPUS / f"{name}.pdf").read_bytes())
                self.assertTrue(result.ok, result.error)
                text = result.model.text()
                self.assertIn("Business Review", text)
                self.assertLess(text.index("operations team"), text.index("first delivery"))
                self.assertEqual(result.metrics["pages_ocr"], 1)

    def test_mixed_pdf_routes_per_page(self):
        result = self.engine.convert((CORPUS / "mixed.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["pages_native"], 1)
        self.assertEqual(result.metrics["pages_ocr"], 1)


    def test_native_and_scanned_text_converge_to_unified_model_runs(self):
        cases = (("native_small.pdf", "native_pdf"), ("scan_1.pdf", "ocr"))
        for filename, source_prefix in cases:
            with self.subTest(filename=filename):
                result = self.engine.convert((CORPUS / filename).read_bytes())
                self.assertTrue(result.ok, result.error)
                page = result.model.pages[0]
                paragraphs = [block for block in page.blocks if isinstance(block, Paragraph)]
                self.assertTrue(paragraphs)
                self.assertTrue(all(paragraph.runs for paragraph in paragraphs if paragraph.text))
                self.assertTrue(all("".join(run.text for run in paragraph.runs) == paragraph.text for paragraph in paragraphs))
                self.assertTrue(any(run.source_region.startswith(source_prefix) for paragraph in paragraphs for run in paragraph.runs))
                self.assertEqual(result.model.metadata["coordinate_system"], "normalized-unrotated-points-top-left")
                self.assertEqual(result.model.metadata["pagination_policy"], "preserve-source-pages")
                # Sentences on hand-broken ragged lines may form one or several
                # paragraphs; the text, headings and list items must converge.
                signature = (" ".join(paragraph.text for paragraph in paragraphs),
                             [(paragraph.kind, paragraph.text) for paragraph in paragraphs if paragraph.kind != "paragraph"])
                if filename == "native_small.pdf":
                    native_signature = signature
                else:
                    self.assertEqual(signature, native_signature)

    def test_invalid_unified_model_fails_before_docx_reconstruction(self):
        broken = Document([Page(0, 595, 842, [
            Paragraph(
                "Expected",
                Box(48, 48, 180, 64),
                0,
                runs=[InlineRun("Different", Box(48, 48, 180, 64), 0, source_region="native_pdf")],
            )
        ])])
        with patch("document_reconstruction.engine.understand", return_value=broken), patch("document_reconstruction.engine.reconstruct") as writer:
            result = self.engine.convert((CORPUS / "native_small.pdf").read_bytes())
        self.assertFalse(result.ok)
        self.assertEqual(result.error["code"], "INVARIANT_VIOLATION")
        self.assertEqual(result.error["stage"], "understanding")
        self.assertEqual(result.error["details"]["issue_count"], 1)
        writer.assert_not_called()

    def test_native_table_enters_model_once_with_cell_provenance(self):
        result = self.engine.convert((CORPUS / "native_table.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)
        table = next(block for block in result.model.pages[0].blocks if isinstance(block, Table))
        self.assertEqual(table.source_region, "native_table")
        self.assertTrue(table.cells)
        self.assertTrue(all(cell.source_page == 0 for cell in table.cells))
        self.assertTrue(all(cell.box is not None for cell in table.cells))
        model_text = result.model.text()
        for cell in table.cells:
            if cell.text.strip():
                self.assertEqual(model_text.count(cell.text), 1)


    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')


    def test_mixed_script_raster_table_uses_targeted_cell_retry_without_degrading_strong_cells(self):
        image = Image.new("RGB", (1654, 2339), "white")
        draw = ImageDraw.Draw(image)
        latin = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", 38)
        arabic = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf", 40)
        han = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 40)
        draw.text((100, 100), "Note above the table remains editable.", font=latin, fill="black")
        xs, ys = [100, 700, 1100, 1550], [420, 670, 1020, 1370, 1720, 2190]
        for y in ys:
            draw.line((xs[0], y, xs[-1], y), fill="black", width=5)
        for x in (xs[0], xs[-1]):
            draw.line((x, ys[0], x, ys[-1]), fill="black", width=5)
        for x in xs[1:-1]:
            draw.line((x, ys[1], x, ys[-1]), fill="black", width=5)
        draw.text((520, 500), "Delivery summary", font=latin, fill="black")
        draw.text((130, 735), "Item line 1", font=latin, fill="black")
        draw.text((130, 790), "continued", font=latin, fill="black")
        draw.text((760, 760), "12", font=latin, fill="black")
        draw.text((1150, 760), "Approved", font=latin, fill="black")
        draw.text((130, 1090), "مرحبا بالعالم", font=arabic, fill="black", direction="rtl")
        draw.text((1150, 1090), "审核完成。", font=han, fill="black")
        draw.text((130, 1440), "Empty cell", font=latin, fill="black")
        draw.text((1150, 1440), "Pending", font=latin, fill="black")
        draw.text((130, 1790), "Boundary row", font=latin, fill="black")
        draw.text((760, 1790), "8", font=latin, fill="black")
        draw.text((1150, 1790), "Complete", font=latin, fill="black")
        draw.text((100, 2240), "Note below table remains editable.", font=latin, fill="black")
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=595, height=842)
            page.insert_image(page.rect, stream=encoded.getvalue())
            payload = pdf.tobytes()
        result = self.engine.convert(payload)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.metrics["retry_pages"], 1)
        retry = next(entry for entry in result.diagnostics if entry.get("retry"))
        self.assertEqual(retry["ocr_partitioning"], "table_cells")
        self.assertEqual(retry["table_cell_reocr_count"], 2)
        table = next(block for block in result.model.pages[0].blocks if isinstance(block, Table))
        self.assertEqual((table.rows, table.columns), (5, 3))
        by_position = {(cell.row, cell.column): cell for cell in table.cells}
        self.assertEqual(by_position[(0, 0)].column_span, 3)
        self.assertEqual(by_position[(1, 0)].text, "Item line 1 continued")
        self.assertEqual(by_position[(1, 2)].text, "Approved")
        self.assertEqual(by_position[(2, 0)].text, "مرحبا بالعالم")
        self.assertEqual(by_position[(2, 0)].direction, "rtl")
        self.assertEqual(by_position[(2, 2)].text, "审核完成。")
        self.assertEqual(by_position[(2, 1)].text, "")
        model_text = result.model.text()
        self.assertEqual(model_text.count("Approved"), 1)
        self.assertEqual(model_text.count("مرحبا بالعالم"), 1)
        self.assertEqual(model_text.count("审核完成。"), 1)
        self.assertIn("Note above the table remains editable.", model_text)
        self.assertIn("Note below table remains editable.", model_text)

    def test_borderless_raster_table_fails_closed_instead_of_silent_flattening(self):
        image = Image.new("RGB", (1654, 2339), "white")
        draw = ImageDraw.Draw(image)
        regular = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", 42)
        bold = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 42)
        xs = (120, 690, 1120)
        for x, text in zip(xs, ("Item", "Qty", "Status")):
            draw.text((x, 420), text, font=bold, fill="black")
        for row_index, values in enumerate((("Core service", "12", "Approved"), ("Guide", "3", "Pending"), ("Notes", "8", "Complete"), ("Audit", "2", "Ready"))):
            y = 560 + row_index * 180
            for x, text in zip(xs, values):
                draw.text((x, y), text, font=regular, fill="black")
        draw.text((120, 120), "Borderless schedule below.", font=regular, fill="black")
        draw.text((120, 1500), "Important note below table.", font=regular, fill="black")
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=595, height=842)
            page.insert_image(page.rect, stream=encoded.getvalue())
            result = self.engine.convert(pdf.tobytes())
        self.assertFalse(result.ok)
        self.assertEqual(result.error["code"], "UNREADABLE_DOCUMENT")
        self.assertEqual(result.error["stage"], "understanding")
        self.assertEqual(result.error["details"].get("reason"), "unresolved_borderless_raster_table")
        self.assertIsNone(result.docx)


    def test_overlapping_color_stamp_with_unreliable_text_fails_closed(self):
        image = Image.new("RGB", (1654, 2339), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", 42)
        draw.text((120, 180), "Approved delivery record 12345 remains editable.", font=font, fill="black")
        draw.ellipse((280, 110, 870, 430), outline=(200, 20, 30), width=18)
        draw.text((430, 250), "PAID", font=font, fill=(200, 20, 30))
        encoded = io.BytesIO(); image.save(encoded, format="PNG")
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=595, height=842)
            page.insert_image(page.rect, stream=encoded.getvalue())
            result = self.engine.convert(pdf.tobytes())
        self.assertFalse(result.ok)
        self.assertEqual(result.error["code"], "UNREADABLE_DOCUMENT")
        # Words under the stamp that cannot be restored leave the text unreadable: never issued as a guess.
        self.assertIn(result.error["details"].get("reason"), ("unresolved_graphic_text_overlap", "text_unreadable"))
        self.assertIsNone(result.docx)

    def test_short_budget_rejects_before_ocr(self):
        result = self.engine.convert((CORPUS / "scan_5.pdf").read_bytes(), timeout_ms=1000)
        self.assertEqual(result.error["code"], "DOCUMENT_TOO_EXPENSIVE")
        self.assertEqual(result.metrics["ocr_ms"], 0)

    def test_mid_request_tail_reserve_exhaustion_is_deadline_failure_not_admission_rejection(self):
        predictor = CostPredictor(**{name: 0 for name in CostPredictor.__dataclass_fields__})
        engine = ConversionEngine(ocr=_ReserveExhaustionBackend(), predictor=predictor)
        result = engine.convert((CORPUS / "scan_1.pdf").read_bytes(), timeout_ms=5_000)
        self.assertFalse(result.ok)
        self.assertEqual(result.error["code"], "FAST_SLA_EXCEEDED")
        self.assertEqual(result.error["stage"], "ocr")

    def test_unresolved_rotation_cannot_pass_on_confidence_alone(self):
        backend = Mock()
        backend.available = {"eng", "osd"}
        backend.detect_script.return_value = ScriptObservation("Latin", 0.5)
        backend.recognize.return_value = [Word("word", Box(50 + index * 10, 100, 54 + index * 10, 150), confidence=0.99, line_id=(index,)) for index in range(10)]
        result = ConversionEngine(ocr=backend).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertEqual(result.error["code"], "UNREADABLE_DOCUMENT")
        self.assertIsNone(result.docx)
        self.assertEqual(result.metrics["retry_pages"], 0)
        self.assertEqual(backend.detect_script.call_count, 1)


class BenchmarkTests(unittest.TestCase):
    def test_success_rejection_and_retry_have_full_schema(self):
        for name in ("lifecycle", "early_rejection", "targeted_retry"):
            record = infrastructure_record(name)
            self.assertTrue(record["expectation_met"])
            for key in ["total_ms", *[f"{stage.value}_ms" for stage in Stage], "pages_total", "pages_native", "pages_ocr", "retry_pages", "ram_peak_bytes", "vram_peak_bytes"]:
                self.assertIn(key, record)

    def test_cer_and_warmup_exclusion(self):
        self.assertAlmostEqual(character_error_rate("hello", "hallo"), 0.2)
        self.assertEqual(character_error_rate("same", "same"), 0)
        self.assertEqual(character_error_rate("123/45،", "12345."), 0)
        self.assertGreater(unicode_error_rate("123/45،", "12345."), 0)
        self.assertEqual(unicode_error_rate("中文，内容。", "中文，内容。"), 0)
        row = infrastructure_record("lifecycle")
        summary = summarize([{**row, "phase": "warmup"}, {**row, "phase": "measured"}])
        self.assertEqual(summary["measured_runs"], 1)


if __name__ == "__main__":
    unittest.main()
