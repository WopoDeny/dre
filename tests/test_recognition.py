import unittest
import io
from pathlib import Path
from unittest.mock import Mock

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from document_reconstruction.core import DeadlineBudget, EngineError, ErrorCode
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.recognition.geometry import recover, rectify_paper
from document_reconstruction.recognition.regions import detect_regions, detect_text_columns, prepare_text_ocr_image
from document_reconstruction.recognition.languages import profile
from document_reconstruction.recognition.registry import capability_report
from document_reconstruction.recognition.tesseract import ScriptObservation, TesseractBackend
from document_reconstruction.recognition.types import RecognizedPage, Word
from document_reconstruction.model import Box, Graphic, Paragraph
from document_reconstruction.understanding import understand_page

CORPUS = Path(__file__).resolve().parents[1] / "benchmarks" / "corpus"


def _fake_word(text="hello"):
    return Word(text, Box(20, 20, 100, 40), confidence=0.95, line_id=(1, 1, 1))


class _RegionalBackend:
    available = {"eng", "rus", "ara", "fas", "chi_sim", "chi_tra", "osd"}

    def __init__(self):
        self.profile = None

    def detect_script(self, image, budget):
        return ScriptObservation("Latin", 3.0, 0, 10.0)

    def detect_script_regions(self, image, budget):
        return [ScriptObservation("Latin", 3.0), ScriptObservation("Cyrillic", 3.0)]

    def recognize(self, image, languages, budget, **kwargs):
        self.profile = languages
        return [_fake_word("hello"), _fake_word("привет")]

    def close(self):
        pass


class _UncertainBackend:
    available = {"eng", "rus", "ara", "fas", "chi_sim", "chi_tra", "osd"}

    def __init__(self):
        self.recognize_calls = 0

    def detect_script(self, image, budget):
        return ScriptObservation()

    def detect_script_regions(self, image, budget):
        return [ScriptObservation(), ScriptObservation()]

    def recognize(self, image, languages, budget, **kwargs):
        self.recognize_calls += 1
        words = [_fake_word("garbage") for _ in range(10)]
        for word in words:
            word.confidence = 0.61
        return words

    def close(self):
        pass


class _UncertainArabicBackend:
    available = {"eng", "rus", "ara", "fas", "chi_sim", "chi_tra", "osd"}

    def __init__(self):
        self.profiles = []

    def detect_script(self, image, budget):
        return ScriptObservation()

    def detect_script_regions(self, image, budget):
        return [ScriptObservation(), ScriptObservation()]

    def recognize(self, image, languages, budget, **kwargs):
        self.profiles.append(languages)
        words = []
        for index in range(10):
            word = _fake_word("مرحبا")
            word.line_id = (index, 1, 1)
            word.confidence = 0.95
            words.append(word)
        return words

    def close(self):
        pass


class _UnsupportedRegionalBackend:
    available = {"Latin", "eng", "osd"}

    def __init__(self):
        self.recognize_calls = 0

    def detect_script(self, image, budget):
        return ScriptObservation("Latin", 8.0, 0, 8.0)

    def detect_script_regions(self, image, budget):
        return [ScriptObservation("Latin", 8.0), ScriptObservation("Devanagari", 6.0), ScriptObservation("Devanagari", 5.5)]

    def recognize(self, image, languages, budget, **kwargs):
        self.recognize_calls += 1
        return [_fake_word()]

    def close(self):
        pass


class _MissingMixedScriptBackend:
    available = {"eng", "rus", "ara", "fas", "chi_sim", "chi_tra", "osd"}

    def __init__(self):
        self.recognize_calls = 0

    def detect_script(self, image, budget):
        return ScriptObservation("Latin", 4.0, 0, 8.0)

    def detect_script_regions(self, image, budget):
        return [ScriptObservation("Latin", 4.0), ScriptObservation("Cyrillic", 4.0), ScriptObservation("Cyrillic", 3.5)]

    def recognize(self, image, languages, budget, **kwargs):
        self.recognize_calls += 1
        words = []
        for index in range(8):
            word = _fake_word("hello")
            word.line_id = (index, 1, 1)
            word.confidence = 0.96
            words.append(word)
        return words

    def close(self):
        pass


class RecognitionRoutingTests(unittest.TestCase):
    def test_unknown_high_confidence_script_is_not_silently_english(self):
        backend = Mock()
        backend.available = {"eng", "osd"}
        backend.detect_script.return_value = ScriptObservation("Devanagari", 8.0, 0, 8.0)
        result = ConversionEngine(ocr=backend).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertEqual(result.error["code"], "LANGUAGE_UNAVAILABLE")
        backend.recognize.assert_not_called()

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_unknown_high_confidence_regional_script_is_not_ignored(self):
        backend = _UnsupportedRegionalBackend()
        result = ConversionEngine(ocr=backend).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertEqual(result.error["code"], "LANGUAGE_UNAVAILABLE")
        self.assertEqual(result.error["details"].get("reason"), "unsupported_script_corroborated")
        self.assertEqual(backend.recognize_calls, 0)

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_complete_script_uncertainty_fails_closed_after_bounded_probe(self):
        backend = _UncertainBackend()
        result = ConversionEngine(ocr=backend).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertFalse(result.ok)
        self.assertEqual(result.error["code"], "LANGUAGE_UNAVAILABLE")
        self.assertEqual(result.error["details"].get("reason"), "script_detection_inconclusive_after_ocr")
        self.assertGreater(backend.recognize_calls, 0)

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_targeted_retry_respects_forced_script_after_uncertain_probe(self):
        backend = _UncertainArabicBackend()
        result = ConversionEngine(ocr=backend).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)
        self.assertEqual(backend.profiles[-1], "ara+fas+eng")
        self.assertEqual(result.metrics["retry_pages"], 1)

    def test_generic_script_models_are_preferred_when_installed(self):
        self.assertEqual(profile("Latin", {"Latin", "eng", "uzb", "aze", "osd"}), "Latin")
        self.assertEqual(profile("Cyrillic", {"Cyrillic", "rus", "tgk", "eng", "osd"}), "Cyrillic")
        self.assertEqual(profile("Arabic", {"Arabic", "ara", "fas", "eng", "osd"}), "Arabic")
        self.assertEqual(profile("Han", {"HanS", "HanT", "chi_sim", "chi_tra", "eng", "osd"}), "HanS+HanT")

    def test_missing_required_script_model_is_structured(self):
        with self.assertRaises(EngineError) as raised:
            profile("Arabic", {"eng", "osd"})
        self.assertEqual(raised.exception.code, ErrorCode.LANGUAGE_UNAVAILABLE)

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_regional_mixed_script_evidence_uses_bounded_mixed_profile(self):
        backend = _RegionalBackend()
        result = ConversionEngine(ocr=backend).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)
        self.assertEqual(backend.profile, "eng+rus+ara+fas+chi_sim+chi_tra")
        self.assertEqual(result.diagnostics[0]["script"], "Mixed")
        self.assertEqual(result.diagnostics[0]["script_evidence"], ["Latin", "Cyrillic"])
        self.assertTrue(result.diagnostics[0]["transformations"])

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_strong_mixed_script_evidence_cannot_succeed_when_one_script_is_missing(self):
        backend = _MissingMixedScriptBackend()
        result = ConversionEngine(ocr=backend).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertFalse(result.ok)
        self.assertEqual(result.error["code"], "UNREADABLE_DOCUMENT")
        self.assertEqual(result.error["stage"], "ocr")
        self.assertIn("Cyrillic", result.error["details"].get("missing_scripts", ""))
        self.assertEqual(result.metrics["retry_pages"], 1)
        self.assertLessEqual(backend.recognize_calls, 8)

    def test_priority_registry_reports_exact_and_generic_route_readiness_separately(self):
        report = capability_report({"eng", "osd", "rus"})
        turkmen = next(item for item in report["priority_languages"] if item["language"] == "Turkmen")
        self.assertFalse(turkmen["ready"])
        self.assertFalse(turkmen["exact_model_ready"])
        self.assertFalse(turkmen["production_route_ready"])
        report = capability_report({"eng", "osd", "Latin"})
        turkmen = next(item for item in report["priority_languages"] if item["language"] == "Turkmen")
        self.assertFalse(turkmen["exact_model_ready"])
        self.assertTrue(turkmen["production_route_ready"])
        self.assertEqual(turkmen["production_route_basis"], "generic_script")
        self.assertEqual(turkmen["production_route_models"], ["Latin"])


class MixedRegionTests(unittest.TestCase):
    def test_detects_four_recurring_text_columns_without_splitting_single_column(self):
        image = Image.new("RGB", (1600, 1200), "white")
        draw = ImageDraw.Draw(image)
        for left in (80, 440, 800, 1160):
            for row in range(8):
                y = 120 + row * 105
                draw.rectangle((left, y, left + 260, y + 32), fill="black")
        columns = detect_text_columns(image)
        self.assertEqual(len(columns), 4)
        self.assertEqual(columns[0][0], 0)
        self.assertEqual(columns[-1][1], image.width)

        single = Image.new("RGB", (1600, 1200), "white")
        draw = ImageDraw.Draw(single)
        for row in range(8):
            y = 120 + row * 105
            draw.rectangle((120, y, 1460, y + 32), fill="black")
        self.assertEqual(detect_text_columns(single), [])


class RecoveryTests(unittest.TestCase):
    def test_stronger_skew_is_corrected_with_transform_metadata(self):
        image = Image.new("RGB", (1400, 900), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        for y in range(120, 760, 80):
            draw.text((150, y), "Business review delivery plan", fill="black", font=font)
        skewed = image.rotate(6.0, expand=True, fillcolor="white")
        result = recover(skewed, DeadlineBudget())
        self.assertTrue(any(op.startswith("deskew:") for op in result.operations))
        self.assertTrue(any(step.kind == "deskew" for step in result.transforms))


    def test_perspective_rectification_allows_uniform_dark_background(self):
        paper = Image.new("RGB", (900, 1200), "white")
        draw = ImageDraw.Draw(paper)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 38)
        for y in range(120, 1050, 95):
            draw.text((90, y), "Perspective recovery keeps readable text", fill="black", font=font)
        canvas = Image.new("RGB", (1250, 1500), (70, 76, 82))
        # A mildly perspective-distorted sheet on a flat dark background.
        corners = [(180, 90), (1080, 160), (1010, 1370), (100, 1300)]
        import numpy as np
        source = [(0, 0), (paper.width, 0), (paper.width, paper.height), (0, paper.height)]
        matrix, values = [], []
        for (x, y), (u, v) in zip(corners, source):
            matrix.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]))
            values.extend((u, v))
        coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(values))
        photo = paper.transform(canvas.size, Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC, fillcolor=(70, 76, 82))
        rectified, transform = rectify_paper(photo)
        self.assertIsNotNone(transform)
        self.assertNotEqual(rectified.size, photo.size)

    def test_perspective_rectification_survives_pdf_like_resampling_edges(self):
        paper = Image.new("RGB", (900, 1200), "white")
        draw = ImageDraw.Draw(paper)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 38)
        for y in range(120, 1050, 95):
            draw.text((90, y), "Perspective recovery keeps readable text", fill="black", font=font)
        import numpy as np
        corners = [(180, 90), (1080, 160), (1010, 1370), (100, 1300)]
        source = [(0, 0), (paper.width, 0), (paper.width, paper.height), (0, paper.height)]
        matrix, values = [], []
        for (x, y), (u, v) in zip(corners, source):
            matrix.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]))
            values.extend((u, v))
        coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(values))
        photo = paper.transform((1250, 1500), Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC, fillcolor=(70, 76, 82))
        # PDF page fitting can introduce long antialiased stripes at the raster
        # frame; those are not readable external content and must not veto crop.
        resampled = photo.resize((1100, 1556), Image.Resampling.BICUBIC)
        rectified, transform = rectify_paper(resampled)
        self.assertIsNotNone(transform)
        self.assertNotEqual(rectified.size, resampled.size)

    def test_perspective_rectification_does_not_crop_external_edge_content(self):
        paper = Image.new("RGB", (900, 1200), "white")
        draw = ImageDraw.Draw(paper)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 38)
        for y in range(120, 1050, 95):
            draw.text((90, y), "Perspective recovery keeps readable text", fill="black", font=font)
        import numpy as np
        corners = [(180, 90), (1080, 160), (1010, 1370), (100, 1300)]
        source = [(0, 0), (paper.width, 0), (paper.width, paper.height), (0, paper.height)]
        matrix, values = [], []
        for (x, y), (u, v) in zip(corners, source):
            matrix.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]))
            values.extend((u, v))
        coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(values))
        photo = paper.transform((1250, 1500), Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC, fillcolor=(70, 76, 82))
        outside = ImageDraw.Draw(photo)
        for y in range(180, 1250, 45):
            outside.line((20, y, 75, y + 18), fill="white", width=5)
            outside.line((1120, y, 1230, y), fill="white", width=4)
        rectified, transform = rectify_paper(photo)
        self.assertIsNone(transform)
        self.assertEqual(rectified.size, photo.size)

    def test_recovery_honors_exhausted_deadline(self):
        class Clock:
            value = 0
            def __call__(self):
                self.value += 2_000_000
                return self.value
        with self.assertRaises(EngineError) as raised:
            recover(Image.new("RGB", (800, 800), "white"), DeadlineBudget(1, clock=Clock()))
        self.assertEqual(raised.exception.code, ErrorCode.DEADLINE_EXCEEDED)


class RealTesseractMetadataTests(unittest.TestCase):
    def test_backend_reports_initialized_capabilities_and_load_timings(self):
        backend = TesseractBackend()
        try:
            capabilities = backend.capabilities
            self.assertEqual(capabilities["backend"], "tesseract")
            self.assertIn("eng", capabilities["initialized_profiles"])
            self.assertIn("osd", capabilities["initialized_profiles"])
            self.assertGreaterEqual(capabilities["initialization_ms"], 0)
            self.assertIn("eng", capabilities["profile_load_ms"])
            self.assertEqual(capabilities["runtime_version"], "5.5.0")
            identity = backend.model_identity("eng")
            self.assertEqual(len(identity["sha256"]), 64)
            self.assertGreater(identity["bytes"], 0)
        finally:
            backend.close()


class GraphicRegionTests(unittest.TestCase):
    @staticmethod
    def _font(size=32):
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)

    def test_overlapping_color_stamp_is_masked_and_does_not_own_body_text(self):
        image = Image.new("RGB", (1000, 1400), "white")
        draw = ImageDraw.Draw(image)
        text = "Approved amount 12345"
        xy = (120, 600)
        font = self._font()
        draw.text(xy, text, font=font, fill="black")
        bbox = draw.textbbox(xy, text, font=font)
        word = Word(text, Box(*bbox), 0.96, size=32, line_id=(1, 1, 1))
        draw.ellipse((180, 540, 520, 760), outline=(200, 20, 30), width=12)
        tables, graphics, diagnostics, issues = detect_regions(image, [word], index=0, width=1000, height=1400)
        self.assertFalse(tables)
        self.assertEqual(len(graphics), 1)
        self.assertEqual(diagnostics["graphics_retained"], 1)
        self.assertEqual(diagnostics["graphics_unresolved"], 0)
        self.assertEqual(graphics[0].source_region, "raster_graphic:color_mask")
        asset = Image.open(io.BytesIO(graphics[0].data))
        self.assertEqual(asset.mode, "RGBA")
        self.assertEqual(asset.getextrema()[3][0], 0)
        self.assertGreater(asset.getextrema()[3][1], 0)
        page = understand_page(RecognizedPage(0, 1000, 1400, [word], graphics=graphics, issues=issues))
        self.assertTrue(any(isinstance(block, Paragraph) and block.text == text for block in page.blocks))
        self.assertTrue(any(isinstance(block, Graphic) for block in page.blocks))


    def test_low_confidence_text_physically_crossed_by_color_mark_is_unresolved(self):
        image = Image.new("RGB", (1000, 1400), "white")
        draw = ImageDraw.Draw(image)
        font = self._font()
        text = "Approved amount 12345"
        xy = (120, 600)
        draw.text(xy, text, font=font, fill="black")
        bbox = draw.textbbox(xy, text, font=font)
        # The word is already unreliable and a retained color mark crosses it.
        word = Word(text, Box(*bbox), 0.52, size=32, line_id=(1, 1, 1))
        draw.ellipse((180, 540, 520, 760), outline=(200, 20, 30), width=12)
        _, graphics, diagnostics, issues = detect_regions(image, [word], index=0, width=1000, height=1400)
        self.assertEqual(len(graphics), 1)
        self.assertGreaterEqual(diagnostics["graphics_text_overlap_unresolved"], 1)
        self.assertTrue(any(issue.code == "unresolved_graphic_text_overlap" and issue.severity == "error" for issue in issues))


    def test_four_column_parallel_text_is_not_flagged_as_borderless_table(self):
        image = Image.new("RGB", (1200, 1400), "white")
        words = []
        for row in range(4):
            for column, text in enumerate(("Delivery plan", "Проверка плана", "مراجعة الخطة", "文档审核计划")):
                x0 = 60 + column * 285
                y0 = 180 + row * 180
                words.append(Word(text, Box(x0, y0, x0 + 190, y0 + 34), 0.94, line_id=(row, column, 0)))
        tables, _, diagnostics, issues = detect_regions(image, words, index=0, width=1200, height=1400)
        self.assertFalse(tables)
        self.assertFalse(diagnostics["unresolved_borderless_table"])
        self.assertFalse(any(issue.code == "unresolved_borderless_raster_table" for issue in issues))

    def test_concentric_color_seal_is_one_graphic_not_duplicate_assets(self):
        image = Image.new("RGB", (1000, 1400), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((300, 850, 610, 1160), outline=(210, 30, 30), width=14)
        draw.ellipse((340, 890, 570, 1120), outline=(210, 30, 30), width=7)
        _, graphics, diagnostics, issues = detect_regions(image, [], index=0, width=1000, height=1400)
        self.assertFalse(issues)
        self.assertEqual(len(graphics), 1)
        self.assertEqual(graphics[0].role, "seal")
        self.assertEqual(diagnostics["graphics_color_candidates"], 1)
        self.assertEqual(diagnostics["graphics_retained"], 1)

    def test_ocr_working_copy_suppresses_sparse_color_mark_not_dense_logo(self):
        image = Image.new("RGB", (1000, 1400), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((180, 500, 620, 760), outline=(200, 20, 30), width=14)
        draw.rectangle((120, 1000, 500, 1180), fill=(20, 120, 190))
        working, diagnostics = prepare_text_ocr_image(image)
        self.assertEqual(diagnostics["ocr_color_mask_candidates"], 1)
        self.assertGreater(diagnostics["ocr_color_masked_pixels"], 0)
        # Sparse red seal/stamp pixels are whitened in the OCR-only copy.
        self.assertEqual(working.getpixel((180, 500)), (255, 255, 255))
        # Dense blue logo remains available to OCR rather than being silently erased.
        self.assertEqual(working.getpixel((200, 1050)), image.getpixel((200, 1050)))

    def test_isolated_black_signature_like_mark_is_retained_and_tracked(self):
        image = Image.new("RGB", (1000, 1400), "white")
        draw = ImageDraw.Draw(image)
        font = self._font()
        words = []
        for index, text in enumerate(("Approved delivery plan", "Signed below")):
            xy = (80, 120 + index * 60)
            draw.text(xy, text, font=font, fill="black")
            words.append(Word(text, Box(*draw.textbbox(xy, text, font=font)), 0.95, size=32, line_id=(index, 1, 1)))
        points = [(180, 1040), (230, 970), (250, 1060), (290, 985), (315, 1045), (365, 995), (410, 1040), (470, 985), (530, 1035)]
        draw.line(points, fill="black", width=7, joint="curve")
        draw.arc((180, 970, 550, 1080), 200, 340, fill="black", width=4)
        tables, graphics, diagnostics, issues = detect_regions(image, words, index=0, width=1000, height=1400)
        self.assertFalse(tables)
        self.assertFalse(issues)
        self.assertEqual(len(graphics), 1)
        self.assertEqual(graphics[0].role, "signature")
        self.assertEqual(graphics[0].source_region, "raster_graphic:black_signature_heuristic")
        self.assertEqual(diagnostics["graphics_black_candidates"], 1)
        self.assertEqual(diagnostics["graphics_retained"], 1)
        self.assertEqual(diagnostics["graphics_roles"].get("signature"), 1)
        self.assertTrue(any(issue.code == "heuristic_black_signature" for issue in graphics[0].issues))

    def test_ambiguous_black_region_is_retained_without_inventing_a_semantic_role(self):
        image = Image.new("RGB", (1000, 1400), "white")
        draw = ImageDraw.Draw(image)
        # A compact isolated black box is graphic-like but not signature-shaped.
        draw.rectangle((395, 895, 525, 1025), outline="black", width=8)
        draw.line((400, 900, 520, 1020), fill="black", width=8)
        draw.line((520, 900, 400, 1020), fill="black", width=8)
        tables, graphics, diagnostics, issues = detect_regions(image, [], index=0, width=1000, height=1400)
        self.assertFalse(tables)
        self.assertEqual(len(graphics), 1)
        self.assertEqual(graphics[0].role, "graphic")
        self.assertEqual(graphics[0].source_region, "raster_graphic:black_unresolved")
        self.assertGreaterEqual(diagnostics["graphics_unresolved"], 1)
        self.assertTrue(any(issue.code == "unresolved_graphic_role" for issue in graphics[0].issues))
        self.assertFalse(any(issue.code == "unresolved_raster_graphic_region" for issue in issues))


if __name__ == "__main__":
    unittest.main()
