import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from document_reconstruction.config import Settings
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.model import Box
from document_reconstruction.recognition.languages import disallowed_letters, letter_scripts, ocr_profile
from document_reconstruction.recognition.tesseract import ScriptObservation
from document_reconstruction.recognition.types import Word

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "corpus"
STRESS = ROOT / "benchmarks" / "stage29" / "stress20" / "pdfs"
DEFAULT = Settings(("Latin", "Cyrillic"), ("Latin", "Cyrillic"), 10)


class ScriptSettingsTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        with patch.dict("os.environ", {"DRE_PAGE_THREADS": "1"}, clear=True):
            self.assertEqual(Settings.from_env(), DEFAULT)  # page threads depend on the machine unless fixed
        with patch.dict("os.environ", {"DRE_OCR_ALLOWED_SCRIPTS": "Latin, Arabic", "DRE_MAX_PAGES": "5"}):
            settings = Settings.from_env()
            self.assertEqual(settings.ocr_scripts, ("Latin", "Arabic"))
            self.assertEqual(settings.max_pages, 5)
        with patch.dict("os.environ", {"DRE_OCR_ALLOWED_SCRIPTS": "Klingon"}):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_letter_scripts_ignore_symbols_and_modifier_letters(self):
        self.assertEqual(letter_scripts("Oʻzbekiston Қазақстан μm 5"), {"Latin": 11, "Cyrillic": 9})
        self.assertEqual(disallowed_letters("Раздел القسم 第一節", ("Latin", "Cyrillic")), {"Arabic": 5, "Han": 3})

    def test_profile_covers_all_allowed_scripts_with_primary_first(self):
        available = {"Latin", "Cyrillic", "eng", "osd"}
        self.assertEqual(ocr_profile(("Latin", "Cyrillic"), available), "Latin+Cyrillic")
        self.assertEqual(ocr_profile(("Latin", "Cyrillic"), available, first="Cyrillic"), "Cyrillic+Latin")


class NativeScriptGateTests(unittest.TestCase):
    def test_native_foreign_script_is_rejected_by_default_and_allowed_by_setting(self):
        payload = (STRESS / "03_zh_cn_en_agreement.pdf").read_bytes()
        rejected = ConversionEngine(ocr=Mock(), settings=DEFAULT).convert(payload)
        self.assertEqual(rejected.error["code"], "LANGUAGE_UNAVAILABLE")
        self.assertEqual(rejected.error["details"]["reason"], "script_not_allowed_native")
        allowed = Settings(DEFAULT.ocr_scripts, ("Latin", "Cyrillic", "Arabic", "Han"), 10)
        self.assertTrue(ConversionEngine(ocr=Mock(), settings=allowed).convert(payload).ok)

    def test_cyrillic_native_pdf_is_accepted(self):
        result = ConversionEngine(ocr=Mock(), settings=DEFAULT).convert((CORPUS / "native_cyrillic.pdf").read_bytes())
        self.assertTrue(result.ok, result.error)

    def test_documents_over_page_limit_are_rejected_before_recognition(self):
        backend = Mock()
        result = ConversionEngine(ocr=backend, settings=DEFAULT).convert((CORPUS / "native_large.pdf").read_bytes())
        self.assertEqual(result.error["code"], "DOCUMENT_TOO_EXPENSIVE")
        self.assertEqual(result.error["details"]["reason"], "too_many_pages")
        backend.recognize.assert_not_called()


class OCRScriptTests(unittest.TestCase):
    def _backend(self, observation):
        backend = Mock()
        backend.available = {"Latin", "Cyrillic", "eng", "osd"}
        backend.detect_script.return_value = observation
        backend.recognize.return_value = [
            Word("Текст", Box(50 + index * 40, 100, 85 + index * 40, 112), confidence=0.95, line_id=(1, 1, 1)) for index in range(8)
        ]
        return backend

    def test_strong_unsupported_osd_script_is_rejected_before_ocr(self):
        backend = self._backend(ScriptObservation("Arabic", 9.0))
        result = ConversionEngine(ocr=backend, settings=DEFAULT).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertEqual(result.error["code"], "LANGUAGE_UNAVAILABLE")
        backend.recognize.assert_not_called()

    def test_weak_osd_uses_one_pass_with_all_allowed_scripts(self):
        backend = self._backend(ScriptObservation("Arabic", 1.2))
        ConversionEngine(ocr=backend, settings=DEFAULT).convert((CORPUS / "scan_1.pdf").read_bytes())
        self.assertEqual(backend.recognize.call_args_list[0].args[1], "Latin+Cyrillic")
        self.assertEqual(backend.detect_script.call_count, 1)


if __name__ == "__main__":
    unittest.main()
