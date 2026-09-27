"""A PDF text layer is copied as it is: its apostrophes are not normalized (only OCR text gets U+02BB)."""

import io
import unittest
from pathlib import Path

from docx import Document

from document_reconstruction.engine import ConversionEngine

NATIVE = Path(__file__).resolve().parents[1] / "src/document_reconstruction/selfcheck/native_uz_sample.pdf"


class NativeApostropheTests(unittest.TestCase):
    def test_text_layer_apostrophes_are_kept(self):
        result = ConversionEngine().convert(NATIVE.read_bytes())
        self.assertTrue(result.ok, result.error)
        text = "\n".join(paragraph.text for paragraph in Document(io.BytesIO(result.docx)).paragraphs)
        self.assertIn("O‘ZBEKISTON", text)
        self.assertNotIn("ʻ", text)


if __name__ == "__main__":
    unittest.main()
