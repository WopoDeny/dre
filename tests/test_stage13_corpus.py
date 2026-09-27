"""Check that synthetic OCR ground truth is actually visible on its source."""

import unittest
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage7_acceptance import FONTS, LANGUAGE_CASES, language_image


class LanguageFixtureVisibilityTests(unittest.TestCase):
    def test_all_language_line_glyphs_fit_inside_page(self):
        for language, (script, text, _) in LANGUAGE_CASES.items():
            with self.subTest(language=language):
                image = language_image(text, script)
                # The helper's rendered result must be nonblank in the three
                # intended bands; its bounding rectangle must stay in margins.
                grayscale = image.convert("L")
                for y in (260, 570, 880):
                    ink = grayscale.crop((80, y, 1574, y + 115))
                    self.assertIsNotNone(ink.getbbox())
                    self.assertLess(ink.getextrema()[0], 220)
                # There must be no nonwhite glyphs in the excluded side strips.
                left = grayscale.crop((0, 0, 80, image.height))
                right = grayscale.crop((1574, 0, image.width, image.height))
                self.assertGreaterEqual(left.getextrema()[0], 240)
                self.assertGreaterEqual(right.getextrema()[0], 240)


if __name__ == "__main__":
    unittest.main()
