import io
import unittest
from unittest.mock import Mock

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from document_reconstruction.config import Settings
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.recognition.regions import _text_lines
from document_reconstruction.recognition.scale import letter_height, ocr_scale


def scan_pdf(letter: int) -> bytes:
    image = Image.new("RGB", (600, 850), "white")
    draw = ImageDraw.Draw(image)
    for row in range(20):
        for column in range(30):
            x, y = 40 + column * (letter + 3), 60 + row * letter * 3
            draw.rectangle((x, y, x + letter - 1, y + letter), fill="black")
    buffer = io.BytesIO(); image.save(buffer, "PNG")
    document = pymupdf.open(); page = document.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return document.tobytes()


class ResolutionTests(unittest.TestCase):
    def test_tiny_letters_are_rejected_before_recognition(self):
        backend = Mock()
        result = ConversionEngine(ocr=backend, settings=Settings(("Latin", "Cyrillic"), ("Latin", "Cyrillic"), 10)).convert(scan_pdf(5))
        self.assertEqual(result.error["details"]["reason"], "low_resolution")
        backend.recognize.assert_not_called()

    def test_letter_height_is_measured_on_the_main_text_not_small_print(self):
        image = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(image)
        for row in range(4):          # small letterhead details: 6 px
            for column in range(60):
                x, y = 40 + column * 9, 40 + row * 14
                draw.rectangle((x, y, x + 4, y + 5), fill="black")
        for row in range(25):         # body text: 18 px
            for column in range(30):
                x, y = 40 + column * 22, 200 + row * 50
                draw.rectangle((x, y, x + 14, y + 17), fill="black")
        self.assertEqual(letter_height(image), 18)

    def test_ocr_scale_targets_readable_letter_size(self):
        self.assertAlmostEqual(ocr_scale(7), 2.0)
        self.assertEqual(ocr_scale(18), 1.0)
        self.assertLess(ocr_scale(40), 1.0)

    def test_text_rows_are_never_a_picture(self):
        ink = np.zeros((200, 300), dtype=bool)
        for row in range(5):
            ink[20 + row * 35:36 + row * 35, 10:290] = True
        self.assertGreaterEqual(_text_lines(ink, 16), 3)
        emblem = np.zeros((200, 200), dtype=bool)
        yy, xx = np.ogrid[:200, :200]
        emblem[(yy - 100) ** 2 + (xx - 100) ** 2 < 90 ** 2] = True
        self.assertEqual(_text_lines(emblem, 16), 0)


if __name__ == "__main__":
    unittest.main()
