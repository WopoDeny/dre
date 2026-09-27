import io
import unittest

import pymupdf
from PIL import Image

from document_reconstruction.intake import image_kind, to_pdf


def encoded(image: Image.Image, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


class ImageIntakeTests(unittest.TestCase):
    def test_formats_are_recognized_by_signature(self):
        image = Image.new("RGB", (40, 30), "white")
        self.assertEqual(image_kind(encoded(image, "JPEG")), "jpeg")
        self.assertEqual(image_kind(encoded(image, "PNG")), "png")
        self.assertEqual(image_kind(encoded(image, "WEBP")), "webp")
        self.assertIsNone(image_kind(b"%PDF-1.7"))

    def test_one_image_becomes_one_a4_page(self):
        pdf = pymupdf.open(stream=to_pdf(encoded(Image.new("RGB", (1240, 1754), "white"), "JPEG")), filetype="pdf")
        self.assertEqual(pdf.page_count, 1)
        self.assertEqual((round(pdf[0].rect.width), round(pdf[0].rect.height)), (595, 842))
        wide = pymupdf.open(stream=to_pdf(encoded(Image.new("RGB", (1754, 1240), "white"), "PNG")), filetype="pdf")
        self.assertGreater(wide[0].rect.width, wide[0].rect.height)

    def test_exif_rotation_is_applied(self):
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise to display
        payload = encoded(Image.new("RGB", (1754, 1240), "white"), "JPEG", exif=exif)
        page = pymupdf.open(stream=to_pdf(payload), filetype="pdf")[0]
        self.assertLess(page.rect.width, page.rect.height)

    def test_transparency_becomes_white_paper(self):
        payload = encoded(Image.new("RGBA", (600, 800), (0, 0, 0, 0)), "PNG")
        page = pymupdf.open(stream=to_pdf(payload), filetype="pdf")[0]
        pixel = page.get_pixmap(dpi=20).pixel(10, 10)
        self.assertEqual(pixel[:3], (255, 255, 255))

    def test_huge_photo_is_reduced(self):
        payload = encoded(Image.new("RGB", (9000, 6000), "white"), "JPEG")
        page = pymupdf.open(stream=to_pdf(payload), filetype="pdf")[0]
        self.assertLessEqual(max(page.get_image_info()[0]["width"], page.get_image_info()[0]["height"]), 4000)


if __name__ == "__main__":
    unittest.main()
