"""Faint straight raster rules must not silently flatten editable table cells."""
from __future__ import annotations
import unittest
import numpy as np
from PIL import Image, ImageDraw
from document_reconstruction.model import Box
from document_reconstruction.recognition.regions import detect_regions
from document_reconstruction.recognition.types import Word


class FaintRasterGridTests(unittest.TestCase):
    @staticmethod
    def _page(*, ink: int, verticals: bool = True, gray_background: int = 255):
        img = Image.new('L', (650, 470), gray_background)
        draw = ImageDraw.Draw(img)
        rows = (140, 190, 240, 290)
        columns = (55, 225, 395, 565)
        for y in rows:
            draw.line((columns[0], y, columns[-1], y), fill=ink, width=2)
        if verticals:
            for x in columns:
                draw.line((x, rows[0], x, rows[-1]), fill=ink, width=2)
        words = []
        for row in range(3):
            for col in range(3):
                x, y = columns[col] + 20, rows[row] + 19
                text = f'R{row}C{col}'
                # Independent OCR evidence: the raster detector is not allowed to invent cell text.
                words.append(Word(text, Box(x, y, x + 65, y + 13), confidence=0.98,
                                  line_id=(row + 1, 1, col + 1)))
                draw.text((x, y), text, fill=65)
        return img, words

    def test_light_printed_grid_produces_real_cell_owned_table(self):
        page, words = self._page(ink=190)
        tables, _, diagnostics, _ = detect_regions(page, words, index=0, width=650, height=470)
        self.assertEqual(len(tables), 1, diagnostics)
        table = tables[0]
        self.assertEqual((table.rows, table.columns), (3, 3))
        self.assertEqual([cell.text for cell in table.cells], [word.text for word in words])
        self.assertEqual(len({cell.text for cell in table.cells}), 9)

    def test_horizontal_printed_rules_without_vertical_grid_are_not_table(self):
        page, words = self._page(ink=190, verticals=False)
        tables, _, _, _ = detect_regions(page, words, index=0, width=650, height=470)
        self.assertEqual(tables, [])

    def test_tinted_photo_background_does_not_promote_weak_marks_to_table(self):
        # A background as dark as a photographed book must not receive the
        # brighter scan threshold; low-contrast ruled-looking artifacts are
        # insufficient to claim a nine-cell editable table.
        page, words = self._page(ink=185, gray_background=200)
        tables, _, _, _ = detect_regions(page, words, index=0, width=650, height=470)
        self.assertEqual(tables, [])

    def test_uniform_gray_background_is_not_a_grid(self):
        page, words = self._page(ink=190, verticals=False, gray_background=220)
        tables, _, _, _ = detect_regions(page, words, index=0, width=650, height=470)
        self.assertEqual(tables, [])


if __name__ == '__main__':
    unittest.main()
