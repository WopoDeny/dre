"""Raster table separators must preserve real cells without inventing split cells."""
from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from document_reconstruction.engine import ConversionEngine
from document_reconstruction.model import Table
from document_reconstruction.recognition.regions import detect_regions

ROOT = Path(__file__).resolve().parents[1]


def thin_grid(*, merged_row: int | None = None) -> Image.Image:
    image = Image.new('RGB', (320, 240), 'white')
    draw = ImageDraw.Draw(image)
    ys = (20, 65, 110, 155)
    for y in ys:
        draw.line((20, y, 260, y), fill='black', width=2)
    for x in (20, 180, 260):
        draw.line((x, ys[0], x, ys[-1]), fill='black', width=1)
    for row, (top, bottom) in enumerate(zip(ys, ys[1:])):
        if row != merged_row:
            draw.line((100, top, 100, bottom), fill='black', width=1)
    return image


class ThinRasterGridTests(unittest.TestCase):
    def test_one_pixel_vertical_lines_keep_all_real_cells(self):
        tables, _, _, _ = detect_regions(thin_grid(), [], index=0, width=320, height=240)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual((table.rows, table.columns), (3, 3))
        self.assertEqual([(c.row, c.column, c.column_span) for c in table.cells],
                         [(r, col, 1) for r in range(3) for col in range(3)])

    def test_missing_separator_preserves_true_single_row_merge(self):
        tables, _, _, _ = detect_regions(thin_grid(merged_row=1), [], index=0, width=320, height=240)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual((table.rows, table.columns), (3, 3))
        self.assertEqual([(c.row, c.column, c.column_span) for c in table.cells],
                         [(0, 0, 1), (0, 1, 1), (0, 2, 1),
                          (1, 0, 2), (1, 2, 1),
                          (2, 0, 1), (2, 1, 1), (2, 2, 1)])

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_frozen_scan_09_keeps_four_and_three_distinct_columns(self):
        source = (ROOT / 'benchmarks/stage29/stress20/pdfs/09_scan_ru_kk_form.pdf').read_bytes()
        engine = ConversionEngine()
        try:
            result = engine.convert(source, timeout_ms=45_000)
        finally:
            engine.close()
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.model.pages), 2)
        for page in result.model.pages:
            tables = [block for block in page.blocks if isinstance(block, Table)]
            self.assertEqual([(t.rows, t.columns) for t in tables], [(4, 4), (5, 3)])
            for table in tables:
                self.assertEqual(len(table.cells), table.rows * table.columns)
                self.assertTrue(all(cell.column_span == 1 for cell in table.cells))


if __name__ == '__main__':
    unittest.main()
