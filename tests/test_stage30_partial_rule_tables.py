"""Native tables with a painted header and row rules remain editable."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
from unittest.mock import Mock
import unittest
import pymupdf
from docx import Document as WordDocument
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.model import Paragraph, Table

SOURCE = Path(__file__).resolve().parents[1] / 'benchmarks/stage29/stress20/pdfs/03_zh_cn_en_agreement.pdf'
SOURCE_SHA256 = '5c858951b6e02b0ebfa9db41ab1b31106bd942f8ca5384927000cf18c8239d13'  # Set from the frozen source below before executing.


class PartialRuleNativeTableTests(unittest.TestCase):
    def test_frozen_multilingual_header_rule_table_is_editable_on_both_pages(self):
        pdf = SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(pdf).hexdigest(), SOURCE_SHA256)
        ocr = Mock()
        result = ConversionEngine(ocr=ocr).convert(pdf)
        self.assertTrue(result.ok, result.error)
        ocr.recognize.assert_not_called()
        ocr.detect_script.assert_not_called()
        for page in result.model.pages:
            tables = [block for block in page.blocks if isinstance(block, Table) and not block.borderless]
            self.assertEqual(len(tables), 2, [(t.rows, t.columns) for t in tables])
            table = next(t for t in tables if t.columns == 4)
            self.assertEqual((table.rows, table.columns), (4, 4))
            self.assertEqual([next(c.text for c in table.cells if c.row == 0 and c.column == col)
                              for col in range(4)], ['序号', '名称 / Item', '批次', '验收数量'])
            self.assertEqual([next(c.text for c in table.cells if c.row == 2 and c.column == col)
                              for col in range(4)], ['02', '校准模块', 'CN-142', '142件']
                              if page.index == 0 else ['03', '配套密封圈', 'CN-009', '2,040件'])
            self.assertFalse(any('验收数量' in p.text for p in page.blocks if isinstance(p, Paragraph)))
        docx = WordDocument(io.BytesIO(result.docx))
        # Borderless layout tables (side-by-side blocks) are not data tables.
        data = [t for t in docx.tables if 'w:val="nil"' not in t._tbl.xml]
        self.assertEqual([(len(t.rows), len(t.columns)) for t in data], [(4, 4), (5, 3), (4, 4), (5, 3)])
        self.assertEqual(data[0].cell(3, 3).text, '2,040件')
        self.assertEqual(data[2].cell(3, 3).text, '124件')

    def test_new_native_table_with_filled_header_and_independent_row_rules(self):
        """An independent layout uses neither the frozen PDF nor its text values."""
        source = pymupdf.open()
        page = source.new_page(width=580, height=740)
        widths = [58, 122, 100]
        boundaries = [55, 113, 235, 335]
        values = [['Code', 'Description', 'Value'],
                  ['B-09', 'Monitor', '45'],
                  ['Q-13', 'Cable', '12'],
                  ['T-31', 'Adapter', '8']]
        for row, texts in enumerate(values):
            top = 180 + row * 24
            for column, text in enumerate(texts):
                left, right = boundaries[column:column+2]
                if row == 0:
                    page.draw_rect(pymupdf.Rect(left, top, right, top+24),
                                   color=None, fill=(0.88, 0.92, 0.97))
                page.draw_line(pymupdf.Point(left, top+24), pymupdf.Point(right, top+24),
                               color=(0.4,0.4,0.4), width=0.5)
                page.insert_text((left+4, top+16), text, fontsize=10)
        page.insert_text((55, 310), 'Free-standing text belongs outside the table.', fontsize=10)
        ocr = Mock()
        result = ConversionEngine(ocr=ocr).convert(source.tobytes())
        self.assertTrue(result.ok, result.error)
        ocr.recognize.assert_not_called()
        tables = [item for item in result.model.pages[0].blocks if isinstance(item, Table)]
        self.assertEqual(len(tables), 1)
        self.assertEqual((tables[0].rows, tables[0].columns), (4, 3))
        self.assertEqual(WordDocument(io.BytesIO(result.docx)).tables[0].cell(3, 1).text, 'Adapter')
        self.assertEqual(result.model.text().count('Adapter'), 1)
        self.assertIn('Free-standing text belongs outside the table.', result.model.text())

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_painted_parallel_headings_without_matching_row_rules_are_not_a_table(self):
        source = pymupdf.open()
        page = source.new_page(width=595, height=842)
        for col in range(4):
            x = 40 + col * 126
            page.draw_rect(pymupdf.Rect(x, 120, x+126, 141), color=None, fill=(0.85,0.85,0.85))
            page.insert_text((x+4, 134), f'Column {col+1}', fontsize=10)
            for row in range(3):
                page.insert_text((x+4, 170+row*25), f'Note{col+1}-{row+1}', fontsize=10)
        ocr = Mock()
        result = ConversionEngine(ocr=ocr).convert(source.tobytes())
        self.assertTrue(result.ok, result.error)
        ocr.recognize.assert_not_called()
        self.assertFalse(any(isinstance(item, Table) for item in result.model.pages[0].blocks))
        self.assertEqual(len(WordDocument(io.BytesIO(result.docx)).tables), 0)

if __name__ == '__main__':
    unittest.main()
