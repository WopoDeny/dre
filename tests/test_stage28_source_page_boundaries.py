"""Word section boundaries must retain model-backed source-page ownership."""

import io
import unittest
import zipfile
from xml.etree import ElementTree as ET

from document_reconstruction.model import Box, Cell, Document, Page, Paragraph, Table
from document_reconstruction.quality import NS, verify
from document_reconstruction.reconstruction.docx import reconstruct


BOX = Box(48, 55, 430, 76)


class SourcePageBoundaryTests(unittest.TestCase):
    @staticmethod
    def _move_boundary_earlier(docx: bytes) -> bytes:
        """Keep the package and section count intact while moving a page break."""
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(docx)) as source, zipfile.ZipFile(output, 'w') as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == 'word/document.xml':
                    root = ET.fromstring(data)
                    body = root.find('w:body', NS)
                    boundary = next(p for p in body.findall('w:p', NS)
                                    if p.find('w:pPr/w:sectPr', NS) is not None)
                    body.remove(boundary)
                    body.insert(1, boundary)
                    data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
                target.writestr(item, data)
        return output.getvalue()

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_earlier_page_break_cannot_move_last_paragraph_to_next_source_page(self):
        model = Document([
            Page(0, 595, 842, [
                Paragraph('Page one first', BOX, 0),
                Paragraph('Page one second', BOX, 0),
            ]),
            Page(1, 595, 842, [Paragraph('Page two first', BOX, 1)]),
        ])
        intact = reconstruct(model)
        self.assertTrue(verify(intact, model).passed)
        shifted = self._move_boundary_earlier(intact)
        self.assertIn('source_section_anchor_mismatch', verify(shifted, model).failures)

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_earlier_page_break_cannot_move_table_to_next_source_page(self):
        model = Document([
            Page(0, 595, 842, [
                Paragraph('Page one title', BOX, 0),
                Table(Box(48, 100, 490, 190), 0, 1, 1, [Cell('Critical table data', 0, 0)]),
            ]),
            Page(1, 595, 842, [Paragraph('Page two text', BOX, 1)]),
        ])
        intact = reconstruct(model)
        self.assertTrue(verify(intact, model).passed)
        shifted = self._move_boundary_earlier(intact)
        self.assertIn('source_section_anchor_mismatch', verify(shifted, model).failures)

    def test_intentional_blank_source_page_remains_valid(self):
        model = Document([
            Page(0, 595, 842, [Paragraph('Before blank page', BOX, 0)]),
            Page(1, 595, 842, [], intentionally_blank=True),
            Page(2, 595, 842, [Paragraph('After blank page', BOX, 2)]),
        ])
        result = verify(reconstruct(model), model)
        self.assertTrue(result.passed, result.failures)


if __name__ == '__main__':
    unittest.main()
