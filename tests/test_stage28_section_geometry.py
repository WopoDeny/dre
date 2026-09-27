"""Protect source-backed Word section geometry from silent corruption."""

import io
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from document_reconstruction.model import Box, Document, Page, Paragraph
from document_reconstruction.quality import NS, W_NS, verify
from document_reconstruction.reconstruction.docx import reconstruct


BOX = Box(48, 50, 430, 72)


class SourceSectionGeometryTests(unittest.TestCase):
    @staticmethod
    def _corrupt(docx: bytes, section_index: int, element: str, attribute: str, value: str) -> bytes:
        output = io.BytesIO()
        with ZipFile(io.BytesIO(docx)) as source, ZipFile(output, 'w') as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == 'word/document.xml':
                    root = ET.fromstring(data)
                    section = root.findall('.//w:sectPr', NS)[section_index]
                    node = section.find(f'w:{element}', NS)
                    assert node is not None
                    node.set(f'{{{W_NS}}}{attribute}', value)
                    data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
                target.writestr(item, data)
        return output.getvalue()

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_source_page_size_cannot_shrink_without_detection(self):
        model = Document([Page(0, 595, 842, [Paragraph('Wide source page', BOX, 0)])])
        source = reconstruct(model)
        self.assertTrue(verify(source, model).passed)
        damaged = self._corrupt(source, 0, 'pgSz', 'w', '6000')
        self.assertIn('source_section_geometry_mismatch', verify(damaged, model).failures)

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_source_page_margins_cannot_be_replaced_without_detection(self):
        model = Document([Page(0, 595, 842, [Paragraph('Source margin', BOX, 0)])])
        source = reconstruct(model)
        self.assertTrue(verify(source, model).passed)
        damaged = self._corrupt(source, 0, 'pgMar', 'left', '3000')
        self.assertIn('source_section_geometry_mismatch', verify(damaged, model).failures)

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_two_page_portrait_landscape_section_geometry_is_valid(self):
        model = Document([
            Page(0, 595, 842, [Paragraph('Portrait page', BOX, 0)]),
            Page(1, 842, 595, [Paragraph('Landscape page', BOX, 1)]),
        ])
        intact = reconstruct(model)
        self.assertTrue(verify(intact, model).passed)
        damaged = self._corrupt(intact, 1, 'pgSz', 'orient', 'portrait')
        self.assertIn('source_section_geometry_mismatch', verify(damaged, model).failures)


if __name__ == '__main__':
    unittest.main()
