"""Regression cases for text/graphics ownership in the generated Word package."""
import io
import unittest
import zipfile
from xml.etree import ElementTree as ET

from PIL import Image
from document_reconstruction.model import Box, Cell, Document, Graphic, Page, Paragraph, Table
from document_reconstruction.quality import NS, W_NS, R_NS, verify
from document_reconstruction.reconstruction.docx import reconstruct


class Stage9QualityTests(unittest.TestCase):
    @staticmethod
    def rewrite(source: bytes, replacements: dict[str, bytes]) -> bytes:
        target = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(source)) as old, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as new:
            for name in old.namelist():
                new.writestr(name, replacements.get(name, old.read(name)))
        return target.getvalue()

    @staticmethod
    def table_model():
        return Document([Page(0, 595, 842, [
            Paragraph('BEFORE', Box(48, 40, 200, 55), 0),
            Table(Box(48, 80, 500, 180), 0, 2, 2, [
                Cell('A', 0, 0), Cell('B', 0, 1), Cell('C', 1, 0), Cell('D', 1, 1),
            ]),
            Paragraph('AFTER', Box(48, 220, 200, 240), 0),
        ])])

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_table_cell_text_cannot_be_moved_into_surrounding_paragraph(self):
        model = self.table_model()
        source = reconstruct(model)
        self.assertTrue(verify(source, model).passed)
        with zipfile.ZipFile(io.BytesIO(source)) as package:
            root = ET.fromstring(package.read('word/document.xml'))
        body = root.find('w:body', NS)
        table = body.find('w:tbl', NS)
        first_cell = table.find('w:tr/w:tc', NS)
        first_text = first_cell.find('.//w:t', NS)
        self.assertEqual(first_text.text, 'A')
        first_text.text = ''
        # Keep all text in the same global order, but move A outside its table cell.
        new_paragraph = ET.Element(f'{{{W_NS}}}p')
        new_run = ET.SubElement(new_paragraph, f'{{{W_NS}}}r')
        ET.SubElement(new_run, f'{{{W_NS}}}t').text = 'A'
        body.insert(list(body).index(table), new_paragraph)
        damaged = self.rewrite(source, {'word/document.xml': ET.tostring(root)})
        self.assertIn('table_cell_text_mismatch', verify(damaged, model).failures)

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_table_rows_cannot_collapse_preserving_global_text(self):
        model = self.table_model()
        source = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(source)) as package:
            root = ET.fromstring(package.read('word/document.xml'))
        table = root.find('.//w:tbl', NS)
        rows = table.findall('w:tr', NS)
        self.assertEqual(len(rows), 2)
        for cell in list(rows[1]):
            if cell.tag == f'{{{W_NS}}}tc':
                rows[1].remove(cell)
                rows[0].append(cell)
        table.remove(rows[1])
        damaged = self.rewrite(source, {'word/document.xml': ET.tostring(root)})
        self.assertIn('table_row_count_mismatch', verify(damaged, model).failures)

    def test_isolated_graphic_must_have_an_embedded_drawing(self):
        image = io.BytesIO()
        Image.new('RGB', (24, 14), 'red').save(image, 'PNG')
        model = Document([Page(0, 595, 842, [Graphic(Box(40, 100, 100, 135), 0, image.getvalue())])])
        source = reconstruct(model)
        self.assertTrue(verify(source, model).passed)
        with zipfile.ZipFile(io.BytesIO(source)) as package:
            root = ET.fromstring(package.read('word/document.xml'))
        drawing = root.find('.//w:drawing', NS)
        self.assertIsNotNone(drawing)
        parent_run = root.find('.//w:r', NS)
        parent_run.remove(drawing)
        damaged = self.rewrite(source, {'word/document.xml': ET.tostring(root)})
        self.assertIn('image_drawing_count_mismatch', verify(damaged, model).failures)

    def test_external_relationships_are_not_accepted_in_local_output(self):
        model = self.table_model()
        source = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(source)) as package:
            rels = ET.fromstring(package.read('word/_rels/document.xml.rels'))
        ns = 'http://schemas.openxmlformats.org/package/2006/relationships'
        ET.SubElement(rels, f'{{{ns}}}Relationship', {
            'Id': 'rIdExternalTest', 'Type': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image',
            'Target': 'https://example.invalid/picture.png', 'TargetMode': 'External',
        })
        damaged = self.rewrite(source, {'word/_rels/document.xml.rels': ET.tostring(rels)})
        self.assertIn('external_relationship:word/_rels/document.xml.rels', verify(damaged, model).failures)

    def test_embedded_graphic_bytes_must_match_the_model(self):
        red = io.BytesIO()
        blue = io.BytesIO()
        Image.new('RGBA', (24, 14), 'red').save(red, 'PNG')
        Image.new('RGBA', (24, 14), 'blue').save(blue, 'PNG')
        model = Document([Page(0, 595, 842, [Graphic(Box(40, 100, 100, 135), 0, red.getvalue())])])
        source = reconstruct(model)
        self.assertTrue(verify(source, model).passed)
        damaged = self.rewrite(source, {'word/media/image1.png': blue.getvalue()})
        self.assertIn('embedded_graphic_content_mismatch', verify(damaged, model).failures)

    def test_reused_image_media_is_valid_for_two_separate_graphic_placements(self):
        image = io.BytesIO()
        Image.new('RGB', (24, 14), 'red').save(image, 'PNG')
        graphic = image.getvalue()
        model = Document([Page(0, 595, 842, [
            Graphic(Box(40, 100, 100, 135), 0, graphic),
            Graphic(Box(40, 200, 100, 235), 0, graphic),
        ])])
        source = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(source)) as package:
            self.assertEqual(len([name for name in package.namelist() if name.startswith('word/media/')]), 1)
        report = verify(source, model)
        self.assertTrue(report.passed, report.failures)
        self.assertEqual(report.images, 2)

    def test_merged_table_cells_and_continuations_keep_text_once(self):
        table = Table(Box(48, 80, 500, 180), 0, 3, 3, [
            Cell('MERGED', 0, 0, row_span=2, column_span=2),
            Cell('TOP', 0, 2), Cell('SIDE', 1, 2),
            Cell('L', 2, 0), Cell('M', 2, 1), Cell('R', 2, 2),
        ])
        model = Document([Page(0, 595, 842, [table])])
        source = reconstruct(model)
        report = verify(source, model)
        self.assertTrue(report.passed, report.failures)


if __name__ == '__main__':
    unittest.main()
