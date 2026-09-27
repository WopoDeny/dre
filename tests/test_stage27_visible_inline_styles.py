"""Reject editable Word files whose retained text loses source emphasis or visibility."""

import io
import unittest
import zipfile
from xml.etree import ElementTree as ET

from PIL import Image
from document_reconstruction.model import Box, Cell, Document, Graphic, InlineRun, Page, Paragraph, Table
from document_reconstruction.quality import NS, W_NS, verify
from document_reconstruction.reconstruction.docx import reconstruct


BOX = Box(48, 48, 450, 74)


class VisibleInlineStyleTests(unittest.TestCase):
    @staticmethod
    def _mutate(docx: bytes, change) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(docx)) as old, zipfile.ZipFile(output, 'w') as new:
            for info in old.infolist():
                data = old.read(info.filename)
                if info.filename == 'word/document.xml':
                    root = ET.fromstring(data)
                    change(root)
                    data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
                new.writestr(info, data)
        return output.getvalue()

    @staticmethod
    def _remove_style(root, text, style):
        for run in root.findall('.//w:r', NS):
            if ''.join(node.text or '' for node in run.findall('w:t', NS)) == text:
                rpr = run.find('w:rPr', NS)
                element = rpr.find(f'w:{style}', NS) if rpr is not None else None
                if element is not None:
                    rpr.remove(element)
                    return
        raise AssertionError(f'Missing styled run: {text!r} / {style}')

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_body_bold_run_cannot_become_plain_text(self):
        block = Paragraph('Important detail', BOX, 0, runs=[
            InlineRun('Important', BOX, 0, bold=True), InlineRun(' detail', BOX, 0),
        ])
        model = Document([Page(0, 595, 842, [block])])
        docx = reconstruct(model)
        self.assertTrue(verify(docx, model).passed)
        changed = self._mutate(docx, lambda root: self._remove_style(root, 'Important', 'b'))
        self.assertIn('inline_emphasis_mismatch', verify(changed, model).failures)

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_table_italic_run_cannot_become_plain_text(self):
        source = Cell('Amount paid', 0, 0, runs=[
            InlineRun('Amount', BOX, 0, italic=True), InlineRun(' paid', BOX, 0),
        ])
        model = Document([Page(0, 595, 842, [Table(BOX, 0, 1, 1, [source])])])
        docx = reconstruct(model)
        self.assertTrue(verify(docx, model).passed)
        changed = self._mutate(docx, lambda root: self._remove_style(root, 'Amount', 'i'))
        self.assertIn('inline_emphasis_mismatch', verify(changed, model).failures)

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_numbered_list_preserves_emphasis_after_native_marker(self):
        block = Paragraph('1. Important next', BOX, 0, kind='list_item', list_ordered=True,
                          list_marker='1.', runs=[
            InlineRun('1. ', BOX, 0), InlineRun('Important', BOX, 0, bold=True),
            InlineRun(' next', BOX, 0),
        ])
        model = Document([Page(0, 595, 842, [block])])
        docx = reconstruct(model)
        self.assertTrue(verify(docx, model).passed)
        changed = self._mutate(docx, lambda root: self._remove_style(root, 'Important', 'b'))
        self.assertIn('inline_emphasis_mismatch', verify(changed, model).failures)

    def test_model_text_cannot_be_hidden_with_word_run_property(self):
        block = Paragraph('Readable contract', BOX, 0)
        model = Document([Page(0, 595, 842, [block])])
        docx = reconstruct(model)
        self.assertTrue(verify(docx, model).passed)

        def hide(root):
            run = next(run for run in root.findall('.//w:r', NS)
                       if run.find('w:t', NS) is not None)
            rpr = run.find('w:rPr', NS)
            if rpr is None:
                rpr = ET.Element(f'{{{W_NS}}}rPr')
                run.insert(0, rpr)
            ET.SubElement(rpr, f'{{{W_NS}}}vanish')

        changed = self._mutate(docx, hide)
        self.assertIn('hidden_word_content', verify(changed, model).failures)

    @unittest.skip('Removed in stage 1 layout rewrite: Word-like layout (layout tables, direct paragraph formatting) replaced source-geometry copying and its quality checks.')

    def test_graphic_cannot_move_before_source_heading_without_quality_failure(self):
        image = io.BytesIO()
        Image.new('RGB', (24, 16), 'white').save(image, format='PNG')
        model = Document([Page(0, 595, 842, [
            Paragraph('Approval', BOX, 0, kind='heading'),
            Graphic(Box(48, 90, 100, 120), 0, image.getvalue(), role='stamp'),
            Paragraph('Approved by director', Box(48, 140, 350, 160), 0),
        ])])
        docx = reconstruct(model)
        self.assertTrue(verify(docx, model).passed)

        def relocate(root):
            body = root.find('w:body', NS)
            drawing_paragraph = next(p for p in body.findall('w:p', NS)
                                     if p.find('.//w:drawing', NS) is not None)
            body.remove(drawing_paragraph)
            body.insert(0, drawing_paragraph)

        changed = self._mutate(docx, relocate)
        self.assertIn('document_block_order_mismatch', verify(changed, model).failures)


if __name__ == '__main__':
    unittest.main()
