"""Do not accept scan DOCX when a strongly evidenced ruled table is flattened."""
from __future__ import annotations
import hashlib
import io

import docx
from pathlib import Path
import unittest
import numpy as np
from document_reconstruction.model import Box
from document_reconstruction.recognition.types import Word
from document_reconstruction.recognition.regions import _unresolved_ruled_raster_table_issue
from document_reconstruction.engine import ConversionEngine

PDF = Path(__file__).resolve().parents[1] / 'benchmarks/stage29/stress20/pdfs/12_scan_ky_tg_register.pdf'
SHA256 = '336e7882aeeb201cdcdc0a5a2d6c7e512d3bec044ac8d9d2fa70385e622c00fd'


class RasterRuleCompletenessTests(unittest.TestCase):
    @staticmethod
    def _simple_page(*, row_content: bool) -> np.ndarray:
        """An independent source-pixel control, not a frozen corpus copy."""
        gray = np.full((700, 600), 255, dtype=np.uint8)
        for y in (140, 170, 200, 230):
            gray[y:y+2, 55:545] = 145
        if row_content:
            for y in (152, 182, 212):
                for x in (65, 230, 395):
                    gray[y:y+4, x:x+36] = 80
        else:
            for y in (152, 182, 212):
                gray[y:y+4, 65:101] = 80
        return gray

    def test_unresolved_source_grid_requires_several_ink_bearing_columns(self):
        observations = [Word(f'w{i}', Box(20+i*4, 40, 23+i*4, 50)) for i in range(9)]
        self.assertIsNone(_unresolved_ruled_raster_table_issue(
            self._simple_page(row_content=False), observations, [], width=600, height=700,
        ))
        issue = _unresolved_ruled_raster_table_issue(
            self._simple_page(row_content=True), observations, [], width=600, height=700,
        )
        self.assertIsNotNone(issue)
        self.assertEqual(issue.code, 'unresolved_ruled_raster_table')
        self.assertEqual(issue.severity, 'error')

    def test_frozen_scanned_table_cannot_return_flattened_docx_as_success(self):
        payload = PDF.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), SHA256)
        result = ConversionEngine().convert(payload)
        if result.ok:
            # Issued (TASK.md section 0: text is restored): the tables must be real Word tables, not flattened text.
            self.assertGreater(len(docx.Document(io.BytesIO(result.docx)).tables), 0)
            return
        self.assertFalse(result.ok, 'Raster table(s) are visibly present but no real Word tables are emitted')
        self.assertIsNone(result.docx)
        self.assertEqual(result.error['code'], 'UNREADABLE_DOCUMENT')
        # When faint rules are recovered, the stricter downstream failure is
        # unreadable ink-bearing cells. All three outcomes reject the same
        # incomplete DOCX; this does not relax the source-completeness gate.
        self.assertIn(result.error['details']['reason'],
                      {'unresolved_ruled_raster_table', 'unresolved_borderless_raster_table',
                       'unresolved_table_cells', 'unreliable_text', 'unrecognized_text', 'text_unreadable'})
        if result.error['details']['reason'] == 'unresolved_table_cells':
            self.assertGreater(result.error['details']['weak_table_cell_count'], 0)
        self.assertEqual(result.metrics['pages_native'], 0)
        self.assertEqual(result.metrics['pages_ocr'], 2)

if __name__ == '__main__': unittest.main()
