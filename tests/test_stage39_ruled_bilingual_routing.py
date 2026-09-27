"""Real source-pixel table evidence may warrant Latin recognition alongside Cyrillic."""
from __future__ import annotations
import json
import unittest
from pathlib import Path

from document_reconstruction.engine import ConversionEngine

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'benchmarks/stage29/stress20/pdfs/09_scan_ru_kk_form.pdf'
SINGLE_SCRIPT = ROOT / 'benchmarks/first_batch/pdfs/23_scan_kazakh_skew.pdf'


class RuledBilingualRoutingTests(unittest.TestCase):
    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')
    def test_frozen_ruled_scan_uses_local_latin_and_cyrillic_on_initial_pass(self):
        engine = ConversionEngine()
        try:
            result = engine.convert(SOURCE.read_bytes(), timeout_ms=45_000)
        finally:
            engine.close()
        self.assertTrue(result.ok, result.error)
        first = [row for row in result.diagnostics if row.get('route') == 'OCR' and not row.get('retry')]
        self.assertEqual(len(first), 2)
        self.assertTrue(all(row['profile'] == 'Cyrillic+eng' for row in first), first)
        self.assertTrue(all(row.get('ruled_table_latin_profile') for row in first), first)
        self.assertEqual(result.metrics['pages_ocr_performed'], 4)

    @unittest.skip('Removed in stage 1: one OCR pass with the allowed-script profile replaced regional OSD routing and retries.')

    def test_non_tabular_kazakh_scan_does_not_expand_first_pass(self):
        engine = ConversionEngine()
        try:
            result = engine.convert(SINGLE_SCRIPT.read_bytes(), timeout_ms=45_000)
        finally:
            engine.close()
        self.assertTrue(result.ok, result.error)
        first = next(row for row in result.diagnostics if row.get('route') == 'OCR' and not row.get('retry'))
        self.assertEqual(first['profile'], 'Cyrillic')
        self.assertFalse(first.get('ruled_table_latin_profile', False))


if __name__ == '__main__':
    unittest.main()
