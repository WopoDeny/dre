"""The same input gives the same DOCX whether pages are recognized one by one or at once."""

import dataclasses
import hashlib
import unittest
from pathlib import Path

import pymupdf

from document_reconstruction.config import Settings
from document_reconstruction.engine import ConversionEngine
from document_reconstruction.resources import cpu_count, page_threads

HIRES = Path(__file__).resolve().parents[1] / "handoff/samples/hires"
PAGES = ("scan_mild/ru_letter_ministry.pdf", "scan_medium/ky_letter_mayor.pdf", "scan_medium/kz_order_school.pdf")


class ParallelPagesTests(unittest.TestCase):
    def test_one_thread_and_three_threads_give_identical_docx(self):
        if not all((HIRES / name).exists() for name in PAGES):
            self.skipTest("hires samples are not available")
        with pymupdf.open() as document:
            for name in PAGES:
                with pymupdf.open(HIRES / name) as source:
                    document.insert_pdf(source)
            payload = document.tobytes()
        digests = []
        for threads in (1, 3):
            engine = ConversionEngine(settings=dataclasses.replace(Settings.from_env(), page_threads=threads))
            try:
                result = engine.convert(payload)
            finally:
                engine.close()
            self.assertTrue(result.ok, result.error)
            digests.append(hashlib.sha256(result.docx).hexdigest())
        self.assertEqual(digests[0], digests[1])

    def test_thread_count_follows_the_setting_and_the_machine(self):
        self.assertEqual(page_threads("1"), 1)
        self.assertEqual(page_threads("9"), 4)
        self.assertLessEqual(page_threads("auto"), max(1, min(4, cpu_count())))


if __name__ == "__main__":
    unittest.main()
