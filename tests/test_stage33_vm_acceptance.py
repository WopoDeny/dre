"""Bounded, cross-platform end-to-end delivery runner regressions."""
from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.vm_acceptance import inspect_docx, run_case


class FakeProcess:
    def __init__(self, *, finish=False, make_docx=None):
        self.returncode = 0 if finish else None
        self.finish = finish
        self.make_docx = make_docx
        self.killed = False

    def communicate(self, timeout=None):
        if not self.finish and not self.killed:
            raise __import__('subprocess').TimeoutExpired('fake', timeout)
        if self.make_docx:
            self.make_docx()
            self.make_docx = None
        return '', ''

    def kill(self):
        self.killed = True
        self.returncode = -9


def sample_docx(destination):
    with zipfile.ZipFile(destination, 'w') as package:
        package.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Editable</w:t></w:r></w:p></w:body></w:document>')


class VMAcceptanceTests(unittest.TestCase):
    def test_real_docx_structure_is_not_equated_to_source_accuracy(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'example.docx'
            sample_docx(path)
            self.assertEqual(inspect_docx(path), {'editable_text_chars': 8, 'tables': 0})

    def test_hung_client_is_killed_and_no_partial_docx_survives(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'example.pdf'
            source.write_bytes(b'%PDF-1.7\n%%EOF')
            fake = FakeProcess()
            with patch('scripts.vm_acceptance.subprocess.Popen', return_value=fake):
                record = run_case(source, index=1, server='http://127.0.0.1:8080',
                                  client=source, output_dir=root, deadline_s=0.2)
            self.assertTrue(fake.killed)
            self.assertEqual(record['status'], 'delivery_deadline_exceeded')
            self.assertFalse((root / 'case-001.docx').exists())

    def test_successful_client_requires_existing_editable_docx(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'example.pdf'
            source.write_bytes(b'%PDF-1.7\n%%EOF')
            output = root / 'case-001.docx'
            fake = FakeProcess(finish=True, make_docx=lambda: sample_docx(output))
            with patch('scripts.vm_acceptance.subprocess.Popen', return_value=fake):
                record = run_case(source, index=1, server='http://127.0.0.1:8080',
                                  client=source, output_dir=root, deadline_s=1)
            self.assertEqual(record['status'], 'technical_docx_only')
            self.assertEqual(record['editable_text_chars'], 8)
            self.assertTrue(output.exists())


if __name__ == '__main__':
    unittest.main()
