"""Stage 33: an idle-socket timeout is not an end-to-end delivery limit."""
from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from examples.windows_client import convert_pdf


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class SlowResponse:
    def __init__(self, clock, content):
        self.clock = clock
        self.content = content
        self.headers = self

    def get_content_type(self):
        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        # A continuously active socket is allowed to take longer than its
        # inactivity timeout; completion must not be reported as in time.
        self.clock.now += 56.0
        return self.content


def small_docx():
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as output:
        output.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>')
    return data.getvalue()


class ClientDeliveryDeadlineTests(unittest.TestCase):
    def test_slow_but_active_response_cannot_create_late_success(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'test.pdf'
            source.write_bytes(b'%PDF-1.7\n%%EOF')
            target = source.with_suffix('.docx')
            with patch('examples.windows_client._monotonic', clock, create=True), patch(
                'examples.windows_client.urllib.request.urlopen',
                return_value=SlowResponse(clock, small_docx()),
            ):
                with self.assertRaises(TimeoutError):
                    convert_pdf(source, base_url='http://127.0.0.1:8080', api_key='test')
            self.assertFalse(target.exists())

    def test_prompt_response_still_creates_valid_docx(self):
        clock = Clock()
        class PromptResponse(SlowResponse):
            def read(self, *_):
                self.clock.now += 0.5
                return self.content
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'test.pdf'
            source.write_bytes(b'%PDF-1.7\n%%EOF')
            with patch('examples.windows_client._monotonic', clock, create=True), patch(
                'examples.windows_client.urllib.request.urlopen',
                return_value=PromptResponse(clock, small_docx()),
            ):
                result = convert_pdf(source, base_url='http://127.0.0.1:8080', api_key='test')
            self.assertEqual(result.read_bytes(), small_docx())


if __name__ == '__main__':
    unittest.main()
