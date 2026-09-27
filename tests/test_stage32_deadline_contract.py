"""New 45-second conversion and 55-second delivery budget contract."""
from __future__ import annotations
import inspect
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from document_reconstruction.core.deadline import DeadlineBudget, INTERNAL_STOP_MS, EXTERNAL_SLA_MS
from document_reconstruction.engine import ConversionEngine, ConversionResult
from document_reconstruction.service.app import Application
from examples.windows_client import convert_pdf


class DeadlineContractTests(unittest.TestCase):
    def test_conversion_cap_is_45_seconds_even_when_caller_requests_longer(self):
        self.assertEqual(INTERNAL_STOP_MS, 45_000)
        self.assertEqual(DeadlineBudget(55_000).timeout_ms, 45_000)
        self.assertEqual(DeadlineBudget(43_000).timeout_ms, 43_000)

    def test_engine_default_and_external_delivery_budget(self):
        self.assertEqual(inspect.signature(ConversionEngine.convert).parameters['timeout_ms'].default, 45_000)
        self.assertEqual(EXTERNAL_SLA_MS, 55_000)

    def test_api_starts_request_with_conversion_budget(self):
        class FakePool:
            def __init__(self):
                self.seen = None
            def convert(self, payload, *, budget, request_id):
                self.seen = (budget.timeout_ms, payload, request_id)
                return ConversionResult(request_id, None, {}, {'code': 'UNREADABLE_DOCUMENT', 'message': 'Unreadable source', 'request_id': request_id})
        pool = FakePool()
        payload = b'%PDF-1.7\n%%EOF'
        environ = {'PATH_INFO': '/convert', 'REQUEST_METHOD': 'POST', 'HTTP_AUTHORIZATION': 'Bearer test-key',
                   'CONTENT_TYPE': 'application/pdf', 'CONTENT_LENGTH': str(len(payload)), 'wsgi.input': io.BytesIO(payload)}
        statuses = []
        result = Application(pool, api_key='test-key')(environ, lambda status, headers: statuses.append(status))
        self.assertEqual(pool.seen[0], 45_000)
        self.assertEqual(pool.seen[1], payload)
        self.assertEqual(statuses, ['422 Unprocessable Content'])
        self.assertTrue(result)

    def test_windows_client_socket_timeout_matches_delivery_ceiling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.pdf'
            path.write_bytes(b'%PDF-1.7\n%%EOF')
            with patch('urllib.request.urlopen', side_effect=RuntimeError('simulated offline')) as mocked:
                with self.assertRaisesRegex(RuntimeError, 'simulated offline'):
                    convert_pdf(path, base_url='http://127.0.0.1:1', api_key='test')
            self.assertEqual(mocked.call_args.kwargs['timeout'], 55)

    def test_windows_client_requires_no_engine_dependencies(self):
        client = Path(__file__).resolve().parents[1] / 'examples' / 'windows_client.py'
        completed = subprocess.run([sys.executable, '-I', '-S', str(client), '--help'],
                                   text=True, capture_output=True, timeout=8)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--server', completed.stdout)


if __name__ == '__main__':
    unittest.main()
