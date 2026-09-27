"""Acceptance exit codes must describe quality, not just successful conversion."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage7_acceptance.py"
spec = importlib.util.spec_from_file_location("stage7_acceptance_under_test", SCRIPT)
assert spec and spec.loader
stage7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage7)


class Stage7LanguageGateTests(unittest.TestCase):
    def _run_case(self, recognized_text: str) -> tuple[int, dict[str, object]]:
        name = "Uzbek Latin"
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            source = root / "language"
            source.mkdir()
            (source / "uzbek_latin.pdf").write_bytes(b"synthetic-pdf-input-placeholder")
            output = root / "case.json"
            result = SimpleNamespace(
                ok=True,
                model=SimpleNamespace(text=lambda: recognized_text, pages=[]),
                docx=None,
                diagnostics=[{"ocr_performed": True, "profile": "Latin", "script": "Latin", "confidence": 0.95}],
                metrics={"retry_pages": 0},
                quality=None,
                error=None,
            )

            class FakeEngine:
                def convert(self, source_bytes: bytes):
                    return result

                def close(self) -> None:
                    pass

            with patch.object(stage7, "generate", return_value=None), patch.object(stage7, "GENERATED", root), patch.object(stage7, "ConversionEngine", FakeEngine):
                status = stage7.language_case(name, output)
            return status, json.loads(output.read_text(encoding="utf-8"))

    def test_successful_conversion_below_quality_threshold_has_failing_exit_status(self):
        status, record = self._run_case("O'zbekiston g'alaba: shahar, o'quvchi, 2026-yil. " * 3)
        self.assertEqual(record["status"], "measured_limitation")
        self.assertFalse(record["accepted"])
        self.assertEqual(status, 1)

    def test_accepted_language_has_success_exit_status(self):
        text = stage7.LANGUAGE_CASES["Uzbek Latin"][1]
        status, record = self._run_case(" ".join([text] * 3))
        self.assertEqual(record["status"], "accepted")
        self.assertTrue(record["accepted"])
        self.assertEqual(status, 0)

    def test_complete_checked_in_corpus_is_not_regenerated_during_acceptance(self):
        source = stage7.GENERATED / "language" / "english.pdf"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with patch.object(stage7, "generate", side_effect=AssertionError("Unnecessary corpus regeneration")):
            stage7.ensure_corpus()
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()