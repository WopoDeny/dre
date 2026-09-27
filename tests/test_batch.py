"""Folder mode: issued documents to DOCX, refused ones to their own folder with the reason."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from document_reconstruction.batch import main

NATIVE = Path(__file__).resolve().parents[1] / "src/document_reconstruction/selfcheck/native_uz_sample.pdf"


class BatchTests(unittest.TestCase):
    def test_folder_is_converted_and_refusals_are_logged(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "in", Path(folder) / "out"
            (source / "sub").mkdir(parents=True)
            shutil.copy(NATIVE, source / "sub" / "letter.pdf")
            (source / "broken.pdf").write_bytes(b"not a pdf")
            (source / "notes.txt").write_text("ignored")
            self.assertEqual(main([str(source), str(output)]), 0)
            self.assertTrue((output / "sub__letter.docx").exists())
            refusal = json.loads((output / "_refused" / "broken.json").read_text())
            self.assertEqual(refusal["outcome"], "refused")
            self.assertTrue((output / "_refused" / "broken.pdf").exists())
            journal = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(record["outcome"] for record in journal), ["issued", "refused"])
            self.assertEqual(main([str(source), str(output)]), 0)  # nothing left to do
            self.assertEqual(len((output / "journal.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
