"""Narrow adjacent horizontal marks are not recoverable raster-table rows."""
from __future__ import annotations
import unittest
import warnings
import numpy as np
from PIL import Image
from document_reconstruction.recognition.regions import detect_regions


class TinyRasterRowTests(unittest.TestCase):
    def test_narrow_row_evidence_must_not_construct_cells_or_emit_numpy_warnings(self):
        pixels = np.full((420, 620), 255, dtype=np.uint8)
        for x in (85, 225, 365, 505):
            pixels[90:241, x:x+2] = 0
        for y in (90, 94, 98, 102, 240):
            pixels[y:y+2, 85:507] = 0
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            tables, _, _, _ = detect_regions(Image.fromarray(pixels), [], index=0, width=620, height=420)
        self.assertFalse(tables, 'Spacing of 4 pixels is not a credible editable table row')


if __name__ == '__main__':
    unittest.main()
