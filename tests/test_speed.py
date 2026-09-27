"""Cost estimates follow the machine speed and the parallel pages."""

import dataclasses
import os
import unittest
from unittest import mock

from document_reconstruction.core import PageRoute
from document_reconstruction.preflight import CostPredictor
from document_reconstruction.speed import speed_factor


class Plan:
    def __init__(self, route, estimated_ms, searchable_scan=False):
        self.route, self.estimated_ms, self.searchable_scan = route, estimated_ms, searchable_scan


class SpeedTests(unittest.TestCase):
    def test_slower_machine_scales_page_estimates(self):
        base = CostPredictor()
        slow = dataclasses.replace(base, speed_factor=3.0)
        self.assertAlmostEqual(slow.page_ms(PageRoute.OCR, 8.0, True), 3 * base.page_ms(PageRoute.OCR, 8.0, True))

    def test_parallel_pages_shorten_the_estimate(self):
        plans = [Plan(PageRoute.OCR, 4000.0) for _ in range(4)]
        one, four = CostPredictor(), CostPredictor(parallel_pages=4)
        self.assertLess(four.remaining_ms(plans), one.remaining_ms(plans) / 3)

    def test_fixed_factor_from_environment(self):
        with mock.patch.dict(os.environ, {"DRE_SPEED_FACTOR": "7"}):
            self.assertEqual(speed_factor(None, "eng"), (7.0, None))


if __name__ == "__main__":
    unittest.main()
