from __future__ import absolute_import

import unittest
import numpy

from cvastrophoto.rops import base


class BaseRopTest(unittest.TestCase):

    def test_effective_roi_rggb_no_margin(self):
        class MockRop(base.BaseRop):
            PROCESSING_MARGIN = 0
            _raw_pattern = numpy.array([[0, 1], [1, 2]])

        rop = MockRop(copy=False)

        exp_roi = in_roi = (10, 20, 30, 40)
        eff_roi = rop.effective_roi(in_roi)
        self.assertEqual(exp_roi, eff_roi)

        exp_roi = in_roi = (10, 20, 30, 40)
        eff_roi = rop.effective_roi(in_roi)
        self.assertEqual(exp_roi, eff_roi)

    def test_effective_roi_rggb_margin(self):
        class MockRop(base.BaseRop):
            PROCESSING_MARGIN = 3
            _raw_pattern = numpy.array([[0, 1], [1, 2]])

        rop = MockRop(copy=False)

        in_roi = (10, 20, 30, 40)
        exp_roi = (6, 16, 34, 44)
        eff_roi = rop.effective_roi(in_roi)
        self.assertEqual(exp_roi, eff_roi)

        in_roi = (10, 20, 30, 40)
        exp_roi = (6, 16, 34, 44)
        eff_roi = rop.effective_roi(in_roi)
        self.assertEqual(exp_roi, eff_roi)
