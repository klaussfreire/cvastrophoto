from __future__ import absolute_import

import unittest

from tests.helpers import indimock

from cvastrophoto.cmd.commands import guide
import cvastrophoto.constants.exposure


class TestCaptureSequence(unittest.TestCase):

    def setUp(self):
        self.ccd = indimock.MockCCD('MockCCD', 32, 32, 'L')
        self.capture_seq = guide.CaptureSequence(None, self.ccd)

        # Mock out delays
        self.capture_seq.sleep = lambda time : None
        self.capture_seq.wait_capture_ready = lambda *p, **kw: None

    def test_find_exposure(self):
        exp = self.capture_seq.find_exposure(3200, cvastrophoto.constants.exposure.FLAT_EXPOSURE_VALUES, 0)
        self.assertEqual(exp, 30)

        exp = self.capture_seq.find_exposure(3000, cvastrophoto.constants.exposure.FLAT_EXPOSURE_VALUES, 0)
        self.assertEqual(exp, 30)

        exp = self.capture_seq.find_exposure(2900, cvastrophoto.constants.exposure.FLAT_EXPOSURE_VALUES, 0)
        self.assertEqual(exp, 20)

        exp = self.capture_seq.find_exposure(30000, cvastrophoto.constants.exposure.FLAT_EXPOSURE_VALUES, 0)
        self.assertEqual(exp, 300)

    def test_auto_flats(self):
        exps = self.capture_seq.auto_flats(5, 200)
        self.assertEqual(exps, {2})
        self.assertGreaterEqual(self.ccd.t_shot_count, 5)
        self.assertLessEqual(self.ccd.t_shot_count, 8)
        self.assertEqual(self.ccd.t_exposures.count(2), 6)
