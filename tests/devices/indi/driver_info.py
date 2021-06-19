from __future__ import absolute_import

import unittest
from cvastrophoto.devices.indi import driver_info


class DriverInfoTest(unittest.TestCase):

    def test_guide_flip(self):
        self.assertEqual(driver_info.get_guide_flip('iEQ', None), (True, True))
        self.assertEqual(driver_info.get_guide_flip('iOptronV3', None), (True, True))
        self.assertEqual(driver_info.get_guide_flip('somedriver', {'Name': 'CEM40'}), (True, True))
        self.assertEqual(driver_info.get_guide_flip('Some Telescope', None), driver_info.DEFAULT_GUIDE_FLIP)

    def test_guess_baud_rate(self):
        self.assertEqual(driver_info.guess_baud_rate('some telescope', '/dev/ttyUSB0', None), 115200)
        self.assertEqual(driver_info.guess_baud_rate('some telescope', '/dev/ttyS0', None), 9600)
