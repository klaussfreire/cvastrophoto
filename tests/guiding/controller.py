from __future__ import absolute_import

import unittest

from cvastrophoto.guiding import controller


class ControllerTest(unittest.TestCase):

    def setUp(self):
        self.controller = controller.GuiderController(None, None)

    def testBacklashState(self):
        c = self.controller
        c.max_gear_state_ns = 1
        c.max_gear_state_we = 2

        c.gear_state_we = 2
        self.assertEqual(0, c.backlash_compensation_ra(0.2))
        self.assertEqual(-4, c.backlash_compensation_ra(-0.2))

        c.gear_state_we = -2
        self.assertEqual(4, c.backlash_compensation_ra(0.2))
        self.assertEqual(0, c.backlash_compensation_ra(-0.2))

        c.gear_state_ns = 1
        self.assertEqual(0, c.backlash_compensation_dec(0.2))
        self.assertEqual(-2, c.backlash_compensation_dec(-0.2))

        c.gear_state_ns = -1
        self.assertEqual(2, c.backlash_compensation_dec(0.2))
        self.assertEqual(0, c.backlash_compensation_dec(-0.2))

    def testSyncState(self):
        c = self.controller
        c.max_gear_state_ns = 1
        c.max_gear_state_we = 2

        c.sync_gear_state_ra(0.1)
        self.assertEqual(0, c.backlash_compensation_ra(0.2))
        self.assertEqual(-4, c.backlash_compensation_ra(-0.2))
        c.sync_gear_state_ra(-0.1)
        self.assertEqual(0, c.backlash_compensation_ra(-0.2))
        self.assertEqual(4, c.backlash_compensation_ra(0.2))

        c.sync_gear_state_dec(0.1)
        self.assertEqual(0, c.backlash_compensation_dec(0.2))
        self.assertEqual(-2, c.backlash_compensation_dec(-0.2))
        c.sync_gear_state_dec(-0.1)
        self.assertEqual(0, c.backlash_compensation_dec(-0.2))
        self.assertEqual(2, c.backlash_compensation_dec(0.2))
