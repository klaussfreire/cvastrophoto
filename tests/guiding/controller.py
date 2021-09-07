from __future__ import absolute_import

import time
import unittest
import threading

from cvastrophoto.guiding import controller

from tests.helpers.indimock import MockST4


class ControllerTest(unittest.TestCase):

    def setUp(self):
        self.controller = c = controller.GuiderController(None, MockST4('MockST4'))

        # Configure for speedy tests
        c.min_pulse = c.min_pulse_ra = c.min_pulse_dec = 0.025
        c.dec_target_pulse = c.ra_target_pulse = 0.05

    def _start_controller(self):
        self.controller.start()
        self.addCleanup(self.controller.stop)

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

    def testResistSwitch(self):
        self._start_controller()

        c = self.controller
        c.dec_switch_resistence = 0.25

        # Fwd-Back, resist back
        resist = c.dec_switch_resistence
        c.add_pulse(resist/2, 0)
        c.wait_pulse(resist*2)

        c.add_pulse(-resist/2, 0)
        c.wait_pulse(resist*2)

        self.assertEqual(c.pull_ignored(), (-resist/2, 0))
        self.assertEqual(c.st4.pull_pulses(), [(int(resist/2*1000), 0)])

        # Pull ignored resets accumulator
        self.assertEqual(c.pull_ignored(), (0, 0))

        # Back-Fwd, after back, back works, resist fwd
        c.add_pulse(-resist/2, 0)
        c.wait_pulse(resist*2)

        self.assertEqual(c.pull_ignored(), (0, 0))
        self.assertEqual(c.st4.pull_pulses(), [(-int(resist/2*1000), 0)])

        c.add_pulse(resist/2, 0)
        c.wait_pulse(resist*2)

        self.assertEqual(c.pull_ignored(), (resist/2, 0))

        # Fwd after ignored fwd works
        c.add_pulse(resist/2, 0)
        c.wait_pulse(resist*2)

        self.assertEqual(c.pull_ignored(), (0, 0))
        self.assertEqual(c.st4.pull_pulses(), [(int(resist/2*1000), 0)])

        # Back little by little, works when total movement reaches the limit
        c.add_pulse(-resist/4, 0)
        c.wait_pulse(resist*2)
        self.assertEqual(c.pull_ignored(), (-resist/4, 0))

        c.add_pulse(-resist/4, 0)
        c.wait_pulse(resist*2)
        self.assertEqual(c.pull_ignored(), (-resist/4, 0))

        c.add_pulse(-resist/4, 0)
        c.wait_pulse(resist*2)
        self.assertEqual(c.pull_ignored(), (-resist/4, 0))

        c.add_pulse(-resist/2, 0)
        c.wait_pulse(resist*2)
        self.assertEqual(c.pull_ignored(), (0, 0))
        self.assertEqual(c.st4.pull_pulses(), [(-int(resist/2*1000), 0)])

    def testResistSwitchOneDirection(self):
        self._start_controller()

        c = self.controller
        c.dec_switch_resistence = 0.25

        # Fwd-Back, resist back, fwd immediately works
        resist = c.dec_switch_resistence
        c.add_pulse(resist/2, 0)
        c.wait_pulse(resist*2)

        c.add_pulse(-resist/2, 0)
        c.wait_pulse(resist*2)

        self.assertEqual(c.pull_ignored(), (-resist/2, 0))

        c.add_pulse(resist/2, 0)
        c.wait_pulse(resist*2)

        self.assertEqual(c.pull_ignored(), (0, 0))
        self.assertEqual(c.st4.pull_pulses(), [(int(resist/2*1000), 0)]*2)

    def testResistDriftLimited(self):
        self._start_controller()

        c = self.controller
        c.dec_switch_resistence = 0.25
        c.backlash_measured = True
        c.eff_max_gear_state_ns = c.max_gear_state_ns = 4

        # Move Fwd, to induce back-resist
        resist = c.dec_switch_resistence
        c.add_pulse(resist/2, 0)
        c.wait_pulse(resist*2)

        # Set back-facing drift, should resist only briefly, not get stuck
        # Be careful to make sure the controller sees this drift setting for the
        # duration the test wants and not longer
        c.wake.set()
        time.sleep(0.1)
        drift_pulses = c.st4.pull_pulses()
        c.set_constant_drift(-1, 0)
        c.add_pulse(-0.01, 0)
        time.sleep(1)
        c.wake.set()
        time.sleep(0.1)

        # Pull ignored only reports direct pulses
        self.assertEqual(c.pull_ignored(), (-0.01, 0))

        drift_pulses = c.st4.pull_pulses()
        total_drift = sum(ns for ns, we in drift_pulses)
        self.assertLessEqual(total_drift, -500)
        self.assertGreaterEqual(total_drift, -1100)
        self.assertTrue(
            all(abs(ns) < 400 for ns, we in drift_pulses),
            "not all pulses below 400ms %r" % drift_pulses)

    def testSetPulse(self):
        self._start_controller()

        c = self.controller

        resist = c.dec_switch_resistence
        c.set_pulse(resist*2, 0)
        c.set_pulse(resist*3, 0)
        c.set_pulse(resist*2, 0)
        c.wait_pulse(resist*8)
        pulse = sum(ns for ns, we in c.st4.pull_pulses())
        self.assertAlmostEqual(pulse, int(resist*2*1000))

    def testAuthoritativeStop(self):
        self._start_controller()

        c = self.controller
        c.max_pulse = 0.5

        c.add_pulse(1000, 0)
        c.wait_pulse(0.1)
        c.set_pulse(0, 0)
        c.wait_pulse(1)
        self.assertEqual(c.st4.pull_pulses(), [(500, 0)])

    def testPulseIgnore(self):
        self._start_controller()

        c = self.controller

        # Test less-than-minimum pulse ignored and reported ignored
        min_pulse = c.min_pulse_dec
        resist = c.dec_switch_resistence
        c.set_pulse(min_pulse/2, 0)
        c.wait_pulse(min_pulse*4)
        pulse = sum(ns for ns, we in c.st4.pull_pulses())
        self.assertEqual(pulse, 0)
        self.assertEqual((min_pulse/2, 0), c.pull_ignored())

        # Test ignored pulse is not executed with later pulses
        c.set_pulse(resist*2, 0)
        c.wait_pulse(resist*8)
        pulse = sum(ns for ns, we in c.st4.pull_pulses())
        self.assertEqual(pulse, int(resist*2*1000))
        self.assertEqual((0, 0), c.pull_ignored())
