from __future__ import absolute_import

import unittest

from cvastrophoto.guiding import controller, backlash


class CalibrationMock:
    guide_exposure = 2.0


class ControllerTest(unittest.TestCase):

    def setUp(self):
        self.calibration = CalibrationMock()
        self.controller = c = controller.GuiderController(None, None)

        c.max_gear_state_ns = 10
        c.max_gear_state_we = 2

        self.bra = backlash.BacklashCompensation.for_controller_ra(self.calibration, c)
        self.bdec = backlash.BacklashCompensation.for_controller_dec(self.calibration, c)

        opts = dict(
            initial_backlash_pulse_ratio = 1.0,
            backlash_ratio_factor = 2.0,

            # Disable shrink by default to simplify tests that don't need it
            shrink_rate = 1.0,
        )

        # Set up known settings that work for this test, even if defaults change later
        for b in (self.bra, self.bdec):
            for k, v in opts.iteritems():
                setattr(b, k, v)
            b.reset()

    def testRampUp(self):
        c = self.controller
        bra = self.bra
        bdec = self.bdec

        # Peaks at 1.3 and then decreases due to aggressiveness
        c.gear_state_we = 2
        self.assertEqual(0, bra.compute_pulse(0.2, 2))
        self.assertAlmostEqual(-0.2, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.2)
        self.assertAlmostEqual(-0.4, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.4)
        self.assertAlmostEqual(-0.8, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.8)
        self.assertAlmostEqual(-1.3, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -1.3)
        self.assertAlmostEqual(-0.65, bra.compute_pulse(-0.2, 2))

        # Peaks at 2.0 and stays there due to max_pulse
        c.gear_state_ns = -10
        self.assertEqual(0, bdec.compute_pulse(-0.2, 2))
        self.assertAlmostEqual(0.2, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.2, 0)
        self.assertAlmostEqual(0.4, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.4, 0)
        self.assertAlmostEqual(0.8, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.8, 0)
        self.assertAlmostEqual(1.6, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(1.6, 0)
        self.assertAlmostEqual(2.0, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(2.0, 0)
        self.assertAlmostEqual(2.0, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(2.0, 0)

    def testEarlyStop(self):
        c = self.controller
        bra = self.bra
        bdec = self.bdec
        bra.shrink_rate = 1
        bdec.shrink_rate = 1

        # As soon as desired pulse drops, backlash compensation stops
        c.gear_state_we = 2
        self.assertEqual(0, bra.compute_pulse(0.2, 2))
        self.assertAlmostEqual(-0.2, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.2)
        self.assertAlmostEqual(-0.4, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.4)
        self.assertEqual(0, bra.compute_pulse(-0.1, 2))
        self.assertEqual(0, bra.compute_pulse(-0.2, 2))

        c.gear_state_ns = -10
        self.assertEqual(0, bdec.compute_pulse(-0.2, 2))
        self.assertAlmostEqual(0.2, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.2, 0)
        self.assertAlmostEqual(0.38, bdec.compute_pulse(0.19, 2)) ; c.add_gear_state(0.38, 0)
        self.assertAlmostEqual(0, bdec.compute_pulse(0.1, 2))
        self.assertAlmostEqual(0, bdec.compute_pulse(0.1, 2))

    def testFlipStop(self):
        c = self.controller
        bra = self.bra
        bdec = self.bdec

        # As soon as desired pulse changes direction, backlash compensation restarts rampup
        c.gear_state_we = 2
        self.assertEqual(0, bra.compute_pulse(0.2, 2))
        self.assertAlmostEqual(-0.2, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.2)
        self.assertAlmostEqual(-0.4, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.4)
        self.assertAlmostEqual(-0.8, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.8)
        self.assertAlmostEqual(0.2, bra.compute_pulse(0.2, 2)) ; c.add_gear_state(0, 0.2)
        self.assertAlmostEqual(0.4, bra.compute_pulse(0.2, 2)) ; c.add_gear_state(0, 0.4)

        # Peaks at 2.0 and stays there due to max_pulse
        c.gear_state_ns = -10
        self.assertEqual(0, bdec.compute_pulse(-0.2, 2))
        self.assertAlmostEqual(0.2, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.2, 0)
        self.assertAlmostEqual(0.4, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.4, 0)
        self.assertAlmostEqual(0.8, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.8, 0)
        self.assertAlmostEqual(-0.2, bdec.compute_pulse(-0.2, 2)) ; c.add_gear_state(-0.2, 0)
        self.assertAlmostEqual(-0.4, bdec.compute_pulse(-0.2, 2)) ; c.add_gear_state(-0.4, 0)

    def testFlipLearn(self):
        c = self.controller
        bra = self.bra
        bdec = self.bdec

        # Enable shrink, disabled by default
        bra.shrink_rate = 0
        bdec.shrink_rate = 0

        # As soon as desired pulse changes direction, backlash compensation restarts rampup
        # After such a flip, controller remembers last travel distance and has reset
        # max gear state accordingly, limiting any successive backlash compensation
        c.gear_state_we = 2
        self.assertEqual(0, bra.compute_pulse(0.2, 2))
        self.assertAlmostEqual(-0.2, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.2)
        self.assertAlmostEqual(-0.4, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.4)
        self.assertAlmostEqual(-0.8, bra.compute_pulse(-0.2, 2)) ; c.add_gear_state(0, -0.8)
        self.assertAlmostEqual(0.2, bra.compute_pulse(0.2, 2)) ; c.add_gear_state(0, 0.2)
        self.assertAlmostEqual(0.4, bra.compute_pulse(0.2, 2)) ; c.add_gear_state(0, 0.4)
        self.assertAlmostEqual(0.4, bra.compute_pulse(0.2, 2)) ; c.add_gear_state(0, 0.4)
        self.assertAlmostEqual(0.2, bra.compute_pulse(0.2, 2)) ; c.add_gear_state(0, 0.2)

        # Peaks at 2.0 and stays there due to max_pulse
        c.gear_state_ns = -10
        self.assertEqual(0, bdec.compute_pulse(-0.2, 2))
        self.assertAlmostEqual(0.2, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.2, 0)
        self.assertAlmostEqual(0.4, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.4, 0)
        self.assertAlmostEqual(0.8, bdec.compute_pulse(0.2, 2)) ; c.add_gear_state(0.8, 0)
        self.assertAlmostEqual(-0.2, bdec.compute_pulse(-0.2, 2)) ; c.add_gear_state(-0.2, 0)
        self.assertAlmostEqual(-0.4, bdec.compute_pulse(-0.2, 2)) ; c.add_gear_state(-0.4, 0)
        self.assertAlmostEqual(-0.4, bdec.compute_pulse(-0.2, 2)) ; c.add_gear_state(-0.4, 0)
        self.assertAlmostEqual(-0.2, bdec.compute_pulse(-0.2, 2)) ; c.add_gear_state(-0.2, 0)
