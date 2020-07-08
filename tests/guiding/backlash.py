from __future__ import absolute_import

import unittest
import logging

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
            backlash_aggressiveness = 0.5,

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

    def testRegression(self):
        c = self.controller
        bdec = self.bdec

        # Enable shrink, disabled by default
        bdec.shrink_rate = 0

        c.max_gear_state_ns = 6
        c.gear_state_ns = -6

        samples = [
            # DEC raw distance, guide distance, pulse ms, invoke-backlash
            (-0.6980564600434437,1.0829373513029397,188,False),
            (-1.035441797292981,1.598925155962586,278,False),
            (-1.3222938476387434,2.0224856140178447,351,False),
            (-1.1958896579114378,3.7000492287011464,643,True),
            (-1.3024061468602472,6.044412409538659,1051,True),
            (-1.4633669650725507,11.319041381877234,1968,True),
            (-1.8807004919952877,11.498548291827465,1999,True),
            (-2.1632441134221163,11.498548291827467,2000,True),
            (-1.389470845267985,11.498548291827467,373,True),
            (-0.3577474719436694,11.498548291827467,96,True),
            (3.4820506791614596,-10.773367630112721,-1873,True),
            (4.1874669397858115,-11.498548291827467,-2000,True),
            (4.942416946027351,-19.14439834500279,-2000,True),
            (3.8013304802604115,-11.822480042196483,-1641,True),
            (0.3913937562756066,-0.6349144373099309,-110,True),
            (0.036454364147541524,-0.07690471584090808,-13,True),
            (-0.07951227943577462,0.12064688670268088,20,False),
            (-0.02696568583482381,0.041410966484924316,7,False),
            (-0.21532810011937628,0.3326004761153988,57,False),
        ]
        dec_speed = 2.9792746114
        dec_agg = 0.8
        max_pulse = 2.0

        for rawd, guided, pulse_ms, do_backlash in samples:
            imm = -rawd / dec_speed * dec_agg
            if do_backlash:
                backlash_pulse = bdec.compute_pulse(imm, max_pulse)
            else:
                backlash_pulse = 0
                bdec.reset()
            total_pulse = imm + backlash_pulse
            logging.debug("imm=%r bl=%r tot=%r exp=%r", imm, backlash_pulse, total_pulse, pulse_ms)
            c.add_gear_state(total_pulse, 0)
            self.assertAlmostEqual(pulse_ms, int(min(max_pulse, max(-max_pulse, total_pulse)) * 1000), delta=10)
