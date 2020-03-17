# -*- coding: utf-8 -*-
from __future__ import absolute_import

import time
import math
import random

from .. import controller

class PEPASimGuiderController(controller.GuiderController):

    pe_amplitude = 0.02
    pe_period = 480.0

    pa_error_ns = 0.005
    pa_error_we = 0.04

    we_speed = 1.0
    ns_speed = 1.0

    random_prob = 0.05
    random_mag = 0.3

    @property
    def eff_drift(self):
        return (
            max(-1, min(1, self.ns_drift * self.ns_speed + self.pa_error_ns + (
                (random.random() * 2 - 1) * self.random_mag
                if random.random() < self.random_prob
                else 0
            ))),
            max(-1, min(1,
                self.we_drift * self.we_speed + self.pa_error_we + self.pe_amplitude * (
                    math.sin(time.time() * 2 * math.pi / self.pe_period)
                ) + (
                    (random.random() * 2 - 1) * self.random_mag
                    if random.random() < self.random_prob
                    else 0
                )
            )),
        )

    def add_pulse(self, ns_s, we_s):
        return super(PEPASimGuiderController, self).add_pulse(
            ns_s * self.ns_speed, we_s * self.we_speed)

    def add_spread_pulse(self, ns_s, we_s, exec_s):
        return super(PEPASimGuiderController, self).add_spread_pulse(
            ns_s * self.ns_speed, we_s * self.we_speed, exec_s)
