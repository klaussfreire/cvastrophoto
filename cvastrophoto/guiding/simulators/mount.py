# -*- coding: utf-8 -*-
from __future__ import absolute_import

import time
import math

from .. import controller

class PEPASimGuiderController(controller.GuiderController):

    pe_amplitude = 0.02
    pe_period = 480.0

    pa_error_ns = 0.005
    pa_error_we = 0.04

    we_speed = 1.0
    ns_speed = 1.0

    @property
    def eff_drift(self):
        return (
            max(-1, min(1, self.ns_drift * self.ns_speed + self.pa_error_ns)),
            max(-1, min(1,
                self.we_drift * self.we_speed + self.pa_error_we + self.pe_amplitude * (
                    math.sin(time.time() * 2 * math.pi / self.pe_period)
                )
            )),
        )
