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

    @property
    def ns_drift(self):
        return self._ns_drift + self.pa_error_ns

    @ns_drift.setter
    def ns_drift(self, value):
        self._ns_drift = value

    @property
    def we_drift(self):
        return self._we_drift + self.pe_amplitude * (
            math.sin(time.time() * math.pi / (2 * self.pe_period))
        )

    @we_drift.setter
    def we_drift(self, value):
        self._we_drift = value
