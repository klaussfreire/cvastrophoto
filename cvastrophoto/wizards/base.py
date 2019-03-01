from __future__ import absolute_import

import numpy
import math

class BaseWizard:

    def _get_raw_instance(self):
        raise NotImplementedError

    def get_image(self):
        img = self._get_raw_instance()

        accum = self.accum
        accum = accum.astype(numpy.float32) * (1.0 / accum.max())
        accum = numpy.clip(accum, 0, 1, out=accum)

        if img.postprocessing_params.no_auto_scale:
            # Must manually gamma-encode with sRGB gamma
            accum += 0.00313
            accum = numpy.power(accum, 0.45)
            accum -= math.pow(0.00313, 0.45) - 0.707 / 255.0
            accum = numpy.clip(accum, 0, 1, out=accum)

        img.set_raw_image(accum * 65535)

        return img
