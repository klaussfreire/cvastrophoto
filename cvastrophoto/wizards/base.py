from __future__ import absolute_import

import numpy

class BaseWizard:

    def _get_raw_instance(self):
        raise NotImplementedError

    def get_image(self):
        img = self._get_raw_instance()

        accum = self.accum
        accum = accum.astype(numpy.float32) * (1.0 / accum.max())
        accum = numpy.clip(accum, 0, 1, out=accum)
        img.set_raw_image(accum * 65535)

        return img
