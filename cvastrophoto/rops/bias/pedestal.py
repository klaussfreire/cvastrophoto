from __future__ import absolute_import

from past.builtins import xrange
import numpy
import skimage.filters
import skimage.morphology

from ..base import BaseRop
from cvastrophoto.util import gaussian

class PedestalRop(BaseRop):

    value = 0.0

    def detect(self, data, **kw):
        return None

    def correct(self, data, bias=None, **kw):
        rv_data = data

        if not isinstance(data, list):
            data = [data]

        for sdata in data:
            if sdata.dtype.kind in 'df':
                sdata[:] += self.value
            else:
                limits = numpy.iinfo(data.dtype)
                if self.value > 0:
                    dmax = sdata.max()
                    if dmax + self.value > limits.max:
                        sdata = numpy.clip(sdata.astype(numpy.float32) + self.value, limits.min, limits.max, out=sdata)
                    else:
                        sdata[:] += int(self.value)
                elif self.value < 0:
                    if dmax + self.value < limits.min:
                        sdata = numpy.clip(sdata.astype(numpy.float32) + self.value, limits.min, limits.max, out=sdata)
                    else:
                        sdata[:] += int(self.value)

        return rv_data
