from past.builtins import xrange

import numpy
import scipy.ndimage

from ..base import BaseRop
from ..tracking.extraction import BackgroundRemovalRop

from cvastrophoto.util import gaussian

class BackgroundNeutralizationRop(BaseRop):

    scale = 128
    sigma = 4.0
    mode = 'smooth'

    def __init__(self, raw, **kw):
        kw.setdefault('despeckle', True)
        kw.setdefault('despeckle_size', 3)
        kw.setdefault('pregauss_size', 3)
        kw.setdefault('aggressive', True)
        kw.setdefault('close_factor', 0)
        self.scale = scale = int(kw.setdefault('scale', self.scale))
        remove_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(BackgroundRemovalRop, k)}
        remove_stars_kw.setdefault('minfilter_size', scale)
        remove_stars_kw.setdefault('gauss_size', scale)
        self._bg_extract = BackgroundRemovalRop(raw, **remove_stars_kw)
        super(BackgroundNeutralizationRop, self).__init__(raw, **kw)

    def correct(self, data, *p, **kw):
        dmax = data.max()
        bg = self._bg_extract.detect(data.copy())
        bg = bg.astype(numpy.float32, copy=False)

        def flatten(bg):
            # Closing pass to avoid artifacts due to any dark spots in the background
            bg = scipy.ndimage.minimum_filter(bg, self.scale, mode='nearest')
            bg = scipy.ndimage.maximum_filter(bg, self.scale, mode='nearest')
            bg = gaussian.fast_gaussian(bg, self.scale)
            return bg
        self.parallel_channel_task(bg, bg, flatten)

        bgmax = bg.max()
        bgavg = numpy.average(bg)
        bgstd = numpy.std(bg)

        if self.mode == 'mult':
            if bgmax > 0.1:
                bg *= 1.0 / bgmax
            bgmin = bg.min()
            if bgmin < 0.1:
                bgmin = numpy.percentile(bg, 25)
                bg = numpy.clip(bg, bgmin, None, out=bg)

            fdata = data.astype(numpy.float32, copy=False)
            ndata = data.astype(numpy.float32, copy=False) / bg
            ndata = numpy.clip(ndata, None, dmax, out=ndata)
            data[:] = (ndata * (self.sigma * (bgavg + bgstd)) + data * fdata) / (self.sigma * (bgavg + bgstd) + fdata)
        elif self.mode == 'add':
            bgthr = bgavg + bgstd * self.sigma
            bgthr = numpy.average(bg[bg < bgthr])
            mask = bg <= bgthr
            data[mask] = numpy.clip(((data + (bgthr - bg))[mask]), None, dmax).astype(data.dtype, copy=False)
        elif self.mode in ('uniform', 'smooth'):
            def threshold(bg):
                bgavg = numpy.average(bg)
                bgstd = numpy.std(bg)
                bgthr = bgavg + bgstd * self.sigma
                bgthr = numpy.average(bg[bg < bgthr])
                return bgthr
            if len(bg.shape) == 3:
                bgthr = numpy.empty((1,1,bg.shape[2]), dtype=bg.dtype)
            else:
                bgthr = numpy.empty(self._raw_pattern.shape, dtype=bg.dtype)
            self.parallel_channel_task(bg, bgthr, threshold)
            bgthr = bgthr.max() - bgthr

            if self.mode == 'uniform':
                def add(data, bgthr):
                    data += bgthr
            elif self.mode == 'smooth':
                def add(data, bgthr):
                    if bgthr > 0:
                        data[:] = data + (bgthr * bgthr * self.sigma) / numpy.clip(data, bgthr * self.sigma, None)
            else:
                raise NotImplementedError(self.mode)

            if len(bg.shape) == 3:
                for c in xrange(bg.shape[2]):
                    add(data[:,:,c], bgthr[0,0,c])
            else:
                raw_pattern = self._raw_pattern
                path, patw = raw_pattern.shape
                for y in xrange(path):
                    for x in xrange(patw):
                        add(data[y::path, x::patw], bgthr[y, x])
        else:
            raise NotImplementedError("Unimplemented neutralization mode %s" % (self.mode,))
        return data
