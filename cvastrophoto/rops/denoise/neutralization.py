import numpy
import scipy.ndimage

from ..base import BaseRop
from ..tracking.extraction import BackgroundRemovalRop

from cvastrophoto.util import gaussian

class BackgroundNeutralizationRop(BaseRop):

    scale = 128
    sigma = 4.0

    def __init__(self, raw, **kw):
        kw.setdefault('despeckle', True)
        kw.setdefault('despeckle_size', 3)
        kw.setdefault('pregauss_size', 3)
        kw.setdefault('aggressive', True)
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
        bgstd = numpy.std(bg)
        if bgmax > 0.1:
            bg *= 1.0 / bgmax
        bgmin = bg.min()
        if bgmin < 0.1:
            bgmin = numpy.percentile(bg, 25)
            bg = numpy.clip(bg, bgmin, None, out=bg)

        fdata = data.astype(numpy.float32, copy=False)
        ndata = data.astype(numpy.float32, copy=False) / bg
        ndata = numpy.clip(ndata, None, dmax, out=ndata)
        data[:] = (ndata * (self.sigma * bgstd) + data * fdata) / (self.sigma * bgstd + fdata)
        return data
