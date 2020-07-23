import math

import numpy.fft
import scipy.ndimage


def fast_gaussian(img, sigma, **kw):
    if not kw and img.dtype.kind in ('d', 'f') and sigma > (math.log(max(img.shape)) / math.log(2)):
        return numpy.fft.irfft2(scipy.ndimage.fourier_gaussian(numpy.fft.rfft2(img), sigma, len(img)))
    else:
        return scipy.ndimage.gaussian_filter(img, sigma, **kw)
