import math

import numpy.fft
import scipy.ndimage
import skimage.util


PADDING_MODE_MAP = {
    'nearest': 'edge',
    'constant': 'constant',
    'mirror': 'mirror',
    'wrap': 'wrap',
    'reflect': 'reflect',
}


def fast_gaussian(img, sigma, mode='reflect', **kw):
    if not kw and img.dtype.kind in ('d', 'f') and sigma > (4 * math.log(max(img.shape)) / math.log(2)):
        skmode = PADDING_MODE_MAP.get(mode)
        if skmode is not None:
            pad_kw = {}
            if mode == 'wrap':
                padding = 0
            else:
                padding = int(sigma * kw.get('truncate', 4))
                if mode == 'constant':
                    pad_kw = {'constant_values': (kw.pop('cval', 0.0),)}

            if padding < max(img.shape):
                # Avoid excessive memory overhead, the slow implementation at least doesn't use extra RAM
                if padding:
                    padded = skimage.util.pad(img, padding, skmode, **pad_kw)
                else:
                    padded = img

                padded = numpy.fft.irfft2(scipy.ndimage.fourier_gaussian(numpy.fft.rfft2(padded), sigma, padded.shape[1]))

                if padding:
                    padded = padded[padding:-padding, padding:-padding].copy()
                return padded

    kw.setdefault('mode', mode)
    return scipy.ndimage.gaussian_filter(img, sigma, **kw)
