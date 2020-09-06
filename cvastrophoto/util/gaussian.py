import math
import os

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

MAX_MEMORY_OVERHEAD = float(os.environ.get('MAX_MEMORY_OVERHEAD', 2.0))


def fast_gaussian(img, sigma, mode='reflect', **kw):
    pad_truncate = kw.pop('pad_truncate', kw.get('truncate', 4))
    if not kw and img.dtype.kind in ('d', 'f') and sigma > (4 * math.log(max(img.shape)) / math.log(2)):
        skmode = PADDING_MODE_MAP.get(mode)
        if skmode is not None:
            pad_kw = {}
            if mode == 'wrap':
                padding = 0
            else:
                padding = int(sigma * pad_truncate)
                if mode == 'constant':
                    pad_kw = {'constant_values': (kw.pop('cval', 0.0),)}

            padded_size = (2*padding+img.shape[0]) * (2*padding+img.shape[1])
            if padding < max(img.shape) and padded_size <= img.size * MAX_MEMORY_OVERHEAD and (img.shape[-1] % 2) == 0:
                # Avoid excessive memory overhead, the slow implementation at least doesn't use extra RAM
                if padding:
                    padded = skimage.util.pad(img, padding, skmode, **pad_kw)
                else:
                    padded = img

                padded = numpy.fft.irfft2(scipy.ndimage.fourier_gaussian(
                    numpy.fft.rfft2(padded), sigma, padded.shape[-1]))

                if padding:
                    padded = padded[padding:-padding, padding:-padding].copy()
                return padded

    kw.setdefault('mode', mode)
    return scipy.ndimage.gaussian_filter(img, sigma, **kw)
