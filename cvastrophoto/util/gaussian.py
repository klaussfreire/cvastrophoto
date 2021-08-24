from __future__ import absolute_import

import math
import os
import platform
from past.builtins import xrange

import numpy.fft
import scipy.ndimage
import skimage.util

from . import pfft


PADDING_MODE_MAP = {
    'nearest': 'edge',
    'constant': 'constant',
    'mirror': 'mirror',
    'wrap': 'wrap',
    'reflect': 'reflect',
}

PLATFORM_CPU_OVERHEAD = {
    'armv7l': 12.0,
    'x86_64': 4.0,
    'i386': 8.0,
    'i586': 6.0,
    'i686': 4.0,
    'unk': 8.0,
}

default_cpu_overhead = PLATFORM_CPU_OVERHEAD.get(platform.machine(), PLATFORM_CPU_OVERHEAD['unk'])

MAX_MEMORY_OVERHEAD = float(os.environ.get('MAX_MEMORY_OVERHEAD', 2.0))
FLOAT_CPU_OVERHEAD = float(os.environ.get('FLOAT_CPU_OVERHEAD', default_cpu_overhead))


def fast_gaussian(img, sigma, mode='reflect', **kw):
    pool = kw.pop('pool', None)
    pad_truncate = kw.pop('pad_truncate', kw.get('truncate', 4))
    fft_dtype = kw.pop('fft_dtype', None)
    if not kw and sigma > (FLOAT_CPU_OVERHEAD * math.log(max(img.shape)) / math.log(2)):
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
                    padded = numpy.pad(img, padding, skmode, **pad_kw)
                else:
                    padded = img

                padded_shape = padded.shape
                padded = pfft.prfft2(pool, padded, outdtype=fft_dtype)
                scipy.ndimage.fourier_gaussian(padded, sigma, padded_shape[-1], output=padded)
                padded = pfft.pirfft2(pool, padded, outdtype=img.dtype)

                if padding:
                    padded = padded[padding:-padding, padding:-padding].copy()
                return padded

    kw.setdefault('mode', mode)
    return scipy.ndimage.gaussian_filter(img, sigma, **kw)
