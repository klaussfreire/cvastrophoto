from __future__ import absolute_import

import math
import os
import platform
from past.builtins import xrange

import numpy.fft
import scipy.ndimage
import skimage.util

from . import pfft

from cvastrophoto.util import vectorize


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

with_cuda_ndimage = os.environ.get('CUDA_NDIMAGE', 'yes') != 'no'


if vectorize.with_cuda:
    from numba import cuda
    import scipy.ndimage
    import scipy.ndimage._nd_image
    import scipy.ndimage._ni_support

    @cuda.jit(fastmath=True)
    def _correlate1d_reflect(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                x2 = x + wpos - origin
                if x2 < 0:
                    x2 = - x2 - 1
                if x2 >= w:
                    x2 = 2 * w - x2 - 1
                if x2 >= 0 and x2 < w:
                    accum += input[x2, y] * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_reflect_t(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                y2 = y + wpos - origin
                if y2 < 0:
                    y2 = - y2 - 1
                if y2 >= h:
                    y2 = 2 * h - y2 - 1
                if y2 >= 0 and y2 < h:
                    accum += input[x, y2] * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_mirror(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                x2 = x + wpos - origin
                if x2 < 0:
                    x2 = - x2
                if x2 >= w:
                    x2 = 2 * w - x2 - 2
                if x2 >= 0 and x2 < w:
                    accum += input[x2, y] * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_mirror_t(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                y2 = y + wpos - origin
                if y2 < 0:
                    y2 = - y2
                if y2 >= h:
                    y2 = 2 * h - y2 - 2
                if y2 >= 0 and y2 < h:
                    accum += input[x, y2] * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_nearest(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                x2 = max(0, min(w-1, x + wpos - origin))
                accum += input[x2, y] * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_nearest_t(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                y2 = max(0, min(h-1, y + wpos - origin))
                accum += input[x, y2] * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_wrap(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                x2 = (x + wpos - origin) % w
                accum += input[x2, y] * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_wrap_t(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                y2 = (y + wpos - origin) % h
                accum += input[x, y2] * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_const(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                x2 = x + wpos - origin
                if x2 < 0 or x2 >= w:
                    inputval = cval
                else:
                    inputval = input[x2, y]
                accum += inputval * weights[wpos]
            output[x, y] = accum

    @cuda.jit(fastmath=True)
    def _correlate1d_const_t(input, weights, output, origin, cval):
        x, y = cuda.grid(2)
        w = input.shape[0]
        h = input.shape[1]
        wsize = weights.shape[0]
        if x < w and y < h:
            accum = 0.0
            for wpos in range(wsize):
                y2 = y + wpos - origin
                if y2 < 0 or y2 >= h:
                    inputval = cval
                else:
                    inputval = input[x, y2]
                accum += inputval * weights[wpos]
            output[x, y] = accum

    _correlate1d = {
        ('reflect', 0): _correlate1d_reflect,
        ('reflect', 1): _correlate1d_reflect_t,
        ('nearest', 0): _correlate1d_nearest,
        ('nearest', 1): _correlate1d_nearest_t,
        ('constant', 0): _correlate1d_const,
        ('constant', 1): _correlate1d_const_t,
        ('mirror', 0): _correlate1d_mirror,
        ('mirror', 1): _correlate1d_mirror_t,
        ('wrap', 0): _correlate1d_wrap,
        ('wrap', 1): _correlate1d_wrap_t,
    }
    if with_cuda_ndimage:
        for mode, axis in list(_correlate1d.keys()):
            _correlate1d[scipy.ndimage._ni_support._extend_mode_to_code(mode), axis] = _correlate1d[mode, axis]

    orig_correlate1d = scipy.ndimage._nd_image.correlate1d

    def correlate1d(input, weights, axis, output, mode, cval, origin):
        if len(input.shape) != 2 or axis not in (0, 1) or input.size < 1024:
            # This CUDA version can only handle 2d relatively large inputs
            return orig_correlate1d(input, weights, axis, output, mode, cval, origin)

        if output is None:
            output = numpy.empty(input.shape, dtype=input.dtype)

        assert input.shape == output.shape

        orig_output = output

        if not input.flags.contiguous:
            input = input.copy()
        if not output.flags.contiguous:
            output = output.copy()
        if not weights.flags.contiguous:
            weights = weights.copy()

        if axis == 1:
            threadsperblock = (1, min(128, weights.size))
        else:
            threadsperblock = (min(128, weights.size), 1)
        blockspergrid_x = math.ceil(input.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(input.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        origin += len(weights) // 2

        _correlate1d[mode, axis][blockspergrid, threadsperblock](input, weights, output, origin, cval)

        if orig_output is not output:
            orig_output[:] = output

        return output

    def separable2d(input, weights, output, mode, cval, origin):
        if len(input.shape) != 2:
            raise ValueError("Only 2D arrays supported")

        assert output is None or input.shape == output.shape

        orig_output = output

        if not input.flags.contiguous:
            input = input.copy()
        if not weights.flags.contiguous:
            weights = weights.copy()

        threadsperblock = (min(128, weights.size), 1)
        threadsperblockt = (1, min(128, weights.size))
        blockspergrid_x = math.ceil(input.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(input.shape[1] / threadsperblock[1])
        blockspergridt_x = math.ceil(input.shape[0] / threadsperblockt[0])
        blockspergridt_y = math.ceil(input.shape[1] / threadsperblockt[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        blockspergridt = (blockspergridt_x, blockspergridt_y)

        origin += len(weights) // 2

        input = cuda.to_device(input)
        weights = cuda.to_device(weights)
        output = cuda.device_array_like(input)

        _correlate1d[mode, 0][blockspergrid, threadsperblock](input, weights, output, origin, cval)
        _correlate1d[mode, 1][blockspergridt, threadsperblockt](output, weights, input, origin, cval)
        return input.copy_to_host(orig_output)

    def cuda_gaussian2d(input, sigma, output=None, mode="reflect", cval=0.0, truncate=4.0):
        sd = float(sigma)
        lw = int(truncate * sd + 0.5)
        weights = scipy.ndimage.filters._gaussian_kernel1d(sigma, 0, lw)[::-1].copy()
        return separable2d(input, weights, output, mode, cval, 0)

    if with_cuda_ndimage:
        scipy.ndimage._nd_image.correlate1d = correlate1d


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
            max_padding = max(img.shape)
            if padding < max_padding and padded_size <= img.size * MAX_MEMORY_OVERHEAD and (img.shape[-1] % 2) == 0:
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

    if vectorize.with_cuda and isinstance(sigma, (int, float)):
        ksize = sigma * pad_truncate * 2
        if img.size > 1024 and ksize > 25:
            # Large kernels benefit from CUDA if available
            return cuda_gaussian2d(img, sigma, **kw)

    return scipy.ndimage.gaussian_filter(img, sigma, **kw)
