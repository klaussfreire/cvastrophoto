from __future__ import division

import multiprocessing.pool
import functools
import threading
import math
from past.builtins import xrange

import numpy.fft

try:
    from numpy.fft import fftpack

    _cook_nd_args = fftpack._cook_nd_args
except ImportError:
    from numpy.fft import _pocketfft

    _cook_nd_args = _pocketfft._cook_nd_args


_global_pool = None
_spawn_lock = threading.Lock()


def _default_pool():
    global _global_pool
    if _global_pool is not None:
        return _global_pool

    with _spawn_lock:
        if _global_pool is not None:
            return _global_pool
        _global_pool = multiprocessing.pool.ThreadPool()

    return _global_pool


def close_default_pool():
    global _global_pool
    if _global_pool is None:
        return

    with _spawn_lock:
        if _global_pool is None:
            return _global_pool
        global_pool = _global_pool
        _global_pool = None

    global_pool.terminate()
    global_pool.join()


def slice_array(a, axis, blocksize=None):
    if blocksize is None:
        ncpu = multiprocessing.cpu_count()
        blocksize = (a.shape[axis] + ncpu - 1) // ncpu

    if axis < 0:
        axis += a.ndim

    base_slice = tuple([slice(None) for _ in xrange(a.ndim)])
    slices = []
    for pos in xrange(0, a.shape[axis], blocksize):
        slices.append(a[base_slice[:axis] + (slice(pos, pos+blocksize),) + base_slice[axis+1:]])

    return slices


def prfft(pool, a, n=None, axis=-1, norm=None, out=None, outdtype=None, paxis=0):
    if pool is None:
        pool = _default_pool()
    if n is None:
        n = a.shape[axis]
    if outdtype is None:
        outdtype = numpy.complex128

    def do_slice(task):
        s, out = task
        out[:] = numpy.fft.rfft(s, n, axis, norm)

    if out is None:
        nshape = list(a.shape)
        nshape[axis] = (n-1)//2 if n % 2 else (n//2) + 1
        out = numpy.empty(nshape, dtype=outdtype)

    aslices = slice_array(a, paxis)
    outslices = slice_array(out, paxis, blocksize=aslices[0].shape[paxis])

    for _ in pool.imap_unordered(do_slice, zip(aslices, outslices)):
        pass

    return out


def pirfft(pool, a, n=None, axis=-1, norm=None, out=None, outdtype=None, paxis=0):
    if pool is None:
        pool = _default_pool()
    if n is None:
        n = (a.shape[axis] - 1) * 2
    if outdtype is None:
        outdtype = numpy.float64

    def do_slice(task):
        s, out = task
        out[:] = numpy.fft.irfft(s, n, axis, norm).real

    if out is None:
        nshape = list(a.shape)
        nshape[axis] = n
        out = numpy.empty(nshape, dtype=outdtype)

    aslices = slice_array(a, paxis)
    outslices = slice_array(out, paxis, blocksize=aslices[0].shape[paxis])

    for _ in pool.imap_unordered(do_slice, zip(aslices, outslices)):
        pass

    return out


def pfft(pool, a, n=None, axis=-1, norm=None, out=None, outdtype=None, paxis=0):
    if pool is None:
        pool = _default_pool()
    if n is None:
        n = a.shape[axis]
    if outdtype is None:
        outdtype = numpy.complex128

    def do_slice(task):
        s, out = task
        out[:] = numpy.fft.fft(s, n, axis, norm)

    if out is None:
        nshape = list(a.shape)
        nshape[axis] = n
        out = numpy.empty(nshape, dtype=outdtype)

    aslices = slice_array(a, paxis)
    outslices = slice_array(out, paxis, blocksize=aslices[0].shape[paxis])

    for _ in pool.imap_unordered(do_slice, zip(aslices, outslices)):
        pass

    return out


def pifft(pool, a, n=None, axis=-1, norm=None, out=None, outdtype=None, paxis=0):
    if pool is None:
        pool = _default_pool()
    if n is None:
        n = a.shape[axis]
    if outdtype is None:
        outdtype = numpy.complex128

    def do_slice(task):
        s, out = task
        out[:] = numpy.fft.ifft(s, n, axis, norm)

    if out is None:
        nshape = list(a.shape)
        nshape[axis] = n
        out = numpy.empty(nshape, dtype=outdtype)

    aslices = slice_array(a, paxis)
    outslices = slice_array(out, paxis, blocksize=aslices[0].shape[paxis])

    for _ in pool.imap_unordered(do_slice, zip(aslices, outslices)):
        pass

    return out


def prfft2(pool, a, **kw):
    if pool is None:
        pool = _default_pool()

    s, axes = _cook_nd_args(a, None, (-2, -1))
    a = prfft(pool, a, s[-1], axes[-1], paxis=axes[0], **kw)
    for ii in xrange(len(axes)-1):
        a = pfft(pool, a, s[ii], axes[ii], paxis=axes[-1], out=a, **kw)
    return a


def pirfft2(pool, a, **kw):
    if pool is None:
        pool = _default_pool()

    s, axes = _cook_nd_args(a, None, (-2, -1), invreal=1)
    a = a.copy()
    for ii in xrange(len(axes)-1):
        a = pifft(pool, a, s[ii], axes[ii], paxis=axes[-1], out=a, **kw)
    kw.setdefault('outdtype', numpy.float64)
    a = pirfft(pool, a, s[-1], axes[-1], paxis=axes[0], **kw)
    return a
