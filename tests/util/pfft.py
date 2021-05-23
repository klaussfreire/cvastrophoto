from __future__ import absolute_import

import functools
import unittest
import operator
import numpy.random
import numpy.fft

from six.moves import reduce

from cvastrophoto.util import pfft


def variant(f, *p, **kw):
    @functools.wraps(f)
    def variant(self):
        return f(self, *p, **kw)
    return variant


class ParallelFFTTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        # Avoid leaving a threadpool around
        pfft.close_default_pool()

    def _testRFFT2(self, dt, shape):
        # The implementation is identical to numpy's,
        # so it should really give the exact same results,
        # floating point error notwithstanding
        if dt().dtype.kind in 'ui':
            size = reduce(operator.mul, shape, 1)
            limits = numpy.iinfo(dt)
            randinput = numpy.random.randint(limits.min, limits.max, size, dt).reshape(shape)
        else:
            randinput = numpy.random.randn(*shape).astype(dt)
        ref = numpy.fft.rfft2(randinput)
        rv = pfft.prfft2(None, randinput)
        self.assertEqual(ref.shape, rv.shape)
        self.assertEqual(ref.dtype, rv.dtype)
        self.assertTrue((ref == rv).all(), "FFT output mismatch")

    def _testIRFFT2(self, shape):
        # The implementation is identical to numpy's,
        # so it should really give the exact same results,
        # floating point error notwithstanding
        randinput = numpy.random.randn(*shape).astype(numpy.float32)
        randinput = numpy.fft.rfft2(randinput)
        ref = numpy.fft.irfft2(randinput)
        rv = pfft.pirfft2(None, randinput)
        self.assertEqual(ref.shape, rv.shape)
        self.assertEqual(ref.dtype, rv.dtype)
        self.assertTrue((ref == rv).all(), "FFT output mismatch")

    def _testIRFFT2OutDT(self, outdt, shape):
        # The implementation is identical to numpy's,
        # so it should really give the exact same results,
        # floating point error notwithstanding
        if outdt().dtype.kind in 'ui':
            size = reduce(operator.mul, shape, 1)
            limits = numpy.iinfo(outdt)
            randinput = numpy.random.randint(limits.min, limits.max, size, outdt).reshape(shape)
        else:
            randinput = numpy.random.randn(*shape).astype(outdt)
        randinput = numpy.fft.rfft2(randinput)
        ref = numpy.fft.irfft2(randinput).astype(outdt)
        rv = pfft.pirfft2(None, randinput, outdtype=outdt)
        self.assertEqual(ref.shape, rv.shape)
        self.assertEqual(ref.dtype, rv.dtype)
        self.assertTrue((ref == rv).all(), "FFT output mismatch")

    pow2shape = (32, 32)
    npow2shape = (50, 50)

    testRFFT2Pow2UI8 = variant(_testRFFT2, numpy.uint8, pow2shape)
    testRFFT2Pow2UI16 = variant(_testRFFT2, numpy.uint16, pow2shape)
    testRFFT2Pow2UI32 = variant(_testRFFT2, numpy.uint32, pow2shape)
    testRFFT2Pow2I8 = variant(_testRFFT2, numpy.int8, pow2shape)
    testRFFT2Pow2I16 = variant(_testRFFT2, numpy.int16, pow2shape)
    testRFFT2Pow2I32 = variant(_testRFFT2, numpy.int32, pow2shape)
    testRFFT2Pow2F32 = variant(_testRFFT2, numpy.float32, pow2shape)
    testRFFT2Pow2F64 = variant(_testRFFT2, numpy.float64, pow2shape)

    testRFFT2NPow2UI8 = variant(_testRFFT2, numpy.uint8, npow2shape)
    testRFFT2NPow2UI16 = variant(_testRFFT2, numpy.uint16, npow2shape)
    testRFFT2NPow2UI32 = variant(_testRFFT2, numpy.uint32, npow2shape)
    testRFFT2NPow2I8 = variant(_testRFFT2, numpy.int8, npow2shape)
    testRFFT2NPow2I16 = variant(_testRFFT2, numpy.int16, npow2shape)
    testRFFT2NPow2I32 = variant(_testRFFT2, numpy.int32, npow2shape)
    testRFFT2NPow2F32 = variant(_testRFFT2, numpy.float32, npow2shape)
    testRFFT2NPow2F64 = variant(_testRFFT2, numpy.float64, npow2shape)

    testIRFFT2Pow2 = variant(_testIRFFT2, pow2shape)
    testIRFFT2NPow2 = variant(_testIRFFT2, npow2shape)

    testIRFFT2Pow2UI8 = variant(_testIRFFT2OutDT, numpy.uint8, pow2shape)
    testIRFFT2Pow2UI16 = variant(_testIRFFT2OutDT, numpy.uint16, pow2shape)
    testIRFFT2Pow2UI32 = variant(_testIRFFT2OutDT, numpy.uint32, pow2shape)
    testIRFFT2Pow2I8 = variant(_testIRFFT2OutDT, numpy.int8, pow2shape)
    testIRFFT2Pow2I16 = variant(_testIRFFT2OutDT, numpy.int16, pow2shape)
    testIRFFT2Pow2I32 = variant(_testIRFFT2OutDT, numpy.int32, pow2shape)
    testIRFFT2Pow2F32 = variant(_testIRFFT2OutDT, numpy.float32, pow2shape)
    testIRFFT2Pow2F64 = variant(_testIRFFT2OutDT, numpy.float64, pow2shape)

    testIRFFT2NPow2UI8 = variant(_testIRFFT2OutDT, numpy.uint8, npow2shape)
    testIRFFT2NPow2UI16 = variant(_testIRFFT2OutDT, numpy.uint16, npow2shape)
    testIRFFT2NPow2UI32 = variant(_testIRFFT2OutDT, numpy.uint32, npow2shape)
    testIRFFT2NPow2I8 = variant(_testIRFFT2OutDT, numpy.int8, npow2shape)
    testIRFFT2NPow2I16 = variant(_testIRFFT2OutDT, numpy.int16, npow2shape)
    testIRFFT2NPow2I32 = variant(_testIRFFT2OutDT, numpy.int32, npow2shape)
    testIRFFT2NPow2F32 = variant(_testIRFFT2OutDT, numpy.float32, npow2shape)
    testIRFFT2NPow2F64 = variant(_testIRFFT2OutDT, numpy.float64, npow2shape)
