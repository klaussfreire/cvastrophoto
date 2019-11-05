# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy.fft

def denoise(laccum, lscale, naccum, nscale, img, amount=1,
        equalize_power=False, debias=False, debias_amount=1, dering=True, ring_freq=10):
    # Fill out nonvisible margin to avoid FFT artifacts
    naccum = img.demargin(naccum.copy())
    laccum = img.demargin(laccum.copy())
    path, patw = img.rimg.raw_pattern.shape
    for y in xrange(path):
        for x in xrange(patw):
            Fn = numpy.fft.rfft2(naccum[y::path,x::patw]) * (float(amount) * lscale / nscale)
            F = numpy.fft.rfft2(laccum[y::path,x::patw])
            if equalize_power and numpy.absolute(Fn[0,0]) > 1:
                Fn *= amount * numpy.absolute(F[0,0]) / numpy.absolute(Fn[0,0])
            absFn = numpy.absolute(Fn)
            if debias:
                absFn[0,0] -= debias_amount * (absFn[:5,:5].sum() - absFn[0,0])
            if dering:
                Fn[0,ring_freq:] = Fn[ring_freq:-ring_freq,0] = absFn[0,ring_freq:] = absFn[ring_freq:-ring_freq,0] = 0
            absF = numpy.absolute(F)
            F[absF <= absFn] = 0
            mask = absF > absFn
            F[mask] *= ((absF[mask] - absFn[mask]) / absF[mask])
            laccum[y::path,x::patw] = numpy.clip(numpy.fft.irfft2(F), 0, None)
    return laccum

def lowpass_kernel_F(shape, thr, steep=2):
    Y, X = numpy.meshgrid(
        numpy.arange(shape[1]) * 2.0 / shape[1],
        numpy.arange(shape[0]) * 2.0 / shape[0])
    Y = numpy.abs(Y - 1) - 1
    X = numpy.abs(X - 1) - 1
    return numpy.clip((thr*thr - (Y*Y + X*X)) / (thr*thr) * steep, 0, 1)
