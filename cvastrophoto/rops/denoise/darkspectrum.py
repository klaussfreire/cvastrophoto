# -*- coding: utf-8 -*-
import numpy.fft

def demargin(accum, img):
    # Fill out nonvisible margin to avoid FFT artifacts
    accum = accum.copy()
    raw_shape = img.rimg.raw_image.shape
    visible_shape = img.rimg.raw_image_visible.shape
    path, patw = img.rimg.raw_pattern.shape
    for y in xrange(path):
        for x in xrange(patw):
            naccum = accum[y::path,x::patw]
            xmargin = (raw_shape[1] - visible_shape[1]) / patw
            if xmargin:
                naccum[:,-xmargin:] = naccum[:,-xmargin-1:-2*xmargin-1:-1]
            ymargin = (raw_shape[0] - visible_shape[0]) / path
            if ymargin:
                naccum[-ymargin:,:] = naccum[-ymargin-1:-2*ymargin-1:-1,:]
    return accum

def denoise(laccum, lscale, naccum, nscale, img, amount=1,
        equalize_power=False, debias=False, debias_amount=1, dering=True, ring_freq=10):
    naccum = demargin(naccum, img)
    laccum = demargin(laccum, img)
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
