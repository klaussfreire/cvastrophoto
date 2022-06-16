# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
import math
import numpy
import random
import scipy.ndimage
import skimage.restoration
import skimage.morphology
import skimage.feature
import PIL.Image

from ..base import PerChannelRop
from ..tracking import extraction
from cvastrophoto.image import Image


class BaseDeconvolutionRop(PerChannelRop):

    method = 'wiener'
    normalize_mode = 'max'
    offset = 0.0005
    protect_low = False
    img_renormalize = False
    target = 'both'
    starless_method = 'localgradient'

    show_k = False

    METHODS = {
        'rl': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False),
        'rls': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False, iterations=100),
        'rlm': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False, iterations=25),
        'rlw': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False, iterations=10),
        'rlf': lambda data, k: skimage.restoration.richardson_lucy(data, k, clip=False, iterations=5),
        'wiener': lambda data, k: skimage.restoration.unsupervised_wiener(data, k)[0],
    }

    NORMALIZERS = {
        'max': lambda k: k.max(),
        'sum': lambda k: k.sum(),
    }

    def get_kernel(self, data, detected=None):
        raise NotImplementedError

    def correct(self, data, *p, **kw):
        orig_data = data

        if self.target != 'both':
            stars_rop = extraction.ExtractPureStarsRop(
                self.raw, copy=False, method=self.starless_method)
            stars = stars_rop.correct(data.copy())
            bg = data - numpy.minimum(stars, data)
            dmax = data.max()
            del stars_rop

            if self.target == 'stars':
                data = stars
            elif self.target == 'bg':
                data = bg
            else:
                raise ValueError("Unrecognized target %r" % (self.target,))

        data = super(BaseDeconvolutionRop, self).correct(data, *p, **kw)

        if self.target != 'both':
            data = numpy.add(bg, numpy.clip(stars, 0, None, out=stars), out=orig_data)
            data = numpy.clip(data, None, dmax, out=data)
            del stars, bg

        return data

    def process_channel(self, data, detected=None, channel=None):
        # Compute kernel
        k = self.get_kernel(data, detected=detected)

        if self.normalize_mode in self.NORMALIZERS:
            knorm = self.NORMALIZERS[self.normalize_mode](k)
            if knorm > 0:
                k /= knorm

        if self.show_k:
            PIL.Image.fromarray((numpy.sqrt(k / k.max()) * 255).astype(numpy.uint8)).show()

        # Apply deconvolution
        mxdata = data.max()
        if self.img_renormalize:
            mndata = data.min()
            avgdata = numpy.average(data)

        if mxdata > 1:
            data = data.astype(numpy.float32, copy=True)
            data /= mxdata
        else:
            data = data.astype(numpy.float32, copy=self.offset != 0)

        if self.offset != 0:
            data += self.offset
        rv = self.METHODS[self.method](data, k)
        if self.offset != 0:
            rv -= self.offset
            if self.protect_low:
                rvm = rv[k.shape[0]:-k.shape[0],k.shape[1]:-k.shape[1]].min()
                if rvm < 0:
                    rv -= rvm
        if mxdata > 1:
            rv *= mxdata
        if self.img_renormalize:
            navgdata = numpy.average(rv)
            rv *= avgdata / navgdata
            rv = numpy.clip(rv, mndata, mxdata, out=rv)

        return rv


class ManualDeconvolutionRop(BaseDeconvolutionRop):

    _sample_region = (0, 0, 0, 0)

    sample_file = ''

    black = 0.0
    gamma = 1.0

    @property
    def sample_region(self):
        return '-'.join(["%s" % r for r in self._sample_region])

    @sample_region.setter
    def sample_region(self, value):
        self._sample_region = tuple(map(int, value.split('-')))

    def get_kernel(self, data, detected=None):
        if self.sample_file:
            k = Image.open(self.sample_file).luma_image(same_shape=False)
        elif any(self._sample_region):
            cx, cy, w, h = self._sample_region

            k = data[cy-h:cy+h, cx-w:cx+w]
        else:
            raise ValueError("No sample provided")

        if self.black:
            k = numpy.clip(k - self.black, 0, None)

        if self.gamma != 1.0:
            k = numpy.power(k.astype(numpy.float32) / k.max(), self.gamma)

        return k



class DrizzleDeconvolutionRop(BaseDeconvolutionRop):

    scale = 2
    size = 0

    def get_kernel(self, data, detected=None):
        scale = self.scale
        size = scale * 3
        if not (size & 1):
            size += 1
            c = (size - 2) // 2
        else:
            c = size // 2

        # The drizzle kernel is a point source doubly-filtered by a uniform filter
        # The first uniform filter models the sensor's pixel itself capturing photons
        # within a rectangle the size of the drizzle factor, and the second filter
        # models the convolution due to multiple such rectangles falling at random
        # locations uniformly during stacking.
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[c,c] = 1
        k = scipy.ndimage.uniform_filter(k, scale)
        return scipy.ndimage.uniform_filter(k, scale)


class GaussianDeconvolutionRop(BaseDeconvolutionRop):

    sigma = 1.0
    gamma = 1.0
    size = 3.0

    def get_kernel(self, data, detected=None):
        sigma = self.sigma
        scale = int(max(1, self.sigma) * self.size)
        size = scale + (scale - 1) * 2
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[size//2, size//2] = 1
        k = scipy.ndimage.gaussian_filter(k, sigma)
        if self.gamma != 1.0:
            k = numpy.power(k, self.gamma)
        return k


class DoubleGaussianDeconvolutionRop(BaseDeconvolutionRop):

    sigma = 1.0
    sigma2 = 3.0
    gamma = 1.0
    w1 = 1.0
    w2 = 0.15
    offx = 0
    offy = 0
    size = 3.0

    def get_kernel(self, data, detected=None):
        scale = int(max(1, self.sigma, self.sigma2) * self.size)
        size = 1 + (scale - 1) * 2
        offx = self.offx
        offy = self.offy
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[size//2, size//2] = 1
        k1 = scipy.ndimage.gaussian_filter(k, self.sigma)
        k2 = scipy.ndimage.gaussian_filter(k, self.sigma2)
        k = k1 * self.w1
        k[offy:,offx:] += k2[:k.shape[0]-offy,:k.shape[1]-offx] * self.w2
        if self.gamma != 1.0:
            k = numpy.power(k, self.gamma)
        return k


class AiryDeconvolutionRop(BaseDeconvolutionRop):

    sigma = 0.5
    r1 = 1.25
    r0 = 0.5
    m0 = 0.2
    scale = 1.0
    size = 5.0

    def get_kernel(self, data, detected=None):
        # A gaussian is a good approximation of the airy power envelope,
        # but it's missing the rings. We get the rings by multiplying by
        # |cos(d)|. We apply the gaussian again to hide aliasing.
        sigma = self.sigma * self.scale
        r1 = self.r1 * self.scale
        r0 = self.r0 * self.scale
        scale = int(max(1, sigma, r1 + r0) * self.size)
        size = 1 + (scale - 1) * 2
        k = numpy.zeros((size, size), dtype=numpy.float32)
        k[size//2, size//2] = 1
        k = scipy.ndimage.gaussian_filter(k, sigma)
        x = numpy.arange(size, dtype=k.dtype)
        x, y = numpy.meshgrid(x, x)
        x -= size//2
        y -= size//2
        d = numpy.sqrt(x*x + y*y)
        k *= self.m0 + numpy.abs(numpy.cos(numpy.clip((d - r0) * (math.pi / 2 / r1), 0, None)))
        return k


class SampledDeconvolutionRop(BaseDeconvolutionRop):

    anisotropic = 0.0
    sigma = 0.5
    gamma = 1.0
    doff = 0.0
    dshrink = 0.0
    size = 64
    min_feature = 0
    sample_size = 64
    weight = 1.0
    envelope = 1.0
    max_samples = 1024
    threshold_rel = 0.01
    sample_threshold_rel = 10.0
    sx = 0
    sy = 0
    sample_region = None

    @property
    def sample_roi(self):
        return '-'.join(map(str, self.sample_region)) if self.sample_region else ''

    @sample_roi.setter
    def sample_roi(self, roi):
        self.sample_region = map(float, roi.split('-'))

    def get_kernel(self, data, detected=None):
        # A gaussian is a good approximation of the airy power envelope,
        # but it's missing the rings. We get the rings by multiplying by
        # |cos(d)|. We apply the gaussian again to hide aliasing.
        luma = data.astype(numpy.float32)

        ksize = 1 + int(self.sample_size + self.doff + self.dshrink) * 2
        size = int(self.sample_size + self.doff + self.dshrink)
        ssize = int(self.size + self.doff + self.dshrink)
        krange = numpy.arange(size)

        pool = self.raw.default_pool
        if pool is not None:
            map_ = pool.map
        else:
            map_ = map

        peaks = []
        footprints = []
        dirs = []

        if not self.sx and not self.sy:
            if self.min_feature:
                feat_luma = scipy.ndimage.minimum_filter(luma, self.min_feature, mode="nearest")
            else:
                feat_luma = luma

            for dy, dx in (
                    (0, 1), (0, -1), (1, 0), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)):
                footprint = numpy.zeros((ksize, ksize), numpy.bool8)
                footprint[ksize//2 + dy*krange, ksize//2 + dx*krange] = True
                footprints.append(footprint)
                dirs.append([dy, dx])

            def find_peaks(footprint):
                sroi = self.sample_region
                if sroi:
                    t,l,b,r = sroi
                    if all(0 <= x <= 1 for x in sroi) and any(0 < x < 1 for x in sroi):
                        t = int(t * luma.shape[0])
                        b = int(b * luma.shape[0])
                        l = int(l * luma.shape[1])
                        r = int(r * luma.shape[1])
                    sluma = feat_luma[t:b, l:r]
                else:
                    sluma = feat_luma
                coords = skimage.feature.peak_local_max(
                    sluma, footprint=footprint, threshold_rel=self.threshold_rel, num_peaks=self.max_samples*8,
                    exclude_border=size)
                if sroi:
                    coords[:,0] += t
                    coords[:,1] += l
                return coords

            for speaks in map_(find_peaks, footprints):
                peaks.append(list(speaks))
        else:
            peaks.append([[self.sy, self.sx]] * 4)

        for speaks in peaks:
            random.shuffle(speaks)

        if not self.anisotropic:
            npeaks = []
            for speaks in peaks:
                npeaks.extend(speaks)
            peaks = [npeaks] * len(dirs)

        ksize = 1 + int(self.size + self.doff + self.dshrink) * 2
        size = ssize
        krange = numpy.arange(size)

        lkerns = []

        for (dy, dx), speaks in zip(dirs, peaks):
            lkern = numpy.zeros(size * 2, numpy.float32)
            lkern[:size] = 1
            nsamples = 0

            for y, x in speaks:
                lsample = luma[y + dy*krange, x + dx*krange]

                if lsample.min() <= 1:
                    continue
                if len(lsample) != size:
                    continue
                if lsample[1:].ptp() <= lsample.min() * self.sample_threshold_rel:
                    continue
                if lsample.max() != lsample[0]:
                    continue
                if lsample[1:].max() >= lsample[0]:
                    continue

                lsample -= lsample.min()
                if not lsample[0]:
                    continue

                lsample /= lsample[0]
                newmin = numpy.minimum(lkern[:size], lsample)
                lkern[:size][newmin > 0] = newmin[newmin > 0]
                nsamples += 1

                if nsamples > self.max_samples:
                    break

            lkerns.append(lkern)

        del peaks, speaks

        ks = []

        x = numpy.arange(ksize, dtype=numpy.float32)
        x, y = numpy.meshgrid(x, x)
        x -= ksize//2
        y -= ksize//2

        d = numpy.sqrt(x*x + y*y)
        kdirx = x / numpy.clip(d, 0.5, None)
        kdiry = y / numpy.clip(d, 0.5, None)

        for (dy, dx), lkern in zip(dirs, lkerns):
            doff = numpy.clip(numpy.clip(d / math.sqrt(dy*dy + dx*dx) + self.doff, 0, None) - self.dshrink, 0, None)
            di = doff.astype(numpy.uint16)
            df = doff - di

            thresh = numpy.percentile(lkern[lkern != 0], 10)
            tail = lkern < thresh
            lkern[~tail] -= thresh
            lkern[tail] = 0

            lkern = numpy.power(lkern, self.gamma)

            k = lkern[di] * (1.0 - df) + lkern[di + 1] * df

            ks.append(k)

        if self.anisotropic:
            def dweight(i):
                rv = (kdirx * dirs[i][1] + kdiry * dirs[i][0]) / (dirs[i][0]**2 + dirs[i][1]**2) - 0.5
                rv = numpy.clip(rv, max(0, 1 - self.anisotropic) * rv.max(), None, out=rv)
                rv[ksize//2, ksize//2] = 1
                return rv
            ksw = dweight(0)
            k = ksw * ks[0]
            for i in xrange(1, len(ks)):
                dw = dweight(i)
                ksw += dw
                k += dw * ks[i]
            k[ksw != 0] /= ksw[ksw != 0]
        else:
            k = ks[0]
            for i in xrange(1, len(ks)):
                k = numpy.minimum(k, ks[i])

        if k.sum():
            k /= k.sum()
            k *= self.weight

        if self.sigma:
            k0 = numpy.zeros(k.shape, k.dtype)
            k0[ksize//2, ksize//2] = 1
            k0 = scipy.ndimage.gaussian_filter(k0, self.sigma)
            k += k0

        return k
