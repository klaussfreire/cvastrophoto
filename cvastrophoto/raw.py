# -*- coding: utf-8 -*-
import os.path
import rawpy
import numpy
import scipy.stats
import PIL.Image
import functools

import logging

logger = logging.getLogger('cvastrophoto.raw')

class Raw(object):

    def __init__(self, path,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT,
            default_pool=None):
        self.name = path
        self.default_pool = default_pool
        self.postprocessing_params = rawpy.Params(
            output_bps=16,
            no_auto_bright=True,
            demosaic_algorithm=demosaic_algorithm,
        )
        self._rimg = None
        self._postprocessed = None

    def close(self):
        if self._rimg is not None:
            self._rimg.close()
            self._rimg = None
        self._postprocessed = None

    @classmethod
    def open_all(cls, dir_path, **kw):
        rv = []
        for path in sorted(os.listdir(dir_path)):
            fullpath = os.path.join(dir_path, path)
            if os.path.isfile(fullpath):
                rv.append(cls(fullpath, **kw))
        return rv

    @property
    def rimg(self):
        if self._rimg is None:
            self._rimg = rawpy.imread(self.name)
        return self._rimg

    def postprocess(self, **kwargs):
        self._postprocessed = self.rimg.postprocess(self.postprocessing_params)
        return self._postprocessed

    @property
    def postprocessed(self):
        if self._postprocessed is None:
            self._postprocessed = self.postprocess()
        return self._postprocessed

    def show(self):
        postprocessed = self.postprocessed
        PIL.Image.fromarray(numpy.clip(
            postprocessed >> 8,
            0, 255,
            out=numpy.empty(postprocessed.shape, numpy.uint8)
        )).show()

    def save(self, path, *p, **kw):
        postprocessed = self.postprocessed
        PIL.Image.fromarray(numpy.clip(
            postprocessed >> 8,
            0, 255,
            out=numpy.empty(postprocessed.shape, numpy.uint8)
        )).save(path, *p, **kw)

    def denoise(self, darks, pool=None, **kw):
        if pool is None:
            pool = self.default_pool
        logger.info("Denoising %s", self)
        raw_image = self.rimg.raw_image
        for dark, k_num, k_denom in find_entropy_weights(self, darks, pool=pool, **kw):
            logger.debug("Applying %s with weight %d/%d", dark, k_num, k_denom)
            dark_weighed = dark.rimg.raw_image.astype(numpy.uint32)
            dark_weighed *= k_num
            dark_weighed /= k_denom
            dark_weighed = numpy.minimum(dark_weighed, raw_image, out=dark_weighed)
            raw_image -= dark_weighed
        logger.info("Finished denoising %s", self)

    def set_raw_image(self, img):
        self.rimg.raw_image[:] = img
        self._postprocessed = None

    def luma_image(self, data=None, renormalize=False, same_shape=True, dtype=numpy.uint32):
        if data is None:
            data = self.rimg.raw_image

        pattern_shape = self.rimg.raw_pattern.shape
        ysize, xsize = pattern_shape
        luma = numpy.zeros((data.shape[0] / ysize, data.shape[1] / xsize), dtype)

        for yoffs in xrange(ysize):
            for xoffs in xrange(xsize):
                luma += data[yoffs::ysize, xoffs::xsize]

        if renormalize:
            factor = xsize * ysize
            if factor in shifts:
                luma >>= shifts[factor]
            else:
                luma /= factor

        if same_shape:
            nluma = numpy.empty(data.shape, dtype)
            for yoffs in xrange(ysize):
                for xoffs in xrange(xsize):
                    nluma[yoffs::ysize, xoffs::xsize] = luma
            luma = nluma

        return luma

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.name)

class RawAccumulator(object):

    def __init__(self):
        self.accum = None
        self.num_images = 0

    def __iadd__(self, raw):
        if self.accum is None:
            self.accum = raw.rimg.raw_image.astype(numpy.uint32)
            self.num_images = 1
        else:
            self.accum += raw.rimg.raw_image
            self.num_images += 1
        return self

    @property
    def average(self):
        if self.accum is not None:
            return self.accum / self.num_images

    @property
    def raw_image(self):
        accum = self.accum
        if accum is not None:
            maxval = accum.max()
            shift = 0
            while maxval > 65535:
                shift += 1
                maxval /= 2

            if shift:
                accum = accum >> shift
            return accum.astype(numpy.uint16)

    @classmethod
    def from_light_dark_set(cls, lights, darks, **kw):
        accum = cls()
        for light in lights:
            light.denoise(darks, **kw)
            accum += light
            light.close()
        return accum

shifts = { 1<<k : k for k in xrange(16) }

def entropy(light, dark, k_denom, k_num, scratch=None, return_params=False, quick=False, quick_size=512):
    raw_image = light.rimg.raw_image_visible
    dark_image = dark.rimg.raw_image_visible
    saturation = light.rimg.raw_image.max()
    if saturation < (1 << 11):
        # Unlikely this is actually a saturated pixel
        saturation = (1 << 16) - 1
    if quick:
        raw_image = raw_image[:quick_size, :quick_size]
        dark_image = dark_image[:quick_size, :quick_size]
    if scratch is None:
        scratch = numpy.empty(raw_image.shape, numpy.int32)
    elif quick:
        scratch = scratch[:quick_size, :quick_size]
    scratch[:] = raw_image
    unsaturated = scratch < saturation
    dark_weighed = dark_image.astype(numpy.int32)
    dark_weighed *= k_num
    if k_denom in shifts:
        dark_weighed >>= shifts[k_denom]
    else:
        dark_weighed /= k_denom
    scratch -= dark_weighed
    scratchmin = scratch.min()
    if scratchmin < 0:
        scratch -= scratch.min()
    scratchmax = scratch.max()
    if scratchmax < (1<<17):
        # bucket sort
        counts = numpy.histogram(scratch[unsaturated], scratchmax + 1, (0, scratchmax + 1))[0]
        counts = counts[counts.nonzero()[0]]
    else:
        # merge sort
        _, counts = numpy.unique(scratch[unsaturated], return_counts=True)
    rv = scipy.stats.entropy(counts)
    if return_params:
        rv = rv, k_num, k_denom
    return rv

def _refine_entropy(light, dark, steps, denom, base, pool=None, **kw):
    base *= steps
    denom *= steps
    _entropy = functools.partial(entropy, light, dark, denom, return_params=True, **kw)
    if pool is None:
        dark_ranges = map(_entropy, xrange(base, base + steps))
    else:
        dark_ranges = pool.map(_entropy, xrange(base, base + steps))
    return min(dark_ranges)

def find_entropy_weights(light, darks, steps=8, maxsteps=512, mink=0.1, maxk=1, **kw):
    ranges = []
    for dark in darks:
        initial_range = _refine_entropy(light, dark, steps, 1, 0, **kw)
        ranges.append((initial_range, dark))

    while ranges:
        best = min(ranges)
        ranges.remove(best)
        (e, base, denom), dark = best

        refined_range = _refine_entropy(light, dark, steps, denom, base, **kw)
        e, base, denom = refined_range

        if denom >= maxsteps:
            k = base / float(denom)

            if k < mink:
                # Close enough
                break
            elif k <= maxk:
                yield dark, base, denom

                # Reset remaining ranges and keep looking
                ranges = [
                    (_refine_entropy(light, dark, steps, 1, 0, **kw), dark)
                    for (_, dark) in ranges
                ]
            else:
                # Bad image, k too high, ignore
                logger.debug("Ignoring %s because k=%d/%d too large", dark, base, denom)
        else:
            ranges.append((refined_range, dark))

