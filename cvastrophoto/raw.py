# -*- coding: utf-8 -*-
import os.path
import rawpy
try:
    from rawpy import enhance
except ImportError:
    enhance = None
import numpy
import scipy.stats
import scipy.ndimage
import PIL.Image
import functools
import random

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
    def is_open(self):
        return self._rimg is not None

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

    def denoise(self, darks, pool=None,
            entropy_weighted=True, stop_at=1, master_bias=None,
            **kw):
        if pool is None:
            pool = self.default_pool
        logger.info("Denoising %s", self)
        raw_image = self.rimg.raw_image
        if entropy_weighted:
            entropy_weights = find_entropy_weights(self, darks, pool=pool, master_bias=master_bias, **kw)
        else:
            entropy_weights = [(dark, 1, 1) for dark in darks]
        applied = 0
        for dark, k_num, k_denom in entropy_weights:
            logger.debug("Applying %s with weight %d/%d", dark, k_num, k_denom)
            dark_weighed = dark.rimg.raw_image.astype(numpy.uint32)
            if k_num != 1 or k_denom != 1:
                if master_bias is not None:
                    bias = numpy.minimum(master_bias, dark_weighed)
                    dark_weighed -= bias
                dark_weighed *= k_num
                dark_weighed /= k_denom
                if master_bias is not None:
                    dark_weighed += bias
            applied += 1
            dark_weighed = numpy.minimum(dark_weighed, raw_image, out=dark_weighed)
            raw_image -= dark_weighed
            if stop_at and applied >= stop_at:
                break
        logger.info("Finished denoising %s", self)

    def demargin(self, accum=None):
        rimg = self.rimg
        if accum is None:
            accum = rimg.raw_image
        raw_shape = rimg.raw_image.shape
        visible_shape = rimg.raw_image_visible.shape
        path, patw = rimg.raw_pattern.shape
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

    @classmethod
    def find_bad_pixels(cls, images, **kw):
        if enhance is None:
            logger.warning("Could not import rawpy.enhance, install dependencies to enable bad pixel detection")
            return None

        logger.info("Analyzing %d images to detect bad pixels...", len(images))
        coords = rawpy.enhance.find_bad_pixels([img.name for img in images], **kw)
        logger.info("Found %d bad pixels", len(coords))
        return coords

    @classmethod
    def find_bad_pixels_from_sets(cls, sets, max_sample_per_set=10, **kw):
        sample_amount = min(map(len, sets) + [max_sample_per_set])
        sample = []
        for images in sets:
            sample.extend(random.sample(images, sample_amount))

        return cls.find_bad_pixels(sample)

    def repair_bad_pixels(self, coords, **kw):
        if coords is None or not len(coords):
            return

        if enhance is None:
            logger.warning("Could not import rawpy.enhance, install dependencies to enable bad pixel correction")
            return

        rawpy.enhance.repair_bad_pixels(self.rimg, coords, **kw)
        logger.info("Done repairing %d bad pixels...", len(coords))

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

    def __init__(self, dtype=numpy.uint32):
        self.accum = None
        self.num_images = 0
        self.dtype = dtype

    def __iadd__(self, raw):
        if isinstance(raw, Raw):
            raw_image = raw.rimg.raw_image
        else:
            raw_image = raw
        if self.accum is None:
            self.accum = raw_image.astype(self.dtype)
            self.num_images = 1
        else:
            self.accum += raw_image
            self.num_images += 1
        return self

    def init(self, shape):
        self.accum = numpy.zeros(shape, self.dtype)

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

def entropy(light, dark, k_denom, k_num, scratch=None, return_params=False,
        light_slice=None, dark_slice=None, noise_prefilter=None, master_bias=None):
    raw_image = light.rimg.raw_image_visible
    dark_image = dark.rimg.raw_image_visible
    saturation = light.rimg.raw_image.max()
    if saturation < (1 << 11):
        # Unlikely this is actually a saturated pixel
        saturation = (1 << 16) - 1
    if light_slice is not None:
        raw_image = light_slice(raw_image)
    if dark_slice is not None:
        dark_image = dark_slice(dark_image)
    if scratch is None:
        scratch = numpy.empty(raw_image.shape, numpy.int32)
    else:
        scratch = scratch[:raw_image.shape[0], :raw_image.shape[1]]
    scratch[:] = raw_image
    unsaturated = scratch < saturation
    dark_weighed = dark_image.astype(numpy.int32)
    if master_bias is not None:
        bias = numpy.minimum(dark_weighed, dark_slice(master_bias))
        dark_weighed -= bias
    else:
        bias = None
    dark_weighed *= k_num
    if k_denom in shifts:
        dark_weighed >>= shifts[k_denom]
    else:
        dark_weighed /= k_denom
    if bias is not None:
        dark_weighed += bias
    scratch -= dark_weighed
    if noise_prefilter is not None:
        scratch = noise_prefilter(scratch)
    scratchmin = scratch.min()
    if scratchmin < 0:
        scratch -= scratchmin
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
    def _entropy_wrapped(*p, **kw):
        try:
            return _entropy(*p, **kw)
        except Exception:
            logger.exception("Error measuring entropy")
            raise
    if pool is None:
        dark_ranges = map(_entropy_wrapped, xrange(base, base + steps))
    else:
        dark_ranges = pool.map(_entropy_wrapped, xrange(base, base + steps))
    return min(dark_ranges)

def find_entropy_weights(light, darks, steps=8, maxsteps=512, mink=0.05, maxk=1,
        prefilter=True, prefilter_size=3, quick=False, quick_size=512,
        **kw):

    light_slice = None
    if quick:
        def measure_slice(raw_image):
            return raw_image[:quick_size, :quick_size]
        kw['light_slice'] = measure_slice
        kw['dark_slice'] = measure_slice

    if prefilter:
        raw_pattern = light.rimg.raw_pattern
        path, patw = raw_pattern.shape
        def noise_prefilter(raw_image):
            for yoffs in xrange(path):
                for xoffs in xrange(patw):
                    raw_image[yoffs::path, xoffs::patw] = scipy.ndimage.white_tophat(
                        raw_image[yoffs::path, xoffs::patw],
                        prefilter_size,
                    )
            return raw_image
        kw['noise_prefilter'] = noise_prefilter

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
        elif (base + 1) / float(denom) <= mink:
            # Hopeless
            pass
        else:
            ranges.append((refined_range, dark))
