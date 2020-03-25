# -*- coding: utf-8 -*-
import os.path
import numpy
import scipy.stats
import scipy.ndimage
import PIL.Image
import functools
import random
import operator
import imageio

import logging
from cvastrophoto.util import srgb

logger = logging.getLogger(__name__)

class BaseImage(object):

    priority = 1000
    concrete = False

    def __init__(self, path, default_pool=None, **kw):
        self.name = path
        self.default_pool = default_pool
        self._kw = kw
        self._rimg = None
        self._postprocessed = None
        self.postprocessing_params = None

    def dup(self):
        rv = type(self)(self.name, default_pool=self.default_pool, **self._kw)
        rv.postprocessing_params = self.postprocessing_params
        return rv

    def close(self):
        if self._rimg is not None:
            self._rimg.close()
            self._rimg = None
        self._postprocessed = None

    def all_frames(self):
        yield self

    @classmethod
    def open_all(cls, dir_path, **kw):
        rv = []
        for path in sorted(os.listdir(dir_path)):
            fullpath = os.path.join(dir_path, path)
            if os.path.isfile(fullpath):
                img = cls.open(fullpath, **kw)
                for frame in img.all_frames():
                    rv.append(frame)
        return rv

    @classmethod
    def open(cls, path, **kw):
        last_supported = None

        concrete_cls = set()
        open_cls = set([cls])
        while open_cls:
            nopen_cls = set()
            for scls in open_cls:
                if scls.concrete:
                    concrete_cls.add(scls)
                nopen_cls.update(scls.__subclasses__())
            open_cls = nopen_cls

        for subcls in sorted(concrete_cls, key=operator.attrgetter('priority')):
            if subcls.supports(path):
                last_supported = subcls
                try:
                    return subcls(path, **kw)
                except:
                    pass
        else:
            return last_supported(path, **kw)

    @classmethod
    def supports(cls, path):
        if cls is BaseImage:
            for subcls in cls.__subclasses__():
                if subcls.supports(path):
                    return True
        return False

    @property
    def is_open(self):
        return self._rimg is not None

    @property
    def rimg(self):
        if self._rimg is None:
            self._rimg = self._open_impl(self.name)
        return self._rimg

    @property
    def black_level(self):
        if any(self.rimg.black_level_per_channel):
            return numpy.array(self.rimg.black_level_per_channel, numpy.uint16)[self.rimg.raw_colors]
        else:
            return 0

    def postprocess(self, **kwargs):
        self._postprocessed = self.rimg.postprocess(self.postprocessing_params)
        return self._postprocessed

    @property
    def postprocessed(self):
        if self._postprocessed is None:
            self._postprocessed = self.postprocess()
        return self._postprocessed

    def _process_gamma(self, postprocessed, gamma=2.4):
        postprocessed = numpy.clip(
            postprocessed, 0, 65535,
            out=numpy.empty(postprocessed.shape, numpy.uint16))
        return srgb.encode_srgb(postprocessed, gamma)

    def get_img(self, gamma=2.4, bright=1.0, component=None, get_array=False):
        postprocessed = self.postprocessed

        if component is None and postprocessed.shape[2] == 1:
            component = 0

        if component is not None:
            postprocessed = postprocessed[:,:,component]

        if bright != 1.0:
            postprocessed = postprocessed * bright

        if gamma != 1.0:
            postprocessed = self._process_gamma(postprocessed, gamma)

        if get_array:
            return postprocessed

        return PIL.Image.fromarray(numpy.clip(
            postprocessed >> 8,
            0, 255,
            out=numpy.empty(postprocessed.shape, numpy.uint8)
        ))

    def show(self, gamma=2.4, bright=1.0):
        img = self.get_img(gamma, bright)
        img.show()

    def save(self, path, gamma=2.4, bright=1.0, meta=dict(compress=6), *p, **kw):
        if path.upper().endswith('TIFF') or path.upper().endswith('TIF'):
            # PIL doesn't support 16-bit tiff, so use imageio
            postprocessed = self.get_img(gamma, bright, get_array=True)
            with imageio.get_writer(path, mode='i', software='cvastrophoto') as writer:
                writer.append_data(postprocessed, meta)
        else:
            img = self.get_img(gamma, bright)
            img.save(path, *p, **kw)

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
            if hasattr(dark, 'rimg'):
                dark_weighed = dark.rimg.raw_image.astype(numpy.uint32)
            else:
                dark_weighed = dark.astype(numpy.uint32)
            if k_num != 1 or k_denom != 1:
                if master_bias is not None:
                    bias = numpy.minimum(master_bias, dark_weighed)
                    dark_weighed -= bias
                dark_weighed *= k_num
                dark_weighed /= k_denom
                if master_bias is not None:
                    dark_weighed += bias
            applied += 1
            dark_weighed = dark_weighed.astype(raw_image.dtype, copy=False)
            dark_weighed = numpy.minimum(dark_weighed, raw_image, out=dark_weighed)
            raw_image -= dark_weighed
            if stop_at and applied >= stop_at:
                break
        logger.info("Finished denoising %s", self)

    def demargin(self, accum=None, raw_pattern=None, sizes=None):
        if raw_pattern is None or accum is None or sizes is None:
            rimg = self.rimg
            if accum is None:
                accum = rimg.raw_image
            if raw_pattern is None:
                raw_pattern = rimg.raw_pattern
            if sizes is None:
                sizes = rimg.sizes
        raw_shape = accum.shape
        path, patw = raw_pattern.shape

        rmargin = (raw_shape[1] - sizes.left_margin - sizes.width) / patw
        lmargin = sizes.left_margin / patw
        bmargin = (raw_shape[0] - sizes.top_margin - sizes.height) / path
        tmargin = sizes.top_margin / path

        for y in xrange(path):
            for x in xrange(patw):
                naccum = accum[y::path,x::patw]
                if rmargin:
                    naccum[:,-rmargin:] = naccum[:,-rmargin-1:-2*rmargin-1:-1]
                if lmargin:
                    naccum[:,:lmargin] = naccum[:,lmargin*2-1:lmargin-1:-1]
                if tmargin:
                    naccum[:tmargin,:] = naccum[2*tmargin-1:tmargin-1:-1,:]
                if bmargin:
                    naccum[-bmargin:,:] = naccum[-bmargin-1:-2*bmargin-1:-1,:]
        return accum

    @classmethod
    def find_bad_pixels(cls, images, **kw):
        return None

    @classmethod
    def find_bad_pixels_from_sets(cls, sets, max_samples_per_set=10, **kw):
        sets = filter(None, sets)
        sample_amount = min(map(len, sets) + [max_samples_per_set])
        sample = []
        for images in sets:
            sample.extend(random.sample(images, sample_amount))

        return cls.find_bad_pixels(sample)

    def repair_bad_pixels(self, coords, **kw):
        return

    def set_raw_image(self, img, add_bias=False):
        if add_bias:
            black_level = self.rimg.black_level_per_channel
            if any(black_level):
                raw_colors = self.rimg.raw_colors
                data = img.astype(numpy.uint32)
                data[:] += numpy.array(black_level, data.dtype)[raw_colors]
                data = numpy.clip(data, 0, 65535, out=data)
                img = data
        self.rimg.raw_image[:] = img
        self._postprocessed = None

    def remove_bias(self, data=None, copy=False):
        if data is None:
            data = self.rimg.raw_image
        black_level = self.rimg.black_level_per_channel
        if any(black_level):
            if copy:
                data = data.copy()
            raw_colors = self.rimg.raw_colors
            data[:] -= numpy.minimum(data, numpy.array(black_level, data.dtype)[raw_colors])
        return data

    def postprocessed_luma(self, dtype=None, copy=False):
        postprocessed = self.postprocessed

        if len(postprocessed.shape) == 3 and postprocessed.shape[2] > 1:
            # RGB image must add all channels
            if dtype is None:
                dtype = numpy.uint32
            luma = numpy.sum(self.postprocessed, axis=2, dtype=dtype)
        else:
            # LUMINANCE image can just be returned as is
            luma = postprocessed
            if len(postprocessed.shape) == 3:
                luma = luma.reshape(postprocessed.shape[:2])
            if dtype is not None:
                luma = luma.astype(dtype, copy=copy)
            elif copy:
                luma = luma.copy()

        return luma

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
            if factor in shifts and dtype is numpy.uint32:
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

class ImageAccumulator(object):

    def __init__(self, dtype=numpy.uint32):
        self.reset(dtype)

    def __iadd__(self, raw):
        if isinstance(raw, BaseImage):
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

    def reset(self, dtype=None):
        self.accum = None
        self.num_images = 0
        if dtype is not None:
            self.dtype = dtype

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
        prefilter=True, prefilter_size=5, quick=False, quick_size=512,
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
