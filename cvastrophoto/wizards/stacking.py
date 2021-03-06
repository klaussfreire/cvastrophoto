from __future__ import absolute_import, division

from past.builtins import xrange
import os.path
import multiprocessing.pool
import numpy
import functools
import tempfile
import PIL.Image
import scipy.ndimage.filters

try:
    import cPickle
except ImportError:
    import pickle as cPickle

from .base import BaseWizard
from cvastrophoto.rops.compound import CompoundRop
from cvastrophoto.rops.tracking import flip
import cvastrophoto.image

import logging

logger = logging.getLogger(__name__)

class BaseStackingMethod(object):

    weight_parts = []
    nonluma_parts = []
    luma_scale = None

    phases = [
        # Phase number - iterations
        (2, 1)
    ]

    def __init__(self, copy_frames=False, pool=None):
        self.copy_frames = copy_frames
        self.phase = 0
        self.pool = pool

    def set_image_shape(self, image):
        # Touch to make sure it's initialized
        image.postprocessed

        # Get raw pattern information
        self.raw_pattern = self.out_raw_pattern = image.rimg.raw_pattern
        self.raw_colors = image.rimg.raw_colors

        self.rmask = self.raw_pattern == 0
        self.gmask = self.raw_pattern == 1
        self.bmask = self.raw_pattern == 2

        self.rmask_image = self.raw_colors == 0
        self.gmask_image = self.raw_colors == 1
        self.bmask_image = self.raw_colors == 2

        self.raw_sizes = image.rimg.sizes

    def get_tracking_image(self, image):
        return image, image

    def extract_frame(self, frame, weights=None):
        return frame

    @property
    def accumulator(self):
        raise NotImplementedError

    def start_phase(self, phase, iteration):
        logger.info("Starting phase %d iteration %d", phase, iteration)
        self.phase = phase
        self.iteration = iteration

    def finish(self):
        logger.info("Finished stacking")
        self.phase = None

    def __iadd__(self, image):
        raise NotImplementedError

    def stack(self, images_callback, extracted=False):
        for phaseno, iterations in self.phases:
            for i in xrange(iterations):
                self.start_phase(phaseno, i)
                for image in images_callback(phaseno, i):
                    if not extracted:
                        image = self.extract_frame(image)
                    self += image
        self.finish()


class AverageStackingMethod(BaseStackingMethod):

    def __init__(self, copy_frames=False, **kw):
        self.light_accum = cvastrophoto.image.ImageAccumulator()
        super(AverageStackingMethod, self).__init__(copy_frames, **kw)

    @property
    def accumulator(self):
        return self.light_accum

    def __iadd__(self, image):
        if isinstance(image, list) and len(image) == 1:
            image = image[0]
        if self.copy_frames and self.light_accum.num_images == 0:
            if isinstance(image, cvastrophoto.image.Image):
                image = image.rimg.raw_image.copy()
        if self.light_accum.num_images == 0 and image.dtype.char in ('f', 'd'):
            self.light_accum.dtype = image.dtype
        self.light_accum += image
        return self


class MaxStackingMethod(BaseStackingMethod):

    def __init__(self, copy_frames=False, **kw):
        self.light_accum = cvastrophoto.image.ImageAccumulator()
        super(MaxStackingMethod, self).__init__(copy_frames, **kw)

    @property
    def accumulator(self):
        return self.light_accum

    def __iadd__(self, image):
        if isinstance(image, cvastrophoto.image.Image):
            image = image.rimg.raw_image
        if self.light_accum.num_images == 0:
            if image.dtype.char in ('f', 'd'):
                self.light_accum.dtype = image.dtype
            self.light_accum += image.copy()
        else:
            numpy.maximum(self.light_accum.accum, image, out=self.light_accum.accum)
        return self


class MinStackingMethod(BaseStackingMethod):

    def __init__(self, copy_frames=False, **kw):
        self.light_accum = cvastrophoto.image.ImageAccumulator()
        super(MinStackingMethod, self).__init__(copy_frames, **kw)

    @property
    def accumulator(self):
        return self.light_accum

    def __iadd__(self, image):
        if isinstance(image, cvastrophoto.image.Image):
            image = image.rimg.raw_image
        if self.light_accum.num_images == 0:
            if image.dtype.char in ('f', 'd'):
                self.light_accum.dtype = image.dtype
            self.light_accum += image.copy()
        else:
            numpy.minimum(self.light_accum.accum, image, out=self.light_accum.accum)
        return self


def dark_spots(cimg, cmin):
    cimg = scipy.ndimage.gaussian_filter(cimg, 8)
    dark_spots = scipy.ndimage.black_tophat(cimg, 64)
    mask = dark_spots > (numpy.average(dark_spots) + numpy.std(dark_spots))
    del cimg, dark_spots

    mask = scipy.ndimage.binary_closing(mask)
    mask = scipy.ndimage.binary_opening(mask)
    return mask


class MedianStackingMethod(BaseStackingMethod):

    def __init__(self, copy_frames=False, **kw):
        self.frames = []
        self.light_accum = None
        super(MedianStackingMethod, self).__init__(copy_frames, **kw)

    def finish(self):
        self.update_accum()

    def update_accum(self):
        frames = numpy.asanyarray(self.frames)
        dtype = self.frames[0].dtype
        del self.frames[:]
        self.light_accum = cvastrophoto.image.ImageAccumulator(dtype)
        self.light_accum += numpy.median(frames, axis=0, overwrite_input=True).astype(dtype, copy=False)

    @property
    def accumulator(self):
        if self.light_accum is None:
            self.update_accum()
        return self.light_accum

    def __iadd__(self, image):
        if isinstance(image, cvastrophoto.image.Image):
            image = image.rimg.raw_image
        if self.copy_frames:
            image = image.copy()
        self.frames.append(image)
        self.light_accum = None
        return self


class ApproxMedianStackingMethod(BaseStackingMethod):

    tier_size = 5

    def __init__(self, copy_frames=False, **kw):
        self.frames = [[]]
        self.light_accum = None
        super(MedianStackingMethod, self).__init__(copy_frames, **kw)

    def finish(self):
        self.update_accum()

    def update_accum(self):
        self.waterfall()

        dtype = self.dtype
        self.light_accum = cvastrophoto.image.ImageAccumulator(dtype)

        tier_weight = 1
        for tier_frames in self.frames:
            for frame in tier_frames:
                self.light_accum += frame.astye(dtype, copy=False) * tier_weight
                self.light_accum.num_frames += (tier_weight - 1)
            tier_weight *= 5

    def waterfall(self):
        for tier, tier_frames in enumerate(self.frames):
            if len(tier_frames) < self.tier_size:
                break
            frames = numpy.asanyarray(tier_frames)
            dtype = tier_frames[0].dtype
            del tier_frames[:]
            self.frames[tier + 1].append(numpy.median(frames, axis=0, overwrite_input=True).astype(dtype, copy=False))

    @property
    def accumulator(self):
        if self.light_accum is None:
            self.update_accum()
        return self.light_accum

    def __iadd__(self, image):
        if isinstance(image, cvastrophoto.image.Image):
            image = image.rimg.raw_image
        if self.copy_frames:
            image = image.copy()
        self.frames[0].append(image)
        self.light_accum = None
        self.waterfall()
        return self


class WeightedAverageStackingMethod(BaseStackingMethod):

    weight_parts = [1]
    nonluma_parts = [1]

    def __init__(self, copy_frames=False, **kw):
        self.final_accumulator = cvastrophoto.image.ImageAccumulator()
        super(WeightedAverageStackingMethod, self).__init__(copy_frames, **kw)

    def extract_frame(self, frame, weights=None):
        if isinstance(frame, cvastrophoto.image.Image):
            frame = frame.rimg.raw_image
        frame = frame.astype(numpy.float32)
        if weights is None:
            weights = numpy.ones(frame.shape, dtype=frame.dtype)
        return [frame, weights]

    def start_phase(self, phase, iteration):
        self.weights = cvastrophoto.image.ImageAccumulator(numpy.float32)
        self.light_accum = cvastrophoto.image.ImageAccumulator(numpy.float32)
        super(WeightedAverageStackingMethod, self).start_phase(phase, iteration)

    def finish_phase(self):
        self.final_accumulator.accum = self.estimate_average()
        self.final_accumulator.accum *= self.light_accum.num_images
        self.final_accumulator.num_images = self.light_accum.num_images

    def finish(self):
        self.finish_phase()
        self.weights = self.light_accum = None
        super(WeightedAverageStackingMethod, self).finish()

    def estimate_average(self, accum=None, weights_accum=None):
        if accum is None:
            accum = self.light_accum
        if weights_accum is None:
            weights_accum = self.weights

        if weights_accum.num_images == 0:
            # Unweighted average
            weights = accum.num_images
        else:
            # Weighted average
            min_weight = weights_accum.accum.min()
            if min_weight <= 0:
                # Must plug holes
                holes = weights_accum.accum <= 0
                weights = weights_accum.accum.copy()
                weights[holes] = 1
            else:
                holes = None
                weights = weights_accum.accum

        return accum.accum / weights

    @property
    def accumulator(self):
        if self.final_accumulator.num_images == 0:
            self.final_accumulator.accum = self.estimate_average()
            self.final_accumulator.accum *= self.light_accum.num_images
            self.final_accumulator.num_images = self.light_accum.num_images
        return self.final_accumulator

    def __iadd__(self, image):
        image, weight = image

        self._add_with_weights(image, weight)

        # Mark final accumulator as dirty so previews recompute the final average
        self.final_accumulator.num_images = 0

        return self

    def _add_with_weights(self, image, weight):
        if weight is not None:
            image *= weight
        else:
            weight = 1
            if self.weights.accum is None:
                # Explicitly initialize so we're able to add a constant weight
                self.weights.init(image.shape)

        self.light_accum += image
        self.weights += weight


class AdaptiveWeightedAverageStackingMethod(BaseStackingMethod):

    kappa_sq = 4
    weight_parts = [1, 2]
    nonluma_parts = [1, 2]

    mask_dark_spots = False

    phases = [
        # Dark phase (0)
        # (0, 1),

        # Outlier trimming phase, 2 iterations
        (1, 2),

        # Final light phase
        (2, 1),
    ]

    def __init__(self, copy_frames=False, **kw):
        self.final_accumulator = cvastrophoto.image.ImageAccumulator()
        self.current_average = None
        self.current_min = [0] * 4
        super(AdaptiveWeightedAverageStackingMethod, self).__init__(copy_frames, **kw)

    def extract_frame(self, frame, weights=None):
        if isinstance(frame, cvastrophoto.image.Image):
            frame = frame.rimg.raw_image
        frame = frame.astype(numpy.float32)
        if self.phase == 0:
            return [frame, frame.copy(), None]
        elif self.phase >= 1:
            return [frame, None, weights]

    def start_phase(self, phase, iteration):
        if phase == 1:
            if iteration:
                self.finish_phase()
                self.invvar = self.estimate_variance(self.light_accum, self.light2_accum, self.weights)
        elif phase == 2:
            self.finish_phase()
            self.invvar = self.estimate_variance(self.light_accum, self.light2_accum, self.weights)
        self.weights = cvastrophoto.image.ImageAccumulator(numpy.float32)
        self.light_accum = cvastrophoto.image.ImageAccumulator(numpy.float32)
        self.light2_accum = cvastrophoto.image.ImageAccumulator(numpy.float32)
        super(AdaptiveWeightedAverageStackingMethod, self).start_phase(phase, iteration)

    def finish_phase(self):
        self.current_average = self.estimate_average()
        self.final_accumulator.accum = self.current_average.copy()
        self.final_accumulator.accum *= self.light_accum.num_images
        self.final_accumulator.num_images = self.light_accum.num_images

        path, patw = self.out_raw_pattern.shape
        self.current_min = numpy.zeros(self.out_raw_pattern.max() + 1, dtype=self.current_average.dtype)
        for y in xrange(path):
            for x in xrange(patw):
                self.current_min[self.out_raw_pattern[y, x]] = self.current_average[y::path, x::patw].min()

        logger.info("Finished phase %r", self.phase)

    def finish(self):
        self.finish_phase()
        self.weights = self.light_accum = self.light2_accum = self.invvar = None
        super(AdaptiveWeightedAverageStackingMethod, self).finish()

    def estimate_variance(self, accum, sq_accum, weight_accum=None):
        if weight_accum is None:
            sq_avg = sq_accum.average
            avg = accum.average
        else:
            sq_avg = self.estimate_average(sq_accum, weight_accum)
            avg = self.estimate_average(accum, weight_accum)
        invvar = sq_avg - numpy.square(avg)
        invvar *= (
            float(max(1, sq_accum.num_images))
            / float(max(1, sq_accum.num_images - 1))
        )
        if invvar.min() <= 0:
            # Fix singularities
            if invvar.max() > 0:
                invvar = numpy.clip(invvar, invvar[invvar > 0].min() * 0.707, None, out=invvar)
            else:
                invvar[:] = 1
        invvar = numpy.reciprocal(invvar, out=invvar)
        return invvar

    def estimate_average(self, accum=None, weights_accum=None):
        if accum is None:
            accum = self.light_accum
        if weights_accum is None:
            weights_accum = self.weights

        if weights_accum.num_images == 0:
            # Unweighted average
            weights = accum.num_images
        else:
            # Weighted average
            min_weight = weights_accum.accum.min()
            if min_weight <= 0:
                # Must plug holes
                holes = weights_accum.accum <= 0
                weights = weights_accum.accum.copy()
                weights[holes] = 1
            else:
                holes = None
                weights = weights_accum.accum

        return accum.accum / weights

    @property
    def accumulator(self):
        if self.final_accumulator.num_images == 0:
            self.final_accumulator.accum = self.estimate_average()
            self.final_accumulator.accum *= self.light_accum.num_images
            self.final_accumulator.num_images = self.light_accum.num_images
        return self.final_accumulator

    def __iadd__(self, image):
        image, weight, imgweight = image
        image_sq = numpy.square(image)

        if self.current_average is not None and self.invvar is not None:
            residue = image - self.current_average
            residue = numpy.square(residue, out=residue)
            residue *= self.invvar
            logger.debug("Residue: %r", residue)

            if self.iteration > 1 or self.phase > 1:
                # Adaptive weighting iterations
                residue += 1
                weight = self.invvar / residue
            else:
                # Kappa-Sigma clipping iteration
                weight = (residue <= self.kappa_sq).astype(numpy.float32)

            if self.mask_dark_spots:
                path, patw = self.out_raw_pattern.shape
                for y in xrange(path):
                    for x in xrange(patw):
                        cweight = weight[y::path, x::patw]
                        cimage = image[y::path, x::patw]
                        bad_pixels = dark_spots(cimage, self.current_min[self.out_raw_pattern[y, x]])
                        if bad_pixels.any():
                            cweight[bad_pixels] *= 0.0001
                        del bad_pixels, cimage, cweight
        elif self.mask_dark_spots:
            if imgweight is None:
                imgweight = numpy.ones(image.shape, dtype=numpy.float32)
            path, patw = self.out_raw_pattern.shape
            for y in xrange(path):
                for x in xrange(patw):
                    cweight = imgweight[y::path, x::patw]
                    cimage = image[y::path, x::patw]
                    bad_pixels = dark_spots(cimage, self.current_min[self.out_raw_pattern[y, x]])
                    if bad_pixels.any():
                        cweight[bad_pixels] *= 0.0001
                    del bad_pixels, cimage, cweight

        if imgweight is not None:
            if weight is None:
                weight = imgweight
            else:
                weight *= imgweight

        self._add_with_weights(image, image_sq, weight)

        # Mark final accumulator as dirty so previews recompute the final average
        self.final_accumulator.num_images = 0

        return self

    def _add_with_weights(self, image, image_sq, weight):
        if weight is not None:
            image *= weight
            image_sq *= weight
        else:
            weight = 1
            if self.weights.accum is None:
                # Explicitly initialize so we're able to add a constant weight
                self.weights.init(image.shape)

        self.light_accum += image
        self.light2_accum += image_sq
        self.weights += weight


class DrizzleStackingMethod(AdaptiveWeightedAverageStackingMethod):
    # WIP, Don't use
    add_bias = True

    # Weight for "fake" color pixels, by phase number
    hole_weight = [1.0, 1.0, 0.5, 0.5]

    # Upscale factor
    scale_factor = 1

    weight_parts = [2, 3]
    nonluma_parts = [1, 2, 3]

    _masks = None

    def get_tracking_image(self, image):
        # Drizzle into an RGB image
        self.raw = image.dup()
        rimg = image.rimg
        raw_image = rimg.raw_image
        scale_factor = self.scale_factor
        shape1x = raw_image.shape
        shape = (raw_image.shape[0] * scale_factor, raw_image.shape[1] * scale_factor)
        if self.raw_pattern.max() > 1:
            channels = 3
            shape += (channels,)
            shape1x += (channels,)
        else:
            channels = 1
        self.channels = channels
        self.rgbshape = shape
        self.rgbshape1x = shape1x

        margins = (
            rimg.sizes.top_margin * scale_factor,
            rimg.sizes.left_margin * scale_factor,
            shape[0] - (rimg.sizes.top_margin + rimg.sizes.iheight) * scale_factor,
            shape[1] - (rimg.sizes.left_margin + rimg.sizes.iwidth) * scale_factor,
        )
        rgbdata = numpy.empty(shape, dtype=raw_image.dtype)
        img = cvastrophoto.image.rgb.RGB(
            None,
            img=rgbdata, margins=margins, default_pool=image.default_pool,
            daylight_whitebalance=rimg.daylight_whitebalance)
        if hasattr(rimg, 'rgb_xyz_matrix'):
            img.lazy_rgb_xyz_matrix = rimg.rgb_xyz_matrix

        self.outimg = img
        self.out_raw_pattern = img.rimg.raw_pattern

        if scale_factor == 1:
            limg = img
        else:
            margins1x = (
                rimg.sizes.top_margin,
                rimg.sizes.left_margin,
                shape1x[0] - (rimg.sizes.top_margin + rimg.sizes.iheight),
                shape1x[1] - (rimg.sizes.left_margin + rimg.sizes.iwidth),
            )
            rgbdata1x = numpy.empty(shape1x, dtype=raw_image.dtype)
            limg = cvastrophoto.image.rgb.RGB(
                None,
                img=rgbdata1x, margins=margins1x, default_pool=image.default_pool,
                daylight_whitebalance=rimg.daylight_whitebalance)
            if hasattr(rimg, 'rgb_xyz_matrix'):
                limg.lazy_rgb_xyz_matrix = rimg.rgb_xyz_matrix

        return limg, img

    def _enlarge_mask(self, img):
        scale = self.scale_factor
        if scale == 1:
            return img

        shape = img.shape
        h, w = shape[:2]
        offset = (scale - 1) // 2

        rv = numpy.zeros((h * scale, w * scale) + shape[2:], dtype=img.dtype)
        rv[offset::scale, offset::scale] = img
        return rv

    def _enlarge_image(self, img):
        scale = self.scale_factor
        if scale == 1:
            return img

        shape = img.shape
        h, w = shape[:2]

        if len(shape) > 2:
            rv = numpy.empty((h * scale, w * scale) + shape[2:], dtype=img.dtype)
            for c in xrange(shape[2]):
                rv[:,:,c] = scipy.ndimage.zoom(img[:,:,c], scale, mode='nearest')
        else:
            rv = scipy.ndimage.zoom(img, scale, mode='nearest')
        return rv

    @property
    def masks(self):
        if self._masks is not None:
            return self._masks

        if self.channels == 3:
            masks = (self.rmask_image, self.gmask_image, self.bmask_image)
        elif self.channels == 1:
            masks = (self.rmask_image,)
        else:
            raise NotImplementedError

        emasks = tuple(map(self._enlarge_mask, masks))
        masks = (masks, emasks)

        self._masks = masks

        return masks

    def extract_frame(self, frame, weights=None):
        if isinstance(frame, cvastrophoto.image.Image):
            frame = frame.rimg.raw_image

        # Demargin to avoid filtering artifacts at the borders
        self.raw.demargin(frame)

        rgbshape = self.rgbshape
        rgbshape1x = self.rgbshape1x
        if len(rgbshape) == 2:
            # Equivalent and more convenient for us
            rgbshape = rgbshape + (1,)
            rgbshape1x = rgbshape1x + (1,)
        self.rawshape = rawshape = (rgbshape[0], rgbshape[1] * rgbshape[2])
        self.rawshape1x = rawshape1x = (rgbshape1x[0], rgbshape1x[1] * rgbshape1x[2])

        # Masked RGB images for accumulation
        rgbimage = numpy.zeros(rgbshape, dtype=frame.dtype)

        if hasattr(weights, 'dtype'):
            rgbweights = numpy.zeros(rgbshape, dtype=weights.dtype)
        elif weights is not None:
            rgbweights = weights
        else:
            rgbweights = None

        masks, emasks = self.masks

        path, patw = self.raw_pattern.shape
        eshape = (path * self.scale_factor, patw * self.scale_factor)

        for c, (mask, emask) in enumerate(zip(masks, emasks)):
            # Create mono image and channel mask
            cimage = rgbimage[:,:,c]
            cimage[emask] = frame[mask]

            if hasattr(rgbweights, 'dtype'):
                cweights = rgbweights[:,:,c]
                cweights[emask] = weights[mask]
            else:
                cweights = None

            # Interpolate between channel samples
            a = scipy.ndimage.filters.uniform_filter(cimage.astype(numpy.float32, copy=False), eshape)
            w = scipy.ndimage.filters.uniform_filter(emask.astype(numpy.float32, copy=False), eshape)
            w = numpy.clip(w, 0.001, None, out=w)
            a /= w
            cimage[~emask] = a[~emask]
            del a

            if cweights is not None:
                cw = scipy.ndimage.filters.uniform_filter(cweights.astype(numpy.float32, copy=False), eshape)
                cw /= w
                cweights[~emask] = cw[~emask]
                del cw, w

        rgbimage, rgbweight, rgbimgweight = super(DrizzleStackingMethod,
            self).extract_frame(rgbimage.reshape(rawshape), rgbweights)
        del rgbweights

        # Compute masked weights
        rgbimage = rgbimage.reshape(rgbshape)
        if rgbweight is not None:
            rgbweight = rgbweight.reshape(rgbshape)
        if rgbimgweight is not None:
            rgbimgweight = rgbimgweight.reshape(rgbshape)

        rvimgweight = numpy.zeros(rgbshape, dtype=numpy.float32)

        imgmask = None
        for c, emask in enumerate(emasks):
            if rgbimgweight is not None:
                cimgweight = rgbimgweight[:,:,c]
            else:
                cimgweight = None

            if self.phase > 0:
                imgmask = emask.astype(numpy.float32)
                imgmask[~emask] = self.hole_weight[self.phase]  # Just to avoid holes
                if cimgweight is not None:
                    imgmask *= cimgweight
                cimgweight = imgmask

            if cimgweight is not None:
                rvimgweight[:,:,c] = cimgweight

        del rgbimgweight, cimgweight, imgmask

        # Extract debayered image to use for alignment
        rsizes = self.raw.rimg.sizes
        luma = self.raw.luma_image(frame, renormalize=True, dtype=numpy.float32)
        rvluma = numpy.empty(rgbshape1x, dtype=frame.dtype)
        for c in xrange(self.channels):
            rvluma[:,:,c] = luma
        del luma

        if self.phase >= 1:
            # Use rough raw luma for border, but actual debayered image for the
            # visible area. We want the extra precision proper debayer gives.
            lframe = frame
            lframemax = lframe.max()
            white = getattr(self.raw.rimg, 'white_level', 16383)
            scale = int((lframemax + 1) / (white + 1))
            if scale > 1:
                lframe = lframe / scale
            self.raw.set_raw_image(lframe, add_bias=self.add_bias)
            del lframe

            rvluma[
                rsizes.top_margin:rsizes.top_margin+rsizes.iheight,
                rsizes.left_margin:rsizes.left_margin+rsizes.iwidth] = self.raw.postprocessed

        # Reshape into patterend RGB
        rvluma = rvluma.reshape(rawshape1x)
        rvframe = rgbimage.reshape(rawshape)
        rvimgweight = rvimgweight.reshape(rawshape)
        rvweight = rgbweight.reshape(rawshape) if rgbweight is not None else None

        return [rvluma, rvframe, rvweight, rvimgweight]

    def __iadd__(self, image):
        return super(DrizzleStackingMethod, self).__iadd__(image[1:4])


class Drizzle2xStackingMethod(DrizzleStackingMethod):
    scale_factor = luma_scale = 2


class Drizzle3xStackingMethod(DrizzleStackingMethod):
    scale_factor = luma_scale = 3


class InterleaveStackingMethod(DrizzleStackingMethod):
    # Weight for "fake" color pixels, by phase number
    hole_weight = [1.0, 0.5, 0.00001, 0.00001]


class Interleave2xStackingMethod(InterleaveStackingMethod):
    scale_factor = luma_scale = 2


class Interleave3xStackingMethod(InterleaveStackingMethod):
    scale_factor = luma_scale = 3


class DrizzleMedianStackingMethod(DrizzleStackingMethod):

    shards = 40
    splits = multiprocessing.cpu_count() * 4

    # This one's single-phase
    phases = [
        (2, 1)
    ]

    def __init__(self, copy_frames=False, **kw):
        super(DrizzleMedianStackingMethod, self).__init__(copy_frames, **kw)
        self.framebuffers = [
            tempfile.TemporaryFile()
            for i in xrange(self.shards)
        ]

    def _add_with_weights(self, image, image_sq, weight):
        # We'll accumulate a shit load of data, so lets dump it to a sharded tempfile
        shards = self.shards
        shardsize = (image.shape[0] + (shards - 1)) // shards
        for framebuffer, row in zip(self.framebuffers, xrange(0, image.shape[0], shardsize)):
            cPickle.dump(
                (image[row:row+shardsize], weight[row:row+shardsize]),
                framebuffer,
                cPickle.HIGHEST_PROTOCOL,
            )
        self.light_accum.num_images += 1

    def start_phase(self, phase, iteration):
        self.current_average = None
        self.invvar = None
        self.weights = cvastrophoto.image.ImageAccumulator(numpy.float32)
        self.light_accum = cvastrophoto.image.ImageAccumulator(numpy.float32)
        self.light2_accum = cvastrophoto.image.ImageAccumulator(numpy.float32)

        # Purposefully skip the superclass
        super(AdaptiveWeightedAverageStackingMethod, self).start_phase(phase, iteration)

    def estimate_variance(self, accum, sq_accum, weight_accum=None):
        return None

    def estimate_average(self, accum=None, weights_accum=None):
        shards = []

        for framebuffer in self.framebuffers:
            # Load shard and compute weighted median
            images = []
            weights = []
            image_shape = None
            framebuffer.seek(0)
            while True:
                try:
                    image, weight = cPickle.load(framebuffer)
                except EOFError:
                    break
                image_shape = image.shape
                images.append(image.ravel())
                weights.append(weight.ravel())

            if not images:
                return None

            images = numpy.asanyarray(images)
            weights = numpy.asanyarray(weights)
            image = numpy.empty(images.shape[1:], dtype=images.dtype)

            if self.pool is not None:
                map_ = self.pool.map
            else:
                map_ = map

            splitsize = (len(image) + (self.splits - 1)) // self.splits

            # Make multiple of 64 to avoid threading issues with concurrent writes across cache line boundaries
            # since this will be writing to a shared array buffer
            splitsize = (splitsize + 63) // 64 * 64

            def split_median(col):
                simages = images[:,col:col+splitsize]
                sweights = weights[:,col:col+splitsize]
                shuffle = numpy.argsort(simages, axis=0)
                pixindex = numpy.arange(sweights.shape[1])
                sweights = sweights[shuffle, pixindex]
                simages = simages[shuffle, pixindex]
                del shuffle

                sweights = numpy.cumsum(sweights, axis=0, out=sweights)
                medweight = sweights[-1,...] / 2
                medposhi = numpy.argmax(sweights >= medweight, axis=0)
                medposlo = medposhi - 1
                medposlo = numpy.clip(medposlo, 0, None, out=medposlo)
                medposfrac = medweight - sweights[medposlo, pixindex]
                medposfrac /= numpy.clip(sweights[medposhi, pixindex] - sweights[medposlo, pixindex], 1e-5, None)
                medposfrac = numpy.clip(medposfrac, 0, 1, out=medposfrac)
                del medweight, sweights

                simage = simages[medposhi, pixindex] * medposfrac
                simage += simages[medposlo, pixindex] * (1 - medposfrac)
                del medposhi, medposlo, pixindex, medposfrac, simages

                image[col:col+splitsize] = simage

            for _ in map_(split_median, xrange(0, len(image), splitsize)):
                pass

            shards.append(image.reshape(image_shape))
            image = images = weights = None

        return numpy.concatenate(shards)


class InterleaveMedianStackingMethod(DrizzleMedianStackingMethod):
    # Weight for "fake" color pixels, by phase number
    hole_weight = [1.0, 0.5, 0.00001, 0.00001]


class StackingWizard(BaseWizard):

    denoise_amount = 1

    def __init__(self, pool=None, input_pool=None,
            denoise=True, quick=True, entropy_weighted_denoise=False, exhaustive_denoise=False,
            fbdd_noiserd=2, fpn_reduction=1,
            tracking_class=None, input_rop=None,
            light_method=AverageStackingMethod,
            dark_method=MedianStackingMethod,
            weight_class=None,
            normalize_weights=True,
            mirror_edges=True,
            pedestal=0,
            remove_bias=True):
        if pool is None:
            pool = multiprocessing.pool.ThreadPool()
        if input_pool is None:
            input_pool = multiprocessing.pool.ThreadPool()
        self.input_pool = input_pool
        self.pool = pool
        self.denoise = denoise
        self.fpn_reduction = fpn_reduction
        self.entropy_weighted_denoise = entropy_weighted_denoise
        self.quick = quick
        self.exhaustive_denoise = exhaustive_denoise
        self.fbdd_noiserd = fbdd_noiserd
        self.tracking_class = tracking_class
        self.weight_class = weight_class
        self.input_rop = input_rop
        self.light_method = light_method
        self.dark_method = dark_method
        self.bad_pixel_coords = None
        self.lights = None
        self.initial_tracking_reference = None
        self.tracking = None
        self.normalize_weights = normalize_weights
        self.mirror_edges = mirror_edges
        self.pedestal = pedestal
        self.remove_bias = remove_bias
        self.dark_library = None
        self.bias_library = None
        self.extra_metadata = {}

    def get_state(self):
        return dict(
            bad_pixel_coords=self.bad_pixel_coords,
            input_rop=self.input_rop.get_state() if self.input_rop is not None else None,
            initial_tracking_reference=self.initial_tracking_reference,
            tracking=self.tracking.get_state() if self.tracking is not None else None,
        )

    def _load_state(self, state):
        self.bad_pixel_coords = state['bad_pixel_coords']
        self.initial_tracking_reference = state['initial_tracking_reference']
        if self.input_rop is not None:
            self.input_rop.load_state(state['input_rop'])
        if self.tracking is not None:
            self.tracking.load_state(state['tracking'])

    def load_set(self,
            base_path='.', light_path='Lights', dark_path='Darks', master_bias=None, bias_shift=0,
            light_files=None, dark_files=None, dark_library=None, auto_dark_library='darklib',
            bias_library=None, weights=None, extra_metadata=None, open_kw={}):
        if light_files:
            self.lights = [
                light_img
                for path in light_files
                for light_img in cvastrophoto.image.Image.open(path, default_pool=self.pool, **open_kw).all_frames()
            ]
        else:
            self.lights = cvastrophoto.image.Image.open_all(
                os.path.join(base_path, light_path), default_pool=self.pool, **open_kw)

        self.light_method_instance = light_method = self.light_method(True, pool=self.pool)
        self.weights = weights

        light_method.set_image_shape(self.lights[0])
        self.stacked_luma_template, self.stacked_image_template = light_method.get_tracking_image(self.lights[0])

        if self.weight_class is not None:
            self.weight_rop = self.weight_class(self.lights[0])
        else:
            self.weight_rop = None

        if extra_metadata:
            self.extra_metadata = extra_metadata

        self.pier_flip_rop = flip.PierFlipTrackingRop(
            self.stacked_image_template,
            lraw=self.stacked_luma_template)

        if self.tracking_class is not None:
            self.tracking = trop = self.tracking_class(
                self.stacked_image_template,
                lraw=self.stacked_luma_template)
            if isinstance(trop, CompoundRop):
                trops = list(trop.rops) + [trop]
            else:
                trops = [trop]
            for trop in trops:
                if not self.mirror_edges and light_method.weight_parts:
                    for partno in light_method.weight_parts:
                        trop.per_part_mode[partno] = 'constant'
                if light_method.luma_scale and light_method.luma_scale != 1:
                    for partno in light_method.nonluma_parts:
                        trop.per_part_scale[partno] = light_method.luma_scale
        else:
            self.tracking = None

        if self.denoise and (dark_path is not None or dark_files is not None):
            if dark_library is None and auto_dark_library:
                dark_library = cvastrophoto.library.darks.DarkLibrary(
                    auto_dark_library, default_pool=self.pool)
                if dark_files:
                    dark_library.build(dark_files)
                else:
                    dark_library.build_recursive(dark_path)
                self.darks = None
            elif dark_files:
                self.darks = [
                    cvastrophoto.image.Image.open(path, default_pool=self.pool)
                    for path in dark_files
                ]
            else:
                self.darks = cvastrophoto.image.Image.open_all(
                    os.path.join(base_path, dark_path), default_pool=self.pool)
            if self.darks:
                self.median_dark = cvastrophoto.image.Image.open(self.darks[0].name)
            else:
                self.median_dark = None
        else:
            self.darks = None

        self.dark_library = dark_library
        self.bias_library = bias_library

        if master_bias is not None:
            if master_bias.lower().endswith('.tif'):
                sizes = self.lights[0].rimg.sizes
                raw_master_bias = numpy.zeros((sizes.raw_height, sizes.raw_width), numpy.uint16)
                raw_master_bias[
                    sizes.top_margin:sizes.top_margin+sizes.iheight,
                    sizes.left_margin:sizes.left_margin+sizes.iwidth] = numpy.array(PIL.Image.open(master_bias))
                self.master_bias = cvastrophoto.image.Image.open(self.lights[0].name)
                self.master_bias.set_raw_image(raw_master_bias)
                self.master_bias.name = master_bias
            else:
                self.master_bias = cvastrophoto.image.Image.open(master_bias)
            raw_master_bias = self.master_bias.rimg.raw_image
            if bias_shift:
                raw_master_bias >>= bias_shift
        else:
            self.master_bias = None

        if self.lights[0].postprocessing_params is not None and self.fbdd_noiserd is not None:
            self.lights[0].postprocessing_params.fbdd_noiserd = self.fbdd_noiserd

    def process(self, progress_callback=None):
        light_method = self.light_method_instance
        bad_pixel_coords = self.bad_pixel_coords

        def enum_darks(phase, iteration, extract=None):
            if self.darks is not None:
                for dark in self.darks:
                    logger.info("Adding dark frame %s", dark.name)
                    if extract is not None:
                        dark = extract(dark)
                    yield dark

        darks = self.darks
        if self.denoise and darks is not None:
            # Stack dark frames
            dark_method = self.dark_method()
            dark_method.stack(enum_darks)
            dark_accum = dark_method.accumulator
            for dark in darks:
                dark.close()
            dark = self.median_dark
            if dark is not None:
                if self.exhaustive_denoise:
                    darks = [dark] + darks
                else:
                    darks = [dark]
                dark.set_raw_image(dark_accum.accum)
            del dark_method

        dark_library = self.dark_library
        bias_library = self.bias_library

        weight_sum = [cvastrophoto.image.ImageAccumulator(numpy.float32)]
        weight_sq_sum = [cvastrophoto.image.ImageAccumulator(numpy.float32)]

        weight_avg = [None]
        weight_istd = [None]

        def safe_reciprocal(x):
            xmin = x.min()
            if xmin <= 0:
                xmask = x > 0
                if xmask.any():
                    x[~xmask] = x[xmask].min()
                else:
                    x[:] = 1
            return numpy.reciprocal(x)

        def weight_normalize(weights):
            if weight_avg[0] is not None:
                weights = weights.astype(numpy.float32, copy=False)
                weights -= weight_avg[0]
                weights *= weight_istd[0] * 4
                bavg = weights <= 0
                weights[bavg] = numpy.reciprocal(1 - weights[bavg])
                weights[~bavg] += 1
            return weights

        def compute_weight_normalization():
            weight_avg[0] = weight_sum[0].average
            weight_istd[0] = safe_reciprocal(
                numpy.sqrt(numpy.maximum(weight_sq_sum[0].average - weight_avg[0], 0)))

            weight_sum[0].reset()
            weight_sq_sum[0].reset()

        def enum_images(phase, iteration, **kw):
            if phase == 0:
                return enum_darks(phase, iteration, **kw)
            else:
                if self.darks is not None and iteration == 0:
                    for dark in self.darks:
                        dark.close()
                return enum_lights(phase, iteration, **kw)

        def enum_lights(phase, iteration, extract=None):
            if weight_avg[0] is None and weight_sum[0].num_images:
                compute_weight_normalization()

            def process_light(light):
                try:
                    # Make sure to get a clean read
                    logger.info("Registering frame %s", light.name)
                    light.close()

                    ldarks = None
                    bias_removed = False
                    if self.denoise and (darks is not None or dark_library is not None or bias_library is not None):
                        if darks is None:
                            if dark_library is not None:
                                ldarks = [dark_library.get_master(dark_library.classify_frame(light.name), raw=light)]
                                ldarks = list(filter(None, ldarks))
                            else:
                                ldarks = []
                            if not ldarks and bias_library is not None:
                                # Can at least remove bias
                                ldarks = [bias_library.get_master(
                                    bias_library.classify_frame(light.name), raw=light)]
                                ldarks = list(filter(None, ldarks))
                        else:
                            ldarks = darks
                        master_bias = self.master_bias
                        if ldarks:
                            if master_bias is None and self.entropy_weighted_denoise and bias_library is not None:
                                master_bias = bias_library.get_master(
                                    bias_library.classify_frame(light.name), raw=light)
                                if master_bias is ldarks[0]:
                                    master_bias = None
                            light.denoise(
                                ldarks,
                                quick=self.quick,
                                master_bias=self.master_bias.rimg.raw_image if self.master_bias is not None else None,
                                entropy_weighted=self.entropy_weighted_denoise,
                                pedestal=self.pedestal)
                            bias_removed = True

                    if bad_pixel_coords is not None:
                        light.repair_bad_pixels(bad_pixel_coords)

                    if ldarks and self.fpn_reduction:
                        darkmean = numpy.mean(ldarks[0].rimg.raw_image)
                        darkstd = numpy.std(ldarks[0].rimg.raw_image)
                        local_bad_pixels = numpy.argwhere(
                            ldarks[0].rimg.raw_image_visible > (darkmean + darkstd / self.fpn_reduction))
                        light.repair_bad_pixels(local_bad_pixels)
                        del local_bad_pixels

                    del ldarks

                    light_basename = os.path.basename(light.name)
                    light_meta = self.extra_metadata.get(light_basename, {}) or {}

                    if self.remove_bias and not bias_removed:
                        light.remove_bias()

                    if self.input_rop is not None:
                        data = self.input_rop.correct(light.rimg.raw_image)
                    else:
                        data = light.rimg.raw_image
                    weights = None
                    if self.tracking is not None:
                        if extract is not None:
                            if self.weight_rop is not None:
                                weights = self.weight_rop.measure_image(data)
                            else:
                                weights = None
                            data = extract(data, weights)
                        if self.pier_flip_rop is not None:
                            data = self.pier_flip_rop.correct(
                                data,
                                light_meta.get('PIERSIDE'),
                                img=light)
                        data = self.tracking.correct(
                            data,
                            img=light)
                        if data is None:
                            logger.warning("Skipping frame %s (rejected by tracking)", light.name)
                            light.close()
                            return light, None, None
                        elif extract is not None and weights is not None and self.normalize_weights:
                            if weights is not None and weight_avg[0] is None:
                                # First pass must compute weight normalization
                                weight_sum[0] += weights
                                weight_sq_sum[0] += numpy.square(weights)
                            elif weights is not None and weight_avg[0] is not None:
                                weights = weight_normalize(weights)

                        if self.weights:
                            explicit_weight = self.weights.get(light_basename, light_meta.get('WEIGHT'))
                            if explicit_weight is not None:
                                if weights is None:
                                    weights = explicit_weight
                                else:
                                    weights *= explicit_weight

                    return light, weights, data
                except Exception:
                    logger.exception("Error registering light %r", light.name)
                    return light, None, None

            added = 0
            rejected = 0

            def processed_lights(lights):
                lights = iter(lights)

                # First frame is the reference frame, so we do it singlethreaded to avoid race conditions
                # while setting the reference frame
                light = next(lights)
                yield process_light(light)

                # Further subs can happen in parallel, limit concurrent jobs to prevent excessive memory usage
                pending = []
                max_tasks = getattr(self.input_pool, '_processes', 4)
                for light in lights:
                    pending.append(self.input_pool.apply_async(process_light, (light,)))
                    if len(pending) >= max_tasks or pending[0].ready():
                        yield pending.pop(0).get()

                # Wait for the last few pending jobs
                for task in pending:
                    yield task.get()

            for i, (light, weights, data) in enumerate(processed_lights(self.lights)):
                if data is None:
                    light.close()
                    rejected += 1
                    continue

                logger.info("Adding frame %s weight %r",
                    light.name,
                    numpy.average(weights) if weights is not None else None)
                logger.debug("Frame data: %r", data)
                yield data
                added += 1

                light.close()

                if progress_callback is not None:
                    progress_callback(phase, iteration, i+1, len(self.lights))
            logger.info("Added %d/%d frames, rejected %d", added, len(self.lights), rejected)

        light_method.set_image_shape(self.lights[0])

        if self.tracking is not None:
            # Must align all extracted components of the image
            self.tracking.set_reference(self.initial_tracking_reference)
            light_method.stack(
                functools.partial(enum_images, extract=light_method.extract_frame),
                extracted=True)
        else:
            light_method.stack(enum_images)

        # Release resources until needed again
        if self.darks is not None:
            for dark in self.darks:
                dark.close()

    @property
    def accumulator(self):
        if hasattr(self, '_accumulator'):
            return self._accumulator
        else:
            return self.light_method_instance.accumulator

    @accumulator.setter
    def accumulator(self, accumulator):
        self._accumulator = accumulator

    @accumulator.deleter
    def accumulator(self):
        if hasattr(self, '_accumulator'):
            del self._accumulator

    @property
    def accum(self):
        return self.accumulator.accum

    def close(self):
        if self.lights is not None:
            for light in self.lights:
                light.close()
        if self.darks is not None:
            for dark in self.darks:
                dark.close()
        self.light_method_instance = None

    def _get_raw_instance(self):
        return self.stacked_image_template
