from __future__ import absolute_import

import os.path
import multiprocessing.pool
import numpy
import functools
import PIL.Image
import scipy.ndimage.filters

from .base import BaseWizard
import cvastrophoto.image

import logging

logger = logging.getLogger(__name__)

class BaseStackingMethod(object):

    phases = [
        # Phase number - iterations
        (2, 1)
    ]

    def __init__(self, copy_frames=False):
        self.copy_frames = copy_frames
        self.phase = 0

    def set_image_shape(self, image):
        # Touch to make sure it's initialized
        image.postprocessed

        # Get raw pattern information
        self.raw_pattern = image.rimg.raw_pattern
        self.raw_colors = image.rimg.raw_colors

        self.rmask = self.raw_pattern == 0
        self.gmask = self.raw_pattern == 1
        self.bmask = self.raw_pattern == 2

        self.rmask_image = self.raw_colors == 0
        self.gmask_image = self.raw_colors == 1
        self.bmask_image = self.raw_colors == 2

        self.raw_sizes = image.rimg.sizes

    def get_tracking_image(self, image):
        return image

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

    def __init__(self, copy_frames=False):
        self.light_accum = cvastrophoto.image.ImageAccumulator()
        super(AverageStackingMethod, self).__init__(copy_frames)

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

    def __init__(self, copy_frames=False):
        self.light_accum = cvastrophoto.image.ImageAccumulator()
        super(MaxStackingMethod, self).__init__(copy_frames)

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

    def __init__(self, copy_frames=False):
        self.light_accum = cvastrophoto.image.ImageAccumulator()
        super(MinStackingMethod, self).__init__(copy_frames)

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


class MedianStackingMethod(BaseStackingMethod):

    def __init__(self, copy_frames=False):
        self.frames = []
        self.light_accum = None
        super(MedianStackingMethod, self).__init__(copy_frames)

    def finish(self):
        self.update_accum()

    def update_accum(self):
        self.light_accum = cvastrophoto.image.ImageAccumulator()
        self.light_accum += numpy.median(self.frames, axis=0).astype(self.frames[0].dtype)

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


class AdaptiveWeightedAverageStackingMethod(BaseStackingMethod):

    kappa_sq = 4

    phases = [
        # Dark phase (0)
        # (0, 1),

        # Outlier trimming phase, 2 iterations
        (1, 2),

        # Final light phase
        (2, 1),
    ]

    def __init__(self, copy_frames=False):
        self.final_accumulator = cvastrophoto.image.ImageAccumulator()
        self.current_average = None
        super(AdaptiveWeightedAverageStackingMethod, self).__init__(copy_frames)

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

        if imgweight is not None:
            if weight is None:
                weight = imgweight
            else:
                weight *= imgweight

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

        # Mark final accumulator as dirty so previews recompute the final average
        self.final_accumulator.num_images = 0

        return self


class DrizzleStackingMethod(AdaptiveWeightedAverageStackingMethod):
    # WIP, Don't use
    add_bias = True

    # Weight for "fake" color pixels, by phase number
    hole_weight = [1.0, 1.0, 0.5, 0.5]

    def get_tracking_image(self, image):
        # Drizzle into an RGB image
        self.raw = image.dup()
        rimg = image.rimg
        raw_image = rimg.raw_image
        shape = (raw_image.shape[0], raw_image.shape[1], 3)
        margins = (
            rimg.sizes.top_margin,
            rimg.sizes.left_margin,
            shape[0] - rimg.sizes.top_margin - rimg.sizes.iheight,
            shape[1] - rimg.sizes.left_margin - rimg.sizes.iwidth,
        )
        rgbdata = numpy.empty(shape, dtype=raw_image.dtype)
        img = cvastrophoto.image.rgb.RGB(
            None,
            img=rgbdata, margins=margins, default_pool=image.default_pool,
            daylight_whitebalance=rimg.daylight_whitebalance)
        img.lazy_rgb_xyz_matrix = rimg.rgb_xyz_matrix
        return img

    def extract_frame(self, frame, weights=None):
        if isinstance(frame, cvastrophoto.image.Image):
            frame = frame.rimg.raw_image

        self.rgbshape = rgbshape = frame.shape + (3,)
        self.rawshape = rawshape = (rgbshape[0], rgbshape[1] * rgbshape[2])

        # Masked RGB images for accumulation
        rgbimage = numpy.zeros(rgbshape, dtype=frame.dtype)

        masks = (self.rmask_image, self.gmask_image, self.bmask_image)
        for c, mask in enumerate(masks):
            # Create mono image and channel mask
            cimage = rgbimage[:,:,c]
            cimage[mask] = frame[mask]

            # Interpolate between channel samples
            a = scipy.ndimage.filters.uniform_filter(cimage.astype(numpy.float32), self.raw_pattern.shape)
            w = scipy.ndimage.filters.uniform_filter(mask.astype(numpy.float32), self.raw_pattern.shape)
            w = numpy.clip(w, 0.001, None, out=w)
            a /= w
            cimage[~mask] = a[~mask]
            del a, w

        rgbimage, rgbweight, rgbimgweight = super(DrizzleStackingMethod,
            self).extract_frame(rgbimage.reshape(rawshape), weights)

        # Compute masked weights
        rgbimage = rgbimage.reshape(rgbshape)
        if rgbweight is not None:
            rgbweight = rgbweight.reshape(rgbshape)
        if rgbimgweight is not None:
            rgbimgweight = rgbimgweight.reshape(rgbshape)

        rvimgweight = numpy.zeros(rgbshape, dtype=numpy.float32)

        imgmask = None
        for c, mask in enumerate(masks):
            if rgbimgweight is not None:
                cimgweight = rgbimgweight[:,:,c]
            else:
                cimgweight = None

            if self.phase > 0:
                imgmask = mask.astype(numpy.float32)
                imgmask[~mask] = self.hole_weight[self.phase]  # Just to avoid holes
                if cimgweight is not None:
                    imgmask *= cimgweight
                cimgweight = imgmask

            if cimgweight is not None:
                rvimgweight[:,:,c] = cimgweight

        del rgbimgweight, cimgweight, imgmask

        # Extract debayered image to use for alignment
        rsizes = self.raw.rimg.sizes
        luma = self.raw.luma_image(frame, renormalize=True)
        rvluma = numpy.empty(rgbimage.shape, dtype=frame.dtype)
        rvluma[:,:,0] = rvluma[:,:,1] = rvluma[:,:,2] = luma
        del luma

        if self.phase >= 1:
            # Use rough raw luma for border, but actual debayered image for the
            # visible area. We want the extra precision proper debayer gives.
            self.raw.set_raw_image(frame, add_bias=self.add_bias)
            rvluma[
                rsizes.top_margin:rsizes.top_margin+rsizes.iheight,
                rsizes.left_margin:rsizes.left_margin+rsizes.iwidth] = self.raw.postprocessed

        # Reshape into patterend RGB
        rvluma = rvluma.reshape(rawshape)
        rvframe = rgbimage.reshape(rawshape)
        rvimgweight = rvimgweight.reshape(rawshape)
        rvweight = rgbweight.reshape(rawshape) if rgbweight is not None else None

        return [rvluma, rvframe, rvweight, rvimgweight]

    def __iadd__(self, image):
        return super(DrizzleStackingMethod, self).__iadd__(image[1:4])


class InterleaveStackingMethod(DrizzleStackingMethod):
    # Weight for "fake" color pixels, by phase number
    hole_weight = [1.0, 0.5, 0.00001, 0.00001]


class StackingWizard(BaseWizard):

    denoise_amount = 1
    save_tracks = False

    def __init__(self, pool=None,
            denoise=True, quick=True, entropy_weighted_denoise=False, exhaustive_denoise=False,
            fbdd_noiserd=2, fpn_reduction=True,
            tracking_class=None, input_rop=None,
            light_method=AverageStackingMethod,
            dark_method=MedianStackingMethod):
        if pool is None:
            pool = multiprocessing.pool.ThreadPool()
        self.pool = pool
        self.denoise = denoise
        self.fpn_reduction = fpn_reduction
        self.entropy_weighted_denoise = entropy_weighted_denoise
        self.quick = quick
        self.exhaustive_denoise = exhaustive_denoise
        self.fbdd_noiserd = fbdd_noiserd
        self.tracking_class = tracking_class
        self.input_rop = input_rop
        self.light_method = light_method
        self.dark_method = dark_method
        self.bad_pixel_coords = None
        self.lights = None
        self.initial_tracking_reference = None
        self.tracking = None

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
            bias_library=None):
        if light_files:
            self.lights = [
                cvastrophoto.image.Image.open(path, default_pool=self.pool)
                for path in light_files
            ]
        else:
            self.lights = cvastrophoto.image.Image.open_all(
                os.path.join(base_path, light_path), default_pool=self.pool)

        self.light_method_instance = light_method = self.light_method(True)

        light_method.set_image_shape(self.lights[0])
        self.stacked_image_template = light_method.get_tracking_image(self.lights[0])

        if self.tracking_class is not None:
            self.tracking = self.tracking_class(self.stacked_image_template)
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

        # For cameras that have a nonzero black level, we have to remove it
        # for denoising calculations or things don't work right
        if darks:
            for dark in darks:
                dark.remove_bias()

        dark_library = self.dark_library
        bias_library = self.bias_library

        def enum_images(phase, iteration, **kw):
            if phase == 0:
                return enum_darks(phase, iteration, **kw)
            else:
                if self.darks is not None and iteration == 0:
                    for dark in self.darks:
                        dark.close()
                return enum_lights(phase, iteration, **kw)

        def enum_lights(phase, iteration, extract=None):
            added = 0
            rejected = 0
            for i, light in enumerate(self.lights):
                # Make sure to get a clean read
                logger.info("Registering frame %s", light.name)
                light.close()

                ldarks = None
                if self.denoise and (darks is not None or dark_library is not None or bias_library is not None):
                    if darks is None:
                        if dark_library is not None:
                            ldarks = [dark_library.get_master(dark_library.classify_frame(light.name), raw=light)]
                            ldarks = filter(None, ldarks)
                        else:
                            ldarks = []
                        if not ldarks and bias_library is not None:
                            # Can at least remove bias
                            ldarks = [bias_library.get_master(
                                bias_library.classify_frame(light.name), raw=light)]
                            ldarks = filter(None, ldarks)
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
                            entropy_weighted=self.entropy_weighted_denoise)

                if bad_pixel_coords is not None:
                    light.repair_bad_pixels(bad_pixel_coords)

                if ldarks is not None and self.fpn_reduction:
                    darkmean = numpy.mean(ldarks[0].rimg.raw_image)
                    darkstd = numpy.std(ldarks[0].rimg.raw_image)
                    local_bad_pixels = numpy.argwhere(ldarks[0].rimg.raw_image_visible > (darkmean + darkstd))
                    light.repair_bad_pixels(local_bad_pixels)
                    del local_bad_pixels

                del ldarks

                light.remove_bias()

                if self.input_rop is not None:
                    data = self.input_rop.correct(light.rimg.raw_image)
                else:
                    data = light.rimg.raw_image
                if self.tracking is not None:
                    if extract is not None:
                        data = extract(data)
                    data = self.tracking.correct(
                        data,
                        img=light,
                        save_tracks=self.save_tracks)
                    if data is None:
                        logger.warning("Skipping frame %s (rejected by tracking)", light.name)
                        light.close()
                        rejected += 1
                        continue

                logger.info("Adding frame %s", light.name)
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
        return self.light_method_instance.accumulator

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
