from __future__ import absolute_import

import os.path
import multiprocessing.pool
import numpy
import functools
import PIL.Image

from .base import BaseWizard
from .. import raw
from ..rops.denoise import darkspectrum

import logging

logger = logging.getLogger('cvastrophoto.wizards.stacking')

class BaseStackingMethod(object):

    phases = [
        # Phase number - iterations
        (2, 1)
    ]

    def __init__(self, copy_frames=False):
        self.copy_frames = copy_frames
        self.phase = 0

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
        self.light_accum = raw.RawAccumulator()
        super(AverageStackingMethod, self).__init__(copy_frames)

    @property
    def accumulator(self):
        return self.light_accum

    def __iadd__(self, image):
        if self.copy_frames and self.light_accum.num_images == 0:
            if isinstance(image, raw.Raw):
                image = image.rimg.raw_image.copy()
        self.light_accum += image
        return self


class MaxStackingMethod(BaseStackingMethod):

    def __init__(self, copy_frames=False):
        self.light_accum = raw.RawAccumulator()
        super(MaxStackingMethod, self).__init__(copy_frames)

    @property
    def accumulator(self):
        return self.light_accum

    def __iadd__(self, image):
        if isinstance(image, raw.Raw):
            image = image.rimg.raw_image
        if self.light_accum.num_images == 0:
            self.light_accum += image.copy()
        else:
            numpy.maximum(self.light_accum.accum, image, out=self.light_accum.accum)
        return self


class MedianStackingMethod(BaseStackingMethod):

    def __init__(self, copy_frames=False):
        self.frames = []
        self.light_accum = None
        super(MedianStackingMethod, self).__init__(copy_frames)

    def finish(self):
        self.update_accum()

    def update_accum(self):
        self.light_accum = raw.RawAccumulator()
        self.light_accum += numpy.median(self.frames, axis=0).astype(self.frames[0].dtype)

    @property
    def accumulator(self):
        if self.light_accum is None:
            self.update_accum()
        return self.light_accum

    def __iadd__(self, image):
        if isinstance(image, raw.Raw):
            image = image.rimg.raw_image
        if self.copy_frames:
            image = image.copy()
        self.frames.append(image)
        self.light_accum = None
        return self


class AdaptiveWeightedAverageStackingMethod(BaseStackingMethod):

    phases = [
        # Dark phase (0)
        (0, 1),

        # Light phase 1, 4 iterations
        (1, 4),

        # Final light phase
        (2, 1),
    ]

    def __init__(self, copy_frames=False):
        self.final_accumulator = raw.RawAccumulator()
        self.current_average = None
        super(AdaptiveWeightedAverageStackingMethod, self).__init__(copy_frames)

    def extract_frame(self, frame, weights=None):
        if isinstance(frame, raw.Raw):
            frame = frame.rimg.raw_image
        frame = frame.astype(numpy.float32)
        if self.phase == 0:
            return [frame, frame, None]
        elif self.phase >= 1:
            if self.current_average is not None:
                weight = numpy.square(frame)
                weight *= self.darkvar
                weight = numpy.clip(weight, 1, None, out=weight)
                weight = numpy.reciprocal(weight, out=weight)
            else:
                # Must first get a regular average
                weight = None
            return [frame, weight, weights]

    def start_phase(self, phase, iteration):
        if phase == 1:
            if iteration:
                self.finish_phase()
                self.invvar = self.estimate_variance(self.light_accum, self.light2_accum, self.weights)
            else:
                self.darkvar = self.estimate_variance(self.weights, self.light_accum)
        elif phase == 2:
            self.finish_phase()
        self.weights = raw.RawAccumulator(numpy.float32)
        self.light_accum = raw.RawAccumulator(numpy.float32)
        self.light2_accum = raw.RawAccumulator(numpy.float32)
        super(AdaptiveWeightedAverageStackingMethod, self).start_phase(phase, iteration)

    def finish_phase(self):
        self.final_accumulator.accum = self.current_average = self.estimate_average()
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
            invvar = numpy.clip(invvar, invvar[invvar > 0].min() * 0.707, None, out=invvar)
        invvar = numpy.reciprocal(invvar, out=invvar)
        return invvar

    def estimate_average(self, accum=None, weights_accum=None):
        if accum is None:
            accum = self.light_accum
        if weights_accum is None:
            weights_accum = self.weights

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
            weight[:] = self.invvar
            residue = image - self.current_average
            residue = numpy.square(residue, out=residue)
            residue *= self.invvar
            residue += 1
            weight /= residue

            if imgweight is not None:
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

        if weight is not None:
            self.weights += weight

        # Mark final accumulator as dirty so previews recompute the final average
        self.final_accumulator.num_images = 0

        return self


class StackingWizard(BaseWizard):

    debias = True
    debias_amount = 1
    denoise_amount = 1
    save_tracks = False

    def __init__(self, pool=None,
            denoise=True, quick=True, entropy_weighted_denoise=True,
            fbdd_noiserd=2,
            tracking_class=None, input_rop=None,
            light_method=AverageStackingMethod,
            dark_method=MedianStackingMethod):
        if pool is None:
            pool = multiprocessing.pool.ThreadPool()
        self.pool = pool
        self.denoise = denoise
        self.entropy_weighted_denoise = entropy_weighted_denoise
        self.quick = quick
        self.fbdd_noiserd = fbdd_noiserd
        self.tracking_class = tracking_class
        self.input_rop = input_rop
        self.light_method = light_method
        self.dark_method = dark_method
        self.bad_pixel_coords = None

    def load_set(self, base_path='.', light_path='Lights', dark_path='Darks', master_bias=None, bias_shift=2):
        self.lights = raw.Raw.open_all(os.path.join(base_path, light_path), default_pool=self.pool)

        if self.tracking_class is not None:
            self.tracking = self.tracking_class(self.lights[0])
        else:
            self.tracking = None

        if self.denoise and dark_path is not None:
            self.darks = raw.Raw.open_all(os.path.join(base_path, dark_path), default_pool=self.pool)
            if self.darks:
                self.median_dark = raw.Raw(self.darks[0].name)
            else:
                self.median_dark = None
        else:
            self.darks = None

        if master_bias is not None:
            if master_bias.lower().endswith('.tif'):
                sizes = self.lights[0].rimg.sizes
                raw_master_bias = numpy.zeros((sizes.raw_height, sizes.raw_width), numpy.uint16)
                raw_master_bias[
                    sizes.top_margin:sizes.top_margin+sizes.iheight,
                    sizes.left_margin:sizes.left_margin+sizes.iwidth] = numpy.array(PIL.Image.open(master_bias))
                self.master_bias = raw.Raw(self.lights[0].name)
                self.master_bias.set_raw_image(raw_master_bias)
                self.master_bias.name = master_bias
            else:
                self.master_bias = raw.Raw(master_bias)
            raw_master_bias = self.master_bias.rimg.raw_image
            raw_master_bias >>= bias_shift
        else:
            self.master_bias = None

        self.lights[0].postprocessing_params.fbdd_noiserd = self.fbdd_noiserd

    def process(self, flat_accum=None, progress_callback=None):
        self.light_method_instance = light_method = self.light_method(True)

        bad_pixel_coords = self.bad_pixel_coords

        def enum_darks(phase, iteration, extract=None):
            if self.darks is not None:
                for dark in self.darks:
                    logger.info("Adding dark frame %s", dark.name)
                    if bad_pixel_coords is not None and not dark.is_open:
                        dark.repair_bad_pixels(bad_pixel_coords)
                    if self.denoise and self.master_bias is not None:
                        dark.denoise(
                            [self.master_bias],
                            quick=self.quick,
                            entropy_weighted=self.entropy_weighted_denoise)
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
                darks = [dark]
                dark.set_raw_image(dark_accum.accum)
            del dark_method

        def enum_images(phase, iteration, **kw):
            if phase == 0:
                return enum_darks(phase, iteration, **kw)
            else:
                if self.darks is not None and iteration == 0:
                    for dark in self.darks:
                        dark.close()
                return enum_lights(phase, iteration, **kw)

        def enum_lights(phase, iteration, extract=None):
            for i, light in enumerate(self.lights):
                # Make sure to get a clean read
                light.close()

                if bad_pixel_coords is not None:
                    light.repair_bad_pixels(bad_pixel_coords)

                if self.debias and flat_accum is not None:
                    light.set_raw_image(
                        darkspectrum.denoise(
                            light.rimg.raw_image, 1,
                            flat_accum.accum, flat_accum.num_images,
                            light,
                            equalize_power=True,
                            debias=True,
                            amount=self.denoise_amount,
                            debias_amount=self.debias_amount,
                        )
                    )
                if self.denoise and darks is not None:
                    light.denoise(
                        darks + filter(None, [self.master_bias]),
                        quick=self.quick,
                        entropy_weighted=self.entropy_weighted_denoise,
                        stop_at_unity=False)
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

                logger.info("Adding frame %s", light.name)
                logger.debug("Frame data: %r", data)
                yield data

                light.close()

                if progress_callback is not None:
                    progress_callback(phase, iteration, i+1, len(self.lights))

        if self.tracking is not None:
            # Must align all extracted components of the image
            self.tracking.set_reference(None)
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
        return self.lights[0]
