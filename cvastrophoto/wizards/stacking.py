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

    def __init__(self, master_bias=None, copy_frames=False):
        self.master_bias = master_bias
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

    def __init__(self, master_bias=None, copy_frames=False):
        self.light_accum = raw.RawAccumulator()
        super(AverageStackingMethod, self).__init__(master_bias, copy_frames)

    @property
    def accumulator(self):
        return self.light_accum

    def __iadd__(self, image):
        if self.copy_frames and self.light_accum.num_images == 0:
            if isinstance(image, raw.Raw):
                image = image.rimg.raw_image.copy()
        self.light_accum += image
        if self.master_bias is not None:
            if isinstance(image, raw.Raw):
                image = image.rimg.raw_image
            self.light_accum.accum -= numpy.minimum(self.master_bias, image)
        return self


class MedianStackingMethod(BaseStackingMethod):

    def __init__(self, master_bias=None, copy_frames=False):
        self.frames = []
        self.light_accum = None
        super(MedianStackingMethod, self).__init__(master_bias, copy_frames)

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
        if self.master_bias is not None:
            image -= numpy.minimum(self.master_bias, image)
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

    def __init__(self, master_bias=None, copy_frames=False):
        self.final_accumulator = raw.RawAccumulator()
        self.current_average = None
        super(AdaptiveWeightedAverageStackingMethod, self).__init__(master_bias, copy_frames)

    def extract_frame(self, frame, weights=None):
        if isinstance(frame, raw.Raw):
            frame = frame.rimg.raw_image
        frame = frame.astype(numpy.float32)
        if self.phase == 0:
            return [frame, frame]
        elif self.phase >= 1:
            weight = numpy.square(frame)
            weight *= self.sigmas
            weight = numpy.clip(weight, 0, 1, out=weight)
            if weights is not None:
                weight *= weights
            return [frame, weight]

    def start_phase(self, phase, iteration):
        if phase == 1:
            if iteration:
                self.finish_phase()
            else:
                self.finish_sigmas()
        elif phase == 2:
            self.finish_phase()
        self.weights = raw.RawAccumulator(numpy.float32)
        self.light_accum = raw.RawAccumulator(numpy.float32)
        super(AdaptiveWeightedAverageStackingMethod, self).start_phase(phase, iteration)

    def finish_phase(self):
        self.final_accumulator.accum = self.current_average = self.estimate_average()
        self.final_accumulator.accum *= self.light_accum.num_images
        self.final_accumulator.num_images = self.light_accum.num_images
        logger.info("Finished phase %r", self.phase)
        super(AdaptiveWeightedAverageStackingMethod, self).finish()

    def finish(self):
        self.finish_phase()
        self.weights = self.light_accum = self.sigmas = None

    def finish_sigmas(self):
        self.sigmas = sigmas = numpy.clip(
            self.light_accum.average - numpy.square(self.weights.average),
            0, None
        ) * (
            float(max(1, self.light_accum.num_images))
            / float(max(1, self.light_accum.num_images - 1))
        )
        if sigmas.min() <= 0:
            # Fix singularities
            sigmas[sigmas <= 0] = sigmas[sigmas > 0].min() * 0.707
        self.sigmas = numpy.reciprocal(sigmas, out=sigmas)

    def estimate_average(self):
        min_weight = self.weights.accum.min()
        if min_weight <= 0:
            # Must plug holes
            holes = self.weights.accum <= 0
            weights = self.weights.accum.copy()
            weights[holes] = 1
        else:
            holes = None
            weights = self.weights.accum

        return self.light_accum.accum / weights

    @property
    def accumulator(self):
        if self.final_accumulator.num_images == 0:
            self.final_accumulator.accum = self.estimate_average()
            self.final_accumulator.accum *= self.light_accum.num_images
            self.final_accumulator.num_images = self.light_accum.num_images
        return self.final_accumulator

    def __iadd__(self, image):
        image, weight = image

        if self.current_average is not None:
            # Apply lorentzian-like factor post-alignment
            residue = image - self.current_average
            residue = numpy.square(residue, out=residue)
            residue *= weight
            residue += 1
            residue = numpy.reciprocal(residue, out=residue)
            weight *= residue
            del residue

        self.light_accum += image * weight
        self.weights += weight
        if self.master_bias is not None:
            if isinstance(image, raw.Raw):
                image = image.rimg.raw_image
            self.light_accum.accum -= numpy.minimum(self.master_bias, image)
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

    def load_set(self, base_path='.', light_path='Lights', dark_path='Darks', master_bias=None):
        self.lights = raw.Raw.open_all(os.path.join(base_path, light_path), default_pool=self.pool)

        if self.tracking_class is not None:
            self.tracking = self.tracking_class(self.lights[0])
        else:
            self.tracking = None

        if self.denoise and dark_path is not None:
            self.darks = raw.Raw.open_all(os.path.join(base_path, dark_path), default_pool=self.pool)
        else:
            self.darks = None

        if master_bias is not None:
            if master_bias.lower().endswith('.tif'):
                sizes = self.lights[0].rimg.sizes
                master_bias = numpy.array(PIL.Image.open(master_bias))
                self.master_bias = numpy.zeros((sizes.raw_height, sizes.raw_width), numpy.uint16)
                self.master_bias[
                    sizes.top_margin:sizes.top_margin+sizes.iheight,
                    sizes.left_margin:sizes.left_margin+sizes.iwidth] = master_bias
            else:
                self.master_bias = raw.Raw(master_bias).raw_image.copy()
        else:
            self.master_bias = None

        self.lights[0].postprocessing_params.fbdd_noiserd = self.fbdd_noiserd

    def process(self, flat_accum=None, progress_callback=None):
        self.light_method_instance = light_method = self.light_method(self.master_bias, True)

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
            dark_method = self.dark_method(self.master_bias)
            dark_method.stack(enum_darks)
            dark_accum = dark_method.accumulator
            for dark in darks:
                dark.close()
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
                        darks,
                        quick=self.quick,
                        entropy_weighted=self.entropy_weighted_denoise)
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
