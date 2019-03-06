from __future__ import absolute_import

import os.path
import multiprocessing.pool
import numpy
import PIL.Image

from .base import BaseWizard
from .. import raw
from ..rops.denoise import darkspectrum


class BaseStackingMethod(object):

    def __init__(self, master_bias=None):
        self.master_bias = master_bias

    @property
    def accumulator(self):
        raise NotImplementedError

    def __iadd__(self, image):
        raise NotImplementedError


class AverageStackingMethod(BaseStackingMethod):

    def __init__(self, master_bias=None):
        self.light_accum = raw.RawAccumulator()
        super(AverageStackingMethod, self).__init__(master_bias)

    @property
    def accumulator(self):
        return self.light_accum

    def __iadd__(self, image):
        self.light_accum += image
        if self.master_bias is not None:
            if isinstance(image, raw.Raw):
                image = image.rimg.raw_image
            self.light_accum.accum -= numpy.minimum(self.master_bias, image)
        return self


class MedianStackingMethod(BaseStackingMethod):

    def __init__(self, master_bias=None):
        self.frames = []
        self.light_accum = None
        super(MedianStackingMethod, self).__init__(master_bias)

    @property
    def accumulator(self):
        if self.light_accum is None:
            self.light_accum = raw.RawAccumulator()
            self.light_accum += numpy.median(self.frames, axis=0).astype(self.frames[0].dtype)
        return self.light_accum

    def __iadd__(self, image):
        if isinstance(image, raw.Raw):
            image = image.rimg.raw_image
        if self.master_bias is not None:
            image -= numpy.minimum(self.master_bias, image)
        self.frames.append(image)
        self.light_accum = None
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
        self.light_method_instance = light_method = self.light_method(self.master_bias)

        darks = self.darks
        if self.denoise and darks is not None:
            # Stack dark frames
            dark_method = self.dark_method(self.master_bias)
            for dark in darks:
                dark_method += dark
            dark_accum = dark_method.accumulator
            for dark in darks:
                dark.close()
            darks = [dark]
            dark.set_raw_image(dark_accum.accum)
            del dark_method

        if self.tracking is not None:
            self.tracking.set_reference(None)

        for i, light in enumerate(self.lights):
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
                data = self.tracking.correct(
                    data,
                    img=light,
                    save_tracks=self.save_tracks)

            light_method += data.copy()
            light.close()

            if progress_callback is not None:
                progress_callback(i+1, len(self.lights))

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
