from __future__ import absolute_import

import os.path
import multiprocessing.pool
import numpy
import functools

from .. import raw

from . import stacking
from .base import BaseWizard
from ..rops.vignette import flats
from ..rops.bias import uniform, localgradient
from ..rops.tracking import centroid
from ..rops import compound, scale

class WhiteBalanceWizard(BaseWizard):

    accumulator = None
    no_auto_scale = True

    # Usually screws color accuracy, but it depends, so user-overrideable
    do_daylight_wb = False

    def __init__(self,
            light_stacker=None, flat_stacker=None, stacker_class=stacking.StackingWizard,
            vignette_class=flats.FlatImageRop,
            debias_class=uniform.UniformBiasRop,
            skyglow_class=localgradient.LocalGradientBiasRop,
            tracking_class=centroid.CentroidTrackingRop,
            pool=None):

        if pool is None:
            pool = multiprocessing.pool.ThreadPool()

        if light_stacker is None:
            light_stacker = stacker_class(pool=pool, tracking_class=tracking_class)
        if flat_stacker is None:
            flat_stacker = stacker_class(pool=pool, light_method=stacking.MedianStackingMethod)

        self.light_stacker = light_stacker
        self.flat_stacker = flat_stacker
        self.vignette_class = vignette_class
        self.debias_class = debias_class
        self.skyglow_class = skyglow_class
        self.tracking_class = tracking_class

    def load_set(self,
            base_path='.',
            light_path='Lights', dark_path='Darks',
            flat_path='Flats', dark_flat_path='Dark Flats',
            master_bias=None):
        self.light_stacker.load_set(base_path, light_path, dark_path, master_bias=master_bias)
        self.flat_stacker.load_set(base_path, flat_path, dark_flat_path, master_bias=master_bias)

        self.vignette = self.vignette_class(self.flat_stacker.lights[0])
        self.debias = self.debias_class(self.light_stacker.lights[0])
        self.skyglow = self.skyglow_class(self.light_stacker.lights[0])

        self.light_stacker.input_rop = compound.CompoundRop(
            self.light_stacker.lights[0],
            self.vignette,
            self.debias,
            scale.ScaleRop(self.light_stacker.lights[0], 65535, numpy.uint16, 0, 65535),
        )

    def process(self, preview=False, preview_path=None):
        self.process_stacks(preview=preview, preview_path=preview_path)
        self.process_rops()

    def process_stacks(self, preview=False, preview_path=None):
        if preview:
            preview_kw = {}
            if preview_path is not None:
                preview_kw['preview_path'] = preview_path
            preview_callback = functools.partial(self.preview, **preview_kw)
        else:
            preview_callback = None

        self.flat_stacker.process()
        self.vignette.set_flat(self.flat_stacker.accum)
        self.flat_stacker.close()

        self.light_stacker.process(
            #flat_accum=self.flat_stacker.accumulator,
            progress_callback=preview_callback)

    def preview(self, done=None, total=None, preview_path='preview.jpg'):
        self.process_rops(quick=True)
        self.get_image().save(preview_path)

    def process_rops(self, quick=False):
        self.accum = self.skyglow.correct(self.light_stacker.accum.copy(), quick=True)

        if self.do_daylight_wb and self.no_auto_scale:
            wb_coeffs = self.debias.raw.rimg.daylight_whitebalance
            if wb_coeffs and all(wb_coeffs[:3]):
                wb_coeffs = numpy.array(wb_coeffs)
                self.accum[:] = self.accum * wb_coeffs[self.debias.raw.rimg.raw_colors]

    def _get_raw_instance(self):
        img = self.light_stacker._get_raw_instance()
        img.postprocessing_params.no_auto_scale = self.no_auto_scale
        return img
