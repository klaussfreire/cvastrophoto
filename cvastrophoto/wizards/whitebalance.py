from __future__ import absolute_import

import os.path
import multiprocessing.pool
import numpy

from .. import raw

from . import stacking
from .base import BaseWizard
from ..rops.vignette import flats
from ..rops.bias import uniform
from ..rops.tracking import centroid

class WhiteBalanceWizard(BaseWizard):

    accumulator = None
    no_auto_scale = True
    do_daylight_wb = True

    def __init__(self,
            light_stacker=None, flat_stacker=None, stacker_class=stacking.StackingWizard,
            vignette_class=flats.FlatImageRop,
            debias_class=uniform.UniformBiasRop,
            tracking_class=centroid.CentroidTrackingRop,
            pool=None):

        if pool is None:
            pool = multiprocessing.pool.ThreadPool()

        if light_stacker is None:
            light_stacker = stacker_class(pool=pool, tracking_class=tracking_class)
        if flat_stacker is None:
            flat_stacker = stacker_class(pool=pool)

        self.light_stacker = light_stacker
        self.flat_stacker = flat_stacker
        self.vignette_class = vignette_class
        self.debias_class = debias_class
        self.tracking_class = tracking_class

    def load_set(self,
            base_path='.',
            light_path='Lights', dark_path='Darks',
            flat_path='Flats', dark_flat_path='Dark Flats'):
        self.light_stacker.load_set(base_path, light_path, dark_path)
        self.flat_stacker.load_set(base_path, flat_path, dark_flat_path)

        self.vignette = self.vignette_class(self.flat_stacker.lights[0])
        self.debias = self.debias_class(self.light_stacker.lights[0])

    def process(self):
        self.process_stacks()
        self.process_rops()

    def process_stacks(self):
        self.light_stacker.process()
        self.flat_stacker.process()
        self.vignette.flat = self.flat_stacker.accum

    def process_rops(self):
        self.accum = self.debias.correct(self.vignette.correct(self.light_stacker.accum))

        if self.do_daylight_wb and self.no_auto_scale:
            wb_coeffs = self.debias.raw.rimg.daylight_whitebalance
            if wb_coeffs and all(wb_coeffs[:3]):
                wb_coeffs = numpy.array(wb_coeffs)
                self.accum *= wb_coeffs[self.debias.raw.rimg.raw_colors]

    def _get_raw_instance(self):
        img = self.light_stacker._get_raw_instance()
        img.postprocessing_params.no_auto_scale = self.no_auto_scale
        return img
