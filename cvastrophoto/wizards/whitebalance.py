from __future__ import absolute_import

import os.path
import multiprocessing.pool
import numpy
import functools
import random

from cvastrophoto import image

from . import stacking
from .base import BaseWizard
from ..rops.vignette import flats
from ..rops.bias import uniform, localgradient
from ..rops.tracking import grid
from ..rops import compound, scale

import logging

logger = logging.getLogger('cvastrophoto.wizards.whitebalance')

class WhiteBalanceWizard(BaseWizard):

    accumulator = None
    no_auto_scale = True

    # Usually screws color accuracy, but it depends, so user-overrideable
    do_daylight_wb = False

    def __init__(self,
            light_stacker=None, flat_stacker=None,
            light_stacker_class=stacking.StackingWizard, light_stacker_kwargs={},
            flat_stacker_class=stacking.StackingWizard, flat_stacker_kwargs={},
            vignette_class=flats.FlatImageRop,
            debias_class=uniform.UniformBiasRop,
            skyglow_class=localgradient.LocalGradientBiasRop,
            tracking_class=grid.GridTrackingRop,
            pool=None):

        if pool is None:
            self.pool = pool = multiprocessing.pool.ThreadPool()

        if light_stacker is None:
            light_stacker = light_stacker_class(pool=pool, tracking_class=tracking_class, **light_stacker_kwargs)
        if flat_stacker is None:
            flat_stacker_kwargs = flat_stacker_kwargs.copy()
            flat_stacker_kwargs.setdefault('light_method', stacking.MedianStackingMethod)
            flat_stacker = flat_stacker_class(pool=pool, **flat_stacker_kwargs)

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

        if flat_path is not None:
            self.flat_stacker.load_set(base_path, flat_path, dark_flat_path, master_bias=master_bias)
            self.vignette = self.vignette_class(self.flat_stacker.lights[0])
        else:
            self.vignette = None

        self.debias = self.debias_class(self.light_stacker.lights[0])
        self.skyglow = self.skyglow_class(self.light_stacker.lights[0])

        if self.vignette is None:
            rops = (
                self.debias,
            )
        else:
            rops = (
                self.vignette,
                scale.ScaleRop(self.light_stacker.lights[0], 65535, numpy.uint16, 0, 65535),
                self.debias,
            )

        self.light_stacker.input_rop = compound.CompoundRop(
            self.light_stacker.lights[0],
            *rops
        )

        # Since we're de-biasing, correct tracking requires that we re-add it
        # before postprocessing raw images for tracking
        self.light_stacker.tracking.add_bias = True

    def process(self, preview=False, preview_kwargs={}):
        self.process_stacks(preview=preview, preview_kwargs=preview_kwargs)
        self.process_rops()

    def detect_bad_pixels(self,
            include_darks=True, include_lights=True, include_flats=True,
            max_samples_per_set=2, **kw):
        # Produce a sampling containing a representative amount of each set
        # Don't use them all, since lights could overpower darks/flats and cause spurious
        # bad pixels in areas of high contrast
        sets = []
        if include_lights:
            sets.extend([self.light_stacker.lights, self.flat_stacker.lights])
        if include_darks:
            sets.extend([self.light_stacker.darks, self.flat_stacker.darks])
        self.bad_pixel_coords = image.Raw.find_bad_pixels_from_sets(sets, max_samples_per_set=max_samples_per_set, **kw)
        self.light_stacker.bad_pixel_coords = self.bad_pixel_coords
        self.flat_stacker.bad_pixel_coords = self.bad_pixel_coords

    def process_stacks(self, preview=False, preview_kwargs={}):
        if preview:
            preview_callback = functools.partial(self.preview, **preview_kwargs)
        else:
            preview_callback = None

        if self.vignette is not None:
            self.flat_stacker.process()
            self.vignette.set_flat(self.flat_stacker.accum)
            self.flat_stacker.close()

        self.light_stacker.process(
            #flat_accum=self.flat_stacker.accumulator,
            progress_callback=preview_callback)

    def preview(self, phase=None, iteration=None, done=None, total=None,
            preview_path='preview-%(phase)d-%(iteration)d.jpg',
            image_kwargs={}):
        if phase < 1:
            return
        preview_path = preview_path % dict(phase=phase, iteration=iteration)
        self.process_rops(quick=True)
        self.get_image(**image_kwargs).save(preview_path)
        logger.info("Saved preview at %s", preview_path)

    def process_rops(self, quick=False):
        self.accum = self.skyglow.correct(self.light_stacker.accum.copy(), quick=True)

        if self.do_daylight_wb and self.no_auto_scale:
            wb_coeffs = self.debias.raw.rimg.daylight_whitebalance
            if wb_coeffs and all(wb_coeffs[:3]):
                wb_coeffs = numpy.array(wb_coeffs)
                self.accum[:] = self.accum * wb_coeffs[self.debias.raw.rimg.raw_colors]

    def _get_raw_instance(self):
        img = self.light_stacker._get_raw_instance()
        if img.postprocessing_params is not None:
            img.postprocessing_params.no_auto_scale = self.no_auto_scale
        return img
