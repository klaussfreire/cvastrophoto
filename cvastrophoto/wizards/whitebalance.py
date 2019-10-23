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
    preview_every_frames = 0
    preview_every_factor = 1.25

    # Usually necessary for good accuracy, but it depends on the camera, so user-overrideable
    do_daylight_wb = True

    WB_SETS = {
        'cls': (1, 0.8, 1.1, 1),
    }

    def __init__(self,
            light_stacker=None, flat_stacker=None,
            light_stacker_class=stacking.StackingWizard, light_stacker_kwargs=dict(
                light_method=stacking.AdaptiveWeightedAverageStackingMethod,
            ),
            flat_stacker_class=stacking.StackingWizard, flat_stacker_kwargs={},
            vignette_class=flats.FlatImageRop,
            debias_class=None,
            skyglow_class=localgradient.LocalGradientBiasRop,
            frame_skyglow_class=None,
            tracking_class=grid.GridTrackingRop,
            tracking_2phase=False,
            dither=False,
            pool=None):

        if pool is None:
            self.pool = pool = multiprocessing.pool.ThreadPool()

        if tracking_2phase:
            tracking_factory = lambda rimg : compound.CompoundRop(
                rimg,
                tracking_class(rimg, median_shift_limit=16),
                tracking_class(rimg),
            )
        else:
            tracking_factory = tracking_class

        if light_stacker is None:
            light_stacker = light_stacker_class(pool=pool, tracking_class=tracking_factory, **light_stacker_kwargs)
        if flat_stacker is None:
            flat_stacker_kwargs = flat_stacker_kwargs.copy()
            flat_stacker_kwargs.setdefault('light_method', stacking.MedianStackingMethod)
            flat_stacker = flat_stacker_class(pool=pool, **flat_stacker_kwargs)

        self.light_stacker = light_stacker
        self.flat_stacker = flat_stacker
        self.vignette_class = vignette_class
        self.debias_class = debias_class
        self.skyglow_class = skyglow_class
        self.frame_skyglow_class = frame_skyglow_class
        self.tracking_class = tracking_class
        self.tracking_2phase = tracking_2phase
        self.dither = dither

        self._reset_preview()

    def _reset_preview(self):
        self._last_preview_iteration = None
        self._last_preview_phase = None
        self._last_preview_done = None

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

        if self.debias_class is not None:
            self.debias = self.debias_class(self.light_stacker.lights[0])
        else:
            self.debias = None

        self.skyglow = self.skyglow_class(self.light_stacker.lights[0])
        self.frame_skyglow = (
            self.frame_skyglow_class(self.light_stacker.lights[0])
            if self.frame_skyglow_class is not None
            else None
        )

        rops = []
        if self.vignette is not None:
            rops.extend((
                self.vignette,
                scale.ScaleRop(self.light_stacker.lights[0], 65535, numpy.uint16, 0, 65535),
            ))

        if self.debias is not None:
            rops.append(self.debias)

        if self.frame_skyglow is not None:
            rops.append(self.frame_skyglow)

        if rops:
            self.light_stacker.input_rop = compound.CompoundRop(
                self.light_stacker.lights[0],
                *rops
            )

        # Since we're de-biasing, correct tracking requires that we re-add it
        # before postprocessing raw images for tracking
        self.light_stacker.tracking.add_bias = True

    def get_state(self):
        return dict(
            light_stacker=self.light_stacker.get_state(),
            flat_stacker=self.flat_stacker.get_state() if self.flat_stacker is not None else None,
        )

    def _load_state(self, state):
        self.light_stacker.load_state(state['light_stacker'])
        if self.flat_stacker is not None:
            self.flat_stacker.load_state(state['flat_stacker'])

    def process(self, preview=False, preview_kwargs={}, rops_kwargs={}):
        self._reset_preview()
        preview_kwargs = preview_kwargs.copy()
        preview_kwargs.setdefault('rops_kwargs', rops_kwargs)
        self.process_stacks(preview=preview, preview_kwargs=preview_kwargs)
        self.process_rops(**rops_kwargs)

    def detect_bad_pixels(self,
            include_darks=True, include_lights=True, include_flats=True,
            max_samples_per_set=2, **kw):
        # Produce a sampling containing a representative amount of each set
        # Don't use them all, since lights could overpower darks/flats and cause spurious
        # bad pixels in areas of high contrast
        sets = []
        if include_lights:
            if isinstance(include_lights, (int, bool)):
                sets.extend([self.light_stacker.lights, self.flat_stacker.lights])
            else:
                sets.extend(include_lights)
        if include_darks:
            if isinstance(include_darks, (int, bool)):
                sets.extend([self.light_stacker.darks, self.flat_stacker.darks])
            else:
                sets.extend(include_darks)
        sets = filter(None, sets)
        self.bad_pixel_coords = image.Raw.find_bad_pixels_from_sets(sets, max_samples_per_set=max_samples_per_set, **kw)
        self.light_stacker.bad_pixel_coords = self.bad_pixel_coords
        self.flat_stacker.bad_pixel_coords = self.bad_pixel_coords

        # Close all resources
        for imgs in sets:
            for img in imgs:
                img.close()

    def process_stacks(self, preview=False, preview_kwargs={}):
        if preview:
            every_frame = preview_kwargs.pop('every_frame', False)
            if every_frame:
                preview_callback = self.preview
            else:
                preview_callback = self._fast_preview
            preview_callback = functools.partial(preview_callback, **preview_kwargs)
        else:
            preview_callback = self._log_progress

        if self.vignette is not None:
            self.flat_stacker.process()
            self.vignette.set_flat(self.flat_stacker.accumulator.average)
            self.flat_stacker.close()

        self.light_stacker.process(
            #flat_accum=self.flat_stacker.accumulator,
            progress_callback=preview_callback)

    def _log_progress(self, phase=None, iteration=None, done=None, total=None, **kw):
        logger.info("Done %s/%s at phase %s iteration %s", done, total, phase, iteration)

    def _fast_preview(self, phase=None, iteration=None, done=None, total=None, **kw):
        self._log_progress(phase, iteration, done, total, **kw)

        frames = self.preview_every_frames
        factor = self.preview_every_factor

        if (self._last_preview_iteration == iteration
                and self._last_preview_phase == phase
                and ((self._last_preview_done + frames) * factor) > done
                and done < (total - 1)):
            return

        self._last_preview_iteration = iteration
        self._last_preview_phase = phase
        self._last_preview_done = done

        return self.preview(
            phase=phase,
            iteration=iteration,
            done=done,
            total=total,
            **kw)

    def preview(self, phase=None, iteration=None, done=None, total=None, quick=True,
            preview_path='preview-%(phase)d-%(iteration)d.jpg',
            image_kwargs={}, rops_kwargs={}):
        if phase < 1:
            return
        preview_path = preview_path % dict(
            phase=phase,
            iteration=iteration,
            done=done,
            total=total,
        )
        self.process_rops(quick=quick, **rops_kwargs)
        self.get_image(**image_kwargs).save(preview_path)
        logger.info("Saved preview at %s", preview_path)

    def process_rops(self, quick=False, extra_wb=None):
        self.accum_prewb = self.accum = self.skyglow.correct(self.light_stacker.accum.copy(), quick=quick)
        self.process_wb(extra_wb=extra_wb)

    def process_wb(self, extra_wb=None):
        if self.do_daylight_wb and self.no_auto_scale:
            raw = self.skyglow.raw
            wb_coeffs = raw.rimg.daylight_whitebalance
            if wb_coeffs and all(wb_coeffs[:3]):
                wb_coeffs = numpy.array(wb_coeffs)
                if isinstance(extra_wb, basestring):
                    extra_wb = self.WB_SETS.get(extra_wb)
                if extra_wb:
                    wb_coeffs *= numpy.array(extra_wb)
                self.accum = self.accum_prewb * wb_coeffs[raw.rimg.raw_colors]

    def _get_raw_instance(self):
        img = self.light_stacker._get_raw_instance()
        if img.postprocessing_params is not None:
            img.postprocessing_params.no_auto_scale = self.no_auto_scale
        return img

    def set_reference_frame(self, frame_index):
        lights = self.light_stacker.lights
        lights[0], lights[frame_index] = lights[frame_index], lights[0]
