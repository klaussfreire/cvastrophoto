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
from ..rops.tracking import grid, correlation, compound as tracking_compound
from ..rops import compound, scale
from ..util import srgb
from ..image import rgb
from ..library import darks, bias

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
        # These are based on the Starguider CLS filter
        'cls': (1, 0.8, 1.1, 1),
        'cls-drizzle-photometric': (0.94107851, 1, 0.67843978, 1),
        'cls-drizzle-perceptive': (1.8, 0.6, 0.67, 1),
    }

    # The bluest component of the CLS filter is a bit greenish, there's no pure blue
    # This matrix accentuates pure blue to restore color balance on blue broadband sources
    CLS_MATRIX = numpy.array([
        [1.0, 0, 0],
        [0, 1, 0],
        [0, -0.56, 1.96],
    ], numpy.float32)

    WB_MATRICES = {
        'cls': CLS_MATRIX,
        'cls-drizzle-photometric': CLS_MATRIX,
        'cls-drizzle-perceptive': CLS_MATRIX,
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
            feature_tracking_class=None,
            tracking_pre_class=None,
            tracking_post_class=None,
            tracking_2phase=False,
            tracking_refinement_phases=0,
            tracking_coarse_distance=1024,
            tracking_fine_distance=512,
            tracking_coarse_limit=16,
            tracking_coarse_downsample=2,
            extra_input_rops=None,
            extra_output_rops=None,
            preskyglow_rops=None,
            dither=False,
            pool=None):

        if pool is None:
            self.pool = pool = multiprocessing.pool.ThreadPool()

        tracking_rop_classes = []

        if feature_tracking_class is not None:
            tracking_rop_classes.append(feature_tracking_class)

        if tracking_pre_class is not None:
            # For 3-phase and up, a first rough translational phase
            # improves rotational accuracy of the second pass
            tracking_rop_classes.append(tracking_pre_class)

        if tracking_2phase:
            # For higher phase count
            # First (N-1) phases with higher tolerance and 2x downsampling
            # Use progressively smaller track distances to improve rotational accuracy
            # as we get closer
            tracking_rop_classes.extend([
                functools.partial(
                    tracking_class,
                    median_shift_limit=tracking_coarse_limit,
                    track_distance=(
                        tracking_fine_distance
                        + (tracking_coarse_distance - tracking_fine_distance) / (i + 1)
                    ),
                    force_pass=True,
                    downsample=tracking_coarse_downsample)
                for i in xrange(tracking_2phase)
            ])

            # Last phase with shorter search distance
            # improves rotational accuracy
            tracking_rop_classes.append(functools.partial(
                tracking_class,
                track_distance=tracking_fine_distance,
                median_shift_limit=1))
        else:
            tracking_rop_classes.append(tracking_class)

        for i in xrange(tracking_refinement_phases):
            tracking_rop_classes.append(tracking_rop_classes[-1])

        if tracking_post_class is not None:
            tracking_rop_classes.append(tracking_post_class)

        if len(tracking_rop_classes) > 1:
            tracking_factory = lambda rimg, **kw : tracking_compound.TrackingCompoundRop(
                rimg,
                *[klass(rimg, **kw) for klass in tracking_rop_classes]
            )
        else:
            tracking_factory = tracking_rop_classes[0]

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
        self.extra_input_rops = extra_input_rops or []
        self.extra_output_rops = extra_output_rops or []
        self.preskyglow_rops = preskyglow_rops or []
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
            master_bias=None,
            dark_library=None, bias_library=None, auto_dark_library='darklib'):

        if dark_library is None and dark_path is None:
            dark_library = darks.DarkLibrary(default_pool=self.pool)
        if bias_library is None and master_bias is None:
            bias_library = bias.BiasLibrary(default_pool=self.pool)

        self.light_stacker.load_set(
            base_path, light_path, dark_path,
            master_bias=master_bias, dark_library=dark_library, bias_library=bias_library,
            auto_dark_library=auto_dark_library)

        if flat_path is not None:
            self.flat_stacker.load_set(
                base_path, flat_path, dark_flat_path,
                master_bias=master_bias, dark_library=dark_library, bias_library=bias_library,
                auto_dark_library=auto_dark_library)
            self.vignette = self.vignette_class(self.flat_stacker.lights[0])
        else:
            self.vignette = None

        if self.debias_class is not None:
            self.debias = self.debias_class(self.light_stacker.lights[0])
        else:
            self.debias = None

        self.skyglow = self.skyglow_class(self.light_stacker.stacked_image_template)
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

        if self.extra_input_rops:
            for rop_class in self.extra_input_rops:
                rops.append(rop_class(self.light_stacker.lights[0]))

        if rops:
            self.light_stacker.input_rop = compound.CompoundRop(
                self.light_stacker.lights[0],
                *rops
            )

        # Since we're de-biasing, correct tracking requires that we re-add it
        # before postprocessing raw images for tracking
        tracking = self.light_stacker.tracking
        if tracking is not None:
            tracking.add_bias = True

    def get_state(self):
        return dict(
            light_stacker=self.light_stacker.get_state(),
            flat_stacker=self.flat_stacker.get_state() if self.flat_stacker is not None else None,
        )

    def _load_state(self, state):
        self.light_stacker.load_state(state['light_stacker'])
        if self.flat_stacker is not None:
            self.flat_stacker.load_state(state['flat_stacker'])

    def process(self, preview=False, preview_kwargs={}, rops_kwargs={}, **kw):
        self._reset_preview()
        preview_kwargs = preview_kwargs.copy()
        preview_kwargs.setdefault('rops_kwargs', rops_kwargs)
        self.process_stacks(preview=preview, preview_kwargs=preview_kwargs, **kw)
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

    def process_stacks(self, preview=False, preview_kwargs={}, on_phase_completed=None):
        if preview:
            every_frame = preview_kwargs.pop('every_frame', False)
            if every_frame:
                preview_callback = self.preview
            else:
                preview_callback = self._fast_preview
            preview_callback = functools.partial(preview_callback, **preview_kwargs)
        else:
            preview_callback = self._log_progress

        if on_phase_completed is not None:
            _preview_callback = preview_callback
            _last_phase = [None]
            def preview_callback(phase=None, iteration=None, *p, **kw):
                if _last_phase[0] is not None and _last_phase[0] != (phase, iteration):
                    try:
                        on_phase_completed(phase, iteration)
                    except Exception:
                        logger.exception("Error in phase callback")
                _last_phase[0] = (phase, iteration)
                return _preview_callback(phase, iteration, *p, **kw)

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
        self.get_image(**image_kwargs).save(preview_path, quality=95)
        logger.info("Saved preview at %s", preview_path)

    def process_rops(self, quick=False, extra_wb=None):
        accum = self.light_stacker.accum.copy()
        if accum.dtype.kind == 'f':
            accum = numpy.nan_to_num(accum, copy=False)
        if self.preskyglow_rops:
            accum = compound.CompoundRop(self.skyglow.raw, *self.preskyglow_rops).correct(accum)
        accum = self.skyglow.correct(accum, quick=quick)
        if self.extra_output_rops:
            accum = compound.CompoundRop(self.skyglow.raw, *self.extra_output_rops).correct(accum)
        self.accum_prewb = self.accum = accum
        self.process_wb(extra_wb=extra_wb)

    def process_wb(self, extra_wb=None):
        img = self.light_stacker._get_raw_instance()
        if isinstance(img, rgb.RGB):
            rgb_xyz_matrix = getattr(img, 'lazy_rgb_xyz_matrix', None)
        else:
            rgb_xyz_matrix = None
        if ((self.do_daylight_wb and self.no_auto_scale)
                or extra_wb is not None
                or rgb_xyz_matrix is not None):
            raw = self.skyglow.raw
            if self.do_daylight_wb and self.no_auto_scale:
                wb_coeffs = raw.rimg.daylight_whitebalance
            else:
                wb_coeffs = None
            if not wb_coeffs and extra_wb is not None:
                wb_coeffs = [1,1,1,1]

            accum = self.accum_prewb

            if rgb_xyz_matrix is not None:
                # Colorspace conversion, since we don't use rawpy's postprocessing we have to do it manually
                accum = accum.reshape((accum.shape[0], accum.shape[1] / 3, 3))
                accum = srgb.camera2rgb(accum, rgb_xyz_matrix, accum.copy()).reshape(self.accum_prewb.shape)

                # The matrix usually already does daylight white balance
                wb_coeffs = [1,1,1,1] if extra_wb is not None else None
            else:
                accum = accum.copy()

            if wb_coeffs and all(wb_coeffs[:3]):
                # Apply white balance coefficients, for both camera and filters
                wb_coeffs = numpy.array(wb_coeffs, numpy.float32)
                if isinstance(extra_wb, basestring):
                    extra_wb_coefs = self.WB_SETS.get(extra_wb)
                else:
                    extra_wb_coefs = extra_wb
                if extra_wb_coefs:
                    wb_coeffs *= numpy.array(extra_wb_coefs, numpy.float32)[:len(wb_coeffs)]
                logger.debug("Applying WB: %r", wb_coeffs)
                raw.postprocessed  # initialize raw pattern
                accum *= wb_coeffs[raw.rimg.raw_colors]

            if isinstance(extra_wb, basestring) and extra_wb in self.WB_MATRICES and isinstance(raw, rgb.RGB):
                accum = accum.reshape((accum.shape[0], accum.shape[1] / 3, 3))
                accum = srgb.color_matrix(accum, self.WB_MATRICES[extra_wb], accum.copy()).reshape(
                    self.accum_prewb.shape)

            self.accum = accum
        else:
            self.accum = self.accum_prewb

    def _get_raw_instance(self):
        img = self.light_stacker._get_raw_instance()
        if img.postprocessing_params is not None:
            img.postprocessing_params.no_auto_scale = self.no_auto_scale
        return img

    def set_reference_frame(self, frame_index):
        lights = self.light_stacker.lights
        lights[0], lights[frame_index] = lights[frame_index], lights[0]
