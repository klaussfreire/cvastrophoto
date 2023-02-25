from __future__ import absolute_import, division

import time
import multiprocessing
import logging
import threading
import os.path

from past.builtins import xrange, basestring

import scipy.ndimage

from cvastrophoto import image
from cvastrophoto.image import metaimage
from cvastrophoto.rops.tracking import correlation, grid
from cvastrophoto.wizards.base import BaseWizard
from .stacking import StackingWizard, WeightInfo


logger = logging.getLogger(__name__)


class NoCache(dict):

    def get(self, k, deflt=None):
        return deflt

    def __setitem__(self, k, v):
        pass

    def setdefault(self, k, v):
        return v


class StackState(object):

    def __init__(self, stacking_wizard, load_set_kw,
            max_time=None, max_subs=None, nocache=True,
            needs_dark_calibration=True):
        self.wiz = stacking_wizard
        self.load_set_kw = load_set_kw
        self.max_time = max_time
        self.max_subs = max_subs
        self.open_time = None
        self.light_method = None
        self.nocache = nocache
        self.first = True
        self.needs_dark_calibration = needs_dark_calibration
        self.dark_calibrated = False
        self._context_state = None
        self._lock = threading.RLock()
        self._weightinfo = WeightInfo()

    def _stacking_context(self):
        light_method = self.light_method
        for phaseno, iterations in light_method.phases:
            for i in xrange(iterations):
                light_method.start_phase(phaseno, i)
                if phaseno != 0:
                    yield phaseno, i

    @property
    def initialized(self):
        return self.light_method is not None and self._context_state is not None

    def init(self, im):
        if self.initialized:
            return

        # Initialize stacking wizard state through load_set
        self.open_time = time.time()
        if self.wiz.lights is None:
            self.wiz.load_set(lights=[im], **self.load_set_kw)

        # Initialize light stacking context
        self.light_method = self.wiz.light_method_instance
        self.light_method.set_image_shape(im)
        self._context_state = self._stacking_context()
        self._cur_phase, self._cur_iteration = next(self._context_state)

        track = self.wiz.tracking
        if self.nocache and track is not None:
            # Disable tracking cache, not useful on live stacking
            if hasattr(track, 'set_tracking_cache'):
                track.set_tracking_cache(NoCache())

    def _add(self, im, dark_calibrated=False):
        if not dark_calibrated and self.needs_dark_calibration:
            if not isinstance(im, metaimage.MetaImage) or not im.dark_calibrated:
                self.wiz.dark_calibration(im, no_repair_pixels=True)
                im.dark_calibrated = True

        if dark_calibrated:
            self.dark_calibrated = True

        if isinstance(im, metaimage.MetaImage):
            data = im
        else:
            data = im.rimg.raw_image

        extracted, weights = self.wiz.process_light(
            im, data,
            extract=self.light_method.extract_frame,
            weightinfo=self._weightinfo,
        )

        if extracted is not None:
            with self._lock:
                self.light_method += extracted

        im.close()

    def add(self, im, keep=True, dark_calibrated=False):
        if not self.initialized:
            with self._lock:
                self.init(im)
        elif keep:
            self.wiz.lights.append(im)

        self._add(im, dark_calibrated=dark_calibrated)

    def should_close(self):
        return self.open_time is not None and time.time() > self.open_time + self.max_time

    def close(self, pool=None):
        wiz = self.wiz
        for self._cur_phase, self._cur_iteration in self._context_state:
            if pool is not None:
                self._add(wiz.lights[0])
                for _ in pool.imap_unordered(self._add, wiz.lights[1:]):
                    pass
            else:
                for light in wiz.lights:
                    self._add(light)
        self.light_method.finish()
        for light in wiz.lights:
            if hasattr(light, 'close'):
                light.close()


class SimpleLiveStackingWizard(BaseWizard):

    max_time = 1800
    max_subs = 40
    resolution = 4

    def __init__(self, pool=None, input_pool=None,
            stacking_wizard_cls=StackingWizard,
            stacking_kw=dict(tracking_class=grid.GridTrackingRop),
            t0_tracking_class=correlation.CorrelationTrackingRop,
            load_set_kw=dict(dark_path=None, auto_dark_library=None),
            open_kw={},
            t0_dir='.', t0_prefix='light_', t0_suffix='.fits'):
        if pool is None:
            pool = multiprocessing.pool.ThreadPool()
        if input_pool is None:
            input_pool = multiprocessing.pool.ThreadPool()
        self.input_pool = input_pool
        self.pool = pool
        self.stacking_wizard_cls = stacking_wizard_cls

        stacking_kw.update(dict(pool=pool, input_pool=input_pool))
        load_set_kw.update(dict(light_path=None))
        self.stacking_kw = stacking_kw
        self.load_set_kw = load_set_kw
        self.open_kw = open_kw.copy()
        self.open_kw.update(dict(default_pool=pool))

        self.t0_tracking_class = t0_tracking_class
        self.t0_tracking = None
        self.t0_dir = t0_dir
        self.t0_prefix = t0_prefix
        self.t0_suffix = t0_suffix
        self._t0_lock = threading.RLock()

        self.stacks = {}
        self._t1_stacking = None
        self._t1_lock = threading.RLock()

        self._nextname = iter(xrange(0xFFFFFFFF))

    def new_t0_stacking_wizard(self):
        kw = self.stacking_kw.copy()
        kw['tracking_class'] = None
        kw['input_rop'] = None
        return self.stacking_wizard_cls(**kw)

    def get_t0_stack_state(self, groupkey):
        state = self.stacks.get(groupkey)
        if state is None:
            state = self.stacks.setdefault(
                groupkey,
                StackState(
                    self.new_t0_stacking_wizard(), self.load_set_kw,
                    self.max_time, self.max_subs,
                ),
            )
        return state

    @property
    def t1_stacking_state(self):
        if self._t1_stacking is None:
            kw = self.stacking_kw.copy()
            kw['remove_bias'] = False
            load_kw = self.load_set_kw.copy()
            load_kw['auto_dark_library'] = None
            load_kw['light_path'] = None
            load_kw['dark_path'] = None
            load_kw['dark_library'] = None
            load_kw['bias_library'] = None
            with self._t1_lock:
                if self._t1_stacking is None:
                    self._t1_stacking = StackState(
                        self.stacking_wizard_cls(**kw), load_kw,
                        needs_dark_calibration=False)
        return self._t1_stacking

    def _close_t0(self, t0):
        t0.close()
        self._add_t1(t0)

    def dark_calibration(self, im):
        for t0 in self.stacks.values():
            wiz = t0.wiz
            break
        else:
            t0 = StackState(self.new_t0_stacking_wizard(), self.load_set_kw)
            t0.init(im)
            wiz = t0.wiz

        return wiz.dark_calibration(im, no_repair_pixels=True)

    def _get_tracking_key(self, im, bias_removed):
        if self.t0_tracking is None:
            self.t0_tracking = self.t0_tracking_class(im)
            if hasattr(self.t0_tracking, 'set_tracking_cache'):
                self.t0_tracking.set_tracking_cache(NoCache())
        bias = self.t0_tracking.detect(im.rimg.raw_image, img=im)
        y, x = self.t0_tracking.translate_coords(bias, 0.0, 0.0)
        return int(y * self.resolution), int(x * self.resolution), bias_removed

    def add_light_async(self, im, do_darks=True):
        if isinstance(im, basestring):
            im = image.Image.open(im, **self.open_kw)
        if do_darks:
            bias_removed = self.dark_calibration(im)
        return self.input_pool.apply_async(self._add_light, (im, bias_removed))

    def add_light(self, im, do_darks=True):
        if isinstance(im, basestring):
            im = image.Image.open(im, **self.open_kw)
        if do_darks:
            bias_removed = self.dark_calibration(im)
        return self._add_light(im, bias_removed)

    def _add_light(self, im, bias_removed):
        groupkey = self._get_tracking_key(im, bias_removed)

        logger.info("tracking key %r for %r", groupkey, im.name)

        with self._t0_lock:
            t0 = self.get_t0_stack_state(groupkey)
            if t0.should_close():
                t0 = self.stacks.pop(groupkey)
                self._close_t0(t0)
                t0 = self.get_t0_stack_state(groupkey)

            t0.add(im, dark_calibrated=bias_removed)

        t1 = self.t1_stacking_state
        if not t1.initialized:
            with self._t1_lock:
                if not t1.initialized:
                    t1.init(t0.wiz.lights[0])
                    del t1.wiz.lights[:]

    def _add_t1(self, t0):
        t0wiz = t0.wiz
        t1 = self.t1_stacking_state
        if not t1.initialized:
            with self._t1_lock:
                if not t1.initialized:
                    t1.init(t0wiz.lights[0])
                    del t1.wiz.lights[:]

        # Extract metaimage and attach image header data and stack offsets
        metaimage = t0wiz.metaimage
        im = t0wiz.lights[0]
        fits_header = getattr(im, 'fits_header', None)
        if fits_header is not None:
            metaimage.fits_header = fits_header
        metaimage.dark_calibrated = t0.dark_calibrated

        fname = self._namegen('t0_')
        metaimage.save(fname)
        metaimage.close()
        metaimage.open(fname)
        logger.info("Saved substack %r", fname)

        t1.add(metaimage, dark_calibrated=t0.dark_calibrated)

    def _namegen(self, prefix=''):
        for nn in self._nextname:
            path = os.path.join(self.t0_dir, '%s%s%03d%s' % (self.t0_prefix, prefix, nn, self.t0_suffix))
            if not os.path.exists(path):
                return path

    @property
    def accumulator(self):
        return self.t1_stacking_state.wiz.accumulator

    @accumulator.setter
    def accumulator(self, accumulator):
        self.t1_stacking_state.wiz.accumulator = accumulator

    @accumulator.deleter
    def accumulator(self):
        del self.t1_stacking_state.wiz.accumulator

    @property
    def metaimage(self):
        return self.t1_stacking_state.wiz.metaimage

    @metaimage.setter
    def metaimage(self, meta):
        self.t1_stacking_state.wiz.metaimage = meta

    @metaimage.deleter
    def metaimage(self):
        del self.t1_stacking_state.wiz.metaimage

    @property
    def accum(self):
        return self.t1_stacking_state.wiz.accumulator.accum

    @property
    def preview_accumulator(self):
        accum = self.accumulator.copy()
        for (y, x, _), t0 in self.stacks.items():
            t0accum = t0.wiz.accumulator.copy()
            t0accum.accum = scipy.ndimage.shift(
                t0accum.accum,
                (y / self.resolution, x / self.resolution),
                order=1, prefilter=False,
            )
            accum += t0accum
        return accum

    def ensure_reference(self):
        with self._t0_lock:
            for groupkey, t0 in self.stacks.items():
                t0 = self.stacks.pop(groupkey)
                self._close_t0(t0)
                break

    def close(self):
        for im in self.input_pool.imap_unordered(self._close_t0, self.stacks.values()):
            pass
        self.stacks.clear()
        self.t1_stacking_state.close(pool=self.input_pool)

    def _get_raw_instance(self):
        return self.t1_stacking_state.wiz.stacked_image_template
