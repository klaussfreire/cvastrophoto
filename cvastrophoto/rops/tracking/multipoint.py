# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
import numpy
import logging

import skimage.transform
import skimage.feature
import scipy.ndimage

from cvastrophoto.image import rgb, Image

from .base import BaseTrackingMatrixRop
from .util import find_transform, TrackMaskMixIn
from . import correlation
from . import extraction

from cvastrophoto.rops.measures.focus import FocusMeasureRop

logger = logging.getLogger(__name__)

class MultipointTrackingRop(TrackMaskMixIn, BaseTrackingMatrixRop):

    points = 5
    add_bias = False
    min_sim = None
    sim_prefilter_size = 64
    median_shift_limit = 2.0
    force_pass = False
    tracking_cache = None
    deglow = None
    masked = True
    mask_sigma = 2.0
    min_overlap = 0.5
    save_tracks = False
    global_pp = True
    transform_type = 'similarity'
    fallback_transform_type = 'euclidean'

    _POPKW = (
        'grid_size',
        'gridsize',
        'margin',
        'add_bias',
        'min_sim',
        'sim_prefilter_size',
        'median_shift_limit',
        'force_pass',
        'deglow',
        'lraw',
    )

    def __init__(self, raw, pool=None,
            tracker_class=correlation.CorrelationTrackingRop,
            order=3,
            mode='reflect',
            median_shift_limit=None,
            track_distance=None,
            **kw):
        super(MultipointTrackingRop, self).__init__(raw, **kw)
        if pool is None:
            pool = raw.default_pool
        self.pool = pool
        self.tracker_class = tracker_class
        self.tracker_kwargs = kw
        self.order = order
        self.mode = mode
        self.lxscale = self.lyscale = None
        self.track_distance = track_distance

        for k in self._POPKW:
            kw.pop(k, None)

        self.color_preprocessing_rop = kw.get('color_preprocessing_rop', None)

        pp_rop = kw.get('luma_preprocessing_rop', False)
        if pp_rop is False:
            pp_rop = extraction.ExtractStarsRop(rgb.Templates.LUMINANCE, copy=False, pre_demargin=False)
        self.luma_preprocessing_rop = pp_rop

        if median_shift_limit is not None:
            self.median_shift_limit = median_shift_limit

        self.trackers = None
        self.trackers_mask = None
        self.ref_luma = None

    def get_state(self):
        return {
            'trackers': [tracker.get_state() for tracker in self.trackers] if self.trackers is not None else None,
            'grid_coords': [tracker.grid_coords for tracker in self.trackers] if self.trackers is not None else None,
            'cache': self.tracking_cache,
        }

    def load_state(self, state):
        trackers = []
        for grid_coords, tracker_state in zip(state['grid_coords'] or [], state['trackers'] or []):
            tracker = self.tracker_class(self.raw, copy=False, lraw=self.lraw, **self.tracker_kwargs)
            tracker.grid_coords = grid_coords
            tracker.set_reference(tracker.grid_coords)
            tracker.load_state(tracker_state)
            trackers.append(tracker)
        self.trackers[:] = trackers
        self.trackers_mask = None
        self.tracking_cache = state.get('cache')

    def set_reference(self, data):
        # Does nothing
        pass

    def _tracking_key(self, data):
        return getattr(data, 'name', id(data))

    def pick_trackers(self, luma):
        sizes = self.lraw.rimg.sizes
        trackers = self.trackers = []

        # peak_local_max does not always respect min_distance, so we'll have to re-check results
        min_distance = self.track_distance or 256
        coords = skimage.feature.peak_local_max(
            luma,
            min_distance=min_distance,
            num_peaks=self.points*10,
            exclude_border=min_distance//2)

        accepted_coords = numpy.ones(len(coords), dtype=numpy.bool8)
        mask = numpy.ones_like(luma, dtype=numpy.bool8)
        for i, (y, x) in enumerate(coords):
            if not mask[y, x]:
                accepted_coords[i] = False
            else:
                mask[max(0, y-min_distance):y+min_distance, max(0, x-min_distance):x+min_distance] = False
        coords = coords[accepted_coords]
        del mask, accepted_coords

        for y, x in coords[:self.points]:
            tracker = self.tracker_class(
                self.raw,
                copy=False, lraw=self.lraw,
                **self.tracker_kwargs)
            if self.track_distance is not None:
                tracker.track_distance = self.track_distance
            tracker.grid_coords = (int(y), int(x))
            tracker.set_reference(tracker.grid_coords)
            trackers.append(tracker)

    def detect(self, data, bias=None, img=None, save_tracks=None, set_data=True, luma=None, **kw):
        if isinstance(data, list):
            data = data[0]

        if self.tracking_cache is None:
            self.tracking_cache = {}

        tracking_key = self._tracking_key(img or data)
        cached = self.tracking_cache.get(tracking_key)

        if cached is None:
            if set_data:
                if self.color_preprocessing_rop is not None:
                    ppdata = self.color_preprocessing_rop.correct(data.copy())
                else:
                    ppdata = data

                self.lraw.set_raw_image(ppdata, add_bias=self.add_bias)
                del ppdata

                # Initialize postprocessed image in the main thread
                self.lraw.postprocessed

            if luma is None:
                luma = self.lraw.postprocessed_luma(copy=True)

                if self.global_pp and self.luma_preprocessing_rop is not None:
                    luma = self.luma_preprocessing_rop.correct(luma)

            luma = self.apply_gray_mask(luma)

            vshape = self.lraw.rimg.raw_image_visible.shape
            lshape = self.lraw.postprocessed.shape
            lyscale = vshape[0] // lshape[0]
            lxscale = vshape[1] // lshape[1]

            if self.trackers is None:
                # Initialize trackers
                if not self.global_pp and self.luma_preprocessing_rop is not None:
                    ppluma = self.luma_preprocessing_rop.correct(luma.copy())
                else:
                    ppluma = luma
                self.pick_trackers(ppluma)
                del ppluma

            def _detect(tracker):
                if save_tracks:
                    save_this_track = tracker is self.trackers[0]
                else:
                    save_this_track = False
                bias = tracker.detect(
                    data,
                    img=img, save_tracks=save_this_track,
                    set_data=False, luma=luma, need_pp=not self.global_pp)

                if bias is None:
                    return None

                # Grid coords are in raw space, translate to luma space
                y, x = tracker.grid_coords
                y //= lyscale
                x //= lxscale
                grid_coords = y, x

                track_distance = getattr(tracker, 'track_distance', None)
                if track_distance is None:
                    track_distance = max(luma.shape) // min(self.grid_size)
                else:
                    track_distance *= getattr(tracker, 'downsample', 1)

                return (
                    list(grid_coords)
                    + list(tracker.translate_coords(bias, *grid_coords))
                    + list(tracker.translate_coords(bias, 0, 0))
                    + [track_distance]
                )

            def detect(tracker):
                try:
                    return _detect(tracker)
                except Exception:
                    logger.exception("Exception in detect")
                    raise

            if self.pool is None:
                map_ = map
            else:
                map_ = self.pool.map

            trackers = self.trackers
            if self.masked and luma is not None:
                trackers_mask = []
                luma_median = numpy.median(luma)
                luma_std = numpy.std(luma)
                luma_std = numpy.std(luma[luma <= (luma_median + self.mask_sigma * luma_std)])
                content_mask = luma > (luma_median + self.mask_sigma * luma_std)
                content_mask = scipy.ndimage.binary_opening(content_mask)
                for tracker in trackers:
                    # Grid coords are in raw space, translate to luma space
                    cy, cx = tracker.grid_coords
                    cy //= lyscale
                    cx //= lxscale

                    # Get tracking distance
                    track_distance = getattr(tracker, 'track_distance', None)
                    if track_distance is None:
                        track_distance = max(luma.shape) // min(self.grid_size)
                    else:
                        track_distance *= getattr(tracker, 'downsample', 1)

                    trackwin = content_mask[
                        max(0, cy-track_distance):cy+track_distance,
                        max(0, cx-track_distance):cx+track_distance
                    ]
                    trackers_mask.append(trackwin.any())
                    del trackwin
                del content_mask

                if self.trackers_mask is None:
                    self.trackers_mask = trackers_mask
                else:
                    trackers_mask = [m1 and m2 for m1, m2 in zip(self.trackers_mask, trackers_mask)]

                logger.info(
                    "Computed tracker mask with %d out of %d trackers enabled",
                    sum(trackers_mask), len(trackers_mask))

                trackers = [tracker for tracker, mask in zip(trackers, self.trackers_mask) if mask]
            translations = numpy.array(list(filter(None, map_(detect, trackers))), dtype=numpy.float64)
            self.tracking_cache[tracking_key] = (translations, vshape, lshape)
            luma = None
        else:
            translations, vshape, lshape = cached
            translations = translations.astype(numpy.float64, copy=False)

        # Filter out translations that exceed min overlap
        if translations.shape[1] > 6:
            translations = translations[
                numpy.maximum(
                    numpy.abs(translations[:,4]),
                    numpy.abs(translations[:,5]),
                ) <= (translations[:,6] * (1 - self.min_overlap))
            ]

        if not len(translations):
            logger.warning("Rejecting frame %s due to no matches", img)
            return None

        translations = translations.copy()
        self.lyscale = lyscale = vshape[0] // lshape[0]
        self.lxscale = lxscale = vshape[1] // lshape[1]

        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape

        # Translate luma space to raw space
        translations[:, [0, 2]] /= ysize // lyscale
        translations[:, [1, 3]] /= xsize // lxscale

        transform = find_transform(
            translations,
            self.transform_type, self.median_shift_limit, self.force_pass, self.fallback_transform_type)

        if transform is None:
            logger.warning("Rejecting frame %s due to poor tracking", img)
            return None

        return transform, lyscale, lxscale

    def translate_coords(self, bias, y, x):
        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape
        transform, lyscale, lxscale = bias
        lyscale *= ysize
        lxscale *= xsize
        (x, y), = transform.inverse([[x / lxscale, y / lyscale]])
        return y * lyscale, x * lxscale

    def needs_data(self, bias=None, img=None, save_tracks=None, **kw):
        if self.tracking_cache is None or img is None:
            return True

        tracking_key = self._tracking_key(img)
        cached = self.tracking_cache.get(tracking_key)
        return cached is None

    def detect_transform(self, data, bias=None, img=None, save_tracks=None, **kw):
        if save_tracks is None:
            save_tracks = self.save_tracks

        if isinstance(data, list):
            data = data[0]

        if bias is None:
            bias = self.detect(data, img=img, save_tracks=save_tracks)
            if bias is None:
                # Frame rejected
                return None

        transform, lyscale, lxscale = bias
        return transform

    def clear_cache(self):
        if self.trackers:
            for rop in self.trackers:
                rop.clear_cache()
        self.tracking_cache = None


class MultipointGuideTrackingRop(MultipointTrackingRop):

    points = 5
    median_shift_limit = 2.0
    force_pass = False
    masked = False
    global_pp = False
    transform_type = 'euclidean'
