# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
import numpy
import logging

import skimage.transform
import scipy.ndimage

from cvastrophoto.image import rgb

from .base import BaseTrackingRop
from .util import find_transform
from . import correlation
from . import extraction

logger = logging.getLogger(__name__)

class GridTrackingRop(BaseTrackingRop):

    grid_size = (3, 3)
    add_bias = False
    min_sim = None
    sim_prefilter_size = 64
    median_shift_limit = 2.0
    force_pass = False
    track_roi = (0, 0, 0, 0)  # (t-margin, l-margin, b-margin, r-margin), normalized
    tracking_cache = None
    deglow = None
    masked = True
    mask_sigma = 2.0
    min_overlap = 0.5
    save_tracks = False

    _POPKW = (
        'grid_size',
        'gridsize',
        'margin',
        'add_bias',
        'min_sim',
        'sim_prefilter_size',
        'median_shift_limit',
        'force_pass',
        'track_roi',
        'deglow',
    )

    @property
    def gridsize(self):
        return self.grid_size[0]

    @gridsize.setter
    def gridsize(self, value):
        self.grid_size = (value, value)

    @property
    def margin(self):
        return float(self.track_roi[0])

    @margin.setter
    def margin(self, value):
        self.track_roi = (value, value, value, value)

    @property
    def tracking_roi(self):
        return '-'.join(map(str, self.track_roi)) if self.track_roi else ''

    @tracking_roi.setter
    def tracking_roi(self, roi):
        self.track_roi = map(float, roi.split('-'))

    def __init__(self, raw, pool=None,
            tracker_class=correlation.CorrelationTrackingRop,
            transform_type='similarity',
            fallback_transform_type='euclidean',
            order=3,
            mode='reflect',
            median_shift_limit=None,
            track_roi=None,
            track_distance=None,
            **kw):
        super(GridTrackingRop, self).__init__(raw, **kw)
        if pool is None:
            pool = raw.default_pool
        self.pool = pool
        self.transform_type = transform_type
        self.fallback_transform_type = fallback_transform_type
        self.tracker_class = tracker_class
        self.order = order
        self.mode = mode
        self.lxscale = self.lyscale = None

        for k in self._POPKW:
            kw.pop(k, None)

        self.color_preprocessing_rop = kw.get('color_preprocessing_rop', None)

        pp_rop = kw.get('luma_preprocessing_rop', False)
        if pp_rop is False:
            pp_rop = extraction.ExtractStarsRop(rgb.Templates.LUMINANCE, copy=False)
        self.luma_preprocessing_rop = pp_rop

        if median_shift_limit is not None:
            self.median_shift_limit = median_shift_limit

        sizes = self.lraw.rimg.sizes

        if track_roi is not None:
            self.track_roi = track_roi

        tmargin, lmargin, bmargin, rmargin = self.track_roi
        t = sizes.top_margin + int(tmargin * sizes.height)
        l = sizes.left_margin + int(lmargin * sizes.width)
        b = sizes.top_margin + sizes.height - int(bmargin * sizes.height)
        r = sizes.left_margin + sizes.width - int(rmargin * sizes.width)

        yspacing = (b-t) // self.grid_size[0]
        xspacing = (r-l) // self.grid_size[1]
        trackers = []
        for y in xrange(t + yspacing//2, b, yspacing):
            for x in xrange(l + xspacing//2, r, xspacing):
                tracker = tracker_class(self.raw, copy=False, **kw)
                if track_distance is not None:
                    tracker.track_distance = track_distance
                tracker.grid_coords = (y, x)
                tracker.set_reference(tracker.grid_coords)
                trackers.append(tracker)

        self.trackers = trackers
        self.trackers_mask = None
        self.ref_luma = None

    def get_state(self):
        return {
            'trackers': [tracker.get_state() for tracker in self.trackers],
            'grid_coords': [tracker.grid_coords for tracker in self.trackers],
            'cache': self.tracking_cache,
        }

    def load_state(self, state):
        trackers = []
        for grid_coords, tracker_state in zip(state['grid_coords'], state['trackers']):
            tracker = self.tracker_class(self.raw, copy=False, lraw=self.lraw)
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

                if self.luma_preprocessing_rop is not None:
                    luma = self.luma_preprocessing_rop.correct(luma)

            vshape = self.lraw.rimg.raw_image_visible.shape
            lshape = self.lraw.postprocessed.shape
            lyscale = vshape[0] // lshape[0]
            lxscale = vshape[1] // lshape[1]

            def _detect(tracker):
                if save_tracks:
                    save_this_track = tracker is self.trackers[4]
                else:
                    save_this_track = False
                bias = tracker.detect(data, img=img, save_tracks=save_this_track, set_data=False, luma=luma)

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
            if self.masked:
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
        x, y = transform([[x / lxscale, y / lyscale]])
        return y * lyscale, x * lxscale

    def correct_with_transform(self, data, bias=None, img=None, save_tracks=None, **kw):
        if save_tracks is None:
            save_tracks = self.save_tracks

        dataset = rvdataset = data
        if isinstance(data, list):
            data = data[0]
        else:
            dataset = [data]

        if bias is None:
            bias = self.detect(data, img=img, save_tracks=save_tracks)
            if bias is None:
                # Frame rejected
                return None, None

        transform, lyscale, lxscale = bias

        rvdataset = self.apply_transform(dataset, transform, img=img, **kw)

        return rvdataset, transform

    def clear_cache(self):
        for rop in self.trackers:
            rop.clear_cache()
        self.tracking_cache = None
