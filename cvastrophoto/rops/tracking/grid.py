# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import logging

import skimage.transform
import skimage.measure

from cvastrophoto.image import rgb

from .base import BaseTrackingRop
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

    _POPKW = (
        'grid_size',
        'gridsize',
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

    def __init__(self, raw, pool=None,
            tracker_class=correlation.CorrelationTrackingRop,
            transform_type='similarity',
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
        self.tracker_class = tracker_class
        self.order = order
        self.mode = mode
        self.lxscale = self.lyscale = None

        for k in self._POPKW:
            kw.pop(k, None)

        pp_rop = kw.get('luma_preprocessing_rop', False)
        if pp_rop is False:
            pp_rop = extraction.ExtractStarsRop(rgb.Templates.LUMINANCE, copy=False)
        self.luma_preprocessing_rop = pp_rop

        if median_shift_limit is not None:
            self.median_shift_limit = median_shift_limit

        sizes = raw.rimg.sizes

        if track_roi is not None:
            self.track_roi = track_roi

        tmargin, lmargin, bmargin, rmargin = self.track_roi
        t = sizes.top_margin + int(tmargin * sizes.height)
        l = sizes.left_margin + int(lmargin * sizes.width)
        b = sizes.top_margin + sizes.height - int(bmargin * sizes.height)
        r = sizes.left_margin + sizes.width - int(rmargin * sizes.width)

        yspacing = (b-t) / self.grid_size[0]
        xspacing = (r-l) / self.grid_size[1]
        trackers = []
        for y in xrange(t + yspacing/2, b, yspacing):
            for x in xrange(l + xspacing/2, r, xspacing):
                tracker = tracker_class(self.raw, copy=False, **kw)
                if track_distance is not None:
                    tracker.track_distance = track_distance
                tracker.grid_coords = (y, x)
                tracker.set_reference(tracker.grid_coords)
                trackers.append(tracker)

        self.trackers = trackers
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
            tracker = self.tracker_class(self.raw, copy=False)
            tracker.grid_coords = grid_coords
            tracker.set_reference(tracker.grid_coords)
            tracker.load_state(tracker_state)
            trackers.append(tracker)
        self.trackers[:] = trackers
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
                if self.deglow is not None:
                    data = self.deglow.correct(data.copy())

                self.raw.set_raw_image(data, add_bias=self.add_bias)

                # Initialize postprocessed image in the main thread
                self.raw.postprocessed

            if luma is None:
                luma = self.raw.postprocessed_luma(copy=True)

                if self.luma_preprocessing_rop is not None:
                    luma = self.luma_preprocessing_rop.correct(luma)

            vshape = self.raw.rimg.raw_image_visible.shape
            lshape = self.raw.postprocessed.shape
            lyscale = vshape[0] / lshape[0]
            lxscale = vshape[1] / lshape[1]

            def detect(tracker):
                if save_tracks:
                    save_this_track = tracker is self.trackers[4]
                else:
                    save_this_track = False
                bias = tracker.detect(data, img=img, save_tracks=save_this_track, set_data=False, luma=luma)

                # Grid coords are in raw space, translate to luma space
                y, x = tracker.grid_coords
                y /= lyscale
                x /= lxscale
                grid_coords = y, x

                return (
                    list(grid_coords)
                    + list(tracker.translate_coords(bias, *grid_coords))
                    + list(tracker.translate_coords(bias, 0, 0))
                )

            if self.pool is None:
                map_ = map
            else:
                map_ = self.pool.map
            translations = numpy.array(map_(detect, self.trackers))
            self.tracking_cache[tracking_key] = (translations, vshape, lshape)
            luma = None
        else:
            translations, vshape, lshape = cached

        translations = translations.copy()
        self.lyscale = lyscale = vshape[0] / lshape[0]
        self.lxscale = lxscale = vshape[1] / lshape[1]

        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape

        # Translate luma space to raw space
        translations[:, [0, 2]] /= ysize / lyscale
        translations[:, [1, 3]] /= xsize / lxscale

        median_shift_mag = float('inf')
        while median_shift_mag > self.median_shift_limit and len(translations) > 3:
            # Estimate transform parameters out of valid measurements
            transform = skimage.transform.estimate_transform(
                self.transform_type,
                translations[:, [3, 2]],
                translations[:, [1, 0]])

            # Weed out outliers
            transformed = transform(translations[:, [3, 2]])
            shift_mags = numpy.sum(numpy.square(translations[:, [1, 0]] - transformed), axis=1)
            median_shift_mag = numpy.median(shift_mags)
            logger.info("Median shift error: %.3f", median_shift_mag)
            if median_shift_mag > self.median_shift_limit:
                # Pick the worst and get it out of the way
                ntranslations = translations[shift_mags < shift_mags.max()]
                if len(ntranslations) >= 3:
                    logger.info("Removed %d bad grid points", len(translations) - len(ntranslations))
                    translations = ntranslations
                else:
                    logger.info("Can't remove any more grid points")
                    logger.warning("Rejecting frame %s due to poor tracking", img)
                    return None

        if median_shift_mag > self.median_shift_limit or len(translations) <= 4 and not self.force_pass:
            logger.warning("Rejecting frame %s due to poor tracking", img)
            return None

        logger.info("Using %d reference grid points", len(translations))

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
