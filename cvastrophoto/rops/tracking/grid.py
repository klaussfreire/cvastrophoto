# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import functools
import logging

import skimage.transform
import skimage.measure
import scipy.ndimage

from ..base import BaseRop
from . import correlation 

logger = logging.getLogger(__name__)

class GridTrackingRop(BaseRop):

    grid_size = (3, 3)
    add_bias = False
    min_sim = None
    sim_prefilter_size = 64
    median_shift_limit = 2

    def __init__(self, raw, pool=None,
            tracker_class=correlation.CorrelationTrackingRop,
            transform_type='similarity',
            order=3,
            mode='reflect',
            median_shift_limit=None):
        super(GridTrackingRop, self).__init__(raw)
        if pool is None:
            pool = raw.default_pool
        self.pool = pool
        self.transform_type = transform_type
        self.tracker_class = tracker_class
        self.order = order
        self.mode = mode

        if median_shift_limit is not None:
            self.median_shift_limit = median_shift_limit

        sizes = raw.rimg.sizes
        yspacing = sizes.iheight / self.grid_size[0]
        xspacing = sizes.iwidth / self.grid_size[1]
        trackers = []
        for y in xrange(sizes.top_margin + yspacing/2, sizes.top_margin + sizes.iheight, yspacing):
            for x in xrange(sizes.left_margin + xspacing/2, sizes.left_margin + sizes.iwidth , xspacing):
                tracker = tracker_class(raw)
                tracker.grid_coords = (y, x)
                tracker.set_reference(tracker.grid_coords)
                trackers.append(tracker)

        self.trackers = trackers
        self.ref_luma = None

    def set_reference(self, data):
        # Does nothing
        pass

    def detect(self, data, bias=None, img=None, save_tracks=None, set_data=True, luma=None, **kw):
        if isinstance(data, list):
            data = data[0]

        if set_data:
            self.raw.set_raw_image(data, add_bias=self.add_bias)

            # Initialize postprocessed image in the main thread
            self.raw.postprocessed

        if luma is None:
            luma = numpy.sum(self.raw.postprocessed, axis=2, dtype=numpy.uint32)

        vshape = self.raw.rimg.raw_image_visible.shape
        lshape = self.raw.postprocessed.shape
        lyscale = vshape[0] / lshape[0]
        lxscale = vshape[1] / lshape[1]

        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape

        def detect(tracker):
            bias = tracker.detect(data, img=img, save_tracks=False, set_data=False, luma=luma)
            return (
                list(tracker.grid_coords)
                + list(tracker.translate_coords(bias, *tracker.grid_coords))
                + list(tracker.translate_coords(bias, 0, 0))
            )

        if self.pool is None:
            map_ = map
        else:
            map_ = self.pool.map
        translations = numpy.array(map_(detect, self.trackers))

        translations[:, [0, 2]] /= lyscale * ysize
        translations[:, [1, 3]] /= lxscale * xsize

        median_shift_mag = 100
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

        if median_shift_mag > self.median_shift_limit or len(translations) <= 4:
            logger.warning("Rejecting frame %s due to poor tracking", img)
            return None

        logger.info("Using %d reference grid points", len(translations))

        return transform, lyscale, lxscale, luma

    def translate_coords(self, bias, y, x):
        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape
        transform, lyscale, lxscale = bias
        lyscale *= ysize
        lxscale *= xsize
        x, y = transform([[x / lxscale, y / lyscale]])
        return y * lyscale, x * lxscale

    def correct(self, data, bias=None, img=None, save_tracks=None, **kw):
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
                return None

        transform, lyscale, lxscale, luma = bias

        # Round to pattern shape to avoid channel crosstalk
        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape

        logger.info("Transform for %s scale %r trans %r rot %r",
            img, transform.scale, transform.translation, transform.rotation)

        # move data - must be careful about copy direction
        imgdata = None
        for sdata in dataset:
            if sdata is None:
                # Multi-component data sets might have missing entries
                continue

            # Put sensible data into image margins to avoid causing artifacts at the edges
            self.raw.demargin(sdata)

            for yoffs in xrange(ysize):
                for xoffs in xrange(xsize):
                    sdata[yoffs::ysize, xoffs::xsize] = skimage.transform.warp(
                        sdata[yoffs::ysize, xoffs::xsize],
                        inverse_map = transform,
                        order=self.order,
                        mode=self.mode,
                        preserve_range=True)

            if imgdata is None:
                imgdata = sdata

        if imgdata is not None:
            self.raw.set_raw_image(imgdata, add_bias=self.add_bias)
            aligned_luma = numpy.sum(self.raw.postprocessed, axis=2, dtype=numpy.uint32)
            aligned_luma[:] = scipy.ndimage.white_tophat(aligned_luma, self.sim_prefilter_size)

            if self.ref_luma is None:
                self.ref_luma = aligned_luma
            else:
                # Exclude a margin proportional to translation amount, to exclude margin artifacts
                margin = int(max(list(numpy.absolute(transform.translation * 2)))) * max(lxscale, lyscale)
                m_aligned_luma = aligned_luma[margin:-margin, margin:-margin]
                m_ref_luma = self.ref_luma[margin:-margin, margin:-margin]

                sim = skimage.measure.compare_nrmse(m_aligned_luma, m_ref_luma, 'mean')
                logging.info("Similarity after alignment: %.8f", sim)

                if self.min_sim is not None and sim < self.min_sim:
                    logging.warning("Rejecting %s due to bad alignment similarity", img)
                    return None

        return rvdataset
