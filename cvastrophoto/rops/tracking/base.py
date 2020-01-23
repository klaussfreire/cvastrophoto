# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import numpy
import skimage.measure
import skimage.transform
import scipy.ndimage

from .. import base

logger = logging.getLogger(__name__)

class BaseTrackingRop(base.BaseRop):

    min_sim = None
    order = 3
    per_part_order = {}
    mode = 'reflect'
    per_part_mode = {}

    def correct(self, data, bias=None, **kw):
        return self.correct_with_transform(data, bias,**kw)[0]

    def _get_lscale(self):
        if self.lyscale is None or self.lxscale is None:
            vshape = self.raw.rimg.raw_image_visible.shape
            lshape = self.raw.postprocessed.shape
            self.lyscale = vshape[0] / lshape[0]
            self.lxscale = vshape[1] / lshape[1]
        return self.lyscale, self.lxscale

    def apply_transform(self, data, transform, img=None, **kw):
        dataset = data
        if isinstance(data, list):
            data = data[0]
        else:
            dataset = [data]

        # Round to pattern shape to avoid channel crosstalk
        raw_pattern = self._raw_pattern
        raw_sizes = self._raw_sizes
        pattern_shape = raw_pattern.shape
        ysize, xsize = pattern_shape

        logger.info("Transform for %s scale %r trans %r rot %r",
            img, transform.scale, transform.translation, transform.rotation)

        if self.raw.default_pool is not None and len(dataset) > 1:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        def transform_data(sdata):
            partno, sdata = sdata
            if sdata is None:
                # Multi-component data sets might have missing entries
                return sdata

            # Put sensible data into image margins to avoid causing artifacts at the edges
            self.demargin(sdata, raw_pattern=raw_pattern, sizes=raw_sizes)

            for yoffs in xrange(ysize):
                for xoffs in xrange(xsize):
                    sdata[yoffs::ysize, xoffs::xsize] = skimage.transform.warp(
                        sdata[yoffs::ysize, xoffs::xsize],
                        inverse_map = transform,
                        order=self.per_part_order.get(partno, self.order),
                        mode=self.per_part_mode.get(partno, self.mode),
                        preserve_range=True)

            return sdata

        # move data - must be careful about copy direction
        imgdata = None
        for sdata in map_(transform_data, enumerate(dataset)):
            if sdata is None:
                # Multi-component data sets might have missing entries
                continue

            if imgdata is None:
                imgdata = sdata

        if imgdata is not None and self.min_sim is not None:
            self.raw.set_raw_image(imgdata, add_bias=self.add_bias)
            aligned_luma = numpy.sum(self.raw.postprocessed, axis=2, dtype=numpy.uint32)
            aligned_luma[:] = scipy.ndimage.white_tophat(aligned_luma, self.sim_prefilter_size)

            if self.ref_luma is None:
                self.ref_luma = aligned_luma
            else:
                # Exclude a margin proportional to translation amount, to exclude margin artifacts
                margin = int(max(list(numpy.absolute(transform.translation * 2)))) * max(self._get_lscale())
                m_aligned_luma = aligned_luma[margin:-margin, margin:-margin]
                m_ref_luma = self.ref_luma[margin:-margin, margin:-margin]

                sim = skimage.measure.compare_nrmse(m_aligned_luma, m_ref_luma, 'mean')
                logging.info("Similarity after alignment: %.8f", sim)

                if self.min_sim is not None and sim < self.min_sim:
                    logging.warning("Rejecting %s due to bad alignment similarity", img)
                    return None

        return dataset

    def clear_cache(self):
        pass
