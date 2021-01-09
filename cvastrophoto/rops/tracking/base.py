# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
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
    per_part_scale = {}

    is_matrix_transform = True

    def __init__(self, raw, *p, **kw):
        lraw = kw.pop('lraw', raw)
        super(BaseTrackingRop, self).__init__(raw, *p, **kw)
        self.lraw = lraw

    def get_lock_pos(self):
        return None

    def get_lock_region(self):
        return None

    def correct(self, data, bias=None, **kw):
        return self.correct_with_transform(data, bias,**kw)[0]

    def _get_lscale(self):
        if self.lyscale is None or self.lxscale is None:
            vshape = self.raw.rimg.raw_image_visible.shape
            lshape = self.raw.postprocessed.shape
            self.lyscale = vshape[0] // lshape[0]
            self.lxscale = vshape[1] // lshape[1]
        return self.lyscale, self.lxscale

    def scale_transform(self, transform, part_scale):
        part_transform = type(transform)(matrix=transform.params.copy())
        part_transform.params[:2, 2] *= part_scale
        return part_transform

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
            img,
            getattr(transform, 'scale', 1.0),
            getattr(transform, 'translation', None),
            getattr(transform, 'rotation', None))

        if self.raw.default_pool is not None and len(dataset) > 1:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        # precache scaled transforms - direct map transforms can be real memory hogs
        scaled_transforms = {}
        for partno in xrange(len(dataset)):
            part_scale = self.per_part_scale.get(partno)
            if part_scale is not None and part_scale not in scaled_transforms:
                scaled_transforms[part_scale] = self.scale_transform(transform, part_scale)

        def transform_data(sdata):
            partno, sdata, yoffs, xoffs = sdata
            if sdata is None:
                # Multi-component data sets might have missing entries
                return sdata

            part_scale = self.per_part_scale.get(partno)
            part_transform = scaled_transforms.get(part_scale, transform)

            sdata[yoffs::ysize, xoffs::xsize] = skimage.transform.warp(
                sdata[yoffs::ysize, xoffs::xsize],
                inverse_map=part_transform,
                order=self.per_part_order.get(partno, self.order),
                mode=self.per_part_mode.get(partno, self.mode),
                preserve_range=True)

            return sdata

        # demargin data and prepare transform tasks
        tasks = []
        for partno, sdata in enumerate(dataset):
            if sdata is None:
                # Multi-component data sets might have missing entries
                continue

            if partno == 0:
                raw = self.lraw
                part_raw_sizes = self.lraw.rimg.sizes
            else:
                raw = self.raw
                part_raw_sizes = raw_sizes

            # Put sensible data into image margins to avoid causing artifacts at the edges
            self.demargin(sdata, raw_pattern=raw_pattern, sizes=part_raw_sizes, raw=raw)

            for yoffs in xrange(ysize):
                for xoffs in xrange(xsize):
                    tasks.append((partno, sdata, yoffs, xoffs))

        # move data - must be careful about copy direction
        imgdata = None
        for sdata in map_(transform_data, tasks):
            if sdata is None:
                # Multi-component data sets might have missing entries
                continue

            if imgdata is None:
                imgdata = sdata

        # Free up the RAM
        scaled_transforms.clear()

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
