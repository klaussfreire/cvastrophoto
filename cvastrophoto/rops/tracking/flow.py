# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
from future.builtins import map as imap
from functools import partial
import os.path
import numpy
import math
import scipy.ndimage
import skimage.registration
import skimage.transform
import logging
import PIL.Image

from cvastrophoto.image import rgb

from .base import BaseTrackingRop
from . import extraction
from .. import compound
from ..colorspace.extract import ExtractChannelRop
from cvastrophoto.util import srgb
from cvastrophoto.util import gaussian

logger = logging.getLogger(__name__)


def optical_flow_cross_correlation(reference, moving, block_size, step_size, max_displacement,
        pool=None, masked=False, mask_sigma=2.0,
        **kw):
    assert reference.shape == moving.shape

    flow = numpy.zeros((2,) + reference.shape, numpy.float32)
    weights = numpy.zeros(reference.shape, numpy.float32)

    def measure_block(task):
        reference, moving, flow, weights, mask = task
        if mask.any():
            corr, err, phase = skimage.registration.phase_cross_correlation(reference, moving, **kw)
            weight = min(reference.sum(), moving.sum()) / max(1.0e-5, err)
            if numpy.abs(corr).max() > max_displacement:
                corr *= max_displacement / numpy.abs(corr).max()
                weight *= 0.1
        else:
            corr = weight = None
        return (reference, moving, flow, weights), corr, weight

    if masked:
        luma_median = numpy.median(moving)
        luma_std = numpy.std(moving)
        luma_std = numpy.std(moving[moving <= (luma_median + mask_sigma * luma_std)])
        content_mask = moving > (luma_median + mask_sigma * luma_std)
        content_mask = scipy.ndimage.binary_opening(content_mask)

    tasks = []
    nblocks = 0
    for ystart in xrange(0, reference.shape[0] - block_size, step_size):
        for xstart in xrange(0, reference.shape[1] - block_size, step_size):
            nblocks += 1
            if masked:
                mask_block = content_mask[ystart:ystart + block_size, xstart:xstart + block_size]
            else:
                mask_block = None
            ref_block = reference[ystart:ystart + block_size, xstart:xstart + block_size]
            moving_block = moving[ystart:ystart + block_size, xstart:xstart + block_size]
            flow_block = flow[:, ystart:ystart + block_size, xstart:xstart + block_size]
            weight_block = weights[ystart:ystart + block_size, xstart:xstart + block_size]
            tasks.append((ref_block, moving_block, flow_block, weight_block, mask_block))

    if pool is None:
        map_ = imap
    else:
        map_ = pool.imap_unordered

    nunmasked = 0
    for (ref_block, moving_block, flow_block, weight_block), corr, weight in map_(measure_block, tasks):
        if weight:
            flow_block[0] -= corr[0] * weight
            flow_block[1] -= corr[1] * weight
            weight_block += weight
            nunmasked += 1

    if masked:
        logger.info("Mask coverage: %.2f%%", nunmasked * 100.0 / nblocks)

    flow /= numpy.clip(weights, 1, None, out=weights)

    flow[0] = gaussian.fast_gaussian(flow[0], step_size/2)
    flow[1] = gaussian.fast_gaussian(flow[1], step_size/2)

    return flow


class OpticalFlowTrackingRop(BaseTrackingRop):

    reference = None
    save_tracks = False
    add_bias = False
    linear_workspace = False
    downsample = 2
    method = 'corr'

    is_matrix_transform = False

    # generic parameters
    postfilter = 0
    iterations = 4
    tolerance = 1.0

    # corr+ilk parameters
    track_distance = 32

    # ilk+tlv1 parameters
    prefilter = True
    num_warp = 5

    # ilk parameters
    ilk_gaussian = False

    # tlv1 parameters
    attachment = 5
    tightness = 0.3
    num_iter = 10

    # corr parameters
    upsample_factor = 16
    track_region = 8

    # mask protection (corr)
    masked = True
    mask_sigma = 2.0

    METHODS = {
        'ilk': lambda self: partial(
            skimage.registration.optical_flow_ilk,
            prefilter=self.prefilter,
            radius=self.track_distance,
            gaussian=self.ilk_gaussian,
        ),
        'tvl1': lambda self: partial(
            skimage.registration.optical_flow_tvl1,
            prefilter=self.prefilter,
            attachment=self.attachment,
            tightness=self.tightness,
            num_iter=self.num_iter,
            num_warp=self.num_warp,
        ),
        'corr': lambda self: partial(
            optical_flow_cross_correlation,
            block_size=self.track_distance * self.track_region,
            step_size=self.track_distance,
            max_displacement=self.track_distance,
            pool=self.raw.default_pool,
            upsample_factor=self.upsample_factor,
            masked=self.masked,
            mask_sigma=self.mask_sigma,
        )
    }

    def __init__(self, *p, **kw):
        pp_rop = kw.pop('luma_preprocessing_rop', False)
        if pp_rop is False:
            pp_rop = extraction.ExtractStarsRop(rgb.Templates.LUMINANCE, copy=False, pre_demargin=False)
        self.luma_preprocessing_rop = pp_rop
        self.color_preprocessing_rop = kw.pop('color_preprocessing_rop', None)

        super(OpticalFlowTrackingRop, self).__init__(*p, **kw)

    def set_reference(self, data):
        if data is not None:
            if isinstance(data, tuple) and len(data) == 2:
                pass
            else:
                self.reference = self.detect(data)[1]
        else:
            self.reference = None

    def _detect(self, data, hint=None, img=None, save_tracks=None, set_data=True, luma=None, initial=False, **kw):
        if save_tracks is None:
            save_tracks = self.save_tracks

        need_pp = False
        if set_data:
            if self.color_preprocessing_rop:
                ppdata = self.color_preprocessing_rop.correct(data.copy())
            else:
                ppdata = data
            self.lraw.set_raw_image(ppdata, add_bias=self.add_bias)
            del ppdata

        if luma is None:
            luma = self.lraw.postprocessed_luma(copy=True)
            need_pp = True

        downsample = self.downsample
        if downsample > 1:
            luma = skimage.transform.downscale_local_mean(luma, (downsample,) * len(luma.shape))

        if need_pp and self.luma_preprocessing_rop is not None:
            luma = self.luma_preprocessing_rop.correct(luma)
            need_pp = False

        luma = srgb.encode_srgb(luma)

        if hint is None:
            # Find the brightest spot to build a tracking window around it
            refluma = luma.copy()
            lyscale = lxscale = None
            initial = True
        else:
            refluma, lyscale, lxscale = hint

        if lxscale is None or lyscale is None:
            vshape = self.lraw.rimg.raw_image_visible.shape
            lshape = luma.shape
            self.lyscale = lyscale = vshape[0] // lshape[0]
            self.lxscale = lxscale = vshape[1] // lshape[1]

        if img is not None and save_tracks:
            try:
                PIL.Image.fromarray(
                        ((refluma - refluma.min()) * 255.0 / refluma.ptp()).astype(numpy.uint8)
                    ).save('Tracks/%s.jpg' % os.path.basename(img.name))
            except Exception:
                logger.exception("Can't save tracks due to error")

        flow = flow_base = None
        if not initial:
            for i in xrange(self.iterations):
                if flow is not None:
                    flow_base = flow
                    iter_luma = self.apply_base_flow(luma, flow_base)
                else:
                    flow_base = None
                    iter_luma = luma

                flow = self.METHODS[self.method](self)(refluma, iter_luma)
                del iter_luma

                if self.postfilter:
                    flow = scipy.ndimage.median_filter(flow, self.postfilter)

                maxflow = max(abs(flow.max()), abs(flow.min()))
                logger.info("Iteration %d max displacement %r", i, maxflow)

                if flow_base is not None:
                    flow += flow_base
                    flow_base = None

                if maxflow < self.tolerance:
                    break

            if downsample > 1:
                flow *= downsample
                flow = skimage.transform.rescale(flow, (1,) + (downsample,) * (len(flow.shape) - 1))

        if flow is not None:
            transform = self.flow_to_transform(flow)
        else:
            transform = None

        return (transform, (refluma, lyscale, lxscale))

    def apply_base_flow(self, luma, flow):
        return skimage.transform.warp(
            luma,
            inverse_map = self.flow_to_transform(flow, copy=True),
            order=self.per_part_order.get(0, self.order),
            mode=self.per_part_mode.get(0, self.mode),
            preserve_range=True)

    def detect(self, data, bias=None, img=None, save_tracks=None, set_data=True, luma=None, **kw):
        if isinstance(data, list):
            data = data[0]

        initial = bias is None and self.reference is None

        if bias is None:
            bias = self._detect(
                data,
                hint=self.reference, save_tracks=save_tracks, img=img, luma=luma, set_data=set_data,
                initial=initial)
            set_data = False

        if self.reference is None:
            self.reference = bias[1]

        return bias

    def translate_coords(self, bias, y, x):
        refimg, transform = bias

        y = min(max(y, 0), transform.shape[0]-1)
        x = min(max(x, 0), transform.shape[1]-1)
        iy = math.floor(y)
        ix = math.floor(x)
        fy = y - iy
        fx = x - ix

        yp1 = min(y+1, transform.shape[0]-1)
        xp1 = min(x+1, transform.shape[1]-1)

        return (
            (transform[y,x] * fy + transform[yp1,x] * (1 - fy)) * fx
            + (transform[y,xp1] * fy + transform[yp1,xp1] * (1 - fy)) * (1 - fx)
        )

    def scale_transform(self, transform, part_scale):
        part_transform = skimage.transform.rescale(transform, (1,) + (part_scale,) * (len(transform.shape) - 1))
        part_transform *= part_scale
        return part_transform

    def flow_to_transform(self, flow, copy=False):
        ygrid = numpy.arange(flow.shape[1], dtype=flow.dtype)
        xgrid = numpy.arange(flow.shape[2], dtype=flow.dtype)

        if copy:
            transform = flow.copy()
        else:
            transform = flow
        transform[0].T[:] += ygrid
        transform[1] += xgrid

        return transform

    def correct_with_transform(self, data, bias=None, img=None, save_tracks=None, **kw):
        if save_tracks is None:
            save_tracks = self.save_tracks

        dataset = rvdataset = data
        if isinstance(data, list):
            data = data[0]
        else:
            dataset = [data]

        if bias is None or self.reference is None:
            bias = self.detect(data, bias=bias, save_tracks=save_tracks, img=img)

        transform, _ = bias
        _, lyscale, lxscale = self.reference

        if transform is not None:
            rvdataset = self.apply_transform(dataset, transform, img=img, **kw)
        else:
            rvdataset = dataset

        return rvdataset, transform