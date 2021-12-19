# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
import os.path
import numpy
import scipy.ndimage
import skimage.transform
import logging
import PIL.Image

from cvastrophoto.image import rgb

from .base import BaseTrackingRop
from .util import TrackMaskMixIn
from . import extraction

logger = logging.getLogger(__name__)

class CenterOfMassTrackingRop(TrackMaskMixIn, BaseTrackingRop):

    reference = None
    track_distance = 256
    recenter_limit = None
    save_tracks = False
    add_bias = False

    def __init__(self, *p, **kw):
        self.luma_preprocessing_rop = kw.pop('luma_preprocessing_rop', None)
        self.color_preprocessing_rop = kw.pop('color_preprocessing_rop', None)
        self.tracking_cache = None

        super(CenterOfMassTrackingRop, self).__init__(*p, **kw)

    def set_reference(self, data):
        pass

    def get_lock_pos(self):
        if self.reference is not None:
            return self.reference[:2]

    def clear_cache(self):
        self.tracking_cache = None

    def _tracking_key(self, data, hint):
        return (
            getattr(data, 'name', id(data)),
            hint[:4] if hint is not None else None,
        )

    def _cache_clean(self, bias):
        return bias

    def _detect(self, data, hint=None, img=None, save_tracks=None, set_data=True, luma=None, **kw):
        if save_tracks is None:
            save_tracks = self.save_tracks

        if set_data:
            if self.color_preprocessing_rop is not None:
                ppdata = self.color_preprocessing_rop.correct(data.copy())
            else:
                ppdata = data
            self.lraw.set_raw_image(ppdata, add_bias=self.add_bias)
            del ppdata
        if luma is None:
            luma = self.lraw.postprocessed_luma(copy=True)

            if self.luma_preprocessing_rop is not None:
                luma = self.luma_preprocessing_rop.correct(luma)
                luma = self.apply_gray_mask(luma)

        if hint is None:
            # Find the brightest spot to build a tracking window around it
            refcentroid = None
            lyscale = lxscale = None
        else:
            refcentroid, lyscale, lxscale = hint

        if lxscale is None or lyscale is None:
            vshape = self.lraw.rimg.raw_image_visible.shape
            lshape = luma.shape
            lyscale = vshape[0] / lshape[0]
            lxscale = vshape[1] / lshape[1]

        trackwin = luma.copy()
        del luma

        # Heighten contrast
        thresh = trackwin.min() + trackwin.ptp()/2
        trackwin -= numpy.minimum(trackwin, thresh.astype(trackwin.dtype))
        trackwin -= trackwin.min()
        trackwin = numpy.multiply(trackwin, 16384.0 / trackwin.ptp(), dtype=numpy.float32)
        trackwin = trackwin.astype(numpy.int32)
        centroid = scipy.ndimage.center_of_mass(trackwin)

        if img is not None and save_tracks:
            try:
                PIL.Image.fromarray(
                        ((trackwin - trackwin.min()) * 255 / trackwin.ptp()).astype(numpy.uint8)
                    ).save('Tracks/%s.jpg' % os.path.basename(img.name))
            except Exception:
                logger.exception("Can't save tracks due to error")

        logger.debug("Center of mass %r for %s", centroid, img)

        return (centroid, lyscale, lxscale)

    def detect(self, data, bias=None, img=None, save_tracks=None, set_data=True, luma=None, **kw):
        if isinstance(data, list):
            data = data[0]

        if self.tracking_cache is None:
            self.tracking_cache = {}

        tracking_key = self._tracking_key(img or data, self.reference)
        if bias is None and self.reference is not None:
            bias = self.tracking_cache.get(tracking_key)

        if bias is None:
            bias = self._detect(data, hint=self.reference, save_tracks=save_tracks, img=img, luma=luma)
            set_data = False

        missing_reference = self.reference is None
        if missing_reference:
            self.reference = bias

        rv = self.tracking_cache.setdefault(tracking_key, bias)
        self.tracking_cache.setdefault(tracking_key, self._cache_clean(rv))
        return rv

    def translate_coords(self, bias, y, x):
        (yoffs, xoffs), _, _ = bias
        (yref, xref), lyscale, lxscale = self.reference

        fydrift = yref - yoffs
        fxdrift = xref - xoffs

        return y + fydrift, x + fxdrift

    def correct_with_transform(self, data, bias=None, img=None, save_tracks=None, **kw):
        if save_tracks is None:
            save_tracks = self.save_tracks

        dataset = rvdataset = data
        if isinstance(data, list):
            data = data[0]
        else:
            dataset = [data]

        if bias is None:
            bias = self.detect(data, save_tracks=save_tracks, img=img)

        (yoffs, xoffs), _, _ = bias
        (yref, xref), lyscale, lxscale = self.reference

        fydrift = yref - yoffs
        fxdrift = xref - xoffs

        # Round to pattern shape to avoid channel crosstalk
        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape
        fxdrift *= lxscale
        fydrift *= lyscale
        xdrift = int(fxdrift / ysize) * ysize
        ydrift = int(fydrift / xsize) * xsize

        transform = skimage.transform.SimilarityTransform(
            translation=(-fxdrift/xsize, -fydrift/ysize))

        logger.info("Tracking offset for %s %r drift %r quantized drift %r",
            img, (xoffs, yoffs), (fxdrift, fydrift), (xdrift, ydrift))

        rvdataset = self.apply_transform(dataset, transform, img=img, **kw)

        return rvdataset, transform
