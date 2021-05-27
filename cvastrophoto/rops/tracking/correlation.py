# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
import os.path
import numpy
import scipy.ndimage
import skimage.feature
import skimage.transform
import logging
import PIL.Image

from cvastrophoto.image import rgb

from .base import BaseTrackingMatrixRop
from . import extraction
from .. import compound
from ..colorspace.extract import ExtractChannelRop
from cvastrophoto.util import srgb

logger = logging.getLogger(__name__)

class CorrelationTrackingRop(BaseTrackingMatrixRop):

    reference = None
    tracking_cache = None
    track_distance = 1024
    resolution = 16
    save_tracks = False
    long_range = False
    add_bias = False
    linear_workspace = True
    downsample = 1

    _lock_region = None

    def __init__(self, *p, **kw):
        pp_rop = kw.pop('luma_preprocessing_rop', False)
        if pp_rop is False:
            pp_rop = extraction.ExtractStarsRop(rgb.Templates.LUMINANCE, copy=False, pre_demargin=False)
        self.luma_preprocessing_rop = pp_rop
        self.color_preprocessing_rop = kw.pop('color_preprocessing_rop', None)

        super(CorrelationTrackingRop, self).__init__(*p, **kw)

    def set_reference(self, data):
        if data is not None:
            if isinstance(data, tuple) and len(data) == 2:
                # Explicit starting point
                self.reference = data + (0, 0, (None, None, None))
            else:
                self.reference = self.detect(data)
        else:
            self.reference = None
        self.tracking_cache = {}

    def get_lock_pos(self):
        if self.reference is not None:
            return self.reference[:2]

    def get_lock_region(self):
        return self._lock_region

    def _tracking_key(self, data, hint):
        return (
            getattr(data, 'name', id(data)),
            hint[:4] if hint is not None else None,
        )

    def _cache_clean(self, bias):
        return bias[:-1] + ((None,) + bias[-1][1:],)

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

        logger.info("Tracking hint for %s: %r", img, hint[:2] if hint is not None else hint)

        if hint is None:
            if need_pp and self.luma_preprocessing_rop is not None:
                luma = self.luma_preprocessing_rop.correct(luma)
                need_pp = False

            # Find the brightest spot to build a tracking window around it
            margin = min(self.track_distance // 2, min(luma.shape) // 4)
            mluma = luma[margin:-margin, margin:-margin]
            pos = numpy.argmax(mluma)

            ymax = pos // mluma.shape[1]
            xmax = pos - ymax * mluma.shape[1]
            ymax += margin
            xmax += margin
            reftrackwin = None
            lyscale = lxscale = None
            del mluma

            vshape = self.lraw.rimg.raw_image_visible.shape
            lshape = luma.shape
            ymax *= vshape[0] // lshape[0]
            xmax *= vshape[1] // lshape[1]
        else:
            ymax, xmax, yref, xref, (reftrackwin, lyscale, lxscale) = hint
            ymax = int(ymax)
            xmax = int(xmax)

        if lxscale is None or lyscale is None:
            vshape = self.lraw.rimg.raw_image_visible.shape
            lshape = luma.shape
            self.lyscale = lyscale = vshape[0] // lshape[0]
            self.lxscale = lxscale = vshape[1] // lshape[1]

        rxmax = xmax
        rymax = ymax
        xmax //= lxscale
        ymax //= lyscale

        track_distance = self.track_distance
        downsample = self.downsample
        if downsample > 1:
            track_distance *= downsample

        wleft = min(xmax, track_distance)
        wright = min(luma.shape[1] - xmax, track_distance)
        wup = min(ymax, track_distance)
        wdown = min(luma.shape[0] - ymax, track_distance)
        trackwin = luma[ymax-wup:ymax+wdown, xmax-wleft:xmax+wright]
        del luma

        self._lock_region = (ymax-wup, xmax-wleft, ymax+wdown, xmax+wright)

        if need_pp and self.luma_preprocessing_rop is not None:
            trackwin = self.luma_preprocessing_rop.correct(trackwin)
            need_pp = False

        logger.info("Tracking window for %s: %d-%d, %d-%d (scale %d, %d)",
            img, xmax-wleft, xmax+wright, ymax-wup, ymax+wdown, lxscale, lyscale)

        # Downsample and heighten contrast
        if downsample > 1:
            trackwin = skimage.transform.downscale_local_mean(
                trackwin, (downsample,) * len(trackwin.shape)).astype(trackwin.dtype, copy=False)

        trackwin -= trackwin.min()
        trackwin = numpy.multiply(trackwin, 1.0 / trackwin.ptp(), dtype=numpy.float32)

        if img is not None and save_tracks:
            try:
                PIL.Image.fromarray(
                        ((trackwin - trackwin.min()) * 255 / trackwin.ptp()).astype(numpy.uint8)
                    ).save('Tracks/%s.jpg' % os.path.basename(img.name))
            except Exception:
                logger.exception("Can't save tracks due to error")

        if reftrackwin is None:
            # Global centroid to center star group in track window
            ytrack = xtrack = xref = yref = 0
            corr = None
        else:
            if initial:
                # No need to even check the first frame
                corr = ((0, 0),)
            else:
                corr_trackwin = trackwin
                corr_reftrackwin = reftrackwin
                if not self.linear_workspace:
                    corr_trackwin = srgb.encode_srgb(corr_trackwin)
                    corr_reftrackwin = srgb.encode_srgb(corr_reftrackwin)
                corr = skimage.feature.register_translation(corr_trackwin, corr_reftrackwin, self.resolution)
            ytrack, xtrack = corr[0]

        # Translate to image space
        xoffs = xtrack + xref
        yoffs = ytrack + yref

        if downsample > 1:
            xoffs *= downsample
            yoffs *= downsample

        logger.info("Correlation offset %r", (yoffs, xoffs))
        logger.debug("Correlation details %r", corr)

        return (rymax, rxmax, yoffs, xoffs, (trackwin, lyscale, lxscale))

    def detect(self, data, bias=None, img=None, save_tracks=None, set_data=True, luma=None, **kw):
        if isinstance(data, list):
            data = data[0]

        if self.tracking_cache is None:
            self.tracking_cache = {}

        tracking_key = self._tracking_key(img or data, self.reference)
        if bias is None and self.reference is not None:
            bias = self.tracking_cache.get(tracking_key)

        initial = bias is None and (self.reference is None or self.reference[-1][0] is None)

        if bias is None or self.reference[-1][0] is None:
            bias = self._detect(
                data,
                hint=self.reference, save_tracks=save_tracks, img=img, luma=luma, set_data=set_data,
                initial=initial)
            set_data = False

        if self.reference is None or self.reference[-1][0] is None:
            self.reference = bias

            # re-detect with hint, as would be done if reference had been initialized above
            # reset reference track window and star information with proper tracking center
            bias = self._detect(
                data,
                hint=bias, save_tracks=False, set_data=set_data, img=img, luma=luma,
                initial=initial)
            self.reference = self.reference[:-3] + bias[-3:]

            bias = self._detect(
                data,
                hint=self.reference, save_tracks=save_tracks, set_data=False, img=img, luma=luma,
                initial=initial)

        self.tracking_cache.setdefault(tracking_key, self._cache_clean(bias))
        return bias

    def get_state(self):
        return dict(reference=self.reference, cache=self.tracking_cache, downsample=self.downsample)

    def load_state(self, state):
        self.reference = state['reference']
        self.tracking_cache = state['cache']
        self.downsample = state.get('downsample', 1)

    def clear_cache(self):
        self.tracking_cache = None

    def translate_coords(self, bias, y, x):
        _, _, yoffs, xoffs, _ = bias
        _, _, yref, xref, (_, lyscale, lxscale) = self.reference

        fydrift = yref - yoffs
        fxdrift = xref - xoffs

        return y + fydrift, x + fxdrift

    def needs_data(self, bias=None, img=None, save_tracks=None, **kw):
        if self.tracking_cache is None or img is None or self.reference is None:
            return True

        tracking_key = self._tracking_key(img, self.reference)
        cached = self.tracking_cache.get(tracking_key)
        return cached is None

    def detect_transform(self, data, bias=None, img=None, save_tracks=None, **kw):
        if save_tracks is None:
            save_tracks = self.save_tracks

        if isinstance(data, list):
            data = data[0]

        if bias is None or self.reference[-1][0] is None:
            bias = self.detect(data, bias=bias, save_tracks=save_tracks, img=img)

        _, _, yoffs, xoffs, _ = bias
        _, _, yref, xref, (_, lyscale, lxscale) = self.reference

        fydrift = yref - yoffs
        fxdrift = xref - xoffs

        if self.long_range and self.reference is not None:
            ymax, xmax = self.reference[:2]
            xmax += int(fxdrift)
            ymax += int(fydrift)
            self.reference = (ymax, xmax,) + self.reference[2:]

        # Round to pattern shape to avoid channel crosstalk
        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape
        fxdrift *= lxscale
        fydrift *= lyscale
        xdrift = int(fxdrift / xsize) * xsize
        ydrift = int(fydrift / ysize) * ysize

        transform = skimage.transform.SimilarityTransform(
            translation=(-fxdrift/xsize, -fydrift/ysize))

        logger.info("Tracking offset for %s %r drift %r quantized drift %r",
            img, (xoffs, yoffs), (fxdrift, fydrift), (xdrift, ydrift))

        return transform


class CometTrackingRop(CorrelationTrackingRop):

    extract_green = True
    star_size = 16

    def __init__(self, raw, *p, **kw):
        # We want to ignore color_preprocessing_rop, instead use comet_preprocessing_rop
        # To track a comet, we remove all stars and leave the background, which hopefully
        # will be dominated by the comet itself. We do this in color preprocessing.
        # Then we extract large-scale features on that background through standard luma preprocessing

        self.star_size = int(kw.setdefault('star_size', self.star_size))
        stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(extraction.RemoveStarsRop, k)}
        kw.pop('color_preprocessing_rop', None)
        super(CometTrackingRop, self).__init__(raw, *p, **kw)

        comet_rop = kw.pop('comet_preprocessing_rop', False)
        if comet_rop is False:
            rops = [extraction.RemoveStarsRop(self.raw, copy=False, **stars_kw)]
            if self.extract_green and self._raw_pattern.max() > 1:
                # In RGB data we will track on G, which is cleaner for comets
                rops.append(ExtractChannelRop(self.raw, copy=False, raw_channels=False, channel=1))
            comet_rop = compound.CompoundRop(self.raw, *rops)
        self.color_preprocessing_rop = comet_rop
