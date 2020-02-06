# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os.path
import numpy
import scipy.ndimage
import skimage.feature
import skimage.transform
import logging
import PIL.Image

from cvastrophoto.image import rgb

from .base import BaseTrackingRop
from ..denoise import median

logger = logging.getLogger(__name__)

class CorrelationTrackingRop(BaseTrackingRop):

    reference = None
    tracking_cache = None
    track_distance = 1024
    resolution = 16
    save_tracks = False
    long_range = False
    add_bias = False

    def __init__(self, *p, **kw):
        pp_rop = kw.pop('luma_preprocessing_rop', False)
        if pp_rop is False:
            pp_rop = median.MaskedMedianFilterRop(rgb.Templates.LUMINANCE, size=3, sigma=1.0, copy=False)
        self.luma_preprocessing_rop = pp_rop

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

        if set_data:
            self.raw.set_raw_image(data, add_bias=self.add_bias)
        if luma is None:
            luma = self.raw.postprocessed_luma(copy=True)

            if self.luma_preprocessing_rop is not None:
                luma = self.luma_preprocessing_rop.correct(luma)

        logger.info("Tracking hint for %s: %r", img, hint[:2] if hint is not None else hint)

        if hint is None:
            # Find the brightest spot to build a tracking window around it
            margin = min(self.track_distance / 2, min(luma.shape) / 4)
            mluma = luma[margin:-margin, margin:-margin]
            pos = numpy.argmax(mluma)
            ymax = pos / mluma.shape[1]
            xmax = pos - ymax * mluma.shape[1]
            ymax += margin
            xmax += margin
            reftrackwin = None
            lyscale = lxscale = None

            vshape = self.raw.rimg.raw_image_visible.shape
            lshape = luma.shape
            ymax *= vshape[0] / lshape[0]
            xmax *= vshape[1] / lshape[1]
        else:
            ymax, xmax, yref, xref, (reftrackwin, lyscale, lxscale) = hint
            ymax = int(ymax)
            xmax = int(xmax)

        if lxscale is None or lyscale is None:
            vshape = self.raw.rimg.raw_image_visible.shape
            lshape = luma.shape
            self.lyscale = lyscale = vshape[0] / lshape[0]
            self.lxscale = lxscale = vshape[1] / lshape[1]

        rxmax = xmax
        rymax = ymax
        xmax /= lxscale
        ymax /= lyscale

        wleft = min(xmax, self.track_distance)
        wright = min(luma.shape[1] - xmax, self.track_distance)
        wup = min(ymax, self.track_distance)
        wdown = min(luma.shape[0] - ymax, self.track_distance)
        trackwin = luma[ymax-wup:ymax+wdown, xmax-wleft:xmax+wright]

        logger.info("Tracking window for %s: %d-%d, %d-%d (scale %d, %d)",
            img, xmax-wleft, xmax+wright, ymax-wup, ymax+wdown, lxscale, lyscale)

        # Heighten contrast
        trackwin = trackwin.astype(numpy.float32)
        trackwin -= trackwin.min()
        trackwin *= (1.0 / trackwin.ptp())

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
                corr = skimage.feature.register_translation(trackwin, reftrackwin, self.resolution)
            ytrack, xtrack = corr[0]

        # Translate to image space
        xoffs = xtrack + xref
        yoffs = ytrack + yref

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
        return dict(reference=self.reference, cache=self.tracking_cache)

    def load_state(self, state):
        self.reference = state['reference']
        self.tracking_cache = state['cache']

    def clear_cache(self):
        self.tracking_cache = None

    def translate_coords(self, bias, y, x):
        _, _, yoffs, xoffs, _ = bias
        _, _, yref, xref, (_, lyscale, lxscale) = self.reference

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
        xdrift = int(fxdrift / ysize) * ysize
        ydrift = int(fydrift / xsize) * xsize

        transform = skimage.transform.SimilarityTransform(
            translation=(-fxdrift/xsize, -fydrift/ysize))

        logger.info("Tracking offset for %s %r drift %r quantized drift %r",
            img, (xoffs, yoffs), (fxdrift, fydrift), (xdrift, ydrift))

        # move data - must be careful about copy direction
        for sdata in dataset:
            if sdata is None:
                # Multi-component data sets might have missing entries
                continue

            if fydrift or fxdrift:
                # Put sensible data into image margins to avoid causing artifacts at the edges
                self.demargin(sdata)

                for yoffs in xrange(ysize):
                    for xoffs in xrange(xsize):
                        scipy.ndimage.shift(
                            sdata[yoffs::ysize, xoffs::xsize],
                            [fydrift/ysize, fxdrift/xsize],
                            mode='reflect',
                            output=sdata[yoffs::ysize, xoffs::xsize])

        return rvdataset, transform
