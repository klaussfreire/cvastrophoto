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

from .base import BaseTrackingMatrixRop
from . import extraction

logger = logging.getLogger(__name__)

class CentroidTrackingRop(BaseTrackingMatrixRop):

    reference = None
    track_distance = 256
    recenter_limit = None
    save_tracks = False
    long_range = False
    add_bias = False

    def __init__(self, *p, **kw):
        pp_rop = kw.pop('luma_preprocessing_rop', False)
        if pp_rop is False:
            pp_rop = extraction.ExtractStarsRop(rgb.Templates.LUMINANCE, copy=False, pre_demargin=False)
        self.luma_preprocessing_rop = pp_rop
        self.color_preprocessing_rop = kw.pop('color_preprocessing_rop', None)

        super(CentroidTrackingRop, self).__init__(*p, **kw)

    def set_reference(self, data):
        if data is not None:
            if isinstance(data, tuple) and len(data) == 2:
                # Explicit starting point
                self.reference = data + data + ((None, None, None, None),)
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

        if hint is None:
            # Find the brightest spot to build a tracking window around it
            margin = min(128 + self.track_distance, min(luma.shape) // 4)
            mluma = luma[margin:-margin, margin:-margin]
            pos = numpy.argmax(mluma)
            ymax = pos // mluma.shape[1]
            xmax = pos - ymax * mluma.shape[1]
            ymax += margin
            xmax += margin
            refstars = refcentroids = None
            lyscale = lxscale = None
        else:
            ymax, xmax, yref, xref, (refstars, refcentroids, lyscale, lxscale) = hint
            ymax = int(ymax)
            xmax = int(xmax)

        if lxscale is None or lyscale is None:
            vshape = self.lraw.rimg.raw_image_visible.shape
            lshape = luma.shape
            lyscale = vshape[0] / lshape[0]
            lxscale = vshape[1] / lshape[1]

        wleft = min(xmax, self.track_distance)
        wright = min(luma.shape[1] - xmax, self.track_distance)
        wup = min(ymax, self.track_distance)
        wdown = min(luma.shape[0] - ymax, self.track_distance)
        originy = ymax-wup
        originx = xmax-wleft
        trackwin = luma[ymax-wup:ymax+wdown, xmax-wleft:xmax+wright]

        self._lock_region = (ymax-wup, xmax-wleft, ymax+wdown, xmax+wright)

        imglog = img if hasattr(img, 'name') else None
        logger.info("Tracking window for %s: %d-%d, %d-%d (scale %d, %d)",
            imglog, xmax-wleft, xmax+wright, ymax-wup, ymax+wdown, lxscale, lyscale)

        # Heighten contrast
        thresh = trackwin.min() + trackwin.ptp()/2
        trackwin -= numpy.minimum(trackwin, thresh.astype(trackwin.dtype))
        trackwin -= trackwin.min()
        trackwin = trackwin.astype(numpy.float32)
        trackwin *= (1.0 / trackwin.ptp())
        stars = scipy.ndimage.label(trackwin >= 0.25)
        trackwin *= 16384
        trackwin = trackwin.astype(numpy.int32)
        centroids = scipy.ndimage.center_of_mass(trackwin, stars[0], range(1, stars[1]+1))

        # Translate to image space
        centroids = [(y+originy, x+originx) for y,x in centroids]

        if img is not None and save_tracks:
            try:
                PIL.Image.fromarray(
                        ((trackwin - trackwin.min()) * 255 / trackwin.ptp()).astype(numpy.uint8)
                    ).save('Tracks/%s.jpg' % os.path.basename(img.name))
            except Exception:
                logger.exception("Can't save tracks due to error")

        logger.debug("Found %d stars with pos %r for %s", stars[1], centroids, img)
        if refcentroids is None:
            # Global centroid to center star group in track window
            ytrack, xtrack = scipy.ndimage.center_of_mass(trackwin)
            ytrack += originy
            xtrack += originx
        else:
            # Find centroid
            xoffs = []
            yoffs = []
            for ytrack, xtrack in centroids:
                refytrack, refxtrack = min(refcentroids, key=lambda c:(
                    (c[0]-ytrack)*(c[0]-ytrack)
                    + (c[1]-xtrack)*(c[1]-xtrack)
                ))
                xoffs.append(xtrack - refxtrack)
                yoffs.append(ytrack - refytrack)
            if xoffs:
                xoffs = numpy.median(xoffs)
                yoffs = numpy.median(yoffs)
            else:
                xoffs = yoffs = 0
            xtrack = xoffs
            ytrack = yoffs

        xoffs = xtrack
        yoffs = ytrack

        return (yoffs, xoffs, yoffs, xoffs, (stars, centroids, lyscale, lxscale))

    def detect(self, data, bias=None, img=None, save_tracks=None, set_data=True, luma=None, **kw):
        if isinstance(data, list):
            data = data[0]

        if self.tracking_cache is None:
            self.tracking_cache = {}

        tracking_key = self._tracking_key(img or data, self.reference)
        if bias is None and self.reference is not None:
            bias = self.tracking_cache.get(tracking_key)

        set_data = True
        if bias is None or self.reference[-1][0] is None:
            bias = self._detect(data, hint=self.reference, save_tracks=save_tracks, img=img, luma=luma)
            set_data = False

        missing_reference = self.reference is None or self.reference[-1][0] is None
        if missing_reference:
            self.reference = bias

            # re-detect with hint, as would be done if reference had been initialized above
            # reset reference track window and star information with proper tracking center
            bias = self._detect(data, hint=bias, save_tracks=False, set_data=set_data, img=img, luma=luma)
            self.reference = self.reference[:-3] + bias[-3:]

            bias = self._detect(data, hint=self.reference, save_tracks=save_tracks, set_data=False, img=img, luma=luma)
            set_data = False

        rv = bias = self.tracking_cache.setdefault(tracking_key, bias)

        yoffs, xoffs, _, _, _ = bias
        _, _, yref, xref, (_, _, lyscale, lxscale) = self.reference

        fydrift = yref - yoffs
        fxdrift = xref - xoffs

        recenter_limit = self.recenter_limit or self.track_distance/4
        recenter_limit = 0
        if self.long_range and not missing_reference and max(abs(fydrift), abs(fxdrift)) > recenter_limit:
            # Rough adjustment of the new reference frame
            ymax, xmax = self.reference[:2]
            xmax -= int(fxdrift)
            ymax -= int(fydrift)
            newref = (ymax, xmax, ymax, xmax, (None, None, None, None))

            # Fine adjustment through re-detection
            # reset reference track window and star information with proper tracking center
            bias = self._detect(data, hint=newref, save_tracks=False, set_data=set_data, img=img, luma=luma)
            bias = self._detect(data, hint=bias, save_tracks=False, set_data=set_data, img=img, luma=luma)
            nymax, nxmax, nyref, nxref, nrefdata = bias
            self.reference = (ymax, xmax, nyref-int(fydrift), nxref-int(fxdrift), nrefdata)

        self.tracking_cache.setdefault(tracking_key, self._cache_clean(rv))
        return rv

    def translate_coords(self, bias, y, x):
        yoffs, xoffs, _, _, _ = bias
        _, _, yref, xref, (_, _, lyscale, lxscale) = self.reference

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
            bias = self.detect(data, bias=self.reference, save_tracks=save_tracks, img=img)

        yoffs, xoffs, _, _, _ = bias
        _, _, yref, xref, (_, _, lyscale, lxscale) = self.reference

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

        imglog = img if hasattr(img, 'name') else None
        logger.info("Tracking offset for %s %r drift %r quantized drift %r",
            imglog, (xoffs, yoffs), (fxdrift, fydrift), (xdrift, ydrift))

        return transform
