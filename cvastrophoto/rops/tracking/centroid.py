# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os.path
import numpy
import scipy.ndimage
import logging
import PIL.Image

from ..base import BaseRop

logger = logging.getLogger('cvastrophoto.tracking')

class CentroidTrackingRop(BaseRop):

    reference = None
    track_distance = 256
    save_tracks = True

    def set_reference(self, data):
        if data is not None:
            self.reference = self.detect(data)
        else:
            self.reference = None
        self.tracking_cache = {}

    def _tracking_key(self, data, hint):
        return (
            getattr(data, 'name', id(data)),
            hint[:4] if hint is not None else None,
        )

    def detect(self, data, hint=None, img=None, save_tracks=None, set_data=True):
        if save_tracks is None:
            save_tracks = self.save_tracks

        if set_data:
            self.raw.set_raw_image(data)
        luma = numpy.sum(self.raw.postprocessed, axis=2, dtype=numpy.uint32)

        if hint is None:
            # Find the brightest spot to build a tracking window around it
            margin = min(128 + self.track_distance, min(luma.shape) / 4)
            mluma = luma[margin:-margin, margin:-margin]
            maxluma = mluma.max()
            maxpos = (mluma == maxluma).nonzero()[0]
            pos = maxpos[len(maxpos)/2]
            del maxpos
            pos = numpy.argmax(mluma)
            ymax = pos / mluma.shape[1]
            xmax = pos - ymax * mluma.shape[1]
            ymax += margin
            xmax += margin
            refstars = refcentroids = None
        else:
            ymax, xmax, yref, xref, (refstars, refcentroids) = hint
            ymax = int(ymax)
            xmax = int(xmax)

        wleft = min(xmax, self.track_distance)
        wright = min(luma.shape[1] - xmax, self.track_distance)
        wup = min(ymax, self.track_distance)
        wdown = min(luma.shape[0] - ymax, self.track_distance)
        trackwin = luma[ymax-wup:ymax+wdown, xmax-wleft:xmax+wright]

        logger.info("Tracking window for %s: %d-%d, %d-%d",
            img, xmax-wleft, xmax+wright, ymax-wup, ymax+wdown)

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

        # Translate to image space
        xoffs = xtrack - wleft + xmax
        yoffs = ytrack - wup + ymax

        return (yoffs, xoffs, yoffs, xoffs, (stars, centroids))

    def correct(self, data, bias=None, img=None, save_tracks=None):
        if save_tracks is None:
            save_tracks = self.save_tracks

        dataset = rvdataset = data
        if isinstance(data, list):
            data = data[0]
        else:
            dataset = [data]

        tracking_key = self._tracking_key(img or data, self.reference)
        if bias is None and self.reference is not None:
            bias = self.tracking_cache.get(tracking_key)

        set_data = True
        if bias is None:
            bias = self.detect(data, hint=self.reference, save_tracks=save_tracks, img=img)
            set_data = False

        if self.reference is None:
            self.reference = bias

            # re-detect with hint, as would be done if reference had been initialized above
            # reset reference track window and star information with proper tracking center
            bias = self.detect(data, hint=bias, save_tracks=False, set_data=set_data, img=img)
            self.reference = self.reference[:-3] + bias[-3:]

            bias = self.detect(data, hint=self.reference, save_tracks=save_tracks, set_data=False, img=img)

        self.tracking_cache.setdefault(tracking_key, bias)

        yoffs, xoffs, _, _, _ = bias
        _, _, yref, xref, _ = self.reference

        fydrift = yref - yoffs
        fxdrift = xref - xoffs

        # Round to pattern shape to avoid channel crosstalk
        pattern_shape = self._raw_pattern.shape
        ysize, xsize = pattern_shape
        xdrift = int(fxdrift / ysize) * ysize
        ydrift = int(fydrift / xsize) * xsize

        logger.info("Tracking offset for %s %r drift %r quantized drift %r",
            img, (xoffs, yoffs), (fxdrift, fydrift), (xdrift, ydrift))

        # move data - must be careful about copy direction
        for sdata in dataset:
            if sdata is None:
                # Multi-component data sets might have missing entries
                continue

            if fydrift or fxdrift:
                # Put sensible data into image margins to avoid causing artifacts at the edges
                self.raw.demargin(sdata)

                for yoffs in xrange(ysize):
                    for xoffs in xrange(xsize):
                        scipy.ndimage.shift(
                            data[yoffs::ysize, xoffs::xsize],
                            [fydrift/ysize, fxdrift/xsize],
                            mode='reflect',
                            output=data[yoffs::ysize, xoffs::xsize])

        return rvdataset
