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
            reftrackwin = refstars = refcentroids = None
        else:
            ymax, xmax, yref, xref, (reftrackwin, refstars, refcentroids) = hint
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
        trackwin *= 16384
        stars = scipy.ndimage.label(trackwin >= 0.25)
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
            xoffs = yoffs = noffs = 0
            for ytrack, xtrack in centroids:
                refytrack, refxtrack = min(refcentroids, key=lambda c:(
                    (c[0]-ytrack)*(c[0]-ytrack)
                    + (c[1]-xtrack)*(c[1]-xtrack)
                ))
                xoffs += xtrack - refxtrack
                yoffs += ytrack - refytrack
                noffs += 1
            if noffs > 0:
                xoffs /= noffs
                yoffs /= noffs
            xtrack = xoffs
            ytrack = yoffs

        # Translate to image space
        xoffs = xtrack - wleft + xmax
        yoffs = ytrack - wup + ymax

        return (yoffs, xoffs, yoffs, xoffs, (trackwin, stars, centroids))

    def correct(self, data, bias=None, img=None, save_tracks=None):
        if save_tracks is None:
            save_tracks = self.save_tracks

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

        yoffs, xoffs, _, _, _ = bias
        _, _, yref, xref, _ = self.reference

        fydrift = yref - yoffs
        fxdrift = xref - xoffs

        # Round to pattern shape to avoid channel crosstalk
        pattern_shape = self._raw_pattern.shape
        xdrift = int(fxdrift / pattern_shape[1]) * pattern_shape[1]
        ydrift = int(fydrift / pattern_shape[0]) * pattern_shape[0]

        logger.info("Tracking offset for %s %r drift %r quantized drift %r",
            img, (xoffs, yoffs), (fxdrift, fydrift), (xdrift, ydrift))

        if ydrift or xdrift:
            # Put sensible data into image margins to avoid causing artifacts at the edges
            self.raw.demargin(data)

        # move data - must be careful about copy direction
        if ydrift > 0:
            if xdrift > 0:
                data[:ydrift-1:-1,xdrift:] = data[-ydrift-1::-1,:-xdrift]
            elif xdrift < 0:
                data[:ydrift-1:-1,:xdrift] = data[-ydrift-1::-1,-xdrift:]
            else:
                data[:ydrift-1:-1] = data[-ydrift-1::-1]
        elif ydrift < 0:
            if xdrift > 0:
                data[:ydrift,xdrift:] = data[-ydrift:,:-xdrift]
            elif xdrift < 0:
                data[:ydrift,:xdrift] = data[-ydrift:,-xdrift:]
            else:
                data[:ydrift] = data[-ydrift:]
        elif xdrift > 0:
            data[:,:xdrift-1:-1] = data[:,-xdrift-1::-1]
        elif xdrift < 0:
            data[:,:xdrift] = data[:,-xdrift:]

        return data
