# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import logging

from ..base import BaseRop

logger = logging.getLogger('cvastrophoto.tracking')

class CentroidTrackingRop(BaseRop):

    reference = None
    track_distance = 128

    def set_reference(self, data):
        if data is not None:
            self.reference = self.detect(data)
        else:
            self.reference = None

    def detect(self, data, hint=None):
        self.raw.set_raw_image(data)
        luma = numpy.sum(self.raw.postprocessed, axis=2, dtype=numpy.uint32)

        margin = min(128, min(luma.shape) / 2)
        luma = luma[margin:-margin, margin:-margin]

        if hint is None:
            # Find the brightest spot to build a tracking window around it
            pos = numpy.argmax(luma)
            ymax = pos / luma.shape[1]
            xmax = pos - ymax * luma.shape[1]
        else:
            ymax, xmax = hint
            ymax = int(ymax - margin)
            xmax = int(xmax - margin)

        wleft = min(xmax, self.track_distance)
        wright = min(luma.shape[1] - xmax, self.track_distance)
        wup = min(ymax, self.track_distance)
        wdown = min(luma.shape[0] - ymax, self.track_distance)
        trackwin = luma[ymax-wup:ymax+wdown, xmax-wleft:xmax+wright]

        # Heighten contrast
        trackwin -= numpy.minimum(trackwin, numpy.average(trackwin).astype(trackwin.dtype))

        # Find centroid
        axx = numpy.arange(trackwin.shape[1])
        axy = numpy.arange(trackwin.shape[0])
        wtrackwin = numpy.sum(trackwin)
        xtrack = numpy.sum(trackwin * axx) / wtrackwin
        ytrack = numpy.sum(trackwin.transpose() * axy) / wtrackwin
        del axx, axy

        # Translate to image space
        xoffs = xtrack - wleft + xmax + margin
        yoffs = ytrack - wup + ymax + margin

        return (yoffs, xoffs)

    def correct(self, data, bias=None, img=None):
        if bias is None:
            bias = self.detect(data, hint=self.reference)
        if self.reference is None:
            # re-detect with hint, initial detection isn't so great
            self.reference = bias
            bias = self.detect(data, hint=bias)

        yoffs, xoffs = bias
        yref, xref = self.reference

        fydrift = yref - yoffs
        fxdrift = xref - xoffs

        # Round to pattern shape to avoid channel crosstalk
        pattern_shape = self._raw_pattern.shape
        xdrift = int(fxdrift / pattern_shape[1]) * pattern_shape[1]
        ydrift = int(fydrift / pattern_shape[0]) * pattern_shape[0]

        logger.info("Tracking offset for %s %r quantized drift %r",
            img, (xoffs, yoffs), (xdrift, ydrift))

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
