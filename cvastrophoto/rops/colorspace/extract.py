# -*- coding: utf-8 -*-
from __future__ import absolute_import

from past.builtins import xrange
import numpy

from ..base import BaseRop
from cvastrophoto.util import demosaic, srgb


class ExtractChannelRop(BaseRop):

    channel = 0
    raw_channels = True

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        raw_pattern = self._raw_pattern

        roi = kw.get('roi')

        def process_data(data):
            if roi is not None:
                data, eff_roi = self.roi_precrop(roi, data)

            ppdata = demosaic.demosaic(data, raw_pattern)

            if not self.raw_channels:
                rgb_xyz_matrix = getattr(self.raw, 'lazy_rgb_xyz_matrix', None)
                if rgb_xyz_matrix is None:
                    rgb_xyz_matrix = getattr(self.raw.rimg, 'rgb_xyz_matrix', None)
                if rgb_xyz_matrix is not None:
                    ppmax = ppdata.max()
                    pptype = ppdata.dtype
                    ppdata = srgb.camera2rgb(ppdata, rgb_xyz_matrix, ppdata.astype(numpy.float32))
                    if pptype.kind in ('i', 'u'):
                        ppdata = numpy.clip(ppdata, 0, ppmax, out=ppdata)
                    ppdata = ppdata.astype(pptype, copy=False)

            cdata = ppdata[:,:,self.channel]
            for c in xrange(ppdata.shape[2]):
                if c != self.channel:
                    ppdata[:,:,c] = cdata

            data = demosaic.remosaic(ppdata, raw_pattern, out=data)

            if roi is not None:
                data = self.roi_postcrop(roi, eff_roi, data)

            return data

        rv = data

        if not isinstance(data, list):
            data = [data]

        for sdata in data:
            if sdata is None:
                continue

            sdata = process_data(sdata)

        return rv
