# -*- coding: utf-8 -*-
from __future__ import absolute_import

from ..base import BaseRop
from cvastrophoto.util import demosaic


class ExtractChannelRop(BaseRop):

    channel = 0

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        raw_pattern = self._raw_pattern

        roi = kw.get('roi')

        def process_data(data):
            if roi is not None:
                data, eff_roi = self.roi_precrop(roi, data)

            ppdata = demosaic.demosaic(data, raw_pattern)

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
