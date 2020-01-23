# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import compound
from .base import BaseTrackingRop

class TrackingCompoundRop(BaseTrackingRop, compound.CompoundRop):

    def correct_with_transform(self, data, detected=None, **kw):
        if detected is None:
            detected = [None] * len(self.rops)

        cumulative_transform = None

        # Make a copy of the data
        if isinstance(data, list):
            orig_data = []
            for sdata in data:
                if sdata is not None:
                    sdata = sdata.copy()
                orig_data.append(sdata)

            # To get the transform, we only need the first element
            data = data[:1]
        elif data is not None:
            orig_data = data.copy()

        for rop, rop_detected in zip(self.rops, detected):
            data, transform = rop.correct_with_transform(data, rop_detected, **kw)
            if data is None:
                return None, None

            if cumulative_transform is None:
                cumulative_transform = transform
            else:
                cumulative_transform += transform

        # Apply the cumulative transform with the last rop
        return rop.apply_transform(orig_data, cumulative_transform, **kw), cumulative_transform

    def apply_transform(self, *p, **kw):
        return self.rops[-1].apply_transform(*p, **kw)

    def clear_cache(self):
        for rop in self.rops:
            rop.clear_cache()
