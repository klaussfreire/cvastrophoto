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
            new_data = []
            for sdata in data:
                if sdata is not None:
                    sdata = sdata.copy()
                new_data.append(sdata)
                break

            # To get the transform, we only need the first element
            orig_data = data
            data = new_data[:1]
            sdata = None
        elif data is not None:
            orig_data = data
            data = data.copy()

        for rop, rop_detected in zip(self.rops, detected):
            data, transform = rop.correct_with_transform(data, rop_detected, **kw)
            if data is None:
                return None, None

            if cumulative_transform is None:
                cumulative_transform = transform
            else:
                cumulative_transform = transform + cumulative_transform

        del data

        # Apply the cumulative transform with the last rop
        return rop.apply_transform(orig_data, cumulative_transform, **kw), cumulative_transform

    @property
    def save_tracks(self):
        for rop in self.rops:
            if rop.save_tracks:
                return True
        else:
            return False

    @save_tracks.setter
    def save_tracks(self, value):
        for rop in self.rops:
            rop.save_tracks = value

    def apply_transform(self, *p, **kw):
        return self.rops[-1].apply_transform(*p, **kw)

    def clear_cache(self):
        for rop in self.rops:
            rop.clear_cache()
