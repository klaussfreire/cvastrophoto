# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import compound
from .base import BaseTrackingRop

class TrackingCompoundRop(BaseTrackingRop, compound.CompoundRop):

    def correct_with_transform(self, data, detected=None, **kw):
        if detected is None:
            detected = [None] * len(self.rops)

        cumulative_transform = None

        def get_new_data(data):
            if isinstance(data, list):
                new_data = []
                for sdata in data:
                    if sdata is not None:
                        sdata = sdata.copy()
                    new_data.append(sdata)
                    break

                # To get the transform, we only need the first element
                return new_data[:1]
            elif data is not None:
                return data.copy()

        # Make a copy of the data
        orig_data = data
        data = None
        prev_mtx_rop = None

        for rop, rop_detected in zip(self.rops, detected):
            if rop.is_matrix_transform:
                if data is None and rop.needs_data(rop_detected, **kw):
                    data = get_new_data(orig_data)
                    if cumulative_transform is not None:
                        data = prev_mtx_rop.apply_transform(data, cumulative_transform, **kw)

                prev_mtx_rop = rop
                transform = rop.detect_transform(data, rop_detected, **kw)
                if transform is None:
                    return None, None

                if cumulative_transform is None:
                    cumulative_transform = transform
                else:
                    cumulative_transform = transform + cumulative_transform

                data = None
            else:
                if cumulative_transform is not None:
                    orig_data = prev_mtx_rop.apply_transform(orig_data, cumulative_transform, **kw)

                orig_data = rop.correct(orig_data, rop_detected, **kw)
                data = None
                cumulative_transform = None

        del data

        if cumulative_transform is None:
            return orig_data, cumulative_transform
        else:
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
        for rop in reversed(self.rops):
            if rop.is_matrix_transform:
                break
        return rop.apply_transform(*p, **kw)

    def clear_cache(self):
        for rop in self.rops:
            rop.clear_cache()

    def set_tracking_cache(self, *p, **kw):
        for rop in self.rops:
            rop.set_tracking_cache(*p, **kw)
