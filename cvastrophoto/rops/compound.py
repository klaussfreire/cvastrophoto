# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .base import BaseRop

class CompoundRop(BaseRop):

    def __init__(self, raw, *rops):
        self.rops = rops
        super(CompoundRop, self).__init__(raw)

    def detect(self, data, **kw):
        rv = []
        for rop in self.rops:
            if data is None:
                rv.append(data)
            else:
                rv.append(rop.detect(data, **kw))
                if rv[-1] is None:
                    data = None
                else:
                    data = rop.correct(data, rv[-1])
        return rv

    def correct(self, data, detected=None, **kw):
        if detected is None:
            detected = [None] * len(self.rops)

        for rop, rop_detected in zip(self.rops, detected):
            data = rop.correct(data, rop_detected, **kw)
            if data is None:
                break

        return data

    def set_reference(self, data):
        for rop in self.rops:
            rop.set_reference(data)

    def get_state(self):
        return [rop.get_state() for rop in self.rops]

    def load_state(self, state):
        for rop_state, rop in zip(state, self.rops):
            rop.load_state(rop_state)
