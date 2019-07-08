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
            rv.append(rop.detect(data, **kw))
            data = rop.correct(data, rv[-1])
        return rv

    def correct(self, data, detected=None, **kw):
        if detected is None:
            detected = [None] * len(self.rops)

        for rop, rop_detected in zip(self.rops, detected):
            data = rop.correct(data, rop_detected, **kw)

        return data

    def set_reference(self, data):
        self.rops[0].set_reference(data)