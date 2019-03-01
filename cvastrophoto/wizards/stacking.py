from __future__ import absolute_import

import os.path
import multiprocessing.pool

from .base import BaseWizard
from .. import raw

class StackingWizard(BaseWizard):

    def __init__(self, pool=None, denoise=True, quick=True, fbdd_noiserd=2):
        if pool is None:
            pool = multiprocessing.pool.ThreadPool()
        self.pool = pool
        self.denoise = denoise
        self.quick = quick
        self.fbdd_noiserd = fbdd_noiserd

    def load_set(self, base_path='.', light_path='Lights', dark_path='Darks'):
        self.lights = raw.Raw.open_all(os.path.join(base_path, light_path), default_pool=self.pool)

        if self.denoise and dark_path is not None:
            self.darks = raw.Raw.open_all(os.path.join(base_path, dark_path), default_pool=self.pool)
        else:
            self.darks = None

        self.lights[0].postprocessing_params.fbdd_noiserd = self.fbdd_noiserd

    def process(self):
        self.light_accum = raw.RawAccumulator()
        for light in self.lights:
            if self.denoise and self.darks is not None:
                light.denoise(self.darks, quick=self.quick)
            self.light_accum += light
            light.close()

    @property
    def accumulator(self):
        return self.light_accum

    @property
    def accum(self):
        return self.accumulator.accum

    def _get_raw_instance(self):
        return self.lights[0]
