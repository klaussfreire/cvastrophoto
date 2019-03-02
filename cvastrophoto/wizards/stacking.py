from __future__ import absolute_import

import os.path
import multiprocessing.pool

from .base import BaseWizard
from .. import raw

class StackingWizard(BaseWizard):

    def __init__(self, pool=None, denoise=True, quick=True, fbdd_noiserd=2,
            tracking_class=None):
        if pool is None:
            pool = multiprocessing.pool.ThreadPool()
        self.pool = pool
        self.denoise = denoise
        self.quick = quick
        self.fbdd_noiserd = fbdd_noiserd
        self.tracking_class = tracking_class

    def load_set(self, base_path='.', light_path='Lights', dark_path='Darks'):
        self.lights = raw.Raw.open_all(os.path.join(base_path, light_path), default_pool=self.pool)

        if self.tracking_class is not None:
            self.tracking = self.tracking_class(self.lights[0])
        else:
            self.tracking = None

        if self.denoise and dark_path is not None:
            self.darks = raw.Raw.open_all(os.path.join(base_path, dark_path), default_pool=self.pool)
        else:
            self.darks = None

        self.lights[0].postprocessing_params.fbdd_noiserd = self.fbdd_noiserd

    def process(self):
        self.light_accum = raw.RawAccumulator()

        if self.tracking is not None:
            self.tracking.set_reference(None)

        for light in self.lights:
            if self.denoise and self.darks is not None:
                light.denoise(self.darks, quick=self.quick)
            if self.tracking is not None:
                self.tracking.correct(light.rimg.raw_image, img=light)

            self.light_accum += light
            light.close()

        # Release resources until needed again
        for dark in self.darks:
            dark.close()

    @property
    def accumulator(self):
        return self.light_accum

    @property
    def accum(self):
        return self.accumulator.accum

    def _get_raw_instance(self):
        return self.lights[0]
