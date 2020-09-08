# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import logging

from ..base import BaseRop
from cvastrophoto.util import demosaic, srgb
from cvastrophoto.image import rgb


logger = logging.getLogger(__name__)


class WhiteBalanceRop(BaseRop):

    protect_white = True
    wb_set = 'custom'

    _wb_coef = [1, 1, 1, 1]

    WB_SETS = {
        # These are based on the Starguider CLS filter
        'cls': (1, 0.8, 1.1, 1),
        'cls-drizzle-photometric': (0.94107851, 1, 0.67843978, 1),
        'cls-drizzle-perceptive': (1.8, 0.6, 0.67, 1),
        'mn34230-rgb': (2.0, 1.0, 2.0),
    }

    WB_ALIASES = {
        'mn34230-rgb': ['qhy163m-rgb', 'zwoasi1600-rgb'],
    }

    # The bluest component of the CLS filter is a bit greenish, there's no pure blue
    # This matrix accentuates pure blue to restore color balance on blue broadband sources
    CLS_MATRIX = numpy.array([
        [1.0, 0, 0],
        [0, 1, 0],
        [0, -0.56, 1.96],
    ], numpy.float32)

    WB_MATRICES = {
        'cls': CLS_MATRIX,
        'cls-drizzle-photometric': CLS_MATRIX,
        'cls-drizzle-perceptive': CLS_MATRIX,
    }

    WB_SETS.update({
        tgt: WB_SETS[src]
        for src, tgts in WB_ALIASES.items()
        for tgt in tgts
    })

    @property
    def wb_coef(self):
        return '/'.join(map(str, self._wb_coef))

    @wb_coef.setter
    def wb_coef(self, value):
        self._wb_coef = map(float, value.split('/'))

    def detect(self, data, **kw):
        pass

    def correct(self, data, detected=None, **kw):
        raw_pattern = self._raw_pattern
        raw_colors = self._raw_colors

        roi = kw.get('roi')

        def process_data(data):
            if roi is not None:
                data, eff_roi = self.roi_precrop(roi, data)

            if isinstance(self.raw, rgb.RGB):
                rgb_xyz_matrix = getattr(self.raw, 'lazy_rgb_xyz_matrix', None)
            else:
                rgb_xyz_matrix = None

            if rgb_xyz_matrix is not None:
                # Colorspace conversion, since we don't use rawpy's postprocessing we have to do it manually
                ppdata = demosaic.demosaic(data, raw_pattern)
                ppshape = ppdata.shape
                ppdata = ppdata.reshape((ppdata.shape[0], ppdata.shape[1] // 3, 3))
                ppdata = srgb.camera2rgb(ppdata, rgb_xyz_matrix, ppdata.copy()).reshape(ppshape)
                data = demosaic.remosaic(ppdata, raw_pattern, out=data)

            wb_coeffs = self.WB_SETS.get(self.wb_set, self._wb_coef)

            # Apply white balance coefficients, for both camera and filters
            wb_coeffs = numpy.array(wb_coeffs, numpy.float32)
            logger.debug("Applying WB: %r", wb_coeffs)

            fdata = data.astype(numpy.float32, copy=False)
            dmax = data.max()
            fdata *= wb_coeffs[raw_colors]

            if self.wb_set in self.WB_MATRICES and isinstance(self.raw, rgb.RGB):
                origshape = fdata.shape
                fdata = fdata.reshape((fdata.shape[0], fdata.shape[1] // 3, 3))
                fdata = srgb.color_matrix(fdata, self.WB_MATRICES[self.wb_set], fdata.copy()).reshape(origshape)

            if self.protect_white:
                fdata = numpy.clip(fdata, None, dmax, out=fdata)
            if data is not fdata:
                data[:] = fdata

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
