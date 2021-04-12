# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import numpy.linalg
import logging

from ..base import BaseRop
from cvastrophoto.util import demosaic, srgb
from cvastrophoto.image import rgb
from cvastrophoto.rops.tracking import extraction


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
        'mn34230-rgb': (4.738198288082122, 1.4819804844645177, 1.0),
        'mn34230-orion5561': (1.6570478579950418, 1.0, 1.0236022889123377, 1.0),
    }

    WB_ALIASES = {
        'mn34230-rgb': ['qhy163m-rgb', 'zwoasi1600-rgb'],
        'mn34230-orion5561': [
            'qhy163m-orion5561', 'zwoasi1600-orion5561',
            'qhy163m-lpro', 'zwoasi1600-lpro',
            'qhy163m-idas-lps', 'zwoasi1600-idas-lps',
        ],
    }

    # The bluest component of the CLS filter is a bit greenish, there's no pure blue
    # This matrix accentuates pure blue to restore color balance on blue broadband sources
    CLS_MATRIX = numpy.array([
        [1.0, 0, 0],
        [0, 1, 0],
        [0, -0.56, 1.96],
    ], numpy.float32)

    ZWORGB_O3_FIX_MATRIX = numpy.array([
        [1.0, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ], numpy.float32)

    SHO_2_HSO_MATRIX = numpy.array([
        [1, 1, 0],
        [0.5, 0, 1],
        [0, 0, 1],
    ], numpy.float32)

    OSC_RGB_XYZ_MATRIX = numpy.array([
        [0.6602, -0.0841, -0.0939],
        [-0.4472,  1.2458,  0.2247],
        [-0.0975,  0.2039,  0.6148],
    ], dtype=numpy.float32)

    CLS_RGB_XYZ_MATRIX = numpy.array([
        [0.6602, -0.0241, -0.0239],
        [-0.4472,  1.2458,  0.2247],
        [-0.0975,  0.2039,  0.6148],
    ], dtype=numpy.float32)

    OSC_RGB_MATRIX = numpy.matmul(srgb.MATRIX_XYZ2RGB, numpy.linalg.inv(OSC_RGB_XYZ_MATRIX))
    CLS_RGB_MATRIX = numpy.matmul(srgb.MATRIX_XYZ2RGB, numpy.linalg.inv(CLS_RGB_XYZ_MATRIX))

    WB_MATRICES = {
        'cls': CLS_MATRIX,
        'cls-drizzle-photometric': CLS_MATRIX,
        'cls-drizzle-perceptive': CLS_MATRIX,
        'zworgb-o3-fix': ZWORGB_O3_FIX_MATRIX,
        'sho-2-hso': SHO_2_HSO_MATRIX,
        'canon-650d': OSC_RGB_MATRIX,
        'osc': OSC_RGB_MATRIX,
        'osc-cls': CLS_RGB_MATRIX,
        'nikon-d5500': srgb.matrix_wb(OSC_RGB_MATRIX, (1.0, 1.1870283084818334, 1.9449060208719295), 4),
    }

    for tgts in WB_MATRICES:
        WB_SETS.setdefault(tgts, (1, 1, 1))

    for src, tgts in WB_ALIASES.items():
        for tgt in tgts:
            if src in WB_SETS:
                WB_SETS[tgt] = WB_SETS[src]
            if src in WB_MATRICES:
                WB_MATRICES[tgt] = WB_MATRICES[src]
    del src, tgt, tgts

    @property
    def wb_coef(self):
        return '/'.join(map(str, self._wb_coef))

    @wb_coef.setter
    def wb_coef(self, value):
        self._wb_coef = tuple(map(float, value.split('/')))

    def detect(self, data, **kw):
        pass

    def _wb_method_auto(self, data):
        star_rop = extraction.ExtractPureStarsRop(self.raw, copy=False)
        star_data = star_rop.correct(data.copy())

        if len(data.shape) == 3:
            ravg = numpy.average(star_data[:,:,0])
            gavg = numpy.average(star_data[:,:,1])
            bavg = numpy.average(star_data[:,:,2])
        else:
            ravg = numpy.average(star_data[self.rmask_image])
            gavg = numpy.average(star_data[self.gmask_image])
            bavg = numpy.average(star_data[self.bmask_image])

        maxavg = float(max(ravg, gavg, bavg))
        if maxavg:
            ravg = maxavg / ravg
            gavg = maxavg / gavg
            bavg = maxavg / bavg
        else:
            ravg = gavg = bavg = 1

        return [ravg, gavg, bavg, gavg]

    def _wb_method_autobg(self, data):
        star_rop = extraction.ExtractPureStarsRop(self.raw, copy=False)
        star_data = star_rop.correct(data.copy())
        bg_data = data - star_data

        if len(data.shape) == 3:
            ravg = numpy.average(bg_data[:,:,0])
            gavg = numpy.average(bg_data[:,:,1])
            bavg = numpy.average(bg_data[:,:,2])
        else:
            ravg = numpy.average(bg_data[self.rmask_image])
            gavg = numpy.average(bg_data[self.gmask_image])
            bavg = numpy.average(bg_data[self.bmask_image])

        maxavg = float(max(ravg, gavg, bavg))
        if maxavg:
            ravg = maxavg / ravg
            gavg = maxavg / gavg
            bavg = maxavg / bavg
        else:
            ravg = gavg = bavg = 1

        return [ravg, gavg, bavg, gavg]

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

            base_coeffs = self._wb_coef
            wb_method = getattr(self, '_wb_method_' + self.wb_set, None)
            if wb_method is not None:
                wb_coeffs = wb_method(data)
                logger.info("Applying WB coefs: %r (%s)", wb_coeffs, self.wb_set)
            else:
                if self.wb_set != 'custom' and self.wb_set not in self.WB_SETS:
                    logger.warning("Unrecognized WB set ignored: %r", self.wb_set)
                wb_coeffs = self.WB_SETS.get(self.wb_set, [1, 1, 1, 1])

            # Apply white balance coefficients, for both camera and filters
            wb_coeffs = numpy.array(wb_coeffs, numpy.float32)
            wb_coeffs *= numpy.array(base_coeffs, numpy.float32)[:len(wb_coeffs)]
            logger.debug("Applying WB: %r", wb_coeffs)

            fdata = data.astype(numpy.float32, copy=False)
            dmax = data.max()

            if len(fdata.shape) == 3:
                for c in range(fdata.shape[2]):
                    fdata[:,:,c] *= wb_coeffs[c]
            else:
                fdata *= wb_coeffs[raw_colors]

            if self.wb_set in self.WB_MATRICES:
                if isinstance(self.raw, rgb.RGB):
                    remosaic_fdata = False
                    origshape = fdata.shape
                    if len(fdata.shape) < 3:
                        fdata = fdata.reshape((fdata.shape[0], fdata.shape[1] // 3, 3))
                else:
                    remosaic_fdata = True
                    fdata = demosaic.demosaic(fdata, self._raw_pattern)

                if len(fdata.shape) == 3 and fdata.shape[2] >= 3:
                    fdata = srgb.color_matrix(fdata, self.WB_MATRICES[self.wb_set], fdata.copy())
                else:
                    logger.warning("Not applying WB matrix to incompatible image shape: %r", fdata.shape)

                if remosaic_fdata:
                    fdata = demosaic.remosaic(fdata, self._raw_pattern)
                else:
                    fdata = fdata.reshape(origshape)

            if self.protect_white:
                fdata = numpy.clip(fdata, None, dmax, out=fdata)
            if data is not fdata:
                if data.dtype.kind in 'iu':
                    limits = numpy.iinfo(data.dtype)
                    fdata = numpy.clip(fdata, limits.min, limits.max, out=fdata)
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
