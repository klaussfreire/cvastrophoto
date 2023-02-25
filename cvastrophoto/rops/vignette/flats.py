from __future__ import absolute_import

import logging
from past.builtins import xrange

import numpy

from ..base import BaseRop

from cvastrophoto.image import Image
from cvastrophoto.util import demosaic, gaussian
from cvastrophoto.library import flats

logger = logging.getLogger(__name__)

class FlatImageRop(BaseRop):

    scale = None
    dtype = numpy.float32
    gauss_size = 0
    pattern_size = 1
    min_luma = 5
    min_luma_ratio = 0.05
    remove_bias = False
    normalize = True
    local_normalization = False
    local_size = 256
    pedestal = 0
    master_flat = ''
    use_lib = False

    def __init__(self, raw=None, flat=None, color=False, flat_rop=None, scale=None, flat_library=None, **kw):
        super(FlatImageRop, self).__init__(raw, **kw)
        self.flat_rop = flat_rop
        if scale is not None:
            self.scale = float(scale)
        if flat is None and self.master_flat:
            flat_img = Image.open(self.master_flat)
            flat = flat_img.rimg.raw_image
            if flat_img.rimg.raw_image.shape != self.raw.rimg.raw_image.shape:
                flat = demosaic.demosaic(flat, flat_img.rimg.raw_pattern)
                flat = demosaic.remosaic(flat, self._raw_pattern)
                sizes = self._raw_sizes
                if flat.shape != (sizes.raw_height, sizes.raw_width):
                    flat = numpy.pad(
                        flat,
                        [
                            (sizes.top_margin, sizes.raw_height - flat.shape[0] - sizes.top_margin),
                            (sizes.left_margin, sizes.raw_width - flat.shape[1] - sizes.left_margin),
                        ],
                        mode='edge',
                    )
        self.set_flat(flat)

        if self.use_lib and flat_library is None:
            flat_library = flats.FlatLibrary()
        self.flat_library = flat_library

    def set_flat(self, flat):
        self.flat = flat
        self.flat_luma = self._flat_luma(flat)

    def _flat_luma(self, flat, scale=None):
        if flat is None:
            return None

        if self.raw.default_pool is not None:
            map_ = self.raw.default_pool.imap_unordered
        else:
            map_ = map

        if flat.max() > 65535:
            flat = flat * (65535.0 / flat.max())
        if scale:
            flat = flat * float(scale)
        self.raw.set_raw_image(flat, add_bias=True)
        flatpp = self.raw.postprocessed
        if (flatpp == 65535).any():
            # Saturated flat fields are bad. Some cameras however have more
            # dynamic range in the raw than accessible through postprocessed
            # values, but issue a warning just in case
            logger.warning(
                "Overexposed flat field, shifting exposure down to try to recover. "
                "Overexposure should be avoided in flat fields in any case.")
        del flatpp

        luma = self.raw.luma_image(flat, dtype=numpy.float32)
        luma = self.demargin(luma)
        if self.pedestal:
            luma += self.pedestal
        luma *= 65535.0 / luma.max()

        def fix_holes(luma):
            min_luma = max(self.min_luma, self.min_luma_ratio * numpy.average(luma))
            if luma.min() <= min_luma:
                # cover holes
                bad_luma = luma <= min_luma
                luma[bad_luma] = luma[~bad_luma].min()
            return luma

        luma = fix_holes(luma)

        path, patw = self._raw_pattern.shape

        if self.flat_rop is not None:
            luma = self.flat_rop.correct(luma)
            luma = self.demargin(luma)
            luma = fix_holes(luma)

        if self.pattern_size > max(path, patw):
            gpath = gpatw = self.pattern_size
        else:
            gpath, gpatw = path, patw
        tasks = [(y, x) for y in xrange(gpath) for x in xrange(gpatw)]

        if self.gauss_size:
            luma = self.demargin(luma)
            def gaussify(task):
                y, x = task
                if self.gauss_size > max(luma.shape):
                    # Simplify gigantic smoothing with an average
                    luma[y::gpath, x::gpatw] = numpy.average(luma[y::gpath, x::gpatw])
                else:
                    luma[y::gpath, x::gpatw] = gaussian.fast_gaussian(
                        luma[y::gpath, x::gpatw], self.gauss_size, mode='nearest')
            for _ in map_(gaussify, tasks):
                pass
            luma = fix_holes(luma)

        if self.local_normalization:
            luma = self.demargin(luma)
            local_luma = numpy.empty_like(luma)
            def local_normalize(task):
                y, x = task
                if self.gauss_size > max(luma.shape):
                    # Simplify gigantic smoothing with an average
                    local_luma[y::gpath, x::gpatw] = numpy.average(luma[y::gpath, x::gpatw])
                else:
                    local_luma[y::gpath, x::gpatw] = gaussian.fast_gaussian(
                        luma[y::gpath, x::gpatw], self.local_size, mode='nearest')
            for _ in map_(local_normalize, tasks):
                pass
            local_luma = fix_holes(local_luma)
            for y, x in tasks:
                local_luma[y::gpath, x::gpatw] *= 1.0 / numpy.average(local_luma[y::gpath, x::gpatw])
            luma = numpy.divide(luma, local_luma)

        luma *= (1.0 / luma.max())

        return luma

    def detect(self, data, **kw):
        pass

    def correct(self, data, flat=None, img=None, **kw):
        if flat is None:
            flat_luma = self.flat_luma
        else:
            flat_luma = self._flat_luma(flat)
        if flat_luma is None and self.use_lib and img is not None:
            master_flat = self.flat_library.get_master(self.flat_library.classify_frame(img), raw=img)
            if master_flat is not None:
                flat_luma = self._flat_luma(master_flat.rimg.raw_image)

        return self.flatten(data, flat_luma)

    def flatten(self, light, luma, dtype=None, scale=None):
        if scale is None:
            scale = self.scale
        if dtype is None:
            dtype = self.dtype

        origmax = light.max()
        flattened = light.astype(numpy.float32)
        if self.remove_bias:
            flattened -= self.raw.black_level
            origmax = light.max()
        flattened /= luma
        if self.normalize:
            if origmax:
                flattened *= 1.0 / origmax
            if light.dtype.kind == 'u':
                # preserve unsignedness
                mn = 0
                mx = 1
            else:
                mn = -1
                mx = 1
        elif dtype().dtype.kind in 'iu':
            iinfo = numpy.iinfo(dtype)
            mn = iinfo.min
            mx = iinfo.max
        else:
            mn = mx = None
        if mn is not None or mx is not None:
            flattened = numpy.clip(flattened, mn, mx, out=flattened)

        if scale is not None:
            flattened *= scale
        if dtype is not numpy.float32:
            flattened = flattened.astype(dtype)

        return flattened


class ColorFlatImageRop(FlatImageRop):

    NAMED_COLOR_MATRICES = {
        'gb': numpy.array([
            [1, 0.125, 0.125],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        'zwo294-bluflat': numpy.array([
            [0, 0.5, 0.5],
            [0, 0.8, 0.2],
            [0, 0, 1],
        ]),
    }

    def __init__(self, raw=None, *p, **kw):
        color_matrix = kw.pop('color_matrix', None)
        self.color_matrix = self.NAMED_COLOR_MATRICES.get(color_matrix, color_matrix)
        super(ColorFlatImageRop, self).__init__(raw, *p, **kw)

    def _flat_luma(self, flat):
        if flat is None:
            return None

        if self._raw_pattern.max() <= 1:
            # Single-channel, it's simpler
            return super(ColorFlatImageRop, self)._flat_luma(flat)

        flat_luma = numpy.empty(flat.shape, dtype=numpy.float32)

        color_matrix = self.color_matrix
        if color_matrix is not None:
            rmtx = color_matrix[0]
            gmtx = color_matrix[1]
            bmtx = color_matrix[2]
        else:
            rmtx = gmtx = bmtx = None

        # Compute flat color balance to neutralize it afterwards
        ravg = numpy.average(flat[self.rmask_image])
        gavg = numpy.average(flat[self.gmask_image])
        bavg = numpy.average(flat[self.bmask_image])
        rgbavg = numpy.array([ravg, gavg, bavg])

        lavg = float(max(ravg, gavg, bavg))

        if color_matrix is not None:
            ravg = (rgbavg * rmtx).sum()
            gavg = (rgbavg * gmtx).sum()
            bavg = (rgbavg * bmtx).sum()

        ravg = ravg or lavg
        gavg = gavg or lavg
        bavg = bavg or lavg

        if color_matrix is None:
            def apply_color_matrix(image, mtx, mask):
                image[~mask] = 0
        else:
            def apply_color_matrix(image, mtx, mask):
                image[self.rmask_image] = image[self.rmask_image] * mtx[0]
                image[self.gmask_image] = image[self.gmask_image] * mtx[1]
                image[self.bmask_image] = image[self.bmask_image] * mtx[2]

        # Compute independent shapes per channel, but mantain a neutral intensity
        rimage = flat.copy()
        apply_color_matrix(rimage, rmtx, self.rmask_image)
        rluma = super(ColorFlatImageRop, self)._flat_luma(rimage, scale=lavg/ravg)
        flat_luma[self.rmask_image] = rluma[self.rmask_image]
        del rluma, rimage

        gimage = flat.copy()
        apply_color_matrix(gimage, gmtx, self.gmask_image)
        gluma = super(ColorFlatImageRop, self)._flat_luma(gimage, scale=lavg/gavg)
        flat_luma[self.gmask_image] = gluma[self.gmask_image]
        del gluma, gimage

        bimage = flat.copy()
        apply_color_matrix(bimage, bmtx, self.bmask_image)
        bluma = super(ColorFlatImageRop, self)._flat_luma(bimage, scale=lavg/bavg)
        flat_luma[self.bmask_image] = bluma[self.bmask_image]
        del bluma, bimage

        return flat_luma
