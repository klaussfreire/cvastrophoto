# -*- coding: utf-8 -*-
from __future__ import division

import numpy
import math
import logging

from astropy.io import fits

from .base import BaseImage, Sizes

from cvastrophoto.util import demosaic, srgb

logger = logging.getLogger(__name__)

class Fits(BaseImage):

    priority = 1
    concrete = True

    def _open_impl(self, path):
        return FitsImage(
            path,
            linear=self._kw.get('linear'),
            autoscale=self._kw.get('autoscale'),
            margins=self._kw.get('margins'),
            mode=self._kw.get('mode'))

    @classmethod
    def supports(cls, path):
        return path.rsplit('.', 1)[-1].lower() in ('fit', 'fits')

    @property
    def fits_header(self):
        fits_header = getattr(self, '_fits_header', None)
        if fits_header is None:
            fits_header = self._fits_header = self.rimg.fits_header
        return fits_header


class FitsImage(object):

    PATTERNS = {
        'L': numpy.array([[0]], numpy.uint8),
        'RGGB': numpy.array([[0,1],[1,2]], numpy.uint8),
        'GRBG': numpy.array([[1,0],[2,1]], numpy.uint8),
        'GBRG': numpy.array([[1,2],[0,1]], numpy.uint8),
        'BGGR': numpy.array([[2,1],[1,0]], numpy.uint8),
        'RGB': numpy.array([[0,1,2]], numpy.uint8),
    }

    black_level_per_channel = (0, 0, 0)
    daylight_whitebalance = (1.0, 1.0, 1.0)

    linear = None
    autoscale = False

    def __init__(self, path=None, hdul=None, margins=None, flip=None, daylight_whitebalance=None,
            linear=None, autoscale=None, mode=None):
        self._path = path

        if hdul is None:
            kw = {}
            if mode is not None:
                kw['mode'] = mode
                if mode == 'update':
                    kw['scale_back'] = True
            hdul = fits.open(path, **kw)

        self.hdul = hdul

        if linear is not None:
            self.linear = linear
        if autoscale is not None:
            self.autoscale = autoscale

        header = hdul[0].header
        if 'BAYERPAT' in header:
            pattern_name = header['BAYERPAT'].strip().upper()
        elif 'NAXIS' in header and header['NAXIS'] == 3:
            pattern_name = 'RGB'
        else:
            pattern_name = 'L'

        if pattern_name in self.PATTERNS:
            pattern = self.PATTERNS[pattern_name]
        elif set(pattern_name) <= set('RGB'):
            cpattern = numpy.array(list(pattern_name))
            dimsq = int(math.sqrt(cpattern.size))
            if dimsq ** 2 == cpattern.size:
                cpattern = cpattern.reshape((dimsq, dimsq))
            pattern = numpy.zeros(cpattern.shape, numpy.uint8)
            pattern[cpattern == 'R'] = 0
            pattern[cpattern == 'G'] = 1
            pattern[cpattern == 'B'] = 2
        else:
            raise ValueError('Unrecognized bayer pattern')

        raw_shape = self.hdul[0].shape
        if len(raw_shape) == 3 and raw_shape[0] <= 3 and raw_shape[2] > 3:
            raw_shape = (raw_shape[1], raw_shape[2] * raw_shape[0])

        self.raw_pattern = pattern
        self.raw_shape = raw_shape
        self._margins = margins or (2, 2, 2, 2)
        self._flip = flip or 0
        self._raw_image = None

        if daylight_whitebalance is not None:
            self.daylight_whitebalance = daylight_whitebalance

    @property
    def sizes(self):
        shape = self.raw_shape
        path, patw = self.raw_pattern.shape
        t, l, b, r = self._margins
        hv = shape[0] - (t + b)
        wv = shape[1] - (l + r) * patw // path
        return Sizes(
            shape[0], shape[1],
            hv, wv,
            t, l * patw // path,
            hv, wv * path // patw,
            path / float(patw),
            self._flip,
        )

    @property
    def raw_image(self):
        if self._raw_image is None:
            linear = self.linear
            if linear is None:
                linear = True

            im = self.hdul[0].data

            if len(im.shape) == 3 and im.shape[0] <= 3 and im.shape[2] > 3:
                im2 = numpy.empty(im.shape[1:] + im.shape[:1], dtype=im.dtype)
                for c in range(im.shape[0]):
                    im2[:,:,c] = im[c]
                im = im2.reshape(self.raw_shape)
                del im2

            if self.autoscale or not linear:
                scaled = im.astype(numpy.float32)
                if self.autoscale:
                    maxval = scaled.max()
                else:
                    maxval = dict(H=65535, I=0xFFFFFFFF, L=0xFFFFFFFFFFFFFFFF, f=1.0, d=1.0)[im.dtype.char]
                if maxval > 0:
                    scaled *= (1.0 / maxval)
                    if not linear:
                        scaled = srgb.decode_srgb(scaled)
                    scaled *= 65535
                im = scaled
            elif im.dtype.kind in 'df' and im.max() <= 1:
                # ROPs work better in the 16-bit data range
                im = im * 65535.0
            self._raw_image = im
        return self._raw_image

    @property
    def raw_image_visible(self):
        if not any(self._margins):
            return self.raw_image
        else:
            sizes = self.sizes
            return self.raw_image[
                sizes.top_margin:sizes.top_margin+sizes.height,
                sizes.left_margin:sizes.left_margin+sizes.width]

    @property
    def raw_colors(self):
        shape = self.raw_shape
        pattern = self.raw_pattern
        return numpy.tile(
            pattern,
            (
                (shape[0] + pattern.shape[0]-1) // pattern.shape[0],
                (shape[1] + pattern.shape[1]-1) // pattern.shape[1],
            )
        )[:shape[0], :shape[1]]

    @property
    def fits_header(self):
        return self.hdul[0].header

    def postprocess(self, params=None):
        if self.raw_pattern.size == 1:
            # Luminance data
            raw_image_visible = self.raw_image_visible
            return raw_image_visible.reshape(raw_image_visible.shape + (1,))
        elif self.raw_pattern is self.PATTERNS['RGB']:
            # RGB data
            raw_image_visible = self.raw_image_visible
            h, w = raw_image_visible.shape
            return raw_image_visible.reshape((h, w//3, 3))

        # Otherwise, bayered multichannel data
        postprocessed = demosaic.demosaic(self.raw_image, self.raw_pattern)

        sizes = self.sizes
        postprocessed = postprocessed[
            sizes.top_margin:sizes.top_margin+sizes.height,
            sizes.left_margin:sizes.left_margin+sizes.width]

        return postprocessed

    def close(self):
        if self.hdul is not None:
            self.hdul.close()
        self.hdul = None
        self._raw_image = None
