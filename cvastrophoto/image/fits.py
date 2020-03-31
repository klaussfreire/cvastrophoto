# -*- coding: utf-8 -*-
import numpy
import math
import logging

from astropy.io import fits
import scipy.ndimage.filters

from .base import BaseImage
from .rgb import Sizes

logger = logging.getLogger(__name__)

class Fits(BaseImage):

    priority = 1
    concrete = True

    def _open_impl(self, path):
        return FitsImage(path)

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
    }

    black_level_per_channel = (0, 0, 0)
    daylight_whitebalance = (1.0, 1.0, 1.0)

    linear = False
    autoscale = True

    def __init__(self, path=None, hdul=None, margins=None, flip=None, daylight_whitebalance=None):
        self._path = path

        if hdul is None:
            hdul = fits.open(path)

        self.hdul = hdul

        header = hdul[0].header
        if 'BAYERPAT' in header:
            pattern_name = header['BAYERPAT'].strip().upper()
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

        self.raw_pattern = pattern
        self.raw_shape = hdul[0].shape
        self._margins = margins or (2, 2, 2, 2)
        self._flip = flip or 0

        if daylight_whitebalance is not None:
            self.daylight_whitebalance = daylight_whitebalance

    @property
    def sizes(self):
        shape = self.hdul[0].shape
        path, patw = self.raw_pattern.shape
        t, l, b, r = self._margins
        hv = shape[0] - (t + b)
        wv = shape[1] - (l + r) * patw / path
        return Sizes(
            shape[0], shape[1],
            hv, wv,
            t, l * patw / path,
            hv, wv * path / patw,
            path / float(patw),
            self._flip,
        )

    @property
    def raw_image(self):
        return self.hdul[0].data

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
                (shape[0] + pattern.shape[0]-1) / pattern.shape[0],
                (shape[1] + pattern.shape[1]-1) / pattern.shape[1],
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

        # Otherwise, multichannel data
        raw_image = self.raw_image
        raw_pattern = self.raw_pattern
        channels = raw_pattern.max() + 1

        postprocessed = numpy.zeros(raw_image.shape + (channels,), raw_image.dtype)
        mask = numpy.zeros(postprocessed.shape, numpy.bool8)

        path, patw = self.raw_pattern.shape
        for y in xrange(path):
            for x in xrange(patw):
                postprocessed[y::path, x::patw, raw_pattern[y,x]] = raw_image[y::path, x::patw]
                mask[y::path, x::patw, raw_pattern[y,x]] = True

        filtered = numpy.empty(postprocessed.shape, numpy.float32)
        filtered_mask = numpy.empty(mask.shape, numpy.float32)
        for c in xrange(channels):
            filtered[:,:,c] = scipy.ndimage.filters.uniform_filter(
                postprocessed[:,:,c].astype(filtered.dtype), (path, patw), mode='constant')
            filtered_mask[:,:,c] = scipy.ndimage.filters.uniform_filter(
                mask[:,:,c].astype(filtered_mask.dtype), (path, patw), mode='constant')

        filtered_mask = numpy.clip(filtered_mask, 0.001, None, out=filtered_mask)
        filtered /= filtered_mask
        filtered = numpy.clip(filtered, raw_image.min(), raw_image.max(), out=filtered)

        postprocessed[~mask] = filtered[~mask]

        sizes = self.sizes
        postprocessed = postprocessed[
            sizes.top_margin:sizes.top_margin+sizes.height,
            sizes.left_margin:sizes.left_margin+sizes.width]

        return postprocessed

    def close(self):
        self.hdul = None
