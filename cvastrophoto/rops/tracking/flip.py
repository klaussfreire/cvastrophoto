# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from past.builtins import xrange
import os.path
import numpy
import scipy.ndimage
import skimage.transform
import logging
import PIL.Image

from cvastrophoto.image import rgb, Image

from .base import BaseTrackingRop
from . import extraction

logger = logging.getLogger(__name__)

class PierFlipTrackingRop(BaseTrackingRop):

    tracking_cache = None
    reference = None

    def set_reference(self, data, img=None):
        if data is not None:
            if isinstance(data, tuple) and len(data) == 2:
                pass
            else:
                if img is None and isinstance(data, Image):
                    img = data
                self.reference = self._detect(data, img=img)
        else:
            self.reference = None

    def clear_cache(self):
        self.tracking_cache = None

    def _tracking_key(self, data, hint):
        return (
            getattr(data, 'name', id(data)),
            hint[:4] if hint is not None else None,
        )

    def _parse_pierside(self, pierside):
        if pierside is not None:
            pierside = pierside.upper()[0]
        return pierside

    def _detect(self, data, img=None):
        headers = getattr(img or data, 'fits_header', None)
        if headers is not None:
            return self._parse_pierside(headers.get('PIERSIDE'))

    def detect(self, data, pierside=None, img=None, save_tracks=None, set_data=None, luma=None, **kw):
        if self.tracking_cache is None:
            self.tracking_cache = {}

        if img is None and isinstance(data, Image):
            img = data

        tracking_key = self._tracking_key(img or data, self.reference)
        if pierside is None:
            pierside = self.tracking_cache.get(tracking_key)

        if pierside is None:
            pierside = self._detect(data, img=img)

        return self.tracking_cache.setdefault(tracking_key, pierside)

    def translate_coords(self, bias, y, x):
        raise NotImplementedError

    def correct_with_transform(self, data, pierside=None, img=None, save_tracks=None, **kw):
        dataset = rvdataset = data
        if isinstance(data, list):
            data = data[0]
        else:
            dataset = [data]

        if pierside is None:
            pierside = self.detect(data, save_tracks=save_tracks, img=img)
        else:
            pierside = self._parse_pierside(pierside)

        if pierside is not None and self.reference is not None and pierside != self.reference:
            logger.info("Flipping %s pier side %r reference %r", img, pierside, self.reference)

            raw_pattern = self._raw_pattern
            raw_sizes = self._raw_sizes
            pattern_shape = raw_pattern.shape
            ysize, xsize = pattern_shape

            for sdata in dataset:
                if sdata is None:
                    # Multi-component data sets might have missing entries
                    continue

                for yoffs in xrange(ysize):
                    for xoffs in xrange(xsize):
                        sdata[yoffs::ysize, xoffs::xsize] = sdata[yoffs::ysize, xoffs::xsize][::-1,::-1].copy()
        elif pierside is not None and self.reference is None:
            self.reference = pierside

        return rvdataset, None
