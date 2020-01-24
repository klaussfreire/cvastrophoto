# -*- coding: utf-8 -*-
import imageio
import logging
import numpy
import math
from collections import namedtuple

from .base import BaseImage
from cvastrophoto.util import srgb

logger = logging.getLogger(__name__)

Sizes = namedtuple(
    'Sizes',
    (
        'raw_height', 'raw_width',
        'height', 'width',
        'top_margin', 'left_margin',
        'iheight', 'iwidth',
        'pixel_aspect',
        'flip',
    )
)

class RGB(BaseImage):

    priority = 10

    def __init__(self, path, **kw):
        raw_template = kw.pop('raw_template', None)

        super(RGB, self).__init__(path, **kw)

        if raw_template is not None:
            self.set_raw_template(raw_template)

    @classmethod
    def from_gray(cls, img, **kw):
        rgb = numpy.empty(img.shape + (3,), dtype=img.dtype)
        rgb[:,:,0] = img
        rgb[:,:,1] = img
        rgb[:,:,2] = img
        return cls(None, img=rgb, **kw)

    def _open_impl(self, path):
        return RGBImage(
            path,
            img=self._kw.get('img'),
            margins=self._kw.get('margins'),
            flip=self._kw.get('flip'),
            daylight_whitebalance=self._kw.get('daylight_whitebalance'),
            linear=self._kw.get('linear'),
            autoscale=self._kw.get('autoscale'))

    def set_raw_template(self, raw):
        rimg = raw.rimg
        raw_image = rimg.raw_image
        shape = (raw_image.shape[0], raw_image.shape[1], 3)
        self._kw['margins'] = (
            rimg.sizes.top_margin,
            rimg.sizes.left_margin,
            shape[0] - rimg.sizes.top_margin - rimg.sizes.iheight,
            shape[1] - rimg.sizes.left_margin - rimg.sizes.iwidth,
        )
        self._kw['daylight_whitebalance'] = rimg.daylight_whitebalance

    @classmethod
    def supports(cls, path):
        return True


class RGBImage(object):

    RGB_PATTERN = numpy.array([[0, 1, 2]], numpy.uint8)
    L_PATTERN = numpy.array([[0]], numpy.uint8)

    black_level_per_channel = (0, 0, 0)
    daylight_whitebalance = (1.0, 1.0, 1.0)

    linear = False
    autoscale = True

    def __init__(self, path=None, img=None, margins=None, flip=None, daylight_whitebalance=None,
            linear=None, autoscale=None):
        self._path = path
        self._img = img
        self._raw_image = None
        self._margins = margins or (0, 0, 0, 0)
        self._flip = flip or 0
        self.img = None

        if daylight_whitebalance is not None:
            self.daylight_whitebalance = daylight_whitebalance
        if linear is not None:
            self.linear = linear
        if autoscale is not None:
            self.autoscale = autoscale

        # Open to load metadata
        self.raw_image

        # Close to free ram until needed
        self.close()

    @property
    def sizes(self):
        shape = self.raw_shape
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
        raw_image = self._raw_image
        if raw_image is None:
            if self.img is None:
                if self._img is not None:
                    self.img = self._img
                elif self._path is not None:
                    self.img = imageio.imread(self._path)
                else:
                    raise ValueError("Either path or image must be given")
            raw_image = self.img
            if len(raw_image.shape) == 3:
                # RGB
                self.raw_pattern = self.RGB_PATTERN
            else:
                # Gray
                self.raw_pattern = self.L_PATTERN
            raw_image = raw_image.reshape((
                raw_image.shape[0] * self.raw_pattern.shape[0],
                raw_image.shape[1] * self.raw_pattern.shape[1],
            ))
            if raw_image.dtype.char == 'B':
                # Transform to 16-bit, decode gamma
                if self.linear:
                    raw_image = raw_image.astype(numpy.uint16)
                    raw_image <<= 8
                else:
                    raw_image = raw_image.astype(numpy.float32)
                    raw_image *= 1.0 / 255.0
                    raw_image = srgb.decode_srgb(raw_image)
                    raw_image *= 65535
                    raw_image = numpy.clip(raw_image, 0, 65535, out=numpy.empty(
                        raw_image.shape, numpy.uint16))
            elif raw_image.dtype.char == 'f':
                # Transform to 16-bit
                scaled = raw_image
                maxval = scaled.max()
                if maxval > 0:
                    scaled = scaled * (65535.0 / maxval)
                raw_image = numpy.clip(scaled, 0, 65535).astype(numpy.uint16)
            elif raw_image.dtype.char == 'H':
                if self.autoscale or not self.linear:
                    scaled = raw_image.astype(numpy.float32)
                    if self.autoscale:
                        maxval = scaled.max()
                    else:
                        maxval = 65535
                    if maxval > 0:
                        scaled *= (1.0 / maxval)
                        if not self.linear:
                            scaled = srgb.decode_srgb(scaled)
                        scaled *= 65535
                else:
                    scaled = raw_image.copy()
                raw_image = scaled
            self._raw_image = raw_image
            self.raw_shape = raw_image.shape
        return raw_image

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

    def postprocess(self, params=None):
        raw_image_visible = self.raw_image_visible
        path, patw = self.raw_pattern.shape
        sizes = self.sizes
        postprocessed = raw_image_visible.reshape((
            sizes.iheight,
            sizes.iwidth,
            patw,
        ))
        return postprocessed

    def close(self):
        self._raw_image = None
        self.img = None
