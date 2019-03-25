# -*- coding: utf-8 -*-
import PIL.Image

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
        super(RGB, self).__init__(path, **kw)

    def _open_impl(self, path):
        return RGBImage(path)

    @classmethod
    def supports(cls, path):
        return True


class RGBImage(object):

    RGB_PATTERN = numpy.array([[0, 1, 2]], numpy.uint8)
    L_PATTERN = numpy.array([[0]], numpy.uint8)

    def __init__(self, path):
        self.img = PIL.Image.open(path)
        self._raw_image = None

    @property
    def sizes(self):
        shape = self.raw_image.shape
        pattern = self.raw_pattern.shape
        return Sizes(
            shape[0], shape[1],
            shape[0], shape[1],
            0, 0,
            shape[0], shape[1],
            pattern[0] / float(pattern[1]),
            0,
        )

    @property
    def raw_image(self):
        raw_image = self._raw_image
        if raw_image is None:
            raw_image = numpy.array(self.img)
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
                raw_image = raw_image.astype(numpy.float32)
                raw_image *= 1.0 / 255.0
                raw_image = srgb.decode_srgb(raw_image)
                raw_image *= 65535
                raw_image = numpy.clip(raw_image, 0, 65535, out=numpy.empty(
                    raw_image.shape, numpy.uint16))
            self._raw_image = raw_image
        return raw_image

    @property
    def raw_image_visible(self):
        return self.raw_image

    @property
    def raw_colors(self):
        shape = self.raw_image.shape
        pattern = self.raw_pattern
        return numpy.tile(
            pattern,
            (
                (shape[0] + pattern.shape[0]-1) / pattern.shape[0],
                (shape[1] + pattern.shape[1]-1) / pattern.shape[1],
            )
        )[:shape[0], :shape[1]]

    def postprocess(self, params=None):
        raw_image = self.raw_image
        pattern = self.raw_pattern
        postprocessed = raw_image.reshape((
            raw_image.shape[0] / pattern.shape[0],
            raw_image.shape[1] / pattern.shape[1],
            self.raw_pattern.shape[1],
        ))
        return postprocessed

    def close(self):
        self._raw_image = None
        self.img.close()
