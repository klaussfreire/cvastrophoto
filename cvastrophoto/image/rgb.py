# -*- coding: utf-8 -*-
from __future__ import division

import imageio
import logging
import numpy
import math

from .base import BaseImage, Sizes
from cvastrophoto.util import srgb

logger = logging.getLogger(__name__)

class RGB(BaseImage):

    priority = 10
    concrete = True

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

    def _open_impl(self, path, img=None, copy=False):
        return RGBImage(
            path,
            img=self._kw.get('img', img),
            margins=self._kw.get('margins'),
            flip=self._kw.get('flip'),
            daylight_whitebalance=self._kw.get('daylight_whitebalance'),
            linear=self._kw.get('linear'),
            autoscale=self._kw.get('autoscale'),
            copy=copy)

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

    def denoise(self, *p, **kw):
        if kw.get('raw_image') is None and kw.get('signed', True):
            self.rimg.ensure_signed()
        return super(RGB, self).denoise(*p, **kw)


class RGBImage(object):

    RGB_PATTERN = numpy.array([[0, 1, 2]], numpy.uint8)
    L_PATTERN = numpy.array([[0]], numpy.uint8)

    black_level_per_channel = (0, 0, 0)
    daylight_whitebalance = (1.0, 1.0, 1.0)

    linear = None
    autoscale = True

    def __init__(self, path=None, img=None, margins=None, flip=None, daylight_whitebalance=None,
            linear=None, autoscale=None, copy=False):
        self._path = path
        self._img = img
        self._copy = copy
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
        wv = shape[1] - (l + r) * patw // path
        return Sizes(
            shape[0], shape[1],
            hv, wv,
            t, l * patw // path,
            hv, wv * path // patw,
            path / float(patw),
            self._flip,
        )

    def ensure_signed(self):
        raw_image = self.raw_image
        if raw_image.dtype.kind == 'u':
            self._raw_image = raw_image.astype(
                {
                    1: numpy.int16,
                    2: numpy.int32,
                    4: numpy.int64,
                    8: numpy.int64,
                }[raw_image.dtype.itemsize]
            )

    @property
    def raw_image(self):
        raw_image = self._raw_image
        linear = self.linear
        if raw_image is None:
            if self.img is None:
                if self._img is not None:
                    if self._copy:
                        self.img = self._img.copy()
                    else:
                        self.img = self._img
                elif self._path is not None:
                    self.img = imageio.imread(self._path)
                else:
                    raise ValueError("Either path or image must be given")
            raw_image = self.img
            if len(raw_image.shape) == 3 and raw_image.shape[2] != 1:
                if raw_image.shape[2] == 4:
                    # RGBA, discard alpha
                    raw_image = raw_image[:,:,:3]
                    self.raw_pattern = self.RGB_PATTERN
                elif raw_image.shape[2] == 3:
                    # RGB
                    self.raw_pattern = self.RGB_PATTERN
                elif raw_image.shape[2] == 2:
                    # LA, discard alpha
                    raw_image = raw_image[:,:,0]
                    self.raw_pattern = self.L_PATTERN
                else:
                    raise ValueError("Unsupported image shape %r" % raw_image.shape)

                if not raw_image.flags.contiguous:
                    # We need it contiguous since we do reshape a lot
                    raw_image = numpy.ascontiguousarray(raw_image)
            else:
                # Gray
                self.raw_pattern = self.L_PATTERN
            raw_image = raw_image.reshape((
                raw_image.shape[0] * self.raw_pattern.shape[0],
                raw_image.shape[1] * self.raw_pattern.shape[1],
            ))
            if linear is None:
                linear = raw_image.dtype.char in 'dfIL'
            if raw_image.dtype.char == 'B':
                # Transform to 16-bit, decode gamma
                if linear:
                    raw_image = raw_image.astype(numpy.uint16)
                    raw_image <<= 8
                else:
                    nraw_image = raw_image.astype(numpy.uint16)
                    nraw_image <<= 8
                    nraw_image |= raw_image
                    raw_image = nraw_image
                    raw_image = srgb.decode_srgb(raw_image)
                    del nraw_image
            elif raw_image.dtype.char in 'df':
                # Transform to 16-bit
                scaled = raw_image
                maxval = scaled.max()
                if maxval > 0:
                    in_scale = 65535.0 / maxval
                else:
                    in_scale = 1.0
                if not linear:
                    raw_image = srgb.decode_srgb(
                        scaled.copy(),
                        in_scale=in_scale / 65535.0, out_scale=65535.0, out_max=65535.0)
                else:
                    raw_image = scaled * in_scale
                    raw_image = numpy.clip(raw_image, 0, 65535, out=raw_image)
                if raw_image.dtype.char == 'f':
                    raw_image = raw_image.astype(numpy.uint16)
            elif raw_image.dtype.char in 'HIL':
                if self.autoscale or not linear:
                    scaled = raw_image.astype(numpy.float32)
                    if self.autoscale:
                        maxval = scaled.max()
                    else:
                        maxval = dict(H=65535, I=0xFFFFFFFF, L=0xFFFFFFFFFFFFFFFF)[raw_image.dtype.char]
                    if maxval > 0:
                        if not linear:
                            scaled = srgb.decode_srgb(scaled, in_scale=1.0 / maxval, out_scale=65535)
                        else:
                            scaled *= 65535.0 / maxval
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
                (shape[0] + pattern.shape[0]-1) // pattern.shape[0],
                (shape[1] + pattern.shape[1]-1) // pattern.shape[1],
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


class Templates:
    pass

Templates.LUMINANCE = RGB(None, img=numpy.zeros((1, 1), dtype=numpy.uint16), linear=True, autoscale=False)
Templates.RGB = RGB(None, img=numpy.zeros((1, 1, 3), dtype=numpy.uint16), linear=True, autoscale=False)

Templates.LUMINANCE.demargin_safe = False
Templates.RGB.demargin_safe = False
