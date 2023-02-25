# -*- coding: utf-8 -*-
from __future__ import absolute_import

from past.builtins import xrange
import functools

import numpy.fft
import scipy.ndimage
import skimage.filters

from ..base import PerChannelRop
from ..tracking.extraction import ExtractPureStarsRop
from cvastrophoto.util import gaussian


class DebandingFilterRop(PerChannelRop):

    direction = 'both'
    hipass = 80
    mask_sigma = 1.0
    tile_size = 0

    def process_tile(self, data):
        if self.mask_sigma is not None:
            tdata = data.copy()
            avg = numpy.average(data)
            std = numpy.std(data)
            thr = avg + self.mask_sigma * std
            tdata[tdata > thr] = avg
        else:
            tdata = data
        fdata = numpy.fft.fft2(tdata)
        del tdata

        direction = self.direction[0].lower()
        if direction == 'v':
            fdata[1:, :] = 0
        elif direction == 'h':
            fdata[:, 1:] = 0
        else:
            fdata[1:, 1:] = 0
        fdata[:self.hipass, :self.hipass] = 0
        banding = numpy.fft.ifft2(fdata).real
        del fdata

        banding -= numpy.average(banding)
        debanded = data - banding
        del banding

        return numpy.clip(debanded, 0, data.max(), out=data)

    def process_channel(self, data, detected=None, channel=None, **kw):
        if self.tile_size:
            tile_size = self.tile_size
            canvas = numpy.zeros_like(data, dtype=numpy.float32)
            weights = numpy.zeros_like(data, dtype=numpy.float32)
            tile_weight = skimage.filters.window('hann', (tile_size, tile_size))
            tile_weight += 1.0e-10

            tasks = []
            for ys in xrange(0, data.shape[0] - tile_size//2, tile_size//2):
                for xs in xrange(0, data.shape[1] - tile_size//2, tile_size//2):
                    tile_data = data[ys:ys+tile_size, xs:xs+tile_size]
                    canvas_data = canvas[ys:ys+tile_size, xs:xs+tile_size]
                    weight_data = weights[ys:ys+tile_size, xs:xs+tile_size]
                    tasks.append((tile_data, canvas_data, weight_data))

            def threaded_task(task):
                tile_data, canvas_data, weight_data = task
                tile_data = tile_data.copy()
                tile_data = self.process_tile(tile_data.copy())
                return tile_data, canvas_data, weight_data

            if self.raw.default_pool is not None:
                imap = self.raw.default_pool.imap_unordered
            else:
                imap = functools.imap

            for tile_data, canvas_data, weight_data in imap(threaded_task, tasks):
                w = tile_weight[:tile_data.shape[0], :tile_data.shape[1]]
                canvas_data += w * tile_data
                weight_data += w

            canvas /= weights
            data[:] = canvas
            return data
        else:
            return self.process_tile(data)


class FlatDebandingFilterRop(DebandingFilterRop):

    mask_sigma = 0.5
    size = 128

    def process_channel(self, data, *p, **kw):
        mn = data.min()
        if mn <= 0:
            mn = min(1, data.max())
        if mn <= 0:
            mn = 1
        envelope = numpy.clip(
            gaussian.fast_gaussian(data, self.size, mode='nearest'),
            mn, None).astype(numpy.float32)
        flatdata = data.astype(numpy.float32) / envelope
        flatdata = super(FlatDebandingFilterRop, self).process_channel(flatdata, *p, **kw)
        flatdata *= envelope

        return numpy.clip(flatdata, data.min(), data.max(), out=flatdata)


class StarlessDebandingFilterRop(FlatDebandingFilterRop):

    mask_sigma = None

    def __init__(self, raw, **kw):
        self._extract_stars_kw = {k: kw.pop(k) for k in list(kw) if hasattr(ExtractPureStarsRop, k)}
        super(StarlessDebandingFilterRop, self).__init__(raw, **kw)

    def correct(self, data, *p, **kw):
        dmax = data.max()
        stars = ExtractPureStarsRop(self.raw, **self._extract_stars_kw).correct(data.copy())
        data -= numpy.clip(stars, None, data, out=stars)
        data = super(StarlessDebandingFilterRop, self).correct(data, *p, **kw)
        data += numpy.clip(stars, None, dmax - data)
        return data
