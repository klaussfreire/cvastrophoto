from __future__ import division, absolute_import

import unittest
import random
import numpy

from cvastrophoto.image import rgb


class TrackingRopTestBase:

    nstars = 60
    nfuzz = 3
    fuzz = 3
    tracking_class = None
    tracking_kwargs = {}
    shape = (512, 512)
    max_delta = 0.25
    min_mag = 8000
    max_mag = 65535

    def setUp(self):
        self.im_tpl = rgb.RGB(None, img=numpy.zeros(self.shape, dtype=numpy.uint16), linear=True, autoscale=False)

    def get_rop(self, **kw):
        for k, v in self.tracking_kwargs.items():
            kw.setdefault(k, v)
        return self.tracking_class(self.im_tpl, **kw)

    def get_starfield(self, seed, offset=(0,0)):
        rnd = random.Random(seed)
        shape = self.shape
        oy, ox = offset

        im = numpy.zeros(shape, dtype=numpy.uint16)

        for i in range(self.nstars):
            # Star coordinates and magnitude
            y, x = rnd.randint(1, shape[0]-2-self.fuzz-5), rnd.randint(1, shape[1]-2-self.fuzz-5)
            mag = rnd.randint(self.min_mag, self.max_mag)
            if i < self.nfuzz:
                y += rnd.randint(0, self.fuzz)
                x += rnd.randint(0, self.fuzz)
            y += oy
            x += ox

            # skip oob
            if not 0<x<shape[1] or not 0<y<shape[0]:
                continue

            # draw star
            im[y-1:y+2, x-1:x+2] = 8000 * mag / 65535
            im[y-1:y+2, x] = 32768 * mag / 65535
            im[y, x-1:x+2] = 32768 * mag / 65535
            im[y, x] = 65535 * mag / 65535

        return im

    def test_simple_offsets(self, **kw):
        for seed in range(1, 4):
            track = self.get_rop(**kw)
            im = self.get_starfield(seed)

            track.clear_cache()
            bias = track.detect(im)
            oy, ox = track.translate_coords(bias, 0, 0)
            self.assertAlmostEqual(oy, 0)
            self.assertAlmostEqual(ox, 0)

            for oseed in range(20, 23):
                track.clear_cache()
                rnd = random.Random(oseed+10)
                offset = (rnd.randint(0,5), rnd.randint(0,5))
                im = self.get_starfield(seed, offset=offset)
                bias = track.detect(im)
                oy, ox = track.translate_coords(bias, 0, 0)
                self.assertAlmostEqual(oy, -offset[0], delta=self.max_delta)
                self.assertAlmostEqual(ox, -offset[1], delta=self.max_delta)

    def test_transform_match(self, **kw):
        for seed in range(1, 4):
            track = self.get_rop(**kw)
            im = self.get_starfield(seed)

            track.clear_cache()
            bias = track.detect(im)

            for oseed in range(20, 23):
                track.clear_cache()
                rnd = random.Random(oseed+10)
                offset = (rnd.randint(0,5), rnd.randint(0,5))
                im = self.get_starfield(seed, offset=offset)
                matched = track.correct(im)
                self.assertTrue((matched == im).all())
