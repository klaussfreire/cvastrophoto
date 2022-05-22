from __future__ import absolute_import, division

import unittest
import numpy
import random
from skimage import morphology

from .base import TrackingRopTestBase

from cvastrophoto.rops.tracking import centroid


class CentroidTrackingTest(TrackingRopTestBase, unittest.TestCase):
    tracking_class = centroid.CentroidTrackingRop
    shape = (512, 512)

    planet_size = 16
    margin = 128

    def get_starfield(self, seed, offset=(0,0)):
        rnd = random.Random(seed)
        shape = self.shape
        oy, ox = offset

        im = numpy.zeros(shape, dtype=numpy.uint16)
        d = morphology.disk(self.planet_size)

        # Planet coordinates
        y = rnd.randint(self.margin, shape[0]-self.margin-d.shape[0])
        x = rnd.randint(self.margin, shape[1]-self.margin-d.shape[1])
        y += oy
        x += ox

        # draw disk
        im[y:y+d.shape[0], x:x+d.shape[1]] = d.astype(im.dtype) * 65535

        return im

class BigImageCentroidTrackingTest(CentroidTrackingTest):
    shape = (1024, 1024)
    margin = 384
