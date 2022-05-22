from __future__ import absolute_import, division

import unittest

from .base import TrackingRopTestBase

from cvastrophoto.rops.tracking import grid


class GridTrackingTest(TrackingRopTestBase, unittest.TestCase):
    shape = (1024, 1024)
    tracking_class = grid.GridTrackingRop
    tracking_kwargs = dict(track_distance=128)


class SparseGridTrackingTest(GridTrackingTest):
    nstars = 18
    nfuzz = 1
