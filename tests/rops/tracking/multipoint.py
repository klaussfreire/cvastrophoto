from __future__ import absolute_import, division

import unittest

from .base import TrackingRopTestBase

from cvastrophoto.rops.tracking import multipoint


class MultipointTrackingTest(TrackingRopTestBase, unittest.TestCase):
    tracking_class = multipoint.MultipointTrackingRop
    tracking_kwargs = dict(track_distance=64)

class DownsampledMultipointTrackingTest(MultipointTrackingTest):
    shape = (1024, 1024)
    max_delta = 1.25
    tracking_kwargs = dict(downsample=2, track_distance=128)

class MultipointGuideTrackingTest(TrackingRopTestBase, unittest.TestCase):
    tracking_class = multipoint.MultipointGuideTrackingRop
    tracking_kwargs = dict(track_distance=64)

class SparseMultipointGuideTrackingTest(MultipointGuideTrackingTest):
    nstars = 1
