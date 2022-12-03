from __future__ import absolute_import, division

import unittest

from .base import TrackingRopTestBase

from cvastrophoto.rops.tracking import correlation


class CorrelationTrackingTest(TrackingRopTestBase, unittest.TestCase):
    tracking_class = correlation.CorrelationTrackingRop

class QuickCorrelationTrackingTest(CorrelationTrackingTest):
    tracking_kwargs = dict(track_distance=128)

class DownsampledCorrelationTrackingTest(CorrelationTrackingTest):
    tracking_kwargs = dict(downsample=2, track_distance=256)
    max_img_delta = 0.2
