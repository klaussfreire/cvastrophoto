from __future__ import absolute_import, division

import unittest

from .base import TrackingRopTestBase

from cvastrophoto.rops.tracking import flow


class OpticalFlowTrackingTest(TrackingRopTestBase, unittest.TestCase):
    tracking_class = flow.OpticalFlowTrackingRop
    tracking_kwargs = dict(track_distance=16, track_region=6, mask_open=False)
    test_simple_offsets = None

class DownsampledOpticalFlowTrackingTest(OpticalFlowTrackingTest):
    shape = (1024, 1024)
    max_delta = 1.25
    max_img_delta = 0.3
    tracking_kwargs = dict(downsample=2, track_distance=32, track_region=6, mask_open=False, order=1)
