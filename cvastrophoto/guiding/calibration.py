# -*- coding: utf-8 -*-
import logging
import time

from cvastrophoto.image import rgb


logger = logging.getLogger(__name__)


class CalibrationSequence(object):

    guide_exposure = 4.0

    drift_cycles = 2
    drift_steps = 10
    save_tracks = False

    def __init__(self, telescope, st4, guide_ccd, main_ccd, controller, tracker_class):
        self.tracker_class = tracker_class
        self.telescope = telescope
        self.st4 = st4
        self.guide_ccd = guide_ccd
        self.main_ccd = main_ccd
        self.controller = controller

    def run(self):
        # Get a reference picture out of the guide_ccd to use on the tracker_class
        self.guide_ccd.expose(self.guide_exposure)
        img = self.guide_ccd.pullImage()

        self.controller.reset()

        # First quick drift measurement to allow precise RA/DEC calibration
        drifty, driftx = self.measure_drift(img, 1, 'pre', self.combine_drift_avg)

    def combine_drift_avg(self, drifts):
        dy = sum(dy for dy, dx in drifts) / len(drifts)
        dx = sum(dx for dy, dx in drifts) / len(drifts)
        return (dy, dx)

    def measure_drift(self, ref_img, cycles, which, combine_mode):
        drifts = []
        for cycle in xrange(cycle):
            tracker = self.tracker_class(ref_img)

            offsets = []
            for step in xrange(self.drift_steps):
                t0 = time.time()
                self.guide_ccd.expose(self.guide_exposure)
                img = self.guide_ccd.pullImage()
                img.name = 'calibration_drift_%s_%d_%d' % (which, cycle, step)

                offset = tracker.detect(img.rimg.raw_image, img=img, save_tracks=self.save_tracks)
                offset = tracker.transform_coords(offset, 0, 0)
                offsets.append((offset, t0))

            # First offset will always be 0 since it's the calibration offset
            offsets = offsets[1:]

            # Compute average drift in pixels/s
            dt = offsets[-1][1] - offsets[0][1]
            dy = (offsets[-1][0][0] - offset[0][0][0])
            dx = (offsets[-1][0][1] - offset[0][0][1])
            drifts.append((dy/dt, dx/dt))

        return combine_mode(drifts)
